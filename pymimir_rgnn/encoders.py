import torch

from array import array
from collections.abc import Iterable
import pymimir as mm
from typing import Any
from weakref import WeakKeyDictionary

from .bases import Encoder, EncodedLists, EncodedTensors, EncodingContext
from .utils import get_action_name, get_atom_name, get_effect_name, get_effect_relation_name, get_predicate_name, relations_to_tensors


# Static atoms are immutable for a problem. Keep their already-grouped,
# problem-local object IDs in compact 32-bit arrays outside encoder/model
# instances so model serialization never retains parsed planning problems.
# Weak keys also let entries disappear with their corresponding problem.
_StaticRelationGroups = tuple[tuple[str, array], ...]
_STATE_STATIC_RELATION_CACHE: WeakKeyDictionary[mm.Problem, _StaticRelationGroups] = WeakKeyDictionary()
_EQUALITY_PREDICATE_NAME = "="


def _type_relation_name(
    type_name: str,
    suffix: str,
    *,
    expressive: bool = False,
) -> str:
    prefix = "expressive_" if expressive else ""
    return f"{prefix}type_relation_{type_name}{suffix}"


def _get_type_relations(
    context: EncodingContext,
    suffix: str,
    *,
    expressive: bool = False,
) -> tuple[tuple[str, list[int]], ...]:
    """Construct unary facts for every declared type and its ancestors."""
    domain = context.problem.domain
    if not domain.uses_typing:
        return ()
    return tuple(
        (
            _type_relation_name(type_name, suffix, expressive=expressive),
            [
                context.get_object_id(obj)
                for obj in context.problem.all_objects
                if domain.is_type_compatible(obj.type_name, type_name)
            ],
        )
        for type_name in domain.type_hierarchy
    )


def _get_equality_relation(
    context: EncodingContext,
    suffix: str,
    *,
    expressive: bool = False,
) -> tuple[str, list[int]] | None:
    """Represent semantic object equality as a reflexive graph relation."""
    if not context.problem.domain.uses_equality:
        return None
    prefix = "expressive_" if expressive else ""
    relation_name = f"{prefix}relation_{_EQUALITY_PREDICATE_NAME}{suffix}"
    repeat = 4 if expressive else 2
    ids = [
        context.get_object_id(obj)
        for obj in context.problem.all_objects
        for _ in range(repeat)
    ]
    return relation_name, ids


def _extend_relation(
    encoding: EncodedLists,
    relation_name: str,
    ids: list[int],
) -> None:
    if relation_name in encoding.flattened_relations:
        encoding.flattened_relations[relation_name].extend(ids)
    else:
        encoding.flattened_relations[relation_name] = ids


def _cache_static_relation_groups(
    problem: mm.Problem,
    static_atoms: Iterable[mm.GroundAtom],
    context: EncodingContext,
) -> _StaticRelationGroups:
    """Group and cache materialized static atoms using problem-local node IDs."""
    grouped_ids: dict[str, array] = {}
    for atom in static_atoms:
        predicate_name = atom.predicate.name
        if predicate_name == _EQUALITY_PREDICATE_NAME:
            continue
        if predicate_name not in grouped_ids:
            # Keep a key even for true nullary predicates, whose ID list is empty.
            grouped_ids[predicate_name] = array("i")
        grouped_ids[predicate_name].extend(
            context.get_object_id(obj) - context.id_offset
            for obj in atom.objects
        )

    cached_groups = tuple(grouped_ids.items())
    _STATE_STATIC_RELATION_CACHE[problem] = cached_groups
    return cached_groups


def _get_dynamic_atoms(state: mm.State) -> tuple[mm.GroundAtom, ...]:
    """Return all state-dependent fluent and derived atoms."""
    return state.fluent_atoms + state.derived_atoms


def _get_all_atoms(state: mm.State) -> tuple[mm.GroundAtom, ...]:
    """Return all enumerable true atoms except semantic equality facts."""
    return tuple(
        atom
        for atom in state.atoms
        if atom.predicate.name != _EQUALITY_PREDICATE_NAME
    )


class StateEncoder(Encoder):
    """Encoder for planning states.

    This encoder transforms a planning state (which contains atoms/facts that are
    currently true) into nodes and relations for the graph neural network.
    Objects become nodes, and atoms become relations between objects.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            Predicate relations followed by namespaced unary type relations.
        """
        relations = [
            (get_predicate_name(predicate, False, True, self.suffix), predicate.arity)
            for predicate in domain.predicates
        ]
        if domain.uses_typing:
            relations.extend(
                (_type_relation_name(type_name, self.suffix), 1)
                for type_name in domain.type_hierarchy
            )
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.State), f'StateEncoder expected a State, got {type(input_value)}'

        dynamic_atoms = _get_dynamic_atoms(input_value)
        problem = input_value.problem

        for type_relation in _get_type_relations(context, self.suffix):
            _extend_relation(encoding, *type_relation)

        equality_relation = _get_equality_relation(context, self.suffix)
        if equality_relation is not None:
            _extend_relation(encoding, *equality_relation)

        static_relation_groups = _STATE_STATIC_RELATION_CACHE.get(problem)
        if static_relation_groups is None:
            static_relation_groups = _cache_static_relation_groups(
                problem,
                problem.initial_static_atoms,
                context,
            )

        # Static atoms are the same in every state of a problem.  Add their
        # cached problem-local IDs at this instance's global node offset.
        for predicate_name, local_object_ids in static_relation_groups:
            relation_name = f'relation_{predicate_name}{self.suffix}'
            offset_object_indices = [local_id + context.id_offset for local_id in local_object_ids]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = offset_object_indices
            else:
                encoding.flattened_relations[relation_name].extend(offset_object_indices)

        # Fluent and derived atoms vary from state to state and must be
        # materialized on every encoding.
        for atom in dynamic_atoms:
            relation_name = get_atom_name(atom, state, False, self.suffix)
            dynamic_object_indices = [context.get_object_id(obj) for obj in atom.objects]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = dynamic_object_indices
            else:
                encoding.flattened_relations[relation_name].extend(dynamic_object_indices)


class GoalEncoder(Encoder):
    """Encoder for goal conditions.

    This encoder transforms a goal condition (conjunctive condition of literals
    that must be satisfied) into relations for the graph neural network.
    It creates both goal-specific relations and marks which atoms are true/false
    in the current state relative to the goal.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for goal conditions.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for goal predicates,
            including both true and false variants.
        """
        relations = []
        relations.extend([(get_predicate_name(predicate, True, False, self.suffix), predicate.arity) for predicate in domain.predicates])
        relations.extend([(get_predicate_name(predicate, True, True, self.suffix), predicate.arity) for predicate in domain.predicates])
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'

        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
            assert literal.is_positive, 'Only positive literals are supported.'
            atom = literal.atom
            relation_name = get_atom_name(atom, state, True, self.suffix)
            object_indices = [context.get_object_id(obj) for obj in atom.objects]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = object_indices
            else:
                encoding.flattened_relations[relation_name].extend(object_indices)



class GroundActionsEncoder(Encoder):
    """Encoder for ground actions.

    This encoder transforms a sequence of available ground actions into nodes and
    relations for the graph neural network. Each action becomes a new node,
    and relations connect actions to their parameter objects.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the ground actions encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for actions.

        Args:
            domain: The PDDL domain containing action definitions.

        Returns:
            List of (relation_name, arity) pairs for all actions, where arity
            is the action's parameter count plus 1 (for the action node itself).
        """
        return [(get_action_name(action, self.suffix), action.arity + 1) for action in domain.actions]

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, (list, tuple)), f'GroundActionsEncoder expected a sequence, got {type(input_value)}'

        for action in input_value:
            assert isinstance(action, mm.GroundAction), f'Expected a GroundAction in the sequence, got {type(action)}'
            relation_name = get_action_name(action, self.suffix)
            action_id = context.new_action_id()
            term_ids = [action_id] + [context.get_object_id(obj) for obj in action.objects]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = term_ids
            else:
                encoding.flattened_relations[relation_name].extend(term_ids)


class TransitionEffectsEncoder(Encoder):
    """Encoder for transition effects.

    This encoder transforms action effects (lists of literals describing state
    changes) into nodes and relations for the graph neural network. Each
    transition becomes a new node, with relations connecting it to affected atoms.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the transition effects encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for transition effects.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for effect relations,
            including positive/negative effects and goal-affecting variants.
        """
        relations = []
        relations.extend([(get_effect_name(predicate, True, False, self.suffix), predicate.arity + 1) for predicate in domain.predicates])
        relations.extend([(get_effect_name(predicate, False, False, self.suffix), predicate.arity + 1) for predicate in domain.predicates])
        relations.extend([(get_effect_name(predicate, True, True, self.suffix), predicate.arity + 1) for predicate in domain.predicates])
        relations.extend([(get_effect_name(predicate, False, True, self.suffix), predicate.arity + 1) for predicate in domain.predicates])
        relations.append((get_effect_relation_name(self.suffix), 2))
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, tuple) and len(input_value) == 3, (
            f'TransitionEffectsEncoder expected a 3-tuple (effects_list, effects_relations, goal_condition), '
            f'got {type(input_value)}'
        )
        effects_list, effects_relations, goal_condition = input_value
        assert isinstance(goal_condition, mm.GroundConjunctiveCondition), (
            f'TransitionEffectsEncoder expected a GroundConjunctiveCondition as the third element, '
            f'got {type(goal_condition)}'
        )
        goal_dictionary = {goal_literal.atom: goal_literal for goal_literal in goal_condition}
        num_transitions = len(effects_list)

        transition_index_to_id: dict[int, int] = dict()

        for transition_index, effects in enumerate(effects_list):
            assert isinstance(effects, list) or isinstance(effects, tuple), 'Expected a list of lists of ground literals.'
            transition_id = context.new_action_id()
            transition_index_to_id[transition_index] = transition_id

            for effect_literal in effects:
                assert isinstance(effect_literal, mm.GroundLiteral), 'Expected a list of lists of ground literals.'
                effect_atom = effect_literal.atom
                effect_name = get_effect_name(effect_atom.predicate, effect_literal.is_positive, False, self.suffix)
                object_ids = [transition_id] + [context.get_object_id(obj) for obj in effect_atom.objects]
                if effect_name not in encoding.flattened_relations:
                    encoding.flattened_relations[effect_name] = object_ids
                else:
                    encoding.flattened_relations[effect_name].extend(object_ids)

                # Add how this transition affects the goal
                if effect_atom in goal_dictionary:
                    goal_literal = goal_dictionary[effect_atom]
                    assert isinstance(goal_literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
                    assert goal_literal.is_positive, 'Only positive literals are supported in the goal condition.'
                    assert effect_atom == goal_literal.atom
                    goal_effect_name = get_effect_name(effect_atom.predicate, effect_literal.is_positive, True, self.suffix)
                    goal_object_ids = [transition_id] + [context.get_object_id(obj) for obj in effect_atom.objects]
                    if goal_effect_name not in encoding.flattened_relations:
                        encoding.flattened_relations[goal_effect_name] = goal_object_ids
                    else:
                        encoding.flattened_relations[goal_effect_name].extend(goal_object_ids)

        # Add relations between transitions if provided
        for from_index, to_index in effects_relations:
            assert isinstance(from_index, int) and isinstance(to_index, int), 'Effect relations must be pairs of integers.'
            assert from_index < num_transitions, f'Invalid from_index {from_index} in effect relations.'
            assert to_index < num_transitions, f'Invalid to_index {to_index} in effect relations.'
            from_id = transition_index_to_id[from_index]
            to_id = transition_index_to_id[to_index]
            effect_relation_name = get_effect_relation_name(self.suffix)
            relation_ids = [from_id, to_id]
            if effect_relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[effect_relation_name] = relation_ids
            else:
                encoding.flattened_relations[effect_relation_name].extend(relation_ids)


class VirtualNodeEncoder(Encoder):
    """Encoder for virtual nodes.

    This encoder adds a virtual node to the representation.
    All original objects are connected to a new virtual node.
    This can help the graph neural network to better capture global context.
    """
    def __init__(self) -> None:
        """Initialize the virtual node encoder."""
        super().__init__()
        self.link_name = 'virtual_node_link'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [(self.link_name, 2)]

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        virtual_id = context.new_virtual_id()
        for obj in state.problem.declared_objects:
            object_id = context.get_object_id(obj)
            if self.link_name not in encoding.flattened_relations:
                encoding.flattened_relations[self.link_name] = [virtual_id, object_id]
            else:
                encoding.flattened_relations[self.link_name].extend([virtual_id, object_id])


def get_relations_from_encoders(domain: mm.Domain, input_specification: tuple[Encoder, ...]) -> list[tuple[str, int]]:
    """Get all relations from a collection of encoders.

    Args:
        domain: The PDDL domain containing predicates and actions.
        input_specification: Tuple of encoder instances.

    Returns:
        Sorted list of (relation_name, arity) pairs from all encoders.
    """
    relations_set: list[tuple[str, int]] = []
    for encoder in input_specification:
        relations_set.extend(encoder.get_relations(domain))
    relations_list = list(relations_set)
    relations_list.sort()  # Ensure that the output is deterministic.
    return relations_list


def get_input_from_encoders(input: list[tuple], input_specification: tuple[Encoder, ...], device: torch.device) -> EncodedTensors:
    """Encode input using a collection of encoders.

    This function processes a batch of input instances using the provided
    encoder specification and returns the encoded graph representation
    ready for use in the graph neural network.

    Args:
        input: List of input tuples, where each tuple contains the inputs
               corresponding to the encoder specification.
        input_specification: Tuple of encoder instances that define how to
                             process each element of the input tuples.
        device: The torch device to place the resulting tensors on.

    Returns:
        EncodedTensors object containing the graph representation.

    Raises:
        AssertionError: If input format doesn't match specification or if
                        no StateEncoder is found in the specification.
    """
    encoding_lists = EncodedLists()

    # Process each input instance
    for instance in input:
        assert isinstance(instance, tuple), 'Input instance must be a tuple.'
        assert len(instance) == len(input_specification), 'Mismatch between the length of an input instance and the input specification.'

        # Find the state of the input
        state = None
        for x in instance:
            if isinstance(x, mm.State):
                state = x
                break

        assert state is not None, 'Input must contain a State.'

        # Track nodes added for this instance
        context = EncodingContext(state.problem, encoding_lists.node_count)

        # Process each encoder with its corresponding input value
        for encoder_index, encoder in enumerate(input_specification):
            input_value = instance[encoder_index]
            encoder.encode(input_value, state, encoding_lists, context)

        # Update global encoding with instance results
        encoding_lists.object_indices.extend(context.get_object_ids())
        encoding_lists.action_indices.extend(context.get_action_ids())
        encoding_lists.virtual_indices.extend(context.get_virtual_ids())
        encoding_lists.auxiliary_indices.extend(context.get_auxiliary_ids())
        encoding_lists.object_sizes.append(context.get_object_count())
        encoding_lists.action_sizes.append(context.get_action_count())
        encoding_lists.virtual_sizes.append(context.get_virtual_count())
        encoding_lists.auxiliary_sizes.append(context.get_auxiliary_count())
        encoding_lists.node_sizes.append(context.get_node_count())
        encoding_lists.node_count += context.get_node_count()

    # Convert the lists to tensors on the correct device
    encoding_tensors = EncodedTensors()
    encoding_tensors.flattened_relations = relations_to_tensors(encoding_lists.flattened_relations, device)
    encoding_tensors.node_count = encoding_lists.node_count
    encoding_tensors.node_sizes = torch.tensor(encoding_lists.node_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.object_indices = torch.tensor(encoding_lists.object_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.object_sizes = torch.tensor(encoding_lists.object_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.action_indices = torch.tensor(encoding_lists.action_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.action_sizes = torch.tensor(encoding_lists.action_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.virtual_indices = torch.tensor(encoding_lists.virtual_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.virtual_sizes = torch.tensor(encoding_lists.virtual_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.auxiliary_indices = torch.tensor(encoding_lists.auxiliary_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.auxiliary_sizes = torch.tensor(encoding_lists.auxiliary_sizes, dtype=torch.int, device=device, requires_grad=False)
    return encoding_tensors


class ExpressiveEncoderBase(Encoder):
    def __init__(self, suffix: str) -> None:
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.prefix = 'expressive_'
        self.suffix = suffix

    def get_relation_name_of_predicate(self, predicate: mm.Predicate, is_goal_predicate: bool, is_true: bool) -> str:
        return self.prefix + get_predicate_name(predicate, is_goal_predicate, is_true, self.suffix)

    def get_relation_name_of_ground_atom(self, ground_atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool) -> str:
        return self.prefix + get_atom_name(ground_atom, state, is_goal_atom, self.suffix)

    @staticmethod
    def get_id(obj1: mm.Object, obj2: mm.Object, context: 'EncodingContext') -> int:
        # We use the already reserved id for the singleton if o1 and o2 are the same.
        # This also makes the encoder compatible with other encoders that are only
        # over these ids. For distinct pairs, we allocate auxiliary ids.
        return context.get_object_id(obj1) if obj1 == obj2 else context.new_or_existing_auxiliary_id((obj1, obj2))


class ExpressiveStateEncoder(ExpressiveEncoderBase):
    """Expressive encoder for planning states.

    This encoder transforms a planning state (which contains atoms — facts that
    are currently true) into nodes and relations for the graph neural network.
    In contrast to StateEncoder, pairs of objects become nodes, and atoms become
    relations between these pairs.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the expressive state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__(suffix)
        self.composition_name = self.prefix + 'composition'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            Predicate relations followed by namespaced unary type relations.
        """
        relations = [
            (
                self.get_relation_name_of_predicate(predicate, False, True),
                predicate.arity * predicate.arity,
            )
            for predicate in domain.predicates
        ]
        if domain.uses_typing:
            relations.extend(
                (_type_relation_name(type_name, self.suffix, expressive=True), 1)
                for type_name in domain.type_hierarchy
            )
        relations.append((self.composition_name, 3))
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.State), f'ExpressiveStateEncoder expected a State, got {type(input_value)}'
        for type_relation in _get_type_relations(
            context,
            self.suffix,
            expressive=True,
        ):
            _extend_relation(encoding, *type_relation)
        equality_relation = _get_equality_relation(
            context,
            self.suffix,
            expressive=True,
        )
        if equality_relation is not None:
            _extend_relation(encoding, *equality_relation)
        # Add atom relations for all atoms in the state
        for atom in _get_all_atoms(input_value):
            relation_name = self.get_relation_name_of_ground_atom(atom, state, False)
            terms = atom.objects
            # Create the relation list if it does not already exist.
            if relation_name not in encoding.flattened_relations:
                flattened_relation: list[int] = []
                encoding.flattened_relations[relation_name] = flattened_relation
            else:
                flattened_relation = encoding.flattened_relations[relation_name]
            # Add all possible pair combinations of the terms as arguments.
            for o1 in terms:
                for o2 in terms:
                    flattened_relation.append(self.get_id(o1, o2, context))
        # Create composition atoms.
        # This is different from the paper, here we add all possible compositions rather than deriving them from the state.
        # TODO: Implement the paper version. This is likely too expensive.
        objects = context.object_to_id.keys()
        composition_relation = []
        for o1 in objects:
            for o2 in objects:
                for o3 in objects:
                    composition_relation.append(self.get_id(o1, o2, context))
                    composition_relation.append(self.get_id(o2, o3, context))
                    composition_relation.append(self.get_id(o1, o3, context))
        if self.composition_name in encoding.flattened_relations:
            encoding.flattened_relations[self.composition_name].extend(composition_relation)
        else:
            encoding.flattened_relations[self.composition_name] = composition_relation


class ExpressiveGoalEncoder(ExpressiveEncoderBase):
    """Expressive encoder for goal conditions.

    This encoder transforms a goal condition (a conjunctive condition of
    literals that must be satisfied) into relations for the graph neural
    network. It creates goal-specific relations and marks which atoms are true
    or false in the current state relative to the goal. Unlike GoalEncoder, the
    conjunctive goal is encoded using object pairs.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__(suffix)

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for goal conditions.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for goal predicates,
            including both true and false variants.
        """
        relations = []
        relations.extend([(self.get_relation_name_of_predicate(predicate, True, False), predicate.arity * predicate.arity) for predicate in domain.predicates])
        relations.extend([(self.get_relation_name_of_predicate(predicate, True, True), predicate.arity * predicate.arity) for predicate in domain.predicates])
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'
        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
            assert literal.is_positive, 'Only positive literals are supported.'
            atom = literal.atom
            relation_name = self.get_relation_name_of_ground_atom(atom, state, True)
            terms = atom.objects
            # Create the relation list if it does not already exist.
            if relation_name not in encoding.flattened_relations:
                flattened_relation: list[int] = []
                encoding.flattened_relations[relation_name] = flattened_relation
            else:
                flattened_relation = encoding.flattened_relations[relation_name]
            # Add all possible pair combinations of the terms as arguments.
            for o1 in terms:
                for o2 in terms:
                    flattened_relation.append(self.get_id(o1, o2, context))
