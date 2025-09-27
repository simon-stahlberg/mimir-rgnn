import pymimir as mm
import torch

from typing import Any

from .utils import get_action_name, get_atom_name, get_effect_name, get_effect_relation_name, get_predicate_name, relations_to_tensors
from .bases import Encoder, EncodedLists, EncodedTensors, EncodingContext


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
            List of (relation_name, arity) pairs for all predicates except 'number'.
        """
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        return [(get_predicate_name(predicate, False, True, self.suffix), predicate.get_arity()) for predicate in predicates]

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.State), f'StateEncoder expected a State, got {type(input_value)}'

        # Add atom relations for all atoms in the state
        for atom in input_value.get_atoms():
            relation_name = get_atom_name(atom, state, False, self.suffix)
            object_indices: list[int] = [context.get_object_id(obj.get_index()) for obj in atom.get_terms()]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = object_indices
            else:
                encoding.flattened_relations[relation_name].extend(object_indices)


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
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        relations = []
        relations.extend([(get_predicate_name(predicate, True, False, self.suffix), predicate.get_arity()) for predicate in predicates])
        relations.extend([(get_predicate_name(predicate, True, True, self.suffix), predicate.get_arity()) for predicate in predicates])
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'

        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
            assert literal.get_polarity(), 'Only positive literals are supported.'
            atom = literal.get_atom()
            relation_name = get_atom_name(atom, state, True, self.suffix)
            object_indices: list[int] = [context.get_object_id(obj.get_index()) for obj in atom.get_terms()]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = object_indices
            else:
                encoding.flattened_relations[relation_name].extend(object_indices)



class GroundActionsEncoder(Encoder):
    """Encoder for ground actions.

    This encoder transforms a list of available ground actions into nodes and
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
        return [(get_action_name(action, self.suffix), action.get_arity() + 1) for action in domain.get_actions()]

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, list), f'GroundActionsEncoder expected a list, got {type(input_value)}'

        for action in input_value:
            assert isinstance(action, mm.GroundAction), f'Expected a GroundAction in the list, got {type(action)}'
            relation_name = get_action_name(action, self.suffix)
            action_id = context.new_action_id()
            term_ids = [action_id] + [context.get_object_id(obj.get_index()) for obj in action.get_objects()]
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
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        relations = []
        relations.extend([(get_effect_name(predicate, True, False, self.suffix), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, False, self.suffix), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, True, True, self.suffix), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, True, self.suffix), predicate.get_arity() + 1) for predicate in predicates])
        relations.append((get_effect_relation_name(self.suffix), 2))
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        if isinstance(input_value, tuple):
            effects_list = input_value[0]
            effects_relations = input_value[1]
        elif isinstance(input_value, list):
            effects_list = input_value
            effects_relations = []
        else:
            raise AssertionError(f'TransitionEffectsEncoder expected a list or tuple, got {type(input_value)}')

        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()
        num_transitions = len(effects_list)

        transition_index_to_id: dict[int, int] = dict()

        for transition_index, effects in enumerate(effects_list):
            assert isinstance(effects, list) or isinstance(effects, tuple), 'Expected a list of lists of ground literals.'
            transition_id = context.new_action_id()
            transition_index_to_id[transition_index] = transition_id

            for effect_literal in effects:
                assert isinstance(effect_literal, mm.GroundLiteral), 'Expected a list of lists of ground literals.'
                effect_atom = effect_literal.get_atom()
                effect_name = get_effect_name(effect_atom.get_predicate(), effect_literal.get_polarity(), False, self.suffix)
                object_ids = [transition_id] + [context.get_object_id(obj.get_index()) for obj in effect_atom.get_terms()]

                if effect_name not in encoding.flattened_relations:
                    encoding.flattened_relations[effect_name] = object_ids
                else:
                    encoding.flattened_relations[effect_name].extend(object_ids)

                # Add literals stating how this transition affects the goal
                for goal_literal in goal_condition:
                    assert isinstance(goal_literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
                    assert goal_literal.get_polarity(), 'Only positive literals are supported in the goal condition.'
                    if effect_atom == goal_literal.get_atom():
                        goal_effect_name = get_effect_name(effect_atom.get_predicate(), effect_literal.get_polarity(), True, self.suffix)
                        goal_object_ids = [transition_id] + [context.get_object_id(obj.get_index()) for obj in effect_atom.get_terms()]
                        if goal_effect_name not in encoding.flattened_relations:
                            encoding.flattened_relations[goal_effect_name] = goal_object_ids
                        else:
                            encoding.flattened_relations[goal_effect_name].extend(goal_object_ids)
                        break  # No need to check other goal literals

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

        # Find the state encoder to get the state
        state = None
        for i, encoder in enumerate(input_specification):
            if isinstance(encoder, StateEncoder):
                state = instance[i]
                assert isinstance(state, mm.State), f'Expected a State at position {i}, got {type(state)}'
                break

        assert state is not None, 'Input specification must contain a StateEncoder.'

        # Track nodes added for this instance
        context = EncodingContext(state.get_problem(), encoding_lists.node_count)

        # Process each encoder with its corresponding input value
        for encoder_index, encoder in enumerate(input_specification):
            input_value = instance[encoder_index]
            encoder.encode(input_value, state, encoding_lists, context)

        # Update global encoding with instance results
        encoding_lists.object_indices.extend(context.get_object_ids())
        encoding_lists.action_indices.extend(context.get_action_ids())
        encoding_lists.object_sizes.append(context.get_object_count())
        encoding_lists.action_sizes.append(context.get_action_count())
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
    return encoding_tensors
