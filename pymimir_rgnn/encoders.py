import pymimir as mm
import torch

from typing import Any

from .utils import get_action_name, get_atom_name, get_effect_name, get_effect_relation_name, get_predicate_name, relations_to_tensors
from .bases import Encoder, EncodedLists, EncodedTensors


class StateEncoder(Encoder):
    """Encoder for planning states.

    This encoder transforms a planning state (which contains atoms/facts that are
    currently true) into nodes and relations for the graph neural network.
    Objects become nodes, and atoms become relations between objects.
    """

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for all predicates except 'number'.
        """
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        return [(get_predicate_name(predicate, False, True), predicate.get_arity()) for predicate in predicates]

    def encode(self, input_value: Any, encoding: 'EncodedLists', state: mm.State) -> int:
        """Encode a planning state into the intermediate representation.

        Args:
            input_value: The state to encode (must be a mm.State).
            encoding: The encoding object to populate with graph structure.
            state: The current planning state (same as input_value).

        Returns:
            Number of object nodes added to the graph.

        Raises:
            AssertionError: If input_value is not a State.
        """
        assert isinstance(input_value, mm.State), f'StateEncoder expected a State, got {type(input_value)}'

        problem = input_value.get_problem()
        domain = problem.get_domain()
        num_objects = len(problem.get_objects()) + len(domain.get_constants())

        # Add atom relations for all atoms in the state
        for atom in input_value.get_atoms():
            self._add_atom_relation(atom, input_value, False, encoding)

        # Track object indices
        encoding.object_indices.extend(range(encoding.node_count, encoding.node_count + num_objects))
        encoding.object_sizes.append(num_objects)

        return num_objects

    def _add_atom_relation(self, atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool, intermediate: 'EncodedLists'):
        """Add an atom relation to the intermediate representation.

        Args:
            atom: The ground atom to add as a relation.
            state: The current planning state.
            is_goal_atom: Whether this atom is part of the goal condition.
            intermediate: The intermediate encoding to update.
        """
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices: list[int] = [term.get_index() + intermediate.node_count for term in atom.get_terms()]
        if relation_name not in intermediate.flattened_relations:
            intermediate.flattened_relations[relation_name] = object_indices
        else:
            intermediate.flattened_relations[relation_name].extend(object_indices)


class GoalEncoder(Encoder):
    """Encoder for goal conditions.

    This encoder transforms a goal condition (conjunctive condition of literals
    that must be satisfied) into relations for the graph neural network.
    It creates both goal-specific relations and marks which atoms are true/false
    in the current state relative to the goal.
    """

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
        relations.extend([(get_predicate_name(predicate, True, False), predicate.get_arity()) for predicate in predicates])
        relations.extend([(get_predicate_name(predicate, True, True), predicate.get_arity()) for predicate in predicates])
        return relations

    def encode(self, input_value: Any, encoding: 'EncodedLists', state: mm.State) -> int:
        """Encode a goal condition into the intermediate representation.

        Args:
            input_value: The goal condition to encode (must be a GroundConjunctiveCondition).
            encoding: The encoding object to populate with graph structure.
            state: The current planning state for context.

        Returns:
            Number of nodes added (0, as goals don't add new nodes).

        Raises:
            AssertionError: If input_value is not a GroundConjunctiveCondition.
        """
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'

        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
            assert literal.get_polarity(), 'Only positive literals are supported.'
            self._add_atom_relation(literal.get_atom(), state, True, encoding)

        return 0  # Goals don't add new nodes, they just add relations

    def _add_atom_relation(self, atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool, intermediate: 'EncodedLists'):
        """Add an atom relation to the intermediate representation.

        Args:
            atom: The ground atom to add as a relation.
            state: The current planning state.
            is_goal_atom: Whether this atom is part of the goal condition.
            intermediate: The intermediate encoding to update.
        """
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices: list[int] = [term.get_index() + intermediate.node_count for term in atom.get_terms()]
        if relation_name not in intermediate.flattened_relations:
            intermediate.flattened_relations[relation_name] = object_indices
        else:
            intermediate.flattened_relations[relation_name].extend(object_indices)


class GroundActionsEncoder(Encoder):
    """Encoder for ground actions.

    This encoder transforms a list of available ground actions into nodes and
    relations for the graph neural network. Each action becomes a new node,
    and relations connect actions to their parameter objects.
    """

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for actions.

        Args:
            domain: The PDDL domain containing action definitions.

        Returns:
            List of (relation_name, arity) pairs for all actions, where arity
            is the action's parameter count plus 1 (for the action node itself).
        """
        return [(get_action_name(action), action.get_arity() + 1) for action in domain.get_actions()]

    def encode(self, input_value: Any, encoding: 'EncodedLists', state: mm.State) -> int:
        """Encode ground actions into the intermediate representation.

        Args:
            input_value: List of ground actions to encode.
            encoding: The encoding object to populate with graph structure.
            state: The current planning state for context.

        Returns:
            Number of action nodes added to the graph.

        Raises:
            AssertionError: If input_value is not a list or contains non-GroundAction items.
        """
        assert isinstance(input_value, list), f'GroundActionsEncoder expected a list, got {type(input_value)}'

        actions = input_value
        num_actions = len(actions)

        for action_index, action in enumerate(actions):
            assert isinstance(action, mm.GroundAction), f'Expected a GroundAction in the list at position {action_index}'

            problem = action.get_problem()
            num_objects = len(problem.get_objects())
            relation_name = get_action_name(action)
            action_local_id = num_objects + action_index
            action_global_id = action_local_id + encoding.node_count
            term_ids = [action_global_id] + [term.get_index() + encoding.node_count for term in action.get_objects()]

            # Add to input relations
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = term_ids
            else:
                encoding.flattened_relations[relation_name].extend(term_ids)

            # Each action adds a new node, remember the id
            encoding.action_indices.append(action_global_id)

        encoding.action_sizes.append(num_actions)
        return num_actions


class TransitionEffectsEncoder(Encoder):
    """Encoder for transition effects.

    This encoder transforms action effects (lists of literals describing state
    changes) into nodes and relations for the graph neural network. Each
    transition becomes a new node, with relations connecting it to affected atoms.
    """

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
        relations.extend([(get_effect_name(predicate, True, False), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, False), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, True, True), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, True), predicate.get_arity() + 1) for predicate in predicates])
        relations.append((get_effect_relation_name(), 2))
        return relations

    def encode(self, input_value: Any, encoding: 'EncodedLists', state: mm.State) -> int:
        """Encode transition effects into the intermediate representation.

        Args:
            input_value: List of effect lists (each inner list contains GroundLiterals).
            encoding: The encoding object to populate with graph structure.
            state: The current planning state for context.

        Returns:
            Number of transition nodes added to the graph.

        Raises:
            AssertionError: If input format is incorrect.
        """
        assert isinstance(input_value, list), f'TransitionEffectsEncoder expected a list, got {type(input_value)}'

        effects_list = input_value
        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()
        num_objects = len(problem.get_objects())
        num_transitions = len(effects_list)

        for transition_index, effects in enumerate(effects_list):
            assert isinstance(effects, list), 'Expected a list of lists of ground literals.'
            transition_local_id = num_objects + transition_index
            transition_global_id = transition_local_id + encoding.node_count

            for effect in effects:
                assert isinstance(effect, mm.GroundLiteral), 'Expected a list of lists of ground literals.'
                effect_name = get_effect_name(effect.get_atom().get_predicate(), effect.get_polarity(), False)
                term_ids = [transition_global_id] + [term.get_index() + encoding.node_count for term in effect.get_atom().get_terms()]

                if effect_name not in encoding.flattened_relations:
                    encoding.flattened_relations[effect_name] = term_ids
                else:
                    encoding.flattened_relations[effect_name].extend(term_ids)

                # Add literals stating how this transition affects the goal
                if any(x == effect.get_atom() for x in goal_condition):
                    goal_effect_name = get_effect_name(effect.get_atom().get_predicate(), effect.get_polarity(), True)
                    goal_term_ids = [transition_global_id] + [term.get_index() + encoding.node_count for term in effect.get_atom().get_terms()]

                    if goal_effect_name not in encoding.flattened_relations:
                        encoding.flattened_relations[goal_effect_name] = goal_term_ids
                    else:
                        encoding.flattened_relations[goal_effect_name].extend(goal_term_ids)

            # Each transition adds a new node, remember the id
            encoding.action_indices.append(transition_global_id)

        encoding.action_sizes.append(num_transitions)
        return num_transitions


def get_relations_from_encoders(domain: mm.Domain, input_specification: tuple[Encoder, ...]) -> list[tuple[str, int]]:
    """Get all relations from a collection of encoders.

    Args:
        domain: The PDDL domain containing predicates and actions.
        input_specification: Tuple of encoder instances.

    Returns:
        Sorted list of (relation_name, arity) pairs from all encoders.
    """
    relations: list[tuple[str, int]] = []
    for encoder in input_specification:
        relations.extend(encoder.get_relations(domain))
    relations.sort()  # Ensure that the output is deterministic.
    return relations


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
    intermediate = EncodedLists()

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
        added_nodes = 0

        # Process each encoder with its corresponding input value
        for encoder_index, encoder in enumerate(input_specification):
            input_value = instance[encoder_index]
            nodes_added = encoder.encode(input_value, intermediate, state)
            added_nodes += nodes_added

        intermediate.node_sizes.append(added_nodes)
        intermediate.node_count += added_nodes

    # Convert the lists to tensors on the correct device
    result = EncodedTensors()
    result.flattened_relations = relations_to_tensors(intermediate.flattened_relations, device)
    result.node_count = intermediate.node_count
    result.node_sizes = torch.tensor(intermediate.node_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.object_indices = torch.tensor(intermediate.object_indices, dtype=torch.int, device=device, requires_grad=False)
    result.object_sizes = torch.tensor(intermediate.object_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.action_indices = torch.tensor(intermediate.action_indices, dtype=torch.int, device=device, requires_grad=False)
    result.action_sizes = torch.tensor(intermediate.action_sizes, dtype=torch.int, device=device, requires_grad=False)
    return result
