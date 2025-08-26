import pymimir as mm
import torch

from enum import Enum

from .utils import get_action_name, get_atom_name, get_effect_name, get_predicate_name, relations_to_tensors


class InputType(Enum):
    Goal = 'goal'
    State = 'state'
    Successors = 'successors'
    GroundActions = 'actions'
    TransitionEffects = 'effects'


class OutputValueType(Enum):
    Scalar = 'scalar'
    Embeddings = 'embeddings'


class OutputNodeType(Enum):
    All = 'all'
    Objects = 'objects'
    Action = 'action'


class ListInput:
    def __init__(self):
        self.flattened_relations: dict[str, list[int]] = {}
        self.node_count: int = 0
        self.node_sizes: list[int] = []
        self.object_sizes: list[int] = []
        self.object_indices: list[int] = []
        self.action_sizes: list[int] = []
        self.action_indices: list[int] = []


class TensorInput:
    def __init__(self):
        self.flattened_relations: dict[str, torch.LongTensor] = {}
        self.node_count: int = 0
        self.node_sizes: torch.LongTensor = torch.LongTensor()
        self.object_sizes: torch.LongTensor = torch.LongTensor()
        self.object_indices: torch.LongTensor = torch.LongTensor()
        self.action_sizes: torch.LongTensor = torch.LongTensor()
        self.action_indices: torch.LongTensor = torch.LongTensor()


def get_encoding(domain: mm.Domain, input_specification: tuple[InputType, ...]) -> list[tuple[str, int]]:
    assert len(input_specification) == len(set(input_specification)), 'Input types must not be repeated.'
    relations: list[tuple[str, int]] = []
    ignored_predicate_names = ['number']
    predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
    for input_type in input_specification:
        if input_type == InputType.State:
            relations.extend([(get_predicate_name(predicate, False, True), predicate.get_arity()) for predicate in predicates])
        elif input_type == InputType.Goal:
            relations.extend([(get_predicate_name(predicate, True, False), predicate.get_arity()) for predicate in predicates])
            relations.extend([(get_predicate_name(predicate, True, True), predicate.get_arity()) for predicate in predicates])
        elif input_type == InputType.GroundActions:
            relations.extend([(get_action_name(action), action.get_arity() + 1) for action in domain.get_actions()])
        elif input_type == InputType.TransitionEffects:
            relations.extend([(get_effect_name(predicate, True), predicate.get_arity() + 1) for predicate in predicates])
            relations.extend([(get_effect_name(predicate, False), predicate.get_arity() + 1) for predicate in predicates])
    relations.sort()  # Ensure that the output is deterministic.
    return relations


def encode_input(input: list[tuple], input_specification: tuple[InputType, ...], device: torch.device) -> TensorInput:  # type: ignore
    intermediate = ListInput()

    # Extract relevant parts
    state_index = None
    goal_index = None
    successors_index = None
    actions_index = None
    effects_index = None

    # This code assumes no duplicates. This is checked in get_encoding().
    for input_index, input_type in enumerate(input_specification):
        assert isinstance(input_type, InputType), f'The item at position {input_index} is not an input type.'
        if input_type == InputType.State:
            state_index = input_index
        elif input_type == InputType.Goal:
            goal_index = input_index
        elif input_type == InputType.Successors:
            successors_index = input_index
        elif input_type == InputType.GroundActions:
            actions_index = input_index
        elif input_type == InputType.TransitionEffects:
            effects_index = input_index

    # Validate specification.
    assert state_index is not None, 'The input specification must contain a state.'
    assert actions_index is None or effects_index is None, "At most one of GroundActions and TransitionEffects can be in the input specification."

    # Functions for populating the result.
    def add_atom_relation(atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool):
        nonlocal intermediate
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices: list[int] = [term.get_index() + intermediate.node_count for term in atom.get_terms()]
        if relation_name not in intermediate.flattened_relations: intermediate.flattened_relations[relation_name] = object_indices
        else: intermediate.flattened_relations[relation_name].extend(object_indices)

    def add_state_relations(instance, state: mm.State) -> int:
        nonlocal intermediate, state_index
        if state_index is not None:
            problem = state.get_problem()
            domain = problem.get_domain()
            num_objects = len(problem.get_objects()) + len(domain.get_constants())
            for atom in state.get_atoms():
                add_atom_relation(atom, state, False)
            intermediate.object_indices.extend(range(intermediate.node_count, intermediate.node_count + num_objects))
            intermediate.object_sizes.append(num_objects)
            return num_objects
        return 0

    def add_goal_relations(instance, state: mm.State) -> int:
        nonlocal intermediate, goal_index
        if goal_index is not None:
            goal: mm.GroundConjunctiveCondition = instance[goal_index]  # type: ignore
            assert isinstance(goal, mm.GroundConjunctiveCondition), f'Mismatch between input and specification: expected a goal at position {goal_index}.'
            for literal in goal:  # type: ignore
                assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
                assert literal.get_polarity(), 'Only positive literals are supported.'
                add_atom_relation(literal.get_atom(), state, True)
        return 0

    def add_successor_relations(instance) -> int:
        if successors_index is not None:
            raise NotImplementedError('State successors are not supported yet.')
        return 0

    def add_action_relations(instance) -> int:
        nonlocal intermediate, actions_index
        if actions_index is not None:
            actions: list[mm.GroundAction] = instance[actions_index]  # type: ignore
            assert isinstance(actions, list), f'Mismatch between input and specification: expected a list at position {actions_index}.'
            num_actions = len(actions)
            for action_index, action in enumerate(actions):  # type: ignore
                assert isinstance(action, mm.GroundAction), f'Mismatch between input and specification: expected a ground action in the list at position {action_index}.'
                problem = action.get_problem()
                num_objects = len(problem.get_objects())
                relation_name = get_action_name(action)
                action_local_id = num_objects + action_index
                action_global_id = action_local_id + intermediate.node_count
                term_ids = [action_global_id] + [term.get_index() + intermediate.node_count for term in action.get_objects()]
                # Add to input relations.
                relations = intermediate.flattened_relations
                if relation_name not in relations: relations[relation_name] = term_ids
                else: relations[relation_name].extend(term_ids)
                # Each action adds a new node, remember the id.
                intermediate.action_indices.append(action_global_id)
            intermediate.action_sizes.append(num_actions)
            return num_actions
        return 0

    def add_effect_relations(instance, problem: mm.Problem) -> int:
        nonlocal intermediate, effects_index
        if effects_index is not None:
            effects_list: list[list[mm.GroundLiteral]] = instance[effects_index]
            assert isinstance(effects_list, list), 'Mismatch between input and specification: expected a list of lists of ground literals.'
            num_objects = len(problem.get_objects())
            num_transitions = len(effects_list)
            for transition_index, effects in enumerate(effects_list):
                assert isinstance(effects, list), 'Mismatch between input and specification: expected a list of lists of ground literals.'
                transition_local_id = num_objects + transition_index
                transition_global_id = transition_local_id + intermediate.node_count
                for effect in effects:
                    assert isinstance(effect, mm.GroundLiteral), 'Mismatch between input and specification: expected a list of lists of ground literals.'
                    effect_name = get_effect_name(effect.get_atom().get_predicate(), effect.get_polarity())
                    term_ids = [transition_global_id] + [term.get_index() + intermediate.node_count for term in effect.get_atom().get_terms()]
                    # Add to input relations.
                    relations = intermediate.flattened_relations
                    if effect_name not in relations: relations[effect_name] = term_ids
                    else: relations[effect_name].extend(term_ids)
                # Each transition adds a new node, remember the id.
                intermediate.action_indices.append(transition_global_id)
            intermediate.action_sizes.append(num_transitions)
            return num_transitions
        return 0

    # Construct input
    for instance in input:  # type: ignore
        assert isinstance(instance, tuple), 'Input instance must be a tuple.'
        assert len(instance) == len(input_specification), 'Mismatch between the length of an input instance and the input specification.'  # type: ignore
        state: mm.State = instance[state_index]  # type: ignore
        problem: mm.Problem = state.get_problem()
        assert isinstance(state, mm.State), f'Mismatch between input and specification: expected a state at position {state_index}.'

        added_nodes = 0
        added_nodes += add_state_relations(instance, state)
        added_nodes += add_goal_relations(instance, state)
        added_nodes += add_successor_relations(instance)
        added_nodes += add_action_relations(instance)
        added_nodes += add_effect_relations(instance, problem)
        intermediate.node_sizes.append(added_nodes)
        intermediate.node_count += added_nodes

    # Convert the lists to tensors on the correct device.
    result = TensorInput()
    result.flattened_relations = relations_to_tensors(intermediate.flattened_relations, device)  # type: ignore
    result.node_count = intermediate.node_count
    result.node_sizes = torch.tensor(intermediate.node_sizes, dtype=torch.int, device=device, requires_grad=False)  # type: ignore
    result.object_indices = torch.tensor(intermediate.object_indices, dtype=torch.int, device=device, requires_grad=False)  # type: ignore
    result.object_sizes = torch.tensor(intermediate.object_sizes, dtype=torch.int, device=device, requires_grad=False)  # type: ignore
    result.action_indices = torch.tensor(intermediate.action_indices, dtype=torch.int, device=device, requires_grad=False)  # type: ignore
    result.action_sizes = torch.tensor(intermediate.action_sizes, dtype=torch.int, device=device, requires_grad=False)  # type: ignore
    return result
