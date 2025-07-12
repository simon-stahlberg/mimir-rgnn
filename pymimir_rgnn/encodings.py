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


class EncodedInput:
    def __init__(self):
        self.flattened_relations: 'dict[str, torch.Tensor]' = {}
        self.node_count: 'int' = 0
        self.node_sizes: 'list[int]' = []
        self.object_sizes: 'list[int]' = []
        self.object_indices: 'list[int]' = []
        self.action_sizes: 'list[int]' = []
        self.action_indices: 'list[int]' = []


def get_encoding(domain: 'mm.Domain', input_specification: 'tuple[InputType, ...]') -> 'list[tuple[str, int]]':
    assert len(input_specification) == len(set(input_specification)), 'Input types must not be repeated.'
    relations = []
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


def encode_input(input: 'list[tuple]', input_specification: 'tuple[InputType, ...]', device: torch.device) -> 'EncodedInput':
    result = EncodedInput()

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
    assert state_index is not None, 'The input specification must contain a state.'

    # Functions for populating the result.
    def add_atom_relation(atom: 'mm.GroundAtom', state: 'mm.State', is_goal_atom: 'bool'):
        nonlocal result
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices = [term.get_index() + result.node_count for term in atom.get_terms()]
        if relation_name not in result.flattened_relations: result.flattened_relations[relation_name] = object_indices
        else: result.flattened_relations[relation_name].extend(object_indices)

    def add_state_relations(state: 'mm.State') -> 'int':
        nonlocal result, state_index
        if state_index is not None:
            problem = state.get_problem()
            domain = problem.get_domain()
            num_objects = len(problem.get_objects()) + len(domain.get_constants())
            for atom in state.get_ground_atoms():
                add_atom_relation(atom, state, False)
            result.object_indices.extend(range(result.node_count, result.node_count + num_objects))
            result.object_sizes.append(num_objects)
            return num_objects
        return 0

    def add_goal_relations(state: 'mm.State') -> 'int':
        nonlocal result, goal_index, instance
        if goal_index is not None:
            goal = instance[goal_index]
            assert isinstance(goal, mm.GroundConjunctiveCondition), f'Mismatch between input and specification: expected a goal at position {goal_index}.'
            for literal in goal:
                assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
                assert literal.get_polarity(), 'Only positive literals are supported.'
                add_atom_relation(literal.get_atom(), state, True)
        return 0

    def add_successor_relations() -> 'int':
        if successors_index is not None:
            raise NotImplementedError('State successors are not supported yet.')
        return 0

    def add_action_relations() -> 'int':
        nonlocal result, actions_index, instance
        if actions_index is not None:
            actions = instance[actions_index]
            assert isinstance(actions, list), f'Mismatch between input and specification: expected a list at position {actions_index}.'
            num_actions = len(actions)
            for action_index, action in enumerate(actions):
                assert isinstance(action, mm.GroundAction), f'Mismatch between input and specification: expected a ground action in the list at position {action_index}.'
                problem = action.get_problem()
                num_objects = len(problem.get_objects())
                relation_name = get_action_name(action)
                action_id = num_objects + action_index
                term_ids = [action_id + result.node_count] + [term.get_index() + result.node_count for term in action.get_objects()]
                if relation_name not in result.flattened_relations: result.flattened_relations[relation_name] = term_ids
                else: result.flattened_relations[relation_name].extend(term_ids)
                result.action_indices.append(action_id + result.node_count)
            result.action_sizes.append(num_actions)
            return num_actions
        return 0

    def add_effect_relations() -> 'int':
        if effects_index is not None:
            raise NotImplementedError('Transition effects are not supported yet.')
        return 0

    # Construct input
    for instance in input:
        assert len(instance) == len(input_specification), 'Mismatch between the length of an input instance and the input specification.'
        state: mm.State = instance[state_index]
        assert isinstance(state, mm.State), f'Mismatch between input and specification: expected a state at position {state_index}.'

        added_nodes = 0
        added_nodes += add_state_relations(state)
        added_nodes += add_goal_relations(state)
        added_nodes += add_successor_relations()
        added_nodes += add_action_relations()
        added_nodes += add_effect_relations()
        result.node_sizes.append(added_nodes)
        result.node_count += added_nodes

    # Convert the lists to tensors on the correct device.
    result.flattened_relations = relations_to_tensors(result.flattened_relations, device)
    result.node_sizes = torch.tensor(result.node_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.object_indices = torch.tensor(result.object_indices, dtype=torch.int, device=device, requires_grad=False)
    result.object_sizes = torch.tensor(result.object_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.action_indices = torch.tensor(result.action_indices, dtype=torch.int, device=device, requires_grad=False)
    result.action_sizes = torch.tensor(result.action_sizes, dtype=torch.int, device=device, requires_grad=False)
    return result
