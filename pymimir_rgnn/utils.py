import pymimir as mm
import torch


def get_atom_name(atom: 'mm.GroundAtom', state: 'mm.State', is_goal_atom: 'bool') -> 'str':
    if is_goal_atom:
        is_in_state = state.contains(atom)
        return get_predicate_name(atom.get_predicate(), True, is_in_state)
    else:
        return get_predicate_name(atom.get_predicate(), False, True)


def get_predicate_name(predicate: 'mm.Predicate', is_goal_predicate: 'bool', is_true: 'bool') -> 'str':
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate:
        truth_value = 'true' if is_true else 'false'
        return f'relation_{predicate.get_name()}_goal_{truth_value}'
    else:
        return f'relation_{predicate.get_name()}'


def get_effect_name(predicate: 'mm.Predicate', positive: 'bool') -> 'str':
    return predicate.get_name() + ('_pos' if positive else '_neg')


def get_action_name(action) -> 'str':
    if  isinstance(action, mm.GroundAction):
        return 'action_' + str(action.get_action().get_name())
    elif isinstance(action, mm.Action):
        return 'action_' + str(action.get_name())


def relations_to_tensors(term_id_groups: 'dict[str, list[int]]', device: 'torch.device') -> 'dict[str, torch.Tensor]':
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result
