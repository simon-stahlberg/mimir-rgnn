import pymimir as mm
import torch


def get_atom_name(atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool) -> str:
    if is_goal_atom:
        is_in_state = state.contains(atom)
        return get_predicate_name(atom.get_predicate(), True, is_in_state)
    else:
        return get_predicate_name(atom.get_predicate(), False, True)


def get_predicate_name(predicate: mm.Predicate, is_goal_predicate: bool, is_true: bool) -> str:
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate:
        truth_value = 'true' if is_true else 'false'
        return f'relation_{predicate.get_name()}_goal_{truth_value}'
    else:
        return f'relation_{predicate.get_name()}'


def get_effect_name(predicate: mm.Predicate, positive: bool, affects_goal: bool) -> str:
    return predicate.get_name() + ('_pos' if positive else '_neg') + ('_goal' if affects_goal else '')


def get_effect_relation_name() -> str:
    return 'effect_relation'


def get_action_name(action: mm.Action | mm.GroundAction) -> str:
    if isinstance(action, mm.GroundAction):
        return 'action_' + str(action.get_action().get_name())
    elif isinstance(action, mm.Action):  # type: ignore
        return 'action_' + str(action.get_name())
    else:
        raise RuntimeError('Argument is not an action.')


def relations_to_tensors(term_id_groups: dict[str, list[int]], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


def gumbel_sigmoid(logits, tau=1.0, hard=False, eps=1e-10):
    """
    Binary Concrete / Gumbel-Sigmoid with double noise trick.
    Args:
        logits: Tensor of pre-sigmoid values.
        tau: Temperature (lower -> harder samples).
        hard: If True, returns hard {0,1} but with straight-through gradients.
    """
    # Sample two Gumbel noises
    u1 = torch.rand_like(logits)
    u2 = torch.rand_like(logits)
    g1 = -torch.log(-torch.log(u1 + eps) + eps)
    g2 = -torch.log(-torch.log(u2 + eps) + eps)

    # Double noise trick
    y_soft = torch.sigmoid((logits + g1 - g2) / tau)

    if hard:
        # Straight-through binarization
        y_hard = (y_soft > 0.5).float()
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y
