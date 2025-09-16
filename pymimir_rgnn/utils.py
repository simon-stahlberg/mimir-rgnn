import pymimir as mm
import torch


def get_atom_name(atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool, suffix: str) -> str:
    """Generate a relation name for an atom based on its context.

    Args:
        atom: The ground atom to generate a name for.
        state: The current planning state.
        is_goal_atom: Whether this atom is part of the goal condition.

    Returns:
        String name for the relation representing this atom.
    """
    if is_goal_atom:
        is_in_state = state.contains(atom)
        return get_predicate_name(atom.get_predicate(), True, is_in_state, suffix)
    else:
        return get_predicate_name(atom.get_predicate(), False, True, suffix)


def get_predicate_name(predicate: mm.Predicate, is_goal_predicate: bool, is_true: bool, suffix: str) -> str:
    """Generate a standardized name for predicate relations.

    Args:
        predicate: The predicate to generate a name for.
        is_goal_predicate: Whether this predicate appears in a goal context.
        is_true: Whether the predicate is true in the current state.

    Returns:
        Standardized string name for the predicate relation.

    Raises:
        AssertionError: If the combination of parameters is invalid.
    """
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate:
        truth_value = 'true' if is_true else 'false'
        return f'relation_{predicate.get_name()}{suffix}_goal_{truth_value}'
    else:
        return f'relation_{predicate.get_name()}{suffix}'


def get_effect_name(predicate: mm.Predicate, positive: bool, affects_goal: bool, suffix: str) -> str:
    """Generate a name for effect relations.

    Args:
        predicate: The predicate affected by the effect.
        positive: Whether this is a positive (add) or negative (delete) effect.
        affects_goal: Whether this effect affects a goal atom.

    Returns:
        String name for the effect relation.
    """
    return predicate.get_name() + suffix + ('_pos' if positive else '_neg') + ('_goal' if affects_goal else '')


def get_effect_relation_name(suffix: str) -> str:
    """Get the standard name for effect relations.

    Returns:
        String name for the general effect relation.
    """
    return 'effect_relation' + suffix


def get_action_name(action: mm.Action | mm.GroundAction, suffix: str) -> str:
    """Generate a standardized name for action relations.

    Args:
        action: The action (grounded or ungrounded) to generate a name for.

    Returns:
        String name for the action relation.

    Raises:
        RuntimeError: If the argument is not a recognized action type.
    """
    if isinstance(action, mm.GroundAction):
        return 'action_' + str(action.get_action().get_name()) + suffix
    elif isinstance(action, mm.Action):  # type: ignore
        return 'action_' + str(action.get_name()) + suffix
    else:
        raise RuntimeError('Argument is not an action.')


def relations_to_tensors(term_id_groups: dict[str, list[int]], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert relation ID lists to tensors on the specified device.

    Args:
        term_id_groups: Dictionary mapping relation names to lists of term IDs.
        device: The torch device to place the tensors on.

    Returns:
        Dictionary mapping relation names to tensor representations.
    """
    result: dict[str, torch.Tensor] = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


def gumbel_sigmoid(logits, tau=1.0, hard=False, eps=1e-10) -> torch.Tensor:
    """
    Binary Concrete / Gumbel-Sigmoid with double noise trick.

    Args:
        logits: Tensor of pre-sigmoid values.
        tau: Temperature (lower -> harder samples).
        hard: If True, returns hard {0,1} but with straight-through gradients.
        eps: Small constant for numerical stability.

    Returns:
        Sampled tensor of same shape as logits.
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
