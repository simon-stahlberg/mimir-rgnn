import pymimir as mm
import torch

from collections.abc import Iterable

from .bases import QuantizationRecord


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


def binarize(logits, tau=1.0) -> torch.Tensor:
    """Deterministic straight-through binarizer mapping each logit into {0, 1}.

    Forward pass (exact, deterministic)::

        value = 1 if logits > 0 else 0

    Backward pass: the gradient of a sigmoid surrogate ``sigmoid(logits / tau)``.

    The forward pass is noise-free, so repeated passes on the same input produce
    identical bits. This is required when the bits are to be decoded into
    symbolic formulas: a bit can only correspond to concept membership if it is
    a deterministic function of the input. It also avoids the sampling variance
    that a Gumbel-based binarizer injects at every layer during training.

    Args:
        logits: Tensor of pre-activation values.
        tau: Temperature of the backward surrogate (lower -> sharper, more local
            gradients around the threshold).

    Returns:
        Tensor of same shape as logits with values in {0, 1} and
        straight-through gradients.
    """
    y_soft = torch.sigmoid(logits / tau)
    y_hard = (logits > 0).float()
    return y_hard.detach() - y_soft.detach() + y_soft


def ternarize(logits, tau=1.0, threshold=1.0) -> torch.Tensor:
    """Deterministic straight-through ternarizer mapping each logit into {-1, 0, 1}.

    Forward pass (exact, deterministic)::

        value = 1 if logits > threshold, -1 if logits < -threshold, else 0

    Backward pass: the gradient of a smooth two-sigmoid surrogate::

        y_soft = sigmoid((logits - threshold) / tau) - sigmoid((-logits - threshold) / tau)

    The forward pass is noise-free, so the three zones are deterministic for
    *any* logit value (not just well-trained ones), and repeated forward passes
    on the same input produce identical outputs. This is required when the
    quantized values are to be decoded into symbolic formulas: a bit can only
    correspond to concept membership if it is a deterministic function of the
    input. It also removes the sampling variance that a Gumbel-based quantizer
    injects at every layer during training.

    The surrogate gradient at ``logits = 0`` is ``2 * sigmoid'(threshold / tau) / tau``
    (~0.39 with the defaults), so logits inside the 0 zone still receive a usable
    gradient and can learn to cross the threshold. Keep ``threshold`` near the
    scale of the incoming activations (~1 after layer normalization); a much
    larger threshold puts all units in the 0 zone at initialization and starves
    the network of signal.

    Args:
        logits: Tensor of pre-activation values.
        tau: Temperature of the backward surrogate (lower -> sharper, more local
            gradients around the thresholds).
        threshold: Location of the +-1 boundary; sets the 0 zone width.

    Returns:
        Tensor of same shape as logits with values in {-1, 0, 1} and
        straight-through gradients.
    """
    y_soft = torch.sigmoid((logits - threshold) / tau) - torch.sigmoid((-logits - threshold) / tau)
    y_hard = (logits > threshold).float() - (logits < -threshold).float()
    return y_hard.detach() - y_soft.detach() + y_soft


def boundary_margin_penalty(
    records: Iterable[QuantizationRecord],
    margin: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Penalize quantization logits that are close to a decision boundary.

    Args:
        records: Quantization records from a forward pass.
        margin: Distance around each decision boundary to penalize.
        device: Device for the zero tensor returned when no records are present.
        dtype: Dtype for the zero tensor returned when no records are present.

    Returns:
        A scalar penalty with gradients flowing to the recorded logits.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    penalties: list[torch.Tensor] = []
    first_tensor: torch.Tensor | None = None
    for record in records:
        logits = record.logits
        if first_tensor is None:
            first_tensor = logits
        if logits.numel() == 0:
            continue
        thresholds = torch.tensor(record.thresholds, device=logits.device, dtype=logits.dtype)
        distances = (logits.unsqueeze(0) - thresholds.view(-1, *([1] * logits.dim()))).abs()
        nearest_boundary = distances.amin(dim=0)
        penalties.append(torch.relu(torch.as_tensor(margin, device=logits.device, dtype=logits.dtype) - nearest_boundary).mean())

    if penalties:
        return torch.stack(penalties).mean()
    if first_tensor is not None:
        return first_tensor.new_tensor(0.0)
    return torch.tensor(0.0, device=device, dtype=dtype)
