from typing import Literal

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with Mish activation function.

    A simple two-layer neural network with Mish activation. This is used
    throughout the library as a basic building block for learnable transformations.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int | None = None):
        """Initialize the MLP.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
            hidden_size: Size of the hidden layer. Defaults to input_size.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size if hidden_size is None else hidden_size
        self._inner = nn.Linear(input_size, self.hidden_size, True)
        self._outer = nn.Linear(self.hidden_size, output_size, True)

    def forward(self, input: torch.Tensor):
        """Forward pass through the MLP.

        Args:
            input: Input tensor of shape (..., input_size).

        Returns:
            Output tensor of shape (..., output_size).
        """
        return self._outer(nn.functional.mish(self._inner(input)))


class SparseMLP(nn.Module):
    """Hard top-k gated linear layer for interpretable input dependencies.

    Each output feature is a linear function of at most k original input
    features. The implementation still uses dense PyTorch linear operations,
    with a hard top-k mask applied to the dense weight matrix. The hard top-k
    gates enforce the exact active-input count; topk_margin_penalty() can be
    added to a training loss to encourage stable separation between selected
    and rejected gate logits.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        k: int,
        tau: float = 1.0,
        gate_mode: Literal["gumbel_topk", "deterministic_topk"] = "gumbel_topk",
    ):
        """Initialize the sparse MLP.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
            k: Maximum number of active input connections per output feature.
            tau: Temperature for the straight-through sigmoid surrogate.
            gate_mode: Training-time hard top-k mode. Evaluation always uses
                deterministic top-k.
        """
        super().__init__()
        if k < 0:
            raise ValueError("k must be non-negative")
        if gate_mode not in ("gumbel_topk", "deterministic_topk"):
            raise ValueError("gate_mode must be 'gumbel_topk' or 'deterministic_topk'")
        self.input_size = input_size
        self.output_size = output_size
        self._k = k
        self._tau = tau
        self._gate_mode = gate_mode
        self._outer = nn.Linear(input_size, output_size, True)
        self._outer_log_alpha = nn.Parameter(torch.zeros(output_size, input_size))

    def _hard_topk_gate(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self._k, scores.size(1))
        if k == 0:
            return torch.zeros_like(scores)
        indices = scores.topk(k, dim=1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
        return mask

    def _gumbel_noise(self, input: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        uniform = torch.rand_like(input)
        return -torch.log(-torch.log(uniform + eps) + eps)

    def _gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        scores = log_alpha
        if self.training and self._gate_mode == "gumbel_topk":
            scores = scores + self._gumbel_noise(scores)
        hard = self._hard_topk_gate(scores)
        if not self.training:
            return hard
        soft = torch.sigmoid(scores / self._tau)
        return hard.detach() - soft.detach() + soft

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sparse MLP.

        Args:
            input: Input tensor of shape (..., input_size).

        Returns:
            Output tensor of shape (..., output_size).
        """
        return nn.functional.linear(input, self._outer.weight * self._gate(self._outer_log_alpha), self._outer.bias)

    def topk_margin_penalty(self, margin: float) -> torch.Tensor:
        """Penalize small gaps at the top-k gate selection boundary.

        The penalty is based on raw deterministic gate logits, not Gumbel-noisy
        training scores. It is zero when the weakest selected logit exceeds the
        strongest rejected logit by at least margin.

        Args:
            margin: Required gap between the kth and (k+1)th gate logits.

        Returns:
            Scalar mean penalty across output rows.
        """
        if margin < 0:
            raise ValueError("margin must be non-negative")
        log_alpha = self._outer_log_alpha
        if log_alpha.numel() == 0 or self._k == 0 or self._k >= log_alpha.size(1):
            return log_alpha.new_tensor(0.0)
        values = log_alpha.topk(self._k + 1, dim=1).values
        kth_selected = values[:, self._k - 1]
        best_rejected = values[:, self._k]
        gap = kth_selected - best_rejected
        return torch.relu(torch.as_tensor(margin, device=log_alpha.device, dtype=log_alpha.dtype) - gap).mean()


class ChannelwiseAffine(nn.Module):
    """Per-channel learnable affine transformation.

    A channel-independent alternative to layer normalization: each channel is
    scaled and shifted by its own learnable parameters, without any cross-channel
    statistics. This keeps each output channel a function of the corresponding
    input channel only, which is required when individual channels are to be
    decoded into symbolic formulas. Followed by a binarizer, it acts as a
    learnable per-channel threshold.
    """

    def __init__(self, size: int):
        """Initialize the channelwise affine transformation.

        Args:
            size: Number of channels.
        """
        super().__init__()
        self._gamma = nn.Parameter(torch.ones(size))
        self._beta = nn.Parameter(torch.zeros(size))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel affine transformation.

        Args:
            input: Input tensor of shape (..., size).

        Returns:
            Output tensor of the same shape.
        """
        return input * self._gamma + self._beta


class SumReadout(nn.Module):
    """Readout module that aggregates embeddings by summing within groups.

    This module is used to aggregate node embeddings within groups (e.g., all
    objects in a state, all actions in an instance) and then applies an MLP
    to produce output values.
    """

    def __init__(self, input_size: int, output_size: int):
        """Initialize the sum readout module.

        Args:
            input_size: Size of the input embeddings.
            output_size: Size of the output features.
        """
        super().__init__()
        self._value = MLP(input_size, output_size)

    def forward(self, node_embeddings: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
        """Aggregate embeddings by sum within groups and apply MLP.

        Args:
            node_embeddings: Node embeddings to aggregate.
            node_sizes: Number of nodes in each group.

        Returns:
            Aggregated and transformed embeddings, one per group.
        """
        if node_sizes.numel() == 1:
            return self._value(node_embeddings.sum(dim=0, keepdim=True))
        cumsum_indices = node_sizes.cumsum(0) - 1
        cumsum_states = node_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)
