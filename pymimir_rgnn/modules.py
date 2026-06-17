import torch
import torch.nn as nn

from .utils import gumbel_sigmoid


class MLP(nn.Module):
    """Multi-layer perceptron with Mish activation function.

    A simple two-layer neural network with Mish activation. This is used
    throughout the library as a basic building block for learnable transformations.
    """

    def __init__(self, input_size: int, output_size: int):
        """Initialize the MLP.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._inner = nn.Linear(input_size, input_size, True)
        self._outer = nn.Linear(input_size, output_size, True)

    def forward(self, input: torch.Tensor):
        """Forward pass through the MLP.

        Args:
            input: Input tensor of shape (..., input_size).

        Returns:
            Output tensor of shape (..., output_size).
        """
        return self._outer(nn.functional.mish(self._inner(input)))


class SparseMLP(nn.Module):
    """Sparse linear layer or two-layer MLP with learned L0-style input-sparsity gates.

    Each output neuron of each linear layer is connected to at most k of its
    direct input neurons. During training, gates are sampled from a Binary
    Concrete distribution (gumbel_sigmoid). At eval time, a deterministic
    top-k hard mask is used so exactly k inputs are active per output row.

    When linear=True the module is a single gated linear layer (no hidden layer,
    no activation). When linear=False (default) it is a two-layer MLP with Mish
    activation and independent gates on both layers.

    Call sparsity_penalty() and add it (scaled) to the training loss.
    """

    def __init__(self, input_size: int, output_size: int, k: int, tau: float = 1.0, linear: bool = False):
        """Initialize the sparse MLP.

        Args:
            input_size: Size of the input features.
            output_size: Size of the output features.
            k: Maximum number of active input connections per output neuron per layer.
            tau: Temperature for the Gumbel-Sigmoid gate sampler (lower = harder).
            linear: If True, use a single gated linear layer with no activation.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._k = k
        self._tau = tau
        self._linear = linear
        self._outer = nn.Linear(input_size, output_size, True)
        self._outer_log_alpha = nn.Parameter(torch.zeros(output_size, input_size))
        if not linear:
            self._inner = nn.Linear(input_size, input_size, True)
            self._inner_log_alpha = nn.Parameter(torch.zeros(input_size, input_size))

    def _hard_topk_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        k = min(self._k, log_alpha.size(1))
        indices = torch.sigmoid(log_alpha).topk(k, dim=1).indices
        mask = torch.zeros_like(log_alpha)
        mask.scatter_(1, indices, 1.0)
        return mask

    def _gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        if self.training:
            return gumbel_sigmoid(log_alpha, tau=self._tau)
        return self._hard_topk_gate(log_alpha)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sparse MLP.

        Args:
            input: Input tensor of shape (..., input_size).

        Returns:
            Output tensor of shape (..., output_size).
        """
        if self._linear:
            x = input
        else:
            x = nn.functional.mish(
                nn.functional.linear(input, self._inner.weight * self._gate(self._inner_log_alpha), self._inner.bias)
            )
        return nn.functional.linear(x, self._outer.weight * self._gate(self._outer_log_alpha), self._outer.bias)

    def sparsity_penalty(self) -> torch.Tensor:
        """Expected excess connections above k per row, summed over active layers."""
        outer_excess = torch.relu(torch.sigmoid(self._outer_log_alpha).sum(dim=1) - self._k)
        if self._linear:
            return outer_excess.sum()
        inner_excess = torch.relu(torch.sigmoid(self._inner_log_alpha).sum(dim=1) - self._k)
        return inner_excess.sum() + outer_excess.sum()


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
        cumsum_indices = node_sizes.cumsum(0) - 1
        cumsum_states = node_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)
