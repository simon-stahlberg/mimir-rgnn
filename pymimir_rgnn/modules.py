import torch
import torch.nn as nn


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
