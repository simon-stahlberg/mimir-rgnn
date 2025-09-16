import torch

from .bases import AggregationFunction


class MeanAggregation(AggregationFunction):
    """Mean aggregation function for graph neural networks.

    This aggregation function computes the mean of all messages sent to each node.
    It provides a stable and commonly-used way to combine information from multiple
    neighboring nodes in the graph.
    """

    def __init__(self) -> None:
        """Initialize the mean aggregation function."""
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Aggregate messages using mean operation.

        Args:
            node_embeddings: The current node embeddings.
            messages: The messages to aggregate.
            indices: Indices indicating which messages belong to which nodes.

        Returns:
            Aggregated messages with the same shape as node_embeddings.
        """
        sum_msg = torch.zeros_like(node_embeddings)
        cnt_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, indices, messages)
        cnt_msg.index_add_(0, indices, torch.ones_like(messages))
        avg_msg = sum_msg / (cnt_msg + 1E-16)  # Avoid division by zero
        return avg_msg


class SumAggregation(AggregationFunction):
    """Sum aggregation function for graph neural networks.

    This aggregation function computes the sum of all messages sent to each node.
    It preserves the magnitude of information flow and is useful when you want
    to accumulate information from all neighbors.
    """

    def __init__(self) -> None:
        """Initialize the sum aggregation function."""
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Aggregate messages using sum operation.

        Args:
            node_embeddings: The current node embeddings.
            messages: The messages to aggregate.
            indices: Indices indicating which messages belong to which nodes.

        Returns:
            Aggregated messages with the same shape as node_embeddings.
        """
        sum_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, indices, messages)
        return sum_msg


class HardMaximumAggregation(AggregationFunction):
    """Hard maximum aggregation function for graph neural networks.

    This aggregation function selects the maximum value from all messages sent
    to each node. It emphasizes the strongest signals and can be useful for
    attention-like mechanisms or when only the most important information matters.
    """

    def __init__(self) -> None:
        """Initialize the hard maximum aggregation function."""
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Aggregate messages using hard maximum operation.

        Args:
            node_embeddings: The current node embeddings.
            messages: The messages to aggregate.
            indices: Indices indicating which messages belong to which nodes.

        Returns:
            Aggregated messages with the same shape as node_embeddings.
        """
        max_msg = torch.full_like(node_embeddings, float('-inf'))  # include_self=False leads to an error for some reason. Use -inf to get the same result.
        max_msg.index_reduce_(0, indices, messages, reduce='amax', include_self=True)
        return max_msg


class SmoothMaximumAggregation(AggregationFunction):
    """Smooth maximum aggregation function using LogSumExp for graph neural networks.

    This aggregation function computes a smooth approximation to the maximum
    using the LogSumExp (log-sum-exponential) operation. It provides a
    differentiable alternative to hard maximum that emphasizes the largest
    values while still considering other inputs.
    """

    def __init__(self) -> None:
        """Initialize the smooth maximum aggregation function."""
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Aggregate messages using smooth maximum (LogSumExp) operation.

        Args:
            node_embeddings: The current node embeddings.
            messages: The messages to aggregate.
            indices: Indices indicating which messages belong to which nodes.

        Returns:
            Aggregated messages with the same shape as node_embeddings.
        """
        exps_max = torch.zeros_like(node_embeddings)
        exps_max.index_reduce_(0, indices, messages, reduce="amax", include_self=False)
        exps_max = exps_max.detach()
        MAXIMUM_SMOOTHNESS = 12.0  # As the value approaches infinity, the hard maximum is attained
        max_offsets = exps_max.index_select(0, indices).detach()
        exps = (MAXIMUM_SMOOTHNESS * (messages - max_offsets)).exp()
        exps_sum = torch.full_like(node_embeddings, 1E-16)
        exps_sum.index_add_(0, indices, exps)
        max_msg = ((1.0 / MAXIMUM_SMOOTHNESS) * exps_sum.log()) + exps_max
        return max_msg
