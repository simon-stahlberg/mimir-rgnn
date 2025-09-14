import torch

from .bases import AggregationFunction


class MeanAggregation(AggregationFunction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        sum_msg = torch.zeros_like(node_embeddings)
        cnt_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, indices, messages)
        cnt_msg.index_add_(0, indices, torch.ones_like(messages))
        avg_msg = sum_msg / cnt_msg
        return avg_msg


class SumAggregation(AggregationFunction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        sum_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, indices, messages)
        return sum_msg


class HardMaximumAggregation(AggregationFunction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        max_msg = torch.full_like(node_embeddings, float('-inf')) # include_self=False leads to an error for some reason. Use -inf to get the same result.
        max_msg.index_reduce_(0, indices, messages, reduce='amax', include_self=True)
        return max_msg


class SmoothMaximumAggregation(AggregationFunction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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
