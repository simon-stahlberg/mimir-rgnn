import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: 'int', output_size: 'int'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._inner = nn.Linear(input_size, input_size, True)
        self._outer = nn.Linear(input_size, output_size, True)

    def forward(self, input: 'torch.Tensor'):
        return self._outer(nn.functional.mish(self._inner(input)))


class SumReadout(nn.Module):
    def __init__(self, input_size: 'int', output_size: 'int'):
        super().__init__()
        self._value = MLP(input_size, output_size)

    def forward(self, node_embeddings: 'torch.Tensor', node_sizes: 'torch.Tensor') -> 'torch.Tensor':
        cumsum_indices = node_sizes.cumsum(0) - 1
        cumsum_states = node_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)
