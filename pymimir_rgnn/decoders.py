import torch

from .bases import Decoder
from .configs import HyperparameterConfig
from .encoders import EncodedTensors
from .modules import SumReadout, MLP


# Decoder implementations
class ActionScalarDecoder(Decoder):
    """Decoder for scalar values over actions."""

    def __init__(self, config: 'HyperparameterConfig'):
        super().__init__()
        self._object_readout = SumReadout(config.embedding_size, config.embedding_size)
        self._action_value = MLP(2 * config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> list[torch.Tensor]:
        action_embeddings = node_embeddings.index_select(0, encoding.action_indices)
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        object_aggregation: torch.Tensor = self._object_readout(object_embeddings, encoding.object_sizes)
        object_aggregation = object_aggregation.repeat_interleave(encoding.action_sizes, dim=0)
        values: torch.Tensor = self._action_value(torch.cat((action_embeddings, object_aggregation), dim=1))
        return [action_values.view(-1) for action_values in values.split(encoding.action_sizes.tolist())]  # type: ignore


class ActionEmbeddingDecoder(Decoder):
    """Decoder for embeddings over actions."""

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        return node_embeddings.index_select(0, encoding.action_indices)


class ObjectsScalarDecoder(Decoder):
    """Decoder for scalar values over objects."""

    def __init__(self, config: 'HyperparameterConfig'):
        super().__init__()
        self._object_readout = SumReadout(config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        return self._object_readout(object_embeddings, encoding.object_sizes).view(-1)


class ObjectsEmbeddingDecoder(Decoder):
    """Decoder for embeddings over objects."""

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        return node_embeddings.index_select(0, encoding.object_indices)
