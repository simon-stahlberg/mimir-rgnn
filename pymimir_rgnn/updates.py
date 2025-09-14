import torch

from .configs import HyperparameterConfig
from .modules import MLP

class MLPUpdates:
    def __init__(self, config: HyperparameterConfig):
        super().__init__()
        self._update = MLP(2 * config.embedding_size, config.embedding_size)

    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        return self._update(torch.cat((aggregated_messages, node_embeddings), 1))
