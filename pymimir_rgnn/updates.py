from typing import Literal

import torch

from .bases import UpdateFunction
from .configs import HyperparameterConfig
from .modules import MLP, SparseMLP


class MLPUpdates(UpdateFunction):
    """Update function using a multi-layer perceptron for node embedding updates.

    This update function uses an MLP to compute new node embeddings based on
    the concatenation of current embeddings and aggregated messages. It provides
    a learnable way to combine the node's current state with incoming information.
    """

    def __init__(self, hparam_config: HyperparameterConfig):
        """Initialize the MLP update function.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
        """
        super().__init__()
        self._update = MLP(2 * hparam_config.embedding_size, hparam_config.embedding_size)

    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """Update node embeddings using the MLP.

        Args:
            node_embeddings: The current node embeddings.
            aggregated_messages: The aggregated messages to incorporate.

        Returns:
            Updated node embeddings with the same shape as the input embeddings.
        """
        return self._update(torch.cat((aggregated_messages, node_embeddings), 1))


class SparseMLPUpdates(UpdateFunction):
    """MLPUpdates variant with a hard top-k gated linear update map.

    Each output feature is a linear function of at most k original input
    features. The implementation still uses dense PyTorch linear operations,
    with a hard top-k mask applied to the dense weight matrix. The top-k margin
    penalty encourages stable separation between selected and rejected gate
    logits.

    Args:
        hparam_config: Hyperparameter configuration.
        k: Maximum number of active input connections per output feature.
        tau: Temperature for the straight-through sigmoid surrogate.
        gate_mode: Training-time hard top-k mode. Evaluation always uses
            deterministic top-k.
    """

    def __init__(
        self,
        hparam_config: HyperparameterConfig,
        k: int,
        tau: float = 1.0,
        gate_mode: Literal["gumbel_topk", "deterministic_topk"] = "gumbel_topk",
    ):
        super().__init__()
        self._update = SparseMLP(2 * hparam_config.embedding_size, hparam_config.embedding_size, k=k, tau=tau, gate_mode=gate_mode)

    def topk_margin_penalty(self, margin: float) -> torch.Tensor:
        """SparseMLP top-k margin penalty for the update network."""
        return self._update.topk_margin_penalty(margin)

    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """Update node embeddings using the sparse MLP.

        Args:
            node_embeddings: The current node embeddings.
            aggregated_messages: The aggregated messages to incorporate.

        Returns:
            Updated node embeddings with the same shape as the input embeddings.
        """
        return self._update(torch.cat((aggregated_messages, node_embeddings), 1))
