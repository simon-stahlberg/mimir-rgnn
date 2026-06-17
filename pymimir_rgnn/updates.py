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
    """MLPUpdates variant where the update MLP is a SparseMLP.

    Each output feature of the update MLP is connected to at most k input
    features per layer. During training a Gumbel-Sigmoid gate produces a soft
    differentiable mask; at eval time a deterministic top-k hard mask is used.
    Call sparsity_penalty() and add it (scaled) to the external training loss.

    Args:
        hparam_config: Hyperparameter configuration.
        k: Maximum number of active input connections per output row per layer.
        tau: Temperature for the Gumbel-Sigmoid gate sampler (lower = harder).
        linear: If True, use a single gated linear layer with no hidden layer or activation.
    """

    def __init__(self, hparam_config: HyperparameterConfig, k: int, tau: float = 1.0, linear: bool = False):
        super().__init__()
        self._update = SparseMLP(2 * hparam_config.embedding_size, hparam_config.embedding_size, k=k, tau=tau, linear=linear)

    def sparsity_penalty(self) -> torch.Tensor:
        """SparseMLP sparsity penalty for the update network."""
        return self._update.sparsity_penalty()

    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """Update node embeddings using the sparse MLP.

        Args:
            node_embeddings: The current node embeddings.
            aggregated_messages: The aggregated messages to incorporate.

        Returns:
            Updated node embeddings with the same shape as the input embeddings.
        """
        return self._update(torch.cat((aggregated_messages, node_embeddings), 1))
