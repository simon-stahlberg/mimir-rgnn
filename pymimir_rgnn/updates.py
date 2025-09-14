import torch

from .bases import UpdateFunction
from .configs import HyperparameterConfig
from .modules import MLP


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
