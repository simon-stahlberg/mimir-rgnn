import torch

from .bases import Decoder
from .configs import HyperparameterConfig
from .encoders import EncodedTensors
from .modules import SumReadout, MLP


class ActionScalarDecoder(Decoder):
    """Decoder for scalar values over actions.

    This decoder produces a scalar value for each action by combining action
    embeddings with object embeddings. Commonly used for Q-values or action
    preferences in reinforcement learning applications.
    """

    def __init__(self, hparam_config: 'HyperparameterConfig'):
        """Initialize the action scalar decoder.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
        """
        super().__init__()
        self._object_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
        self._action_value = MLP(2 * hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> list[torch.Tensor]:
        """Compute scalar values for each action.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing action and object indices.

        Returns:
            List of tensors, where each tensor contains scalar values for the
            actions in the corresponding input instance.
        """
        action_embeddings = node_embeddings.index_select(0, encoding.action_indices)
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        object_aggregation: torch.Tensor = self._object_readout(object_embeddings, encoding.object_sizes)
        object_aggregation = object_aggregation.repeat_interleave(encoding.action_sizes, dim=0)
        values: torch.Tensor = self._action_value(torch.cat((action_embeddings, object_aggregation), dim=1))
        return [action_values.view(-1) for action_values in values.split(encoding.action_sizes.tolist())]  # type: ignore


class ActionEmbeddingDecoder(Decoder):
    """Decoder for embeddings over actions.

    This decoder extracts the learned embeddings for action nodes without
    any additional processing. Useful when you want to access the raw
    action representations learned by the graph neural network.
    """

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        """Extract action node embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing action indices.

        Returns:
            Tensor containing the embeddings of all action nodes.
        """
        return node_embeddings.index_select(0, encoding.action_indices)


class ObjectsScalarDecoder(Decoder):
    """Decoder for scalar values over objects.

    This decoder produces a single scalar value by aggregating all object
    embeddings within each input instance. Commonly used for state value
    estimation or global state assessment.
    """

    def __init__(self, hparam_config: 'HyperparameterConfig'):
        """Initialize the objects scalar decoder.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
        """
        super().__init__()
        self._object_readout = SumReadout(hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        """Compute scalar values by aggregating object embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing object indices and sizes.

        Returns:
            Tensor containing one scalar value per input instance.
        """
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        return self._object_readout(object_embeddings, encoding.object_sizes).view(-1)


class ObjectsEmbeddingDecoder(Decoder):
    """Decoder for embeddings over objects.

    This decoder extracts the learned embeddings for object nodes without
    any additional processing. Useful when you want to access the raw
    object representations learned by the graph neural network.
    """

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        """Extract object node embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing object indices.

        Returns:
            Tensor containing the embeddings of all object nodes.
        """
        return node_embeddings.index_select(0, encoding.object_indices)
