import pymimir as mm

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bases import AggregationFunction, MessageFunction, UpdateFunction


@dataclass
class HyperparameterConfig:
    """Configuration class for R-GNN model hyperparameters.

    This class contains all the hyperparameters needed to configure a Relational
    Graph Neural Network, including model architecture parameters and training settings.

    Attributes:
        domain: The PDDL domain for the planning problem.
        embedding_size: The size of the node embeddings.
        num_layers: The number of message passing layers.
        normalize_updates: Whether to apply layer normalization to embedding updates.
        global_readout: Whether to use a global readout for the node embeddings.
        residual_updates: Whether to use residual updates for the node embeddings.
        binarize_updates: Whether to binarize the updates for the node embeddings.
    """

    domain: mm.Domain = field(
        metadata={'doc': 'The domain of the planning problem.'}
    )

    embedding_size: int = field(
        default=32,
        metadata={'doc': 'The size of the node embeddings.'}
    )

    num_layers: int = field(
        default=30,
        metadata={'doc': 'The number of message passing layers.'}
    )

    normalize_updates: bool = field(
        default=True,
        metadata={'doc': 'Whether to apply layer normalization to the embedding updates.'}
    )

    global_readout: bool = field(
        default=False,
        metadata={'doc': 'Whether to use a global readout for the node embeddings.'}
    )

    residual_updates: bool = field(
        default=True,
        metadata={'doc': 'Whether to use residual updates for the node embeddings.'}
    )

    binarize_updates: bool = field(
        default=False,
        metadata={'doc': 'Whether to binarize the updates for the node embeddings.'}
    )


@dataclass
class ModuleConfig:
    """Configuration for neural network modules used in the RGNN.

    This class specifies which neural network modules to use for the three
    main components of the graph neural network: aggregation, message computation,
    and node updates.

    Attributes:
        aggregation_function: The aggregation function used to combine messages.
        message_function: The message function used to compute messages between nodes.
        update_function: The update function used to update node embeddings.
    """

    aggregation_function: 'AggregationFunction' = field(
        metadata={'doc': 'The aggregation function used to combine messages.'}
    )

    message_function: 'MessageFunction' = field(
        metadata={'doc': 'The message function used to compute messages between nodes.'}
    )

    update_function: 'UpdateFunction' = field(
        metadata={'doc': 'The update function used to update node embeddings.'}
    )
