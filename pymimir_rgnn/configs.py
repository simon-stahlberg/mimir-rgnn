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
        normalize_updates: Whether to apply normalization to embedding updates.
        channelwise_normalization: Whether to use a per-channel learnable affine
            instead of layer normalization for the embedding updates. Keeps each
            channel independent, which is required for per-bit concept decoding.
            Requires normalize_updates.
        global_readout: Whether to use a global readout for the node embeddings.
        residual_updates: Whether to use residual updates for the node embeddings.
        or_residual_updates: Whether to combine consecutive binary embeddings with
            an elementwise OR. Keeps embeddings binary while making concepts
            persist monotonically across layers. Requires binarize_updates and
            excludes residual_updates.
        binarize_updates: Whether to binarize the updates for the node embeddings.
        ternarize_messages: Whether to cast predicate-MLP messages into {-1, 0, 1}
            via a deterministic straight-through ternarizer before aggregation.

    Raises:
        ValueError: If an invalid combination of flags is given.
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
        metadata={'doc': 'Whether to apply normalization to the embedding updates.'}
    )

    channelwise_normalization: bool = field(
        default=False,
        metadata={'doc': 'Whether to use a per-channel learnable affine instead of layer '
                         'normalization for the embedding updates. Requires normalize_updates.'}
    )

    global_readout: bool = field(
        default=False,
        metadata={'doc': 'Whether to use a global readout for the node embeddings.'}
    )

    residual_updates: bool = field(
        default=True,
        metadata={'doc': 'Whether to use residual updates for the node embeddings.'}
    )

    or_residual_updates: bool = field(
        default=False,
        metadata={'doc': 'Whether to combine consecutive binary embeddings with an elementwise '
                         'OR. Requires binarize_updates and excludes residual_updates.'}
    )

    binarize_updates: bool = field(
        default=False,
        metadata={'doc': 'Whether to binarize the updates for the node embeddings.'}
    )

    ternarize_messages: bool = field(
        default=False,
        metadata={'doc': 'Whether to cast predicate-MLP messages into {-1, 0, 1} via '
                         'a deterministic straight-through ternarizer before aggregation.'}
    )

    def __post_init__(self) -> None:
        if self.or_residual_updates and not self.binarize_updates:
            raise ValueError('or_residual_updates requires binarize_updates: '
                             'an elementwise OR is only meaningful on binary embeddings.')
        if self.or_residual_updates and self.residual_updates:
            raise ValueError('or_residual_updates and residual_updates are mutually exclusive: '
                             'enable at most one residual scheme.')
        if self.channelwise_normalization and not self.normalize_updates:
            raise ValueError('channelwise_normalization requires normalize_updates: '
                             'the flag would otherwise have no effect.')
        if self.residual_updates and self.binarize_updates:
            raise ValueError('residual_updates and binarize_updates are incompatible: '
                             'additive residuals over binary updates produce integer embeddings, '
                             'not bits. Use or_residual_updates instead.')


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
