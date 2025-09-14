import pymimir as mm

from dataclasses import dataclass, field


@dataclass
class HyperparameterConfig:
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


# TODO: Introduce a ModuleConfig class for AggregationFunction, MessageFunction, UpdateFunction, etc.
