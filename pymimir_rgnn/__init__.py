"""Mimir-RGNN: Relational Graph Neural Networks for AI Planning.

This package implements Relational Graph Neural Networks (R-GNN) for AI planning
applications. It provides a flexible and typed interface for building graph neural
networks that operate on PDDL planning domains and problems.

The main components include:

- **RelationalGraphNeuralNetwork**: The main R-GNN model class
- **Encoders**: Transform PDDL structures (states, goals, actions) into graphs
- **Decoders**: Extract outputs (values, embeddings) from node representations
- **Aggregation Functions**: Combine messages during graph neural network computation
- **Message/Update Functions**: Define how nodes communicate and update

Key Features:
- Seamless integration with PDDL via Mimir
- Flexible encoder/decoder architecture
- GPU acceleration via PyTorch
- Type-safe interface
- Batched processing support

Example:
    >>> import pymimir as mm
    >>> import pymimir_rgnn as rgnn
    >>>
    >>> # Load PDDL domain
    >>> domain = mm.Domain.from_file('path/to/domain.pddl')
    >>>
    >>> # Configure the R-GNN
    >>> hparam_config = rgnn.HyperparameterConfig(
    ...     domain=domain,
    ...     embedding_size=64,
    ...     num_layers=30,
    ... )
    >>>
    >>> # Define input/output specifications
    >>> input_spec = (rgnn.StateEncoder(), rgnn.GoalEncoder())
    >>> output_spec = [('value', rgnn.ObjectsScalarDecoder(hparam_config))]
    >>>
    >>> # Configure modules
    >>> module_config = rgnn.ModuleConfig(
    ...     aggregation_function=rgnn.MeanAggregation(),
    ...     message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
    ...     update_function=rgnn.MLPUpdates(hparam_config)
    ... )
    >>>
    >>> # Create model
    >>> model = rgnn.RelationalGraphNeuralNetwork(
    ...     hparam_config, module_config, input_spec, output_spec
    ... )
"""

from .aggregations import HardMaximumAggregation, MeanAggregation, SmoothMaximumAggregation, SumAggregation
from .bases import AggregationFunction, MessageFunction, UpdateFunction, Encoder, Decoder, QuantizationRecord
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import ActionScalarDecoder, ActionEmbeddingDecoder, ObjectsScalarDecoder, ObjectsEmbeddingDecoder
from .encoders import EncoderRelation, EncoderRelationKind, StateEncoder, GoalEncoder, GroundActionsEncoder, TransitionEffectsEncoder, ExpressiveStateEncoder, ExpressiveGoalEncoder
from .messages import PredicateMLPMessages, PredicateLinearMessages, SenderOnlyMLPMessages, SparseMLPMessages, AttentionMessages, AttentionMessagesBase
from .model import ForwardState, RelationalGraphNeuralNetwork
from .modules import MLP, SparseMLP, ChannelwiseAffine, SumReadout
from .updates import MLPUpdates, LinearUpdates, SparseMLPUpdates
from .utils import boundary_margin_penalty

__all__ = [
    "ActionEmbeddingDecoder",
    "ActionScalarDecoder",
    "AggregationFunction",
    "AttentionMessages",
    "AttentionMessagesBase",
    "ChannelwiseAffine",
    "Decoder",
    "Encoder",
    "EncoderRelation",
    "EncoderRelationKind",
    "ExpressiveGoalEncoder",
    "ExpressiveStateEncoder",
    "ForwardState",
    "GoalEncoder",
    "GroundActionsEncoder",
    "HardMaximumAggregation",
    "HyperparameterConfig",
    "LinearUpdates",
    "MeanAggregation",
    "MessageFunction",
    "MLP",
    "MLPUpdates",
    "ModuleConfig",
    "ObjectsEmbeddingDecoder",
    "ObjectsScalarDecoder",
    "PredicateLinearMessages",
    "PredicateMLPMessages",
    "QuantizationRecord",
    "RelationalGraphNeuralNetwork",
    "SenderOnlyMLPMessages",
    "SmoothMaximumAggregation",
    "SparseMLP",
    "SparseMLPMessages",
    "SparseMLPUpdates",
    "StateEncoder",
    "SumAggregation",
    "SumReadout",
    "TransitionEffectsEncoder",
    "UpdateFunction",
    "boundary_margin_penalty",
]
