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
    >>> domain = mm.Domain('path/to/domain.pddl')
    >>>
    >>> # Configure the R-GNN
    >>> hparam_config = rgnn.HyperparameterConfig(
    ...     domain=domain,
    ...     embedding_size=64,
    ...     num_layers=30,
    ... )
    >>>
    >>> # Configure modules
    >>> module_config = rgnn.ModuleConfig(
    ...     aggregation_function=rgnn.MeanAggregation(),
    ...     message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
    ...     update_function=rgnn.MLPUpdates(hparam_config)
    ... )
    >>>
    >>> # Define input/output specifications
    >>> input_spec = (rgnn.StateEncoder(), rgnn.GoalEncoder())
    >>> output_spec = [('q_values', rgnn.ActionScalarDecoder(hparam_config))]
    >>>
    >>> # Create model
    >>> model = rgnn.RelationalGraphNeuralNetwork(
    ...     hparam_config, module_config, input_spec, output_spec
    ... )
"""

from .aggregations import HardMaximumAggregation, MeanAggregation, SmoothMaximumAggregation, SumAggregation
from .bases import AggregationFunction, MessageFunction, UpdateFunction, Encoder, Decoder
from .decoders import ActionScalarDecoder, ActionEmbeddingDecoder, ObjectsScalarDecoder, ObjectsEmbeddingDecoder
from .encoders import StateEncoder, GoalEncoder, GroundActionsEncoder, TransitionEffectsEncoder
from .configs import HyperparameterConfig, ModuleConfig
from .model import ForwardState, RelationalGraphNeuralNetwork
from .messages import PredicateMLPMessages, AttentionMessages
from .updates import MLPUpdates

__all__ = [
    "ActionEmbeddingDecoder",
    "ActionScalarDecoder",
    "AggregationFunction",
    "AttentionMessages",
    "Decoder",
    "Encoder",
    "ForwardState",
    "GoalEncoder",
    "GroundActionsEncoder",
    "HardMaximumAggregation",
    "HyperparameterConfig",
    "MeanAggregation",
    "MessageFunction",
    "MLPUpdates",
    "ModuleConfig",
    "ObjectsEmbeddingDecoder",
    "ObjectsScalarDecoder",
    "PredicateMLPMessages",
    "RelationalGraphNeuralNetwork",
    "SmoothMaximumAggregation",
    "StateEncoder",
    "SumAggregation",
    "TransitionEffectsEncoder",
    "UpdateFunction",
]
