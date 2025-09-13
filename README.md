# Mimir-RGNN

[![PyPI Version](https://img.shields.io/pypi/v/pymimir-rgnn)](https://pypi.org/project/pymimir-rgnn/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymimir-rgnn)](https://pypi.org/project/pymimir-rgnn/)
[![License](https://img.shields.io/pypi/l/pymimir-rgnn)](https://github.com/simon-stahlberg/mimir-rgnn/blob/master/LICENSE)
[![Tests](https://github.com/simon-stahlberg/mimir-rgnn/actions/workflows/test.yml/badge.svg)](https://github.com/simon-stahlberg/mimir-rgnn/actions/workflows/test.yml)

**Mimir-RGNN** is a Python library that implements Relational Graph Neural Networks (R-GNN) for AI planning applications. Built on PyTorch and Mimir, it provides a powerful and flexible interface for learning on structured relational data, particularly PDDL planning domains.

## Key Features

- **🧠 Relational Graph Neural Networks**: R-GNN implementation for structured reasoning
- **📋 PDDL Integration**: Seamless integration with PDDL planning domains and problems via Mimir
- **⚡ PyTorch Backend**: Built on PyTorch for GPU acceleration
- **🔧 Flexible Configuration**: Declarative configuration system with class-based encoder API
- **🎯 Planning-Focused**: Designed specifically for AI planning and reinforcement learning applications
- **📊 Multiple Aggregation Functions**: Support for various message aggregation strategies
- **🏗️ Professional API**: Clean, type-safe interface with comprehensive documentation
- **🔌 Extensible**: Inherit from base encoder classes to create custom input/output processing

## Installation

Install Mimir-RGNN from PyPI:

```bash
pip install pymimir-rgnn
```

### Requirements

- Python 3.11+
- PyTorch 2.6.0+
- Mimir 0.13.42+

## Quick Start

```python
import pymimir as mm
import pymimir_rgnn as rgnn

# Load a PDDL domain
domain = mm.Domain('path/to/domain.pddl')

# Configure using the class-based encoder/decoder API
config = rgnn.RelationalGraphNeuralNetworkConfig(
    domain=domain,
    input_specification=(
        rgnn.StateEncoder(), 
        rgnn.GoalEncoder(),
        rgnn.GroundActionsEncoder()
    ),
    output_specification=[
        ('q_values', rgnn.ActionScalarDecoder(64)),
        ('state_value', rgnn.ObjectsScalarDecoder(64))
    ],
    embedding_size=64,
    num_layers=5,
    message_aggregation=rgnn.AggregationFunction.HardMaximum
)

# Create the model
model = rgnn.RelationalGraphNeuralNetwork(config)

# Use the model for inference
problem = mm.Problem(domain, 'path/to/problem.pddl')
state = problem.get_initial_state()
actions = state.generate_applicable_actions()
goal = problem.get_goal_condition()

output = model([(state, goal, actions)])
q_values = output.readout('q_values')
state_value = output.readout('state_value')
```

## API Overview

### Core Components

#### `RelationalGraphNeuralNetworkConfig`
Central configuration class that defines:
- **Domain**: The PDDL domain for the planning problem
- **Input Specification**: Types of inputs using encoder classes
- **Output Specification**: Named outputs with decoder classes
- **Model Parameters**: Embedding size, number of layers, aggregation functions

#### `RelationalGraphNeuralNetwork`
The main R-GNN model class that:
- Processes relational graph structures from PDDL problems
- Supports various input types (states, goals, actions, effects)
- Provides flexible output configurations for different applications
- Handles batched inference efficiently

### Input Encoders

- **`StateEncoder()`**: Current state of the planning problem
- **`GoalEncoder()`**: Goal specification
- **`GroundActionsEncoder()`**: Available ground actions
- **`TransitionEffectsEncoder()`**: Action effects and transitions
- **`SuccessorsEncoder()`**: State successors

### Output Decoders

- **`ActionScalarDecoder(embedding_size)`**: Scalar values over actions
- **`ActionEmbeddingDecoder()`**: Embeddings over actions  
- **`ObjectsScalarDecoder(embedding_size)`**: Scalar values over objects
- **`ObjectsEmbeddingDecoder()`**: Embeddings over objects

### Aggregation Functions

- **`AggregationFunction.Add`**: Sum aggregation
- **`AggregationFunction.Mean`**: Mean aggregation  
- **`AggregationFunction.HardMaximum`**: Hard maximum
- **`AggregationFunction.SmoothMaximum`**: Smooth maximum (LogSumExp)

## Extensibility

The class-based encoder/decoder API allows you to extend the library by inheriting from base classes:

### Custom Input Encoders

```python
class CustomStateEncoder(rgnn.StateEncoder):
    """Custom state encoder with additional relations."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        # Get standard state relations
        relations = super().get_relations(domain)
        
        # Add custom relations
        relations.append(("custom_spatial_relation", 3))
        
        return relations
    
    def encode(self, input_value: Any, intermediate: rgnn.EncodedInput, state: mm.State) -> int:
        # Get standard encoding
        nodes_added = super().encode(input_value, intermediate, state)
        
        # Add custom encoding logic here
        # ... custom processing ...
        
        return nodes_added

# Use in configuration
config = rgnn.RelationalGraphNeuralNetworkConfig(
    domain=domain,
    input_specification=(CustomStateEncoder(), rgnn.GoalEncoder()),
    output_specification=[('value', rgnn.ObjectsScalarDecoder(64))],
    # ... other parameters
)
```

### Custom Output Decoders  

```python
class CustomActionDecoder(rgnn.ActionScalarDecoder):
    """Custom action decoder with specialized readout logic."""
    
    def __init__(self, embedding_size: int):
        super().__init__(embedding_size)
        # Add custom readout components
        from pymimir_rgnn.modules import MLP
        self._custom_layer = MLP(embedding_size, embedding_size)
    
    def forward(self, node_embeddings: torch.Tensor, input: rgnn.EncodedInput) -> torch.Tensor:
        # Implement custom readout logic
        action_embeddings = node_embeddings.index_select(0, input.action_indices)
        custom_embeddings = self._custom_layer(action_embeddings)
        # ... custom processing ...
        return custom_embeddings

# Use in configuration  
config = rgnn.RelationalGraphNeuralNetworkConfig(
    domain=domain,
    input_specification=(rgnn.StateEncoder(),),
    output_specification=[('custom_q', CustomActionDecoder(64))],
    # ... other parameters
)
```

## Examples and Tutorials

For an comprehensive example, visit:

- [Example Project Repository](https://github.com/simon-stahlberg/relational-neural-network-python/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Development setup
- Coding standards
- Testing requirements  
- Pull request process

## License

This project is licensed under the GNU General Public License v3.0 or later. See the [LICENSE](LICENSE) file for details.

## Citation

If you use Mimir-RGNN in your research, please cite:

```bibtex
@software{stahlberg2024mimir_rgnn,
  author = {Simon St\r{a}hlberg},
  title = {Mimir-RGNN: Relational Graph Neural Networks for AI Planning},
  url = {https://github.com/simon-stahlberg/mimir-rgnn},
  version = {<package version>},
  year = {<year package version was released>}
}
```

## Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/simon-stahlberg/mimir-rgnn/issues)
- 📧 **Contact**: simon.stahlberg@gmail.com
