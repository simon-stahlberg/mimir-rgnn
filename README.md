# Mimir-RGNN

[![PyPI Version](https://img.shields.io/pypi/v/pymimir-rgnn)](https://pypi.org/project/pymimir-rgnn/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymimir-rgnn)](https://pypi.org/project/pymimir-rgnn/)
[![License](https://img.shields.io/pypi/l/pymimir-rgnn)](https://github.com/simon-stahlberg/mimir-rgnn/blob/master/LICENSE)
[![Tests](https://github.com/simon-stahlberg/mimir-rgnn/actions/workflows/test.yml/badge.svg)](https://github.com/simon-stahlberg/mimir-rgnn/actions/workflows/test.yml)

**Mimir-RGNN** is a Python library that implements Relational Graph Neural Networks (R-GNN) for AI planning applications. Built on PyTorch and Mimir, it provides a powerful and flexible interface for learning on structured relational data, particularly PDDL planning domains.

## Key Features

- **🧠 Relational Graph Neural Networks**: State-of-the-art R-GNN implementation for structured reasoning
- **📋 PDDL Integration**: Seamless integration with PDDL planning domains and problems via Mimir
- **⚡ PyTorch Backend**: Built on PyTorch for efficient computation and GPU acceleration
- **🔧 Flexible Configuration**: Declarative configuration system for input/output specifications
- **🎯 Planning-Focused**: Designed specifically for AI planning and reinforcement learning applications
- **📊 Multiple Aggregation Functions**: Support for various message aggregation strategies
- **🏗️ Professional API**: Clean, type-safe interface with comprehensive documentation

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

# Configure the R-GNN model
config = rgnn.RelationalGraphNeuralNetworkConfig(
    domain=domain,
    input_specification=(rgnn.InputType.State, rgnn.InputType.Goal),
    output_specification=[('q_values', rgnn.OutputNodeType.Action, rgnn.OutputValueType.Scalar)],
    embedding_size=64,
    num_layers=5,
    message_aggregation=rgnn.AggregationFunction.HardMaximum
)

# Create and initialize the model
model = rgnn.RelationalGraphNeuralNetwork(config)

# Use the model for inference
# states = [...]  # List of Mimir State objects
# goals = [...]   # List of Mimir GroundConjunctiveCondition objects
# actions = [...]  # List of lists of Mimir GroundAction objects
# 
# inputs = list(zip(states, actions, goals))
# outputs = model(inputs)
```

## API Overview

### Core Components

#### `RelationalGraphNeuralNetworkConfig`
Central configuration class that defines:
- **Domain**: The PDDL domain for the planning problem
- **Input Specification**: Types of inputs (State, Goal, GroundActions, etc.)
- **Output Specification**: Named outputs with node types and value types
- **Model Parameters**: Embedding size, number of layers, aggregation functions

#### `RelationalGraphNeuralNetwork`
The main R-GNN model class that:
- Processes relational graph structures from PDDL problems
- Supports various input types (states, goals, actions, effects)
- Provides flexible output configurations for different applications
- Handles batched inference efficiently

### Input Types

- **`InputType.State`**: Current state of the planning problem
- **`InputType.Goal`**: Goal specification
- **`InputType.GroundActions`**: Available ground actions
- **`InputType.TransitionEffects`**: Action effects and transitions

### Output Specifications

Configure outputs by node type and value type:

```python
output_specification = [
    ('actor', rgnn.OutputNodeType.Action, rgnn.OutputValueType.Scalar),
    ('critic', rgnn.OutputNodeType.Objects, rgnn.OutputValueType.Scalar),
    ('embeddings', rgnn.OutputNodeType.All, rgnn.OutputValueType.Embeddings)
]
```

### Aggregation Functions

- **`AggregationFunction.Add`**: Sum aggregation
- **`AggregationFunction.Mean`**: Mean aggregation  
- **`AggregationFunction.HardMaximum`**: Hard maximum
- **`AggregationFunction.SmoothMaximum`**: Smooth maximum (Gumbel)

## Applications

Mimir-RGNN is particularly well-suited for:

- **Reinforcement Learning**: Value function approximation and policy learning
- **Planning**: Heuristic learning and action selection
- **Graph Neural Networks**: Learning on structured relational data
- **AI Research**: Experiments with neural-symbolic integration

## Examples and Tutorials

For comprehensive examples and tutorials, visit:

- [Example Project Repository](https://github.com/simon-stahlberg/relational-neural-network-python/)
- [Documentation](#) _(coming soon)_

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
  author = {Simon Stahlberg},
  title = {Mimir-RGNN: Relational Graph Neural Networks for AI Planning},
  url = {https://github.com/simon-stahlberg/mimir-rgnn},
  version = {0.1.3},
  year = {2024}
}
```

## Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/simon-stahlberg/mimir-rgnn/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/simon-stahlberg/mimir-rgnn/discussions)
- 📧 **Contact**: simon.stahlberg@gmail.com
