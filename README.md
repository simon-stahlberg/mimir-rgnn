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
- **🔧 Flexible Configuration**: Declarative configuration system for input/output specifications
- **🎯 Planning-Focused**: Designed specifically for AI planning and reinforcement learning applications
- **📊 Multiple Aggregation Functions**: Support for various message aggregation strategies
- **🏗️ Typed API**: Clean and type-safe interface

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

# Configure the R-GNN hyperparameters
hparam_config = rgnn.HyperparameterConfig(
    domain=domain,
    embedding_size=64,
    num_layers=30,
)

# Define input and output specifications using encoder/decoder classes
input_spec = (rgnn.StateEncoder(), rgnn.GroundActionsEncoder(), rgnn.GoalEncoder())
output_spec = [('q_values', rgnn.ActionScalarDecoder(hparam_config))]

# Configure the R-GNN modules (aggregation, message, and update functions)
module_config = rgnn.ModuleConfig(
    aggregation_function=rgnn.MeanAggregation(),
    message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
    update_function=rgnn.MLPUpdates(hparam_config)
)

# Create and initialize the model
model = rgnn.RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)

# Use the model for inference
# problem = mm.Problem(domain, 'path/to/problem.pddl')
# state = problem.get_initial_state()
# actions = state.generate_applicable_actions()
# goal = problem.get_goal_condition()
#
# inputs = [(state, actions, goal)]  # Input tuple matching input_spec order
# outputs = model(inputs)
# q_values = outputs.readout('q_values')
```

## API Overview

### Core Components

#### `HyperparameterConfig`
Configuration class for R-GNN model hyperparameters:
- **Domain**: The PDDL domain for the planning problem
- **Model Parameters**: Embedding size, number of layers
- **Training Settings**: Normalization, global readout options

#### `ModuleConfig`
Configuration class for R-GNN neural network modules:
- **Aggregation Function**: How messages are aggregated (mean, sum, max, etc.)
- **Message Function**: How messages are computed between related nodes
- **Update Function**: How node embeddings are updated with aggregated messages

#### Encoder/Decoder Classes
Extensible class-based system for defining inputs and outputs:
- **Input Specification**: Tuple of encoder instances (StateEncoder, GoalEncoder, etc.)
- **Output Specification**: List of named decoder instances with custom readout logic

#### `RelationalGraphNeuralNetwork`
The main R-GNN model class that:
- Takes hyperparameter config, module config, input specification, and output specification
- Processes relational graph structures from PDDL problems
- Supports extensible encoder/decoder system for custom input/output handling
- Handles batched inference efficiently

### Encoder Classes

Inherit from `Encoder` base class to define custom input processing:

- **`StateEncoder`**: Current state of the planning problem
- **`GoalEncoder`**: Goal specification  
- **`GroundActionsEncoder`**: Available ground actions
- **`TransitionEffectsEncoder`**: Action effects and transitions

### Decoder Classes

Inherit from `Decoder` base class to define custom output readout:

```python
input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
output_spec = [
    ('actor', ActionScalarDecoder(hparam_config)),
    ('critic', ObjectsScalarDecoder(hparam_config)), 
    ('embeddings', ActionEmbeddingDecoder())
]
```

### Aggregation Functions

Available in the `ModuleConfig`:

- **`MeanAggregation()`**: Mean aggregation
- **`SumAggregation()`**: Sum aggregation
- **`HardMaximumAggregation()`**: Hard maximum
- **`SmoothMaximumAggregation()`**: Smooth maximum (LogSumExp)

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
@inproceedings{stahlberg-bonet-geffner-icaps2022,
  author       = {Simon St{\aa}hlberg and Blai Bonet and Hector Geffner},
  title        = {Learning General Optimal Policies with Graph Neural Networks: Expressive Power, Transparency, and Limits},
  booktitle    = {Proceedings of the Thirty-Second International Conference on Automated Planning and Scheduling, {ICAPS} 2022, Singapore (virtual), June 13-24, 2022},
  pages        = {629--637},
  year         = {2022}
}
```

## Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/simon-stahlberg/mimir-rgnn/issues)
- 📧 **Contact**: simon.stahlberg@gmail.com
