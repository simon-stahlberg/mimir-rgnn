# Mimir-RGNN

Mimir-RGNN is a Python library that implements Relational Graph Neural Networks (R-GNN) using PyTorch and Mimir, offering a streamlined interface for building and using these networks.

## API

One notable feature of Mimir-RGNN is its ability to define both input and output specifications directly during model construction.
This allows for tailored configurations, as demonstrated below:

```python
import pymimir_rgnn as rgnn

domain = ...
embedding_size = ...
num_layers = ...

config = rgnn.RelationalGraphNeuralNetworkConfig(
    domain=domain,
    input_specification=(rgnn.InputType.State, rgnn.InputType.Goal),
    output_specification=[('value', rgnn.OutputNodeType.Objects, rgnn.OutputValueType.Scalar)],
    embedding_size=embedding_size,
    num_layers=num_layers,
    message_aggregation=rgnn.AggregationFunction.HardMaximum
)

model = rgnn.RelationalGraphNeuralNetwork(config)
```

In this example, the configuration (`RelationalGraphNeuralNetworkConfig`) allows specifying input types (e.g., state, goal) and output requirements (e.g., scalar values from object embeddings).
This flexibility supports various applications, including reinforcement learning (RL) and auxiliary loss functions.

## Getting Started

### Installation

Mimir-RGNN is available on PyPi and can be installed via pip:

```console
pip install pymimir-rgnn
```

### Examples

For an example, refer to the GitHub repository:

[R-GNN Example Project](https://github.com/simon-stahlberg/relational-neural-network-python/)
