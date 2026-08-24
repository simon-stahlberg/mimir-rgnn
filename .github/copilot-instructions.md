# GitHub Copilot Instructions for Mimir-RGNN

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

**Mimir-RGNN** is a professional Python library implementing Relational Graph Neural Networks (R-GNN) for AI planning applications. The project integrates PyTorch deep learning with Mimir's PDDL planning capabilities to enable neural learning on structured relational data.

## Working Effectively

### Build, and Test the Repository

**CRITICAL TIMING**: Set timeouts appropriately - NEVER CANCEL builds/tests prematurely.

1. **Use virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run tests** - takes ~3 seconds:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Run type checking** - takes ~16 seconds:
   ```bash
   mypy pymimir_rgnn/
   ```

### Key Design Principles
1. **Type Safety First**: All public interfaces must be fully typed
2. **Declarative Configuration**: Use dataclasses with metadata for configuration
3. **Class-Based Extensibility**: Use inheritance for encoders and decoders instead of enums
4. **Separation of Concerns**: Clean module boundaries and responsibilities
5. **PyTorch Integration**: Leverage PyTorch ecosystem and conventions
6. **PDDL Integration**: Deep integration with Mimir's PDDL capabilities

## Project Structure

```
mimir-rgnn/
├── pymimir_rgnn/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── model.py           # Core R-GNN model and configuration classes
│   ├── bases.py           # Base classes for encoders and decoders
│   ├── encoders.py        # Input encoder implementations
│   ├── decoders.py        # Output decoder implementations
│   ├── configs.py         # Configuration classes and enums
│   ├── modules.py         # Reusable PyTorch neural network modules
│   └── utils.py           # Helper functions and utilities
├── tests/                  # Test suite - 22 parameterized tests
│   ├── data/              # Test PDDL domains (blocks, gripper)
│   │   ├── blocks/        # Blocksworld domain and problem
│   │   └── gripper/       # Gripper domain and problem
│   └── test_model.py      # Model tests with extensive parametrization
├── .github/workflows/     # CI/CD workflows (test.yml, mypy.yml, publish.yml)
├── pyproject.toml         # Project configuration and dependencies
└── .gitignore            # Excludes __pycache__, *.pt, build/, dist/, etc.
```

## Core Architecture

### Configuration System
Use dataclasses with field metadata for all configuration:

```python
@dataclass
class HyperparameterConfig:
    domain: mm.Domain = field(metadata={'doc': 'The domain of the planning problem.'})
    embedding_size: int = field(default=32, metadata={'doc': 'Size of node embeddings.'})
    num_layers: int = field(default=3, metadata={'doc': 'Number of message passing layers.'})
    # ... other configuration fields

@dataclass
class ModuleConfig:
    aggregation_function: AggregationFunction = field(metadata={'doc': 'Message aggregation function.'})
    message_function: MessageFunction = field(metadata={'doc': 'Message computation function.'})
    update_function: UpdateFunction = field(metadata={'doc': 'Node update function.'})
```

**Key Patterns**:
- Use `field()` with metadata for documentation
- Type all attributes with proper generics
- Provide sensible defaults for optional parameters
- Validate configurations in `__post_init__` if needed

### 2. Encoder/Decoder System (bases.py, encoders.py, decoders.py)

The encoding system transforms PDDL structures into graph neural network inputs using class-based encoders:

**Encoder Classes**:
- `StateEncoder()`: Current planning state
- `GoalEncoder()`: Goal specification
- `GroundActionsEncoder()`: Available actions
- `TransitionEffectsEncoder()`: Action effects

**Decoder Classes**:
- `ActionScalarDecoder(config)`: Scalar values over actions
- `ActionEmbeddingDecoder()`: Embeddings over actions
- `ObjectsScalarDecoder(config)`: Scalar values over objects
- `ObjectsEmbeddingDecoder()`: Embeddings over objects

**Critical Functions**:
- `Encoder.get_relations()`: Determines relational structure from domain
- `Encoder.encode()`: Transforms PDDL data into intermediate format
- `Decoder.forward()`: Implements direct readout logic from node embeddings

### 3. Neural Network Modules (modules.py)

Standard PyTorch modules following library conventions:
- Inherit from `nn.Module`
- Implement `forward()` method
- Use proper tensor typing
- Support batched operations

### 4. Main R-GNN Model (model.py)

**Aggregation Functions**:
- `MeanAggregation()`: Mean aggregation
- `SumAggregation()`: Sum aggregation
- `HardMaximumAggregation()`: Hard maximum
- `SmoothMaximumAggregation()`: Smooth maximum (LogSumExp)

### Main R-GNN Model
The `RelationalGraphNeuralNetwork` class:
- Takes hyperparameter config, input specification tuple, and output specification list
- Implements PyTorch `nn.Module` interface
- Supports batch processing of planning instances
- Returns structured outputs based on decoder configuration

## Development Standards

### Type Safety (CRITICAL)
**All public interfaces MUST be fully typed.** This is non-negotiable:

```python
# ✅ Correct - fully typed
def encode_input(
    input: list[tuple],
    input_specification: tuple[Encoder, ...],
    device: torch.device
) -> EncodedTensors:
    """Process planning instances into tensor format."""

# ❌ Wrong - missing types
def encode_input(input, input_specification, device):
    """Process planning instances into tensor format."""
```

### Configuration Pattern
Use dataclasses with field metadata for all configuration:

```python
@dataclass
class MyConfig:
    required_param: str = field(
        metadata={'doc': 'Required parameter description.'}
    )

    optional_param: int = field(
        default=42,
        metadata={'doc': 'Optional parameter with default.'}
    )
```

### Class-Based Encoder/Decoder Pattern
Use inheritance for extensibility instead of enums:

```python
class CustomEncoder(Encoder):
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        relations = super().get_relations(domain) if hasattr(super(), 'get_relations') else []
        relations.append(("custom_relation", 2))
        return relations

    def encode(self, input_value: Any, intermediate: EncodedLists, state: mm.State) -> int:
        # Custom encoding implementation
        return nodes_added

class CustomDecoder(Decoder):
    def __init__(self, hparam_config: HyperparameterConfig):
        super().__init__()
        self._readout = MLP(hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, input: EncodedTensors) -> torch.Tensor:
        # Direct readout implementation
        return self._readout(node_embeddings)
```

### Error Handling
Use assertions with descriptive messages for validation:

### Testing Patterns
Use pytest parametrization for comprehensive testing:

```python
@pytest.mark.parametrize("domain,aggregation,layers,embedding_size", [
    ('blocks', HardMaximumAggregation(), 2, 32),
    ('gripper', MeanAggregation(), 4, 64),
])
def test_model_configuration(domain: str, aggregation: AggregationFunction,
                           layers: int, embedding_size: int):
    """Test various model configurations."""
```

### Import Organization
```python
# 1. imports
import os
import torch
import torch.nn as nn
import pymimir as mm

# 2. Third-party from imports
from pathlib import Path
from typing import Any, Callable, Union

# 3. Local from imports
from .encoders import StateEncoder
from .decoders import ActionScalarDecoder
from .modules import MLP, SumReadout
```

### Error Handling
Use assertions with descriptive messages for validation:

```python
assert len(input_specification) == len(set(input_specification)), \
    'Input types must not be repeated.'
```

## Common Development Tasks

### Adding New Encoder Classes
1. Create new class inheriting from `Encoder` in encoders.py
2. Implement `get_relations()` to define graph relations
3. Implement `encode()` to process input format
4. Add corresponding tests with parameterization

### Adding New Decoder Classes
1. Create new class inheriting from `Decoder` in decoders.py
2. Implement `forward()` method for direct readout logic
3. Add tests covering the new decoder functionality

### PDDL Integration Patterns
```python
# Load a problem and access its initial state
problem = mm.Problem.from_files("domain.pddl", "problem.pddl")
state = problem.initial_state
atoms = state.atoms

# Goals are conjunctive conditions
goal = problem.goal

# Applicable ground actions expose their object arguments
action = state.applicable_actions()[0]
objects = action.objects
```

### Graph Neural Network Concepts
- **Nodes**: Objects, actions, or combined entity representations
- **Relations**: Derived from PDDL predicates and action structures
- **Message Passing**: Communication between related nodes
- **Aggregation**: Combining messages (sum, mean, max variants)
- **Readout**: Extracting final outputs from node embeddings

## Common Development Tasks

### Adding New Encoder Classes
1. Create new class inheriting from `Encoder` in encoders.py
2. Implement `get_relations()` to define graph relations
3. Implement `encode()` to process input format
4. Add corresponding tests with parameterization

### Adding New Decoder Classes
1. Create new class inheriting from `Decoder` in decoders.py
2. Implement `forward()` method for direct readout logic
3. Add tests covering the new decoder functionality

### Adding New Aggregation Functions
1. Add enum value to `AggregationFunction` in configs.py
2. Update model implementation to handle new function
3. Add tests covering the new aggregation method

### Extending Configuration
1. Consider if new dataclass fields needed in HyperparameterConfig
2. Ensure backward compatibility with existing configs
3. Update validation logic if needed

### Performance Optimization
- Focus on tensor operations efficiency
- Batch processing for multiple planning instances
- GPU memory management for large graphs
- Sparse tensor usage where appropriate

### Tensor Device Consistency
```python
# ✅ Correct - consistent device placement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor1 = torch.tensor(data, device=device)
tensor2 = torch.tensor(other_data, device=device)
```

## Dependencies & Compatibility

**Core Dependencies**:
- **Python**: 3.11+ (required for type system features)
- **PyTorch**: 2.6.0+ (core ML framework)
- **Pymimir**: 0.14.0b2 (PDDL planning library)

**Development Dependencies**:
- **pytest**: Testing framework
- **mypy**: Type checking
- **build**: Package building (may fail in CI environments)
- **twine**: PyPI publishing

## Common Pitfalls & Solutions

### Type System Issues
Always validate Mimir object types:
```python
# ❌ Wrong - will cause mypy errors
def process_data(data):  # Missing type annotation
    return data.some_method()

# ✅ Correct - fully typed
def process_data(data: EncodedTensors) -> torch.Tensor:
    return data.flattened_relations['predicate_name']
```

### PDDL Integration Errors
```python
# ❌ Wrong - not checking types from Mimir
goal = instance[goal_index]  # Could be anything

# ✅ Correct - validate types from Mimir
goal = instance[goal_index]
assert isinstance(goal, mm.GroundConjunctiveCondition), \
    f'Expected goal at position {goal_index}, got {type(goal)}'
```

### Performance Considerations
- Use sparse tensors for large, sparse relational graphs
- Batch processing to amortize PyTorch overhead
- Efficient tensor operations (avoid loops when possible)
- Ensure all tensors are device-agnostic

## Documentation Standards

Use Google-style docstrings for all public interfaces:

```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of function purpose.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Dictionary containing processed results.

    Raises:
        ValueError: If param1 is empty string.
    """
```

## Debugging & Development Tips

1. Run tests frequently: `python -m pytest tests/ -v`
2. Check types: `mypy pymimir_rgnn/`
3. Validate against multiple test domains (blocks, gripper)
4. Test with different batch sizes and configurations

### Common Debug Scenarios
- **Shape mismatches**: Check tensor dimensions in forward pass
- **Device errors**: Ensure all tensors on same device
- **Type errors**: Validate Mimir object types before processing
- **Configuration errors**: Validate input specifications are consistent

## Version Information

- **Version source**: `project.version` in `pyproject.toml`
- **Semantic versioning**: MAJOR.MINOR.PATCH with PEP 440 prereleases
- **License**: GPL-3.0-or-later

This document should guide all development decisions and ensure consistency across the codebase. When in doubt, follow existing patterns and prioritize type safety and clear documentation.
