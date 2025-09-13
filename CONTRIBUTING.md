# Contributing to Mimir-RGNN

Thank you for your interest in contributing to Mimir-RGNN! This document provides guidelines for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Type Safety Requirements](#type-safety-requirements)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/simon-stahlberg/mimir-rgnn.git
   cd mimir-rgnn
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify installation**:
   ```bash
   python -m pytest tests/ -v
   ```

All tests should pass before you begin development.

## Coding Standards

### Code Style

Mimir-RGNN follows professional Python development practices:

- **Naming Conventions**:
  - Use `snake_case` for functions, variables, and module names
  - Use `PascalCase` for classes and enums
  - Use `UPPER_CASE` for constants
  - Prefix private attributes with underscore: `_private_attr`

- **Import Organization**:
  ```python
  # Standard library imports first
  import os
  from pathlib import Path
  from typing import Any, List, Optional, Union
  
  # Third-party imports second
  import torch
  import torch.nn as nn
  import pymimir as mm
  
  # Local imports last
  from .encoders import StateEncoder
  from .decoders import ActionScalarDecoder
  from .modules import MLP
  ```

- **Line Length**: Keep lines under 120 characters when practical

### Documentation Standards

- All public interfaces **must** include docstrings
- Use Google-style docstrings:
  ```python
  def encode_input(input: list[tuple], input_specification: tuple[Encoder, ...], device: torch.device) -> EncodedTensors:
      """Encode input data into tensor format for R-GNN processing.
      
      Args:
          input: List of input tuples containing states, goals, actions, etc.
          input_specification: Specification of encoders and their order.
          device: Target PyTorch device for tensor placement.
          
      Returns:
          EncodedTensors object containing encoded relational graph data.
          
      Raises:
          AssertionError: If input specification validation fails.
      """
  ```

### Configuration with Dataclasses

Use dataclasses with metadata for configuration objects:

```python
@dataclass
class MyConfig:
    domain: mm.Domain = field(
        metadata={'doc': 'The planning domain specification.'}
    )
    
    embedding_size: int = field(
        default=32,
        metadata={'doc': 'Size of node embeddings.'}
    )
```

### Class-Based Encoder/Decoder System

Use inheritance from base classes for extensibility:

```python
class CustomEncoder(Encoder):
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [("custom_relation", 2)]
    
    def encode(self, input_value: Any, intermediate: EncodedLists, state: mm.State) -> int:
        # Custom encoding logic
        return nodes_added

class CustomDecoder(Decoder):
    def __init__(self, config: HyperparameterConfig):
        super().__init__()
        self._readout = MLP(config.embedding_size, 1)
    
    def forward(self, node_embeddings: torch.Tensor, input: EncodedTensors) -> torch.Tensor:
        # Direct readout implementation
        return self._readout(node_embeddings)
```

## Type Safety Requirements

**Critical**: All public interfaces must be fully typed. This is a strict requirement.

### Required Type Annotations

- **Function signatures**: All parameters and return types
- **Class attributes**: Public attributes must be typed
- **Variable declarations**: When type cannot be inferred

### Type Annotation Examples

```python
# Function with full type annotations
def create_model(config: HyperparameterConfig, input_spec: tuple[Encoder, ...], output_spec: list[tuple[str, Decoder]]) -> RelationalGraphNeuralNetwork:
    """Create R-GNN model from configuration."""
    return RelationalGraphNeuralNetwork(config, input_spec, output_spec)

# Class with typed attributes  
class EncodedTensors:
    def __init__(self) -> None:
        self.flattened_relations: dict[str, torch.Tensor] = {}
        self.node_count: int = 0
        self.node_sizes: torch.Tensor = torch.LongTensor()

# Generic types when needed
from typing import Generic, TypeVar

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item
```

### Union Types and Optionals

```python
from typing import Union, Optional

# Use | syntax for Python 3.10+ (our minimum is 3.11)
def process_input(data: str | bytes) -> str:
    """Process string or bytes input."""
    
# Optional parameters
def configure_model(size: int, activation: Optional[str] = None) -> None:
    """Configure model with optional activation function."""
```

## Testing Guidelines

### Test Structure

- Tests are located in the `tests/` directory
- Use pytest for all testing
- Follow the existing parameterized test pattern:

```python
import pytest
from pymimir_rgnn import *

@pytest.mark.parametrize("domain,aggregation,layers,embedding_size", [
    ('blocks', AggregationFunction.HardMaximum, 2, 32),
    ('gripper', AggregationFunction.Mean, 4, 64),
])
def test_model_creation(domain: str, aggregation: AggregationFunction, 
                       layers: int, embedding_size: int):
    """Test R-GNN model creation with various configurations."""
    # Test implementation using encoder/decoder classes
    config = HyperparameterConfig(
        domain=load_domain(domain),
        embedding_size=embedding_size,
        num_layers=layers,
        message_aggregation=aggregation
    )
    input_spec = (StateEncoder(), GoalEncoder())
    output_spec = [('q_values', ActionScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
```

### Test Requirements

1. **Coverage**: New features require corresponding tests
2. **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple configurations
3. **Test Data**: Use the existing test data structure in `tests/data/`
4. **Assertions**: Use descriptive assertion messages

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model.py -v

# Run with coverage (if installed)
python -m pytest tests/ --cov=pymimir_rgnn
```

## Project Structure

Understanding the project organization:

```
mimir-rgnn/
├── pymimir_rgnn/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── model.py           # Core R-GNN model implementation
│   ├── bases.py           # Base classes for encoders and decoders
│   ├── encoders.py        # Input encoder implementations
│   ├── decoders.py        # Output decoder implementations  
│   ├── configs.py         # Configuration classes and enums
│   ├── modules.py         # Neural network modules
│   └── utils.py           # Utility functions
├── tests/                  # Test suite
│   ├── data/              # Test data (PDDL domains)
│   └── test_model.py      # Model tests
├── .github/workflows/     # CI/CD workflows
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── LICENSE                # GPL-3.0 license
└── CONTRIBUTING.md        # This file
```

### Module Responsibilities

- **`model.py`**: Core R-GNN implementation and main model class
- **`bases.py`**: Base classes for Encoder and Decoder abstractions  
- **`encoders.py`**: Input encoder implementations (StateEncoder, GoalEncoder, etc.)
- **`decoders.py`**: Output decoder implementations (ActionScalarDecoder, etc.)
- **`configs.py`**: Configuration classes and enumeration types
- **`modules.py`**: Reusable PyTorch modules (MLP, readout functions)
- **`utils.py`**: Helper functions for tensor operations, naming conventions

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass locally
   ```bash
   python -m pytest tests/ -v
   ```

2. **Type checking**: Verify type annotations (if using mypy):
   ```bash
   mypy pymimir_rgnn/
   ```

3. **Code review**: Review your changes for compliance with coding standards

### PR Checklist

- [ ] All public interfaces are properly typed
- [ ] Tests are included for new functionality
- [ ] Documentation is updated for API changes
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] PR description clearly explains the changes

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] All public interfaces are typed
- [ ] Documentation updated
```

## Issue Reporting

### Bug Reports

Use the GitHub issue template and include:

- **Environment**: Python version, PyTorch version, Mimir version
- **Reproduction**: Minimal example that demonstrates the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Stack trace**: If applicable

### Feature Requests

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: Your ideas for addressing it
- **Alternatives**: Other approaches you've considered
- **Impact**: Who would benefit from this feature

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a professional tone in all interactions

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/simon-stahlberg/mimir-rgnn/discussions)
- **Bugs**: Create a [GitHub Issue](https://github.com/simon-stahlberg/mimir-rgnn/issues)
- **Direct Contact**: simon.stahlberg@gmail.com

Thank you for contributing to Mimir-RGNN! 🚀