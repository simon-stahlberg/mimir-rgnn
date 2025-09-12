# Copilot Instructions for Mimir-RGNN

## Project Overview

**Mimir-RGNN** is a professional Python library implementing Relational Graph Neural Networks (R-GNN) for AI planning applications. The project integrates PyTorch deep learning with Mimir's PDDL planning capabilities to enable neural learning on structured relational data.

### Core Purpose
- Enable neural network learning on PDDL planning domains
- Provide R-GNN implementations for reinforcement learning in planning
- Bridge symbolic AI planning with deep learning approaches
- Support research in neural-symbolic integration

## Architecture & Design Patterns

### Project Structure
```
pymimir_rgnn/
├── __init__.py          # Public API - exports all user-facing classes/functions
├── model.py             # Core R-GNN model and configuration classes
├── encodings.py         # Input/output type definitions and encoding logic  
├── modules.py           # Reusable PyTorch neural network modules
└── utils.py             # Helper functions and utilities
```

### Key Design Principles
1. **Type Safety First**: All public interfaces must be fully typed
2. **Declarative Configuration**: Use dataclasses with metadata for configuration
3. **Enum-Based Options**: Use enums for categorical configuration choices
4. **Separation of Concerns**: Clean module boundaries and responsibilities
5. **PyTorch Integration**: Leverage PyTorch ecosystem and conventions
6. **PDDL Integration**: Deep integration with Mimir's PDDL capabilities

## Core Components Deep Dive

### 1. RelationalGraphNeuralNetworkConfig (model.py)

The central configuration class using dataclass pattern:

```python
@dataclass
class RelationalGraphNeuralNetworkConfig:
    domain: mm.Domain = field(metadata={'doc': 'The domain of the planning problem.'})
    input_specification: tuple[InputType, ...] = field(...)
    output_specification: list[tuple[str, OutputNodeType, OutputValueType]] = field(...)
    # ... other configuration fields
```

**Key Patterns**:
- Use `field()` with metadata for documentation
- Type all attributes with proper generics
- Provide sensible defaults for optional parameters
- Validate configurations in `__post_init__` if needed

### 2. Input/Output System (encodings.py)

The encoding system transforms PDDL structures into graph neural network inputs:

**Input Types**:
- `InputType.State`: Current planning state
- `InputType.Goal`: Goal specification  
- `InputType.GroundActions`: Available actions
- `InputType.TransitionEffects`: Action effects

**Output Configuration**:
- Named outputs: `('q_values', OutputNodeType.Action, OutputValueType.Scalar)`
- Node types: All, Objects, Action
- Value types: Scalar, Embeddings

**Critical Functions**:
- `get_encoding()`: Determines relational structure from domain/input spec
- `encode_input()`: Transforms PDDL data into tensor format

### 3. Neural Network Modules (modules.py)

Standard PyTorch modules following library conventions:
- Inherit from `nn.Module`
- Implement `forward()` method
- Use proper tensor typing
- Support batched operations

### 4. Main R-GNN Model (model.py)

The `RelationalGraphNeuralNetwork` class:
- Takes configuration object in constructor
- Implements PyTorch `nn.Module` interface
- Supports batch processing of planning instances
- Returns structured outputs based on configuration

## Coding Standards & Patterns

### Type Annotations (CRITICAL)
All public interfaces MUST be typed. This is non-negotiable:

```python
# ✅ Correct - fully typed
def encode_input(
    input: list[tuple], 
    input_specification: tuple[InputType, ...], 
    device: torch.device
) -> TensorInput:
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

### Enum Usage
Use enums for categorical options:

```python
class MessageFunction(Enum):
    PredicateMLP = 'predicate_mlp'
    # Use descriptive names with string values
```

### Error Handling
Use assertions with descriptive messages for validation:

```python
assert len(input_specification) == len(set(input_specification)), \
    'Input types must not be repeated.'
```

### Import Organization
```python
# 1. Standard library
import os
from pathlib import Path
from typing import Any, Callable, Union

# 2. Third-party (PyTorch, Mimir)
import torch
import torch.nn as nn
import pymimir as mm

# 3. Local imports
from .encodings import InputType, OutputValueType
from .modules import MLP, SumReadout
```

## Testing Patterns

### Parameterized Testing
Use pytest parametrization for comprehensive testing:

```python
@pytest.mark.parametrize("domain,aggregation,layers,embedding_size", [
    ('blocks', AggregationFunction.HardMaximum, 2, 32),
    ('gripper', AggregationFunction.Mean, 4, 64),
])
def test_model_configuration(domain: str, aggregation: AggregationFunction, 
                           layers: int, embedding_size: int):
    """Test various model configurations."""
```

### Test Data Structure
```
tests/data/
├── blocks/domain.pddl     # Test planning domains
└── gripper/domain.pddl    # Multiple domains for robustness
```

## Domain-Specific Knowledge

### PDDL Integration
Understanding of PDDL concepts is crucial:
- **Domain**: Defines predicates, actions, types
- **Problem**: Specific instance with objects, initial state, goal
- **Ground Actions**: Instantiated actions with specific objects
- **Ground Literals**: Instantiated predicates (positive/negative)

### Planning-Specific Patterns
```python
# States contain atoms (ground literals that are true)
state = mm.State(...)
atoms = state.get_atoms()

# Goals are conjunctive conditions
goal = mm.GroundConjunctiveCondition(...)

# Actions have parameters and can be grounded
action = mm.GroundAction(...)
objects = action.get_objects()
```

### Graph Neural Network Concepts
- **Nodes**: Objects, actions, or combined entity representations
- **Relations**: Derived from PDDL predicates and action structures
- **Message Passing**: Communication between related nodes
- **Aggregation**: Combining messages (sum, mean, max variants)
- **Readout**: Extracting final outputs from node embeddings

## Common Development Tasks

### Adding New Input Types
1. Add enum value to `InputType` in encodings.py
2. Update `get_encoding()` to handle new type
3. Update `encode_input()` to process new input format
4. Add corresponding tests with parameterization

### Adding New Aggregation Functions
1. Add enum value to `AggregationFunction`
2. Update model implementation to handle new function
3. Add tests covering the new aggregation method

### Extending Output Specifications
1. Consider if new `OutputNodeType` or `OutputValueType` needed
2. Update readout logic in model
3. Ensure backward compatibility with existing configs

### Performance Optimization
- Focus on tensor operations efficiency
- Batch processing for multiple planning instances
- GPU memory management for large graphs
- Sparse tensor usage where appropriate

## Dependencies & Compatibility

### Core Dependencies
- **PyTorch**: 2.6.0+ (core ML framework)
- **Mimir**: 0.13.42+ (PDDL planning library)
- **Python**: 3.11+ (type system features)

### Development Dependencies  
- **pytest**: Testing framework
- **build**: Package building
- **twine**: PyPI publishing

### Version Compatibility
- Maintain backward compatibility within minor versions
- Follow semantic versioning
- Test against multiple PyTorch versions when possible

## Performance Considerations

### Memory Efficiency
- Use sparse tensors for large, sparse relational graphs
- Batch processing to amortize PyTorch overhead
- Efficient tensor operations (avoid loops when possible)

### GPU Utilization
- Ensure all tensors are device-agnostic  
- Support CUDA when available
- Use proper tensor types (LongTensor for indices, FloatTensor for embeddings)

## Common Pitfalls & Solutions

### Type System Issues
```python
# ❌ Wrong - will cause mypy errors
def process_data(data):  # Missing type annotation
    return data.some_method()

# ✅ Correct - fully typed
def process_data(data: TensorInput) -> torch.Tensor:
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

### Tensor Device Consistency
```python
# ❌ Wrong - device mismatch potential
tensor1 = torch.tensor(data, device='cpu')
tensor2 = torch.tensor(other_data)  # Default device

# ✅ Correct - consistent device placement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor1 = torch.tensor(data, device=device)
tensor2 = torch.tensor(other_data, device=device)
```

## Debugging & Development Tips

### Logging Strategy
- Use assertions for validation (will be optimized out in production)
- Add informative error messages
- Consider tensor shapes in error messages

### Development Workflow
1. Run tests frequently: `python -m pytest tests/ -v`
2. Check types if using mypy: `mypy pymimir_rgnn/`
3. Validate against multiple test domains
4. Test with different batch sizes and configurations

### Common Debug Scenarios
- **Shape mismatches**: Check tensor dimensions in forward pass
- **Device errors**: Ensure all tensors on same device
- **Type errors**: Validate Mimir object types before processing
- **Configuration errors**: Validate input specifications are consistent

## Documentation Standards

### Docstring Format (Google Style)
```python
def complex_function(param1: str, param2: int, param3: Optional[bool] = None) -> Dict[str, Any]:
    """Brief description of function purpose.
    
    Longer description explaining the function's behavior, algorithms used,
    or important implementation details.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.  
        param3: Optional parameter description.
        
    Returns:
        Dictionary containing processed results with keys 'output1', 'output2'.
        
    Raises:
        ValueError: If param1 is empty string.
        TypeError: If param2 is negative.
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['output1'])
        processed_test
    """
```

### Code Comments
- Explain WHY, not WHAT (code should be self-documenting)
- Complex algorithms deserve explanation
- Non-obvious design decisions should be documented

## Release & Maintenance

### Version Scheme
- Semantic versioning: MAJOR.MINOR.PATCH
- Current: 0.1.3 (pre-1.0, API may change)
- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Version number updated in pyproject.toml
- [ ] CHANGELOG updated
- [ ] Documentation current
- [ ] PyPI publishing workflow ready

This document should guide all development decisions and ensure consistency across the codebase. When in doubt, follow existing patterns and prioritize type safety and clear documentation.