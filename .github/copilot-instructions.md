# GitHub Copilot Instructions for Mimir-RGNN

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

# GitHub Copilot Instructions for Mimir-RGNN

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

**Mimir-RGNN** is a professional Python library implementing Relational Graph Neural Networks (R-GNN) for AI planning applications. The project integrates PyTorch deep learning with Mimir's PDDL planning capabilities to enable neural learning on structured relational data.

## Working Effectively

### Bootstrap, Build, and Test the Repository

**CRITICAL TIMING**: Set timeouts appropriately - NEVER CANCEL builds/tests prematurely.

1. **Install development dependencies** - takes ~2 minutes, NEVER CANCEL:
   ```bash
   pip install -e ".[dev]"  # Takes 1m55s - set timeout to 5+ minutes
   ```

2. **Run tests** - takes ~3 seconds:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Run type checking** - takes ~16 seconds:
   ```bash
   mypy pymimir_rgnn/
   ```

### Alternative CI-Style Setup (if main setup fails)

If the regular setup fails due to network issues, use the CI approach:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # Takes ~28s
pip install pymimir
pip install pytest mypy
pip install -e .[dev]
```

### Validation

**ALWAYS run these validation scenarios after making changes:**

1. **Test basic import**:
   ```bash
   python -c "import pymimir_rgnn as rgnn; print('✓ Import successful')"
   ```

2. **Test core functionality** - create and run this validation script:
   ```python
   import pymimir as mm
   import pymimir_rgnn as rgnn
   from pathlib import Path
   
   # Test with blocks domain
   test_dir = Path('tests/data')
   domain = mm.Domain(test_dir / 'blocks' / 'domain.pddl')
   problem = mm.Problem(domain, test_dir / 'blocks' / 'problem.pddl')
   
   config = rgnn.RelationalGraphNeuralNetworkConfig(
       domain=domain,
       input_specification=(rgnn.InputType.State, rgnn.InputType.GroundActions, rgnn.InputType.Goal),
       output_specification=[('q_values', rgnn.OutputNodeType.Action, rgnn.OutputValueType.Scalar)],
       embedding_size=32,
       num_layers=3,
       message_aggregation=rgnn.AggregationFunction.Mean
   )
   
   model = rgnn.RelationalGraphNeuralNetwork(config)
   initial_state = problem.get_initial_state()
   initial_actions = initial_state.generate_applicable_actions()
   goal = problem.get_goal_condition()
   
   output = model.forward([(initial_state, initial_actions, goal)])
   q_values = output.readout('q_values')
   print(f"✓ Got Q-values for {len(q_values[0])} actions")
   ```

### Build Limitations

**DO NOT attempt `python -m build`** - this fails due to network timeouts in CI environments. The package builds work in production but not in sandboxed environments.

### Pre-commit Validation

Always run before finishing changes:
```bash
python -m pytest tests/ -v     # Takes 3 seconds
mypy pymimir_rgnn/            # Takes 16 seconds  
```

No linting tools (black/flake8/ruff) are configured - rely on mypy for code quality.

## Project Structure

```
mimir-rgnn/
├── pymimir_rgnn/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── model.py           # Core R-GNN model and configuration classes
│   ├── encodings.py       # Input/output type definitions and encoding logic  
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
class RelationalGraphNeuralNetworkConfig:
    domain: mm.Domain = field(metadata={'doc': 'The domain of the planning problem.'})
    input_specification: tuple[InputType, ...] = field(...)
    output_specification: list[tuple[str, OutputNodeType, OutputValueType]] = field(...)
    # ... other configuration fields
```

### Key Enums and Types

**Input Types** (encodings.py):
- `InputType.State`: Current planning state
- `InputType.Goal`: Goal specification  
- `InputType.GroundActions`: Available actions
- `InputType.TransitionEffects`: Action effects

**Output Configuration**:
- Named outputs: `('q_values', OutputNodeType.Action, OutputValueType.Scalar)`
- Node types: All, Objects, Action
- Value types: Scalar, Embeddings

**Aggregation Functions**:
- `AggregationFunction.Add`: Sum aggregation
- `AggregationFunction.Mean`: Mean aggregation  
- `AggregationFunction.HardMaximum`: Hard maximum
- `AggregationFunction.SmoothMaximum`: Smooth maximum (LogSumExp)

### Main R-GNN Model
The `RelationalGraphNeuralNetwork` class:
- Takes configuration object in constructor
- Implements PyTorch `nn.Module` interface
- Supports batch processing of planning instances
- Returns structured outputs based on configuration

## Development Standards

### Type Safety (CRITICAL)
**All public interfaces MUST be fully typed.** This is non-negotiable:

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

### Testing Patterns
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

### Error Handling
Use assertions with descriptive messages for validation:

```python
assert len(input_specification) == len(set(input_specification)), \
    'Input types must not be repeated.'
```

## Common Development Tasks

### Adding New Input Types
1. Add enum value to `InputType` in encodings.py
2. Update `get_encoding()` to handle new type
3. Update `encode_input()` to process new input format
4. Add corresponding tests with parameterization

### Extending Output Specifications
1. Consider if new `OutputNodeType` or `OutputValueType` needed
2. Update readout logic in model
3. Ensure backward compatibility with existing configs
4. Add tests covering the new output type

### PDDL Integration Patterns
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
- **Mimir**: 0.13.42+ (PDDL planning library)

**Development Dependencies**:
- **pytest**: Testing framework
- **mypy**: Type checking
- **build**: Package building (may fail in CI environments)
- **twine**: PyPI publishing

## Common Pitfalls & Solutions

### Type System Issues
Always validate Mimir object types:
```python
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

- **Current**: 0.1.3 (pre-1.0, API may change)
- **Semantic versioning**: MAJOR.MINOR.PATCH
- **License**: GPL-3.0-or-later

This document should guide all development decisions and ensure consistency across the codebase. When in doubt, follow existing patterns and prioritize type safety and clear documentation.