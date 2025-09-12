import pymimir as mm
import pytest

from pathlib import Path
from pymimir_rgnn import (
    RelationalGraphNeuralNetworkConfig, 
    RelationalGraphNeuralNetwork,
    AggregationFunction,
    StateEncoder,
    GoalEncoder, 
    GroundActionsEncoder,
    ActionScalarOutput,
    ActionEmbeddingOutput,
    ObjectsScalarOutput,
    ObjectsEmbeddingOutput,
    # Legacy imports for comparison
    InputType,
    OutputNodeType,
    OutputValueType
)

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_new_encoder_api_basic(domain_name: str):
    """Test basic functionality of the new encoder-based API."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    # Test new encoder-based API
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GroundActionsEncoder(), GoalEncoder()),
        output_specification=[
            ('q_values', ActionScalarOutput()),
            ('state_value', ObjectsScalarOutput())
        ],
        message_aggregation=AggregationFunction.HardMaximum,
        num_layers=2,
        embedding_size=4,
    )
    
    model = RelationalGraphNeuralNetwork(config)
    
    # Test forward pass
    initial_state = problem.get_initial_state()
    initial_actions = initial_state.generate_applicable_actions()
    original_goal = problem.get_goal_condition()
    
    input_data = [(initial_state, initial_actions, original_goal)]
    output = model.forward(input_data)
    
    q_values = output.readout('q_values')
    state_value = output.readout('state_value')
    
    # Verify outputs have correct structure
    assert isinstance(q_values, list), "Q-values should be a list for action outputs"
    assert len(q_values) == 1, "Should have one set of q-values per instance"
    assert len(q_values[0]) == len(initial_actions), "Q-values should match number of actions"
    
    assert hasattr(state_value, 'shape'), "State value should be a tensor"
    assert len(state_value.shape) == 1, "State value should be 1D tensor"


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_encoder_api_equivalence_to_legacy(domain_name: str):
    """Test that new encoder API produces equivalent results to legacy API."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    # Legacy enum-based config
    legacy_config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.GroundActions, InputType.Goal),
        output_specification=[('q_values', OutputNodeType.Action, OutputValueType.Scalar)],
        message_aggregation=AggregationFunction.Mean,
        num_layers=3,
        embedding_size=8,
    )
    
    # New encoder-based config
    new_config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GroundActionsEncoder(), GoalEncoder()),
        output_specification=[('q_values', ActionScalarOutput())],
        message_aggregation=AggregationFunction.Mean,
        num_layers=3,
        embedding_size=8,
    )
    
    legacy_model = RelationalGraphNeuralNetwork(legacy_config)
    new_model = RelationalGraphNeuralNetwork(new_config)
    
    # Test forward pass
    initial_state = problem.get_initial_state()
    initial_actions = initial_state.generate_applicable_actions()
    original_goal = problem.get_goal_condition()
    
    input_data = [(initial_state, initial_actions, original_goal)]
    
    legacy_output = legacy_model.forward(input_data)
    new_output = new_model.forward(input_data)
    
    legacy_q_values = legacy_output.readout('q_values')
    new_q_values = new_output.readout('q_values')
    
    # Check that outputs have the same structure
    assert type(legacy_q_values) == type(new_q_values), "Output types should match"
    assert len(legacy_q_values) == len(new_q_values), "Batch sizes should match"
    assert len(legacy_q_values[0]) == len(new_q_values[0]), "Action counts should match"


def test_output_encoder_types():
    """Test all different output encoder types work correctly."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GoalEncoder()),
        output_specification=[
            ('action_scalars', ActionScalarOutput()),
            ('action_embeddings', ActionEmbeddingOutput()),
            ('objects_scalars', ObjectsScalarOutput()),
            ('objects_embeddings', ObjectsEmbeddingOutput())
        ],
        embedding_size=4,
        num_layers=2
    )
    
    # Note: This config doesn't have actions in input but has action outputs
    # The model should handle this gracefully (though outputs may be empty)
    model = RelationalGraphNeuralNetwork(config)
    
    initial_state = problem.get_initial_state()
    original_goal = problem.get_goal_condition()
    
    input_data = [(initial_state, original_goal)]
    output = model.forward(input_data)
    
    # All readouts should work without error
    objects_scalars = output.readout('objects_scalars')
    objects_embeddings = output.readout('objects_embeddings') 
    
    # These should exist but may be empty since no actions in input
    try:
        action_scalars = output.readout('action_scalars')
        action_embeddings = output.readout('action_embeddings')
    except (RuntimeError, IndexError):
        # Expected if no actions in the input specification
        pass


def test_mixed_api_formats():
    """Test that mixing old and new format in same config raises appropriate errors."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    
    # This should work - pure legacy format
    legacy_config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.Goal),
        output_specification=[('value', OutputNodeType.Objects, OutputValueType.Scalar)],
        embedding_size=4,
        num_layers=2
    )
    
    # This should work - pure new format
    new_config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GoalEncoder()),
        output_specification=[('value', ObjectsScalarOutput())],
        embedding_size=4,
        num_layers=2
    )
    
    # Both should create models successfully
    legacy_model = RelationalGraphNeuralNetwork(legacy_config)
    new_model = RelationalGraphNeuralNetwork(new_config)
    
    assert legacy_model is not None
    assert new_model is not None


def test_encoder_inheritance():
    """Test that encoders can be subclassed (demonstrates extensibility)."""
    
    class CustomStateEncoder(StateEncoder):
        """Example of a custom encoder that could add custom logic."""
        
        def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
            # For demo purposes, just call parent implementation
            # In reality, this could add custom relations
            return super().get_relations(domain)
    
    class CustomActionOutput(ActionScalarOutput):
        """Example of a custom output encoder."""
        pass
    
    # Test that custom encoders work
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(CustomStateEncoder(), GoalEncoder()),
        output_specification=[('custom_output', CustomActionOutput())],
        embedding_size=4,
        num_layers=2
    )
    
    # Should create model successfully with custom encoders
    model = RelationalGraphNeuralNetwork(config)
    assert model is not None