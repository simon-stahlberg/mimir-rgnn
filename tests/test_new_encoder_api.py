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
    ActionScalarDecoder,
    ActionEmbeddingDecoder,
    ObjectsScalarDecoder,
    ObjectsEmbeddingDecoder
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
    embedding_size = 4
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GroundActionsEncoder(), GoalEncoder()),
        output_specification=[
            ('q_values', ActionScalarDecoder(embedding_size)),
            ('state_value', ObjectsScalarDecoder(embedding_size))
        ],
        message_aggregation=AggregationFunction.HardMaximum,
        num_layers=2,
        embedding_size=embedding_size,
    )
    
    model = RelationalGraphNeuralNetwork(config)
    
    # Test forward pass
    initial_state = problem.get_initial_state()
    goal_condition = problem.get_goal_condition()
    ground_actions = initial_state.generate_applicable_actions()
    
    input_data = [(initial_state, ground_actions, goal_condition)]
    result = model.forward(input_data)
    
    # Test action values output
    q_values = result.readout('q_values')
    assert isinstance(q_values, list)
    assert len(q_values) == 1  # One instance in batch
    assert len(q_values[0]) == len(ground_actions)
    
    # Test state value output
    state_value = result.readout('state_value')
    
    assert hasattr(state_value, 'shape'), "State value should be a tensor"
    assert len(state_value.shape) == 1, "State value should be 1D tensor"


def test_decoder_types():
    """Test all different decoder types work correctly."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    embedding_size = 4
    
    # Test ActionScalarDecoder
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GroundActionsEncoder()),
        output_specification=[('q_values', ActionScalarDecoder(embedding_size))],
        embedding_size=embedding_size,
        num_layers=2
    )
    model = RelationalGraphNeuralNetwork(config)
    
    # Test ObjectsScalarDecoder  
    config2 = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GoalEncoder()),
        output_specification=[('object_values', ObjectsScalarDecoder(embedding_size))],
        embedding_size=embedding_size,
        num_layers=2
    )
    model2 = RelationalGraphNeuralNetwork(config2)
    
    # Test ActionEmbeddingDecoder
    config3 = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GroundActionsEncoder()),
        output_specification=[('action_embeddings', ActionEmbeddingDecoder())],
        embedding_size=embedding_size,
        num_layers=2
    )
    model3 = RelationalGraphNeuralNetwork(config3)
    
    # Test ObjectsEmbeddingDecoder
    config4 = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(StateEncoder(), GoalEncoder()),
        output_specification=[('object_embeddings', ObjectsEmbeddingDecoder())],
        embedding_size=embedding_size,
        num_layers=2
    )
    model4 = RelationalGraphNeuralNetwork(config4)


class CustomStateEncoder(StateEncoder):
    """Custom state encoder for testing inheritance."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        relations = super().get_relations(domain)
        # Add a custom relation (this is just for testing - it won't work in practice)
        relations.append(("custom_relation", 2))
        return relations


def test_custom_encoder_inheritance():
    """Test that users can inherit from encoder classes and customize behavior."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    
    embedding_size = 4
    
    # Test that custom encoder can be used
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(CustomStateEncoder(), GoalEncoder()),
        output_specification=[('value', ObjectsScalarDecoder(embedding_size))],
        embedding_size=embedding_size,
        num_layers=2
    )
    
    model = RelationalGraphNeuralNetwork(config)
    assert model is not None


def test_encoder_position_matters():
    """Test that encoder position determines which input argument it processes."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    embedding_size = 4
    
    # Different order: Goal, State, Actions
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(GoalEncoder(), StateEncoder(), GroundActionsEncoder()),
        output_specification=[('q_values', ActionScalarDecoder(embedding_size))],
        embedding_size=embedding_size,
        num_layers=2
    )
    
    model = RelationalGraphNeuralNetwork(config)
    
    initial_state = problem.get_initial_state()
    goal_condition = problem.get_goal_condition()
    ground_actions = initial_state.generate_applicable_actions()
    
    # Input order must match encoder order: goal, state, actions
    input_data = [(goal_condition, initial_state, ground_actions)]
    result = model.forward(input_data)
    
    q_values = result.readout('q_values')
    assert isinstance(q_values, list)
    assert len(q_values) == 1
    assert len(q_values[0]) == len(ground_actions)