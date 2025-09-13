import pymimir as mm
import pytest

from pathlib import Path
from pymimir_rgnn import *


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm", [
    ('blocks', AggregationFunction.HardMaximum, 2, 2, False, False),
    ('blocks', AggregationFunction.SmoothMaximum, 3, 3, False, False),
    ('blocks', AggregationFunction.Mean, 4, 4, False, True),
    ('blocks', AggregationFunction.Add, 5, 5, True, False),
    ('gripper', AggregationFunction.HardMaximum, 2, 2, True, False),
    ('gripper', AggregationFunction.SmoothMaximum, 3, 3, True, False),
    ('gripper', AggregationFunction.Mean, 4, 4, True, True),
    ('gripper', AggregationFunction.Add, 5, 5, True, True),
])
def test_create_model(dom: str, agg: AggregationFunction, layers: int, size: int, gro: bool, norm: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    domain = mm.Domain(domain_path)
    config = HyperparameterConfig(
        domain=domain,
        message_aggregation=agg,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
    assert model is not None


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm", [
    ('blocks', AggregationFunction.HardMaximum, 2, 2, False, False),
    ('blocks', AggregationFunction.SmoothMaximum, 3, 3, False, False),
    ('blocks', AggregationFunction.Mean, 4, 4, False, True),
    ('blocks', AggregationFunction.Add, 5, 5, True, False),
    ('gripper', AggregationFunction.HardMaximum, 2, 2, True, False),
    ('gripper', AggregationFunction.SmoothMaximum, 3, 3, True, False),
    ('gripper', AggregationFunction.Mean, 4, 4, True, True),
    ('gripper', AggregationFunction.Add, 5, 5, True, True),
])
def test_forward_model(dom: str, agg: AggregationFunction, layers: int, size: int, gro: bool, norm: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    problem_path = DATA_DIR / dom / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    config = HyperparameterConfig(
        domain=domain,
        message_aggregation=agg,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
    initial_state = problem.get_initial_state()
    initial_actions = initial_state.generate_applicable_actions()
    original_goal = problem.get_goal_condition()
    input = [(initial_state, initial_actions, original_goal)]
    output = model.forward(input)
    q_values = output.readout('q_values')
    assert isinstance(q_values, list)
    assert len(q_values) == 1
    assert len(q_values[0]) == len(initial_actions)

@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_forward_hook(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
    initial_state = problem.get_initial_state()
    original_goal = problem.get_goal_condition()
    input = [(initial_state, original_goal)]
    hook_output = []
    def hook_function(x: ForwardState):
        layer_index = x.get_layer_index()
        layer_readout = x.readout('value')
        assert layer_index == len(hook_output)
        assert layer_readout is not None
        hook_output.append((layer_index, layer_readout))
    model.add_hook(hook_function)
    output = model.forward(input)
    final_index = output.get_layer_index()
    final_readout = output.readout('value')
    assert hook_output[-1][0] == final_index
    assert hook_output[-1][1] == final_readout


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_forward_identical_batch(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
    initial_state = problem.get_initial_state()
    original_goal = problem.get_goal_condition()
    batch_size = 4
    input = [(initial_state, original_goal)] * batch_size
    output = model.forward(input)
    readout = output.readout('value')
    assert len(readout) == batch_size
    assert readout.var() < 0.0000001


@pytest.mark.parametrize("domain_name", [('blocks')])
def test_forward_different_batch(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(config))]
    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)
    initial_state = problem.get_initial_state()
    original_goal = problem.get_goal_condition()
    different_goals = [mm.GroundConjunctiveCondition.new([literal], problem) for literal in original_goal]
    input = [(initial_state, different_goal) for different_goal in different_goals]
    output = model.forward(input)
    readout = output.readout('value')
    assert len(readout) == len(different_goals)
    assert readout.var() > 0.0000001


def test_save_and_load():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    # Create a model.
    config_1 = HyperparameterConfig(
        domain=domain,
        message_aggregation=AggregationFunction.Mean,
        num_layers=2,
        embedding_size=4,
        global_readout=True,
        normalize_updates=False
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(config_1))]
    model_1 = RelationalGraphNeuralNetwork(config_1, input_spec, output_spec)
    # Save the model with some extras.
    extras_1 = {'foo': 42, 'bar': 'baz'}
    model_1.save('test.pt', extras_1)
    # Load the saved file back.
    device = model_1.get_device()
    model_2, extras_2 = RelationalGraphNeuralNetwork.load(domain, 'test.pt', device)
    # Check that the loaded file matches the saved one, and that the extras are identical.
    # Note: We can't directly compare configs because encoder objects have different identities after serialization
    assert model_1._config.embedding_size == model_2._config.embedding_size
    assert model_1._config.num_layers == model_2._config.num_layers
    assert model_1._config.message_aggregation == model_2._config.message_aggregation
    assert extras_1 == extras_2


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_simple_forward(domain_name: str):
    """Test basic functionality of the new encoder-based API."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)

    # Test new encoder-based API
    embedding_size = 4
    config = HyperparameterConfig(
        domain=domain,
        message_aggregation=AggregationFunction.HardMaximum,
        num_layers=2,
        embedding_size=embedding_size,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(config)), ('state_value', ObjectsScalarDecoder(config))]

    model = RelationalGraphNeuralNetwork(config, input_spec, output_spec)

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


def test_decoder_constructors():
    """Test all different decoders construct properly."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    embedding_size = 4

    # Test ActionScalarDecoder
    config_1 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_1=(StateEncoder(), GroundActionsEncoder())
    output_spec_1=[('q_values', ActionScalarDecoder(config_1))]
    model_1 = RelationalGraphNeuralNetwork(config_1, input_spec_1, output_spec_1)

    # Test ObjectsScalarDecoder
    config_2 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_2=(StateEncoder(), GoalEncoder())
    output_spec_2=[('object_values', ObjectsScalarDecoder(config_2))]
    model_2 = RelationalGraphNeuralNetwork(config_2, input_spec_2, output_spec_2)

    # Test ActionEmbeddingDecoder
    config3 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_3=(StateEncoder(), GroundActionsEncoder())
    output_spec_3=[('action_embeddings', ActionEmbeddingDecoder())]
    model_3 = RelationalGraphNeuralNetwork(config3, input_spec_3, output_spec_3)

    # Test ObjectsEmbeddingDecoder
    config_4 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_4=(StateEncoder(), GoalEncoder())
    output_spec_4=[('object_embeddings', ObjectsEmbeddingDecoder())]
    model_4 = RelationalGraphNeuralNetwork(config_4, input_spec_4, output_spec_4)
