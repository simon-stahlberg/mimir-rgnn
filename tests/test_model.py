import pymimir as mm
import pytest

from pathlib import Path
from pymimir_rgnn import *


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm, rand", [
    ('blocks', AggregationFunction.HardMaximum, 2, 2, False, False, False),
    ('blocks', AggregationFunction.SmoothMaximum, 3, 3, False, False, True),
    ('blocks', AggregationFunction.Mean, 4, 4, False, True, False),
    ('blocks', AggregationFunction.Add, 5, 5, True, False, False),
    ('gripper', AggregationFunction.HardMaximum, 2, 2, True, False, False),
    ('gripper', AggregationFunction.SmoothMaximum, 3, 3, True, False, True),
    ('gripper', AggregationFunction.Mean, 4, 4, True, True, False),
    ('gripper', AggregationFunction.Add, 5, 5, True, True, False),
])
def test_create_model(dom: str, agg: str, layers: int, size: int, gro: bool, norm: bool, rand: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    domain = mm.Domain(domain_path)
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.GroundActions, InputType.Goal),
        output_specification=[('q_values', OutputNodeType.Action, OutputValueType.Scalar)],
        message_aggregation=agg,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
        random_initialization=rand
    )
    model = RelationalGraphNeuralNetwork(config)
    assert model is not None


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm, rand", [
    ('blocks', AggregationFunction.HardMaximum, 2, 2, False, False, False),
    ('blocks', AggregationFunction.SmoothMaximum, 3, 3, False, False, True),
    ('blocks', AggregationFunction.Mean, 4, 4, False, True, False),
    ('blocks', AggregationFunction.Add, 5, 5, True, False, False),
    ('gripper', AggregationFunction.HardMaximum, 2, 2, True, False, False),
    ('gripper', AggregationFunction.SmoothMaximum, 3, 3, True, False, True),
    ('gripper', AggregationFunction.Mean, 4, 4, True, True, False),
    ('gripper', AggregationFunction.Add, 5, 5, True, True, False),
])
def test_forward_model(dom: str, agg: str, layers: int, size: int, gro: bool, norm: bool, rand: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    problem_path = DATA_DIR / dom / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.GroundActions, InputType.Goal),
        output_specification=[('q_values', OutputNodeType.Action, OutputValueType.Scalar)],
        message_aggregation=agg,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
        random_initialization=rand
    )
    model = RelationalGraphNeuralNetwork(config)
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
    config = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.Goal),
        output_specification=[('value', OutputNodeType.Objects, OutputValueType.Scalar)],
        num_layers=4,
        embedding_size=8
    )
    model = RelationalGraphNeuralNetwork(config)
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


def test_save_and_load():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain(domain_path)
    # Create a model.
    config_1 = RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(InputType.State, InputType.GroundActions, InputType.Goal),
        output_specification=[('q_values', OutputNodeType.Action, OutputValueType.Scalar)],
        message_aggregation=AggregationFunction.Mean,
        num_layers=2,
        embedding_size=4,
        global_readout=True,
        normalize_updates=False,
        random_initialization=True
    )
    model_1 = RelationalGraphNeuralNetwork(config_1)
    # Save the model with some extras.
    extras_1 = {'foo': 42, 'bar': 'baz'}
    model_1.save('test.pt', extras_1)
    # Load the saved file back.
    device = model_1.get_device()
    model_2, extras_2 = RelationalGraphNeuralNetwork.load(domain, 'test.pt', device)
    # Check that the loaded file matches the saved one, and that the extras are identical.
    assert model_1._config == model_2._config
    assert extras_1 == extras_2
