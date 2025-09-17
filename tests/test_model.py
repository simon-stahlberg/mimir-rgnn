import pymimir as mm
import pytest
import torch

from pathlib import Path
from pymimir_rgnn import *


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm", [
    ('blocks', HardMaximumAggregation(), 2, 2, False, False),
    ('blocks', SmoothMaximumAggregation(), 3, 3, False, False),
    ('blocks', MeanAggregation(), 4, 4, False, True),
    ('blocks', SumAggregation(), 5, 5, True, False),
    ('gripper', HardMaximumAggregation(), 2, 2, True, False),
    ('gripper', SmoothMaximumAggregation(), 3, 3, True, False),
    ('gripper', MeanAggregation(), 4, 4, True, True),
    ('gripper', SumAggregation(), 5, 5, True, True),
])
def test_create_model(dom: str, agg: AggregationFunction, layers: int, size: int, gro: bool, norm: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    domain = mm.Domain(domain_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
    )
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec = [('q_values', ActionScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=agg,
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    assert model is not None


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm", [
    ('blocks', HardMaximumAggregation(), 2, 2, False, False),
    ('blocks', SmoothMaximumAggregation(), 3, 3, False, False),
    ('blocks', MeanAggregation(), 4, 4, False, True),
    ('blocks', SumAggregation(), 5, 5, True, False),
    ('gripper', HardMaximumAggregation(), 2, 2, True, False),
    ('gripper', SmoothMaximumAggregation(), 3, 3, True, False),
    ('gripper', MeanAggregation(), 4, 4, True, True),
    ('gripper', SumAggregation(), 5, 5, True, True),
])
def test_forward_model(dom: str, agg: AggregationFunction, layers: int, size: int, gro: bool, norm: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    problem_path = DATA_DIR / dom / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=agg,
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
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
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=SmoothMaximumAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
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
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
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
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec=(StateEncoder(), GoalEncoder())
    output_spec=[('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=SumAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
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
    hparam_config_1 = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
        global_readout=True,
        normalize_updates=False
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(hparam_config_1))]
    module_config_1 = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config_1, input_spec),
        update_function=MLPUpdates(hparam_config_1)
    )
    model_1 = RelationalGraphNeuralNetwork(hparam_config_1, module_config_1, input_spec, output_spec)  # type: ignore
    # Save the model with some extras.
    extras_1 = {'foo': 42, 'bar': 'baz'}
    model_1.save('test.pt', extras_1)
    # Load the saved file back.
    device = model_1.get_device()
    model_2, extras_2 = RelationalGraphNeuralNetwork.load(domain, 'test.pt', device)
    # Check that the loaded file matches the saved one, and that the extras are identical.
    # Note: We can't directly compare configs because encoder objects have different identities after serialization
    assert model_1._hparam_config.embedding_size == model_2._hparam_config.embedding_size
    assert model_1._hparam_config.num_layers == model_2._hparam_config.num_layers
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
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=embedding_size,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(hparam_config)), ('state_value', ObjectsScalarDecoder(hparam_config))]

    module_config = ModuleConfig(
        aggregation_function=HardMaximumAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore

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
    hparam_config_1 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_1=(StateEncoder(), GroundActionsEncoder())
    output_spec_1=[('q_values', ActionScalarDecoder(hparam_config_1))]
    module_config_1 = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config_1, input_spec_1),
        update_function=MLPUpdates(hparam_config_1)
    )
    model_1 = RelationalGraphNeuralNetwork(hparam_config_1, module_config_1, input_spec_1, output_spec_1)  # type: ignore

    # Test ObjectsScalarDecoder
    hparam_config_2 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_2=(StateEncoder(), GoalEncoder())
    output_spec_2=[('object_values', ObjectsScalarDecoder(hparam_config_2))]
    module_config_2 = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config_2, input_spec_2),
        update_function=MLPUpdates(hparam_config_2)
    )
    model_2 = RelationalGraphNeuralNetwork(hparam_config_2, module_config_2, input_spec_2, output_spec_2)  # type: ignore

    # Test ActionEmbeddingDecoder
    hparam_config_3 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_3=(StateEncoder(), GroundActionsEncoder())
    output_spec_3=[('action_embeddings', ActionEmbeddingDecoder())]
    module_config_3 = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config_3, input_spec_3),
        update_function=MLPUpdates(hparam_config_3)
    )
    model_3 = RelationalGraphNeuralNetwork(hparam_config_3, module_config_3, input_spec_3, output_spec_3)  # type: ignore

    # Test ObjectsEmbeddingDecoder
    hparam_config_4 = HyperparameterConfig(
        domain=domain,
        embedding_size=embedding_size,
        num_layers=2
    )
    input_spec_4=(StateEncoder(), GoalEncoder())
    output_spec_4=[('object_embeddings', ObjectsEmbeddingDecoder())]
    module_config_4 = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config_4, input_spec_4),
        update_function=MLPUpdates(hparam_config_4)
    )
    model_4 = RelationalGraphNeuralNetwork(hparam_config_4, module_config_4, input_spec_4, output_spec_4)  # type: ignore


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_attention_messages(domain_name: str):
    """Test that AttentionMessages class does not crash and produces reasonable output."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=8,
    )
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec = [('q_values', ActionScalarDecoder(hparam_config))]
    
    # Create model with AttentionMessages
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=AttentionMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    
    # Test forward pass
    initial_state = problem.get_initial_state()
    goal_condition = problem.get_goal_condition()
    ground_actions = initial_state.generate_applicable_actions()
    
    input_data = [(initial_state, ground_actions, goal_condition)]
    result = model.forward(input_data)
    
    # Verify output
    q_values = result.readout('q_values')
    assert isinstance(q_values, list)
    assert len(q_values) == 1  # One instance in batch
    assert len(q_values[0]) == len(ground_actions)
    
    # Check that we get valid tensors
    assert all(isinstance(val, torch.Tensor) for val in q_values[0])
    assert all(val.numel() == 1 for val in q_values[0])  # Each should be a scalar


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_curry_forward(domain_name: str):
    """Test that curry_forward is equivalent to forward but allows separated computation."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    
    # Setup model
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=3,
        embedding_size=8
    )
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec = [('q_values', ActionScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore

    # Prepare input data
    initial_state = problem.get_initial_state()
    goal_condition = problem.get_goal_condition()
    ground_actions = initial_state.generate_applicable_actions()
    input_data = [(initial_state, ground_actions, goal_condition)]

    # Test 1: Verify curry_forward returns a callable
    curried_func = model.curry_forward(input_data)
    assert callable(curried_func), "curry_forward should return a callable"

    # Test 2: Verify equivalence - model.forward(x) == model.curry_forward(x)()
    direct_result = model.forward(input_data)
    curried_result = curried_func()

    # Both should return ForwardState objects
    assert isinstance(direct_result, ForwardState), "forward should return ForwardState"
    assert isinstance(curried_result, ForwardState), "curry_forward() should return ForwardState"

    # Layer indices should match
    assert direct_result.get_layer_index() == curried_result.get_layer_index()

    # Readouts should produce equivalent results
    direct_q_values = direct_result.readout('q_values')
    curried_q_values = curried_result.readout('q_values')

    assert isinstance(direct_q_values, list)
    assert isinstance(curried_q_values, list)
    assert len(direct_q_values) == len(curried_q_values)
    assert len(direct_q_values[0]) == len(curried_q_values[0])

    # Values should be close (allowing for small numerical differences)
    for direct_val, curried_val in zip(direct_q_values[0], curried_q_values[0]):
        assert torch.allclose(direct_val, curried_val, atol=1e-6)

    # Test 3: Verify that the curried function can be called multiple times
    curried_result_2 = curried_func()
    curried_q_values_2 = curried_result_2.readout('q_values')
    
    # Should produce the same results
    for val1, val2 in zip(curried_q_values[0], curried_q_values_2[0]):
        assert torch.allclose(val1, val2, atol=1e-6)
