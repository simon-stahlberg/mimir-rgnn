import gc
import pymimir as mm
import pytest
import threading
import torch
import warnings

from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Any, Literal, cast
from pymimir_rgnn import *
from pymimir_rgnn.encoders import get_input_from_encoders
import pymimir_rgnn.model as rgnn_model
from pymimir_rgnn.model import MIN_REPLAYED_LAYERS, RelationalLayerStackModule


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
    domain = mm.Domain.from_file(domain_path)
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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=layers,
        embedding_size=size,
        global_readout=gro,
        normalize_updates=norm,
    )
    input_spec=(StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec=[('q_values', ActionScalarDecoder(hparam_config)), ('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=agg,
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    initial_state = problem.initial_state
    initial_actions = initial_state.applicable_actions()
    original_goal = problem.goal
    input = [(initial_state, initial_actions, original_goal)]
    output = model.forward(input)
    q_values = output.readout('q_values')
    assert isinstance(q_values, list)
    assert len(q_values) == 1
    assert len(q_values[0]) == len(initial_actions)
    value = output.readout('value')
    assert isinstance(value, torch.Tensor)
    assert not torch.isnan(value).any()
    assert value.shape == (1,)
    assert value.numel() == 1


def test_forward_model_uses_model_dtype_for_node_embeddings():
    domain = mm.Domain.from_file(DATA_DIR / 'blocks' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'blocks' / 'problem.pddl')
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=1,
        embedding_size=4,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config),
    )
    model = RelationalGraphNeuralNetwork(
        hparam_config,
        module_config,
        input_spec,
        [('value', ObjectsScalarDecoder(hparam_config))],
    ).to(dtype=torch.float64)

    output = model([(problem.initial_state, problem.goal)]).readout('value')

    assert output.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layers", [0, 2])
def test_forward_model_supports_parameter_free_message_passing(dtype, layers):
    class ConstantMessages(MessageFunction):
        def forward(self, node_embeddings, relations):
            indices = torch.arange(node_embeddings.shape[0], device=node_embeddings.device)
            return torch.ones_like(node_embeddings), indices

    class MessageUpdates(UpdateFunction):
        def forward(self, node_embeddings, aggregated_messages):
            return aggregated_messages

    domain = mm.Domain.from_file(DATA_DIR / 'blocks' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'blocks' / 'problem.pddl')
    config = HyperparameterConfig(
        domain=domain,
        num_layers=layers,
        embedding_size=4,
        normalize_updates=False,
    )
    model = RelationalGraphNeuralNetwork(
        config,
        ModuleConfig(MeanAggregation(), ConstantMessages(), MessageUpdates()),
        (StateEncoder(),),
        [('embeddings', ObjectsEmbeddingDecoder())],
    ).to(dtype=dtype)
    inputs = [(problem.initial_state,)]

    output, = model(inputs).readout('embeddings')
    prepared_output, = model.curry_forward(inputs)().readout('embeddings')

    assert output.dtype == dtype
    assert output.numel() > 0
    assert torch.equal(output, torch.full_like(output, layers))
    assert torch.equal(prepared_output, output)


def test_forward_with_lifted_nullary_facts_and_unary_transition_effect() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain nullary-forward)
    (:requirements :strips)
    (:predicates (ready))
    (:action clear
        :parameters ()
        :precondition (ready)
        :effect (not (ready))
    )
)
""".strip()
    )
    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem nullary-forward-problem)
    (:domain nullary-forward)
    (:objects first second)
    (:init (ready))
    (:goal (ready))
)
""".strip(),
    )
    state = problem.initial_state
    clear = state.applicable_actions()[0]
    successor = clear.apply(state)
    transition = ([successor], [clear], (), problem.goal)
    input_spec = (
        StateEncoder(),
        GoalEncoder(),
        ExpressiveStateEncoder(),
        ExpressiveGoalEncoder(),
        TransitionEffectsEncoder(),
    )
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
    )
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config),
    )
    model = RelationalGraphNeuralNetwork(
        hparam_config,
        module_config,
        input_spec,
        [("value", ObjectsScalarDecoder(hparam_config))],
    )

    output = model.forward(
        [(state, problem.goal, state, problem.goal, transition)]
    )
    value = output.readout("value")
    assert isinstance(value, torch.Tensor)
    assert value.shape == (1,)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_forward_hook(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
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
    initial_state = problem.initial_state
    original_goal = problem.goal
    input = [(initial_state, original_goal)]
    hook_output: list[tuple[int, torch.Tensor]] = []
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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
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
    initial_state = problem.initial_state
    original_goal = problem.goal
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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
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
    initial_state = problem.initial_state
    original_goal = problem.goal
    different_goals = [problem.ground_condition(literal) for literal in original_goal]
    input = [(initial_state, different_goal) for different_goal in different_goals]
    output = model.forward(input)
    readout = output.readout('value')
    assert len(readout) == len(different_goals)
    assert readout.var() > 0.0000001

def test_save_and_load():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    # Create a model.
    hparam_config_1 = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
        message_hidden_size=3,
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
    assert model_1._hparam_config.message_hidden_size == model_2._hparam_config.message_hidden_size
    assert model_1._hparam_config.num_layers == model_2._hparam_config.num_layers
    assert extras_1 == extras_2
    assert hasattr(model_1, '_input_spec')
    assert hasattr(model_2, '_input_spec')
    assert isinstance(model_1._input_spec, tuple)
    assert isinstance(model_2._input_spec, tuple)
    assert len(model_1._input_spec) > 0
    assert len(model_2._input_spec) > 0
    assert len(model_1._input_spec) == len(model_2._input_spec)
    assert isinstance(model_1._input_spec[0], StateEncoder)
    assert isinstance(model_2._input_spec[0], StateEncoder)


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_simple_forward(domain_name: str):
    """Test basic functionality of the new encoder-based API."""
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)

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
    initial_state = problem.initial_state
    goal_condition = problem.goal
    ground_actions = initial_state.applicable_actions()

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


def test_sum_readout_single_group_matches_batched_values_and_gradients():
    readout = SumReadout(4, 3).double()
    nodes = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    other_nodes = torch.randn(3, 4, dtype=torch.float64)

    single = readout(nodes, torch.tensor([5]))
    batched = readout(torch.cat((nodes, other_nodes)), torch.tensor([5, 3]))[:1]

    torch.testing.assert_close(single, batched)
    inputs = (nodes, *readout.parameters())
    single_gradients = torch.autograd.grad(single.sum(), inputs)
    batched_gradients = torch.autograd.grad(batched.sum(), inputs)
    for single_gradient, batched_gradient in zip(single_gradients, batched_gradients, strict=True):
        torch.testing.assert_close(single_gradient, batched_gradient)


@pytest.mark.parametrize("empty_actions", [False, True])
def test_decoders_preserve_single_and_batched_outputs(empty_actions):
    domain = mm.Domain.from_file(DATA_DIR / 'blocks' / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / 'blocks' / 'problem.pddl')
    config = HyperparameterConfig(domain=domain, embedding_size=4, num_layers=2)
    input_spec = (StateEncoder(), GroundActionsEncoder())
    model = RelationalGraphNeuralNetwork(
        config,
        ModuleConfig(
            MeanAggregation(),
            PredicateMLPMessages(config, input_spec),
            MLPUpdates(config),
        ),
        input_spec,
        [
            ('q_values', ActionScalarDecoder(config)),
            ('actions', ActionEmbeddingDecoder()),
            ('objects', ObjectsEmbeddingDecoder()),
        ],
    )
    state = problem.initial_state
    actions = [] if empty_actions else list(state.applicable_actions())
    instance = (state, actions)

    single = model([instance])
    batched = model([instance, instance])

    for name, container_type in [('q_values', list), ('actions', tuple), ('objects', tuple)]:
        single_outputs = single.readout(name)
        batch_outputs = batched.readout(name)
        assert isinstance(single_outputs, container_type)
        assert isinstance(batch_outputs, container_type)
        assert len(single_outputs) == 1
        assert len(batch_outputs) == 2
        for output in batch_outputs:
            torch.testing.assert_close(single_outputs[0], output)
    assert single.readout('q_values')[0].shape == (len(actions),)
    assert single.readout('actions')[0].shape == (len(actions), 4)


def test_decoder_constructors():
    """Test all different decoders construct properly."""
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)

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
    initial_state = problem.initial_state
    goal_condition = problem.goal
    ground_actions = initial_state.applicable_actions()

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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)

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
    initial_state = problem.initial_state
    goal_condition = problem.goal
    ground_actions = initial_state.applicable_actions()
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


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_expressive_encoders(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8
    )
    input_spec = (ExpressiveStateEncoder(), ExpressiveGoalEncoder())
    output_spec = [('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=SmoothMaximumAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    assert model is not None
    input = [(problem.initial_state, problem.goal)]
    output = model.forward(input)
    value = output.readout('value')
    assert isinstance(value, torch.Tensor)
    assert value.numel() == 1
    assert not torch.isnan(value).any()


def test_predicate_linear_messages_forward_model():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    output_spec = [('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateLinearMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    output = model.forward([(problem.initial_state, problem.goal)])
    value = output.readout('value')
    assert isinstance(value, torch.Tensor)
    assert value.shape == (1,)
    assert not torch.isnan(value).any()


def test_linear_updates_forward_model():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    output_spec = [('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=LinearUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    output = model.forward([(problem.initial_state, problem.goal)])
    value = output.readout('value')
    assert isinstance(value, torch.Tensor)
    assert value.shape == (1,)
    assert not torch.isnan(value).any()


def test_predicate_linear_messages_shape_matches_predicate_messages():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=3)
    input_spec = (StateEncoder(), GoalEncoder())
    linear_messages = PredicateLinearMessages(hparam_config, input_spec)
    predicate_messages = PredicateMLPMessages(hparam_config, input_spec)
    relation_name, module = next(iter(linear_messages._relation_linears.items()))
    relation_module = cast(torch.nn.Linear, module)
    arity = relation_module.in_features // hparam_config.embedding_size
    argument_indices = torch.arange(arity, dtype=torch.long)
    node_embeddings = torch.randn(arity, hparam_config.embedding_size)

    linear_output, linear_indices = linear_messages(node_embeddings, {relation_name: argument_indices})
    predicate_output, predicate_indices = predicate_messages(node_embeddings, {relation_name: argument_indices})

    assert linear_output.shape == predicate_output.shape
    assert torch.equal(linear_indices, predicate_indices)


@pytest.mark.parametrize(("configured_hidden_size", "expected_hidden_size"), [(3, 3), (5, 5)])
def test_predicate_mlp_messages_uses_configured_hidden_size(
    configured_hidden_size: int | None,
    expected_hidden_size: int,
):
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        embedding_size=3,
        message_hidden_size=configured_hidden_size,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    predicate_messages = PredicateMLPMessages(hparam_config, input_spec)
    relation_name, relation_module = next(
        (name, cast(MLP, module))
        for name, module in predicate_messages._relation_mlps.items()
        if cast(MLP, module).input_size > hparam_config.embedding_size
    )
    arity = relation_module.input_size // hparam_config.embedding_size
    argument_indices = torch.arange(arity, dtype=torch.long)
    node_embeddings = torch.randn(arity, hparam_config.embedding_size)

    output_messages, output_indices = predicate_messages(
        node_embeddings,
        {relation_name: argument_indices},
    )

    assert hparam_config.message_hidden_size == expected_hidden_size
    assert relation_module.hidden_size == expected_hidden_size
    assert relation_module._inner.in_features == arity * hparam_config.embedding_size
    assert relation_module._inner.out_features == expected_hidden_size
    assert relation_module._outer.in_features == expected_hidden_size
    assert relation_module._outer.out_features == arity * hparam_config.embedding_size
    assert output_messages.shape == (arity, hparam_config.embedding_size)
    assert torch.equal(output_indices, argument_indices)


def test_predicate_linear_messages_has_no_argument_residual():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=3)
    input_spec = (StateEncoder(), GoalEncoder())
    linear_messages = PredicateLinearMessages(hparam_config, input_spec)
    relation_name, module = next(iter(linear_messages._relation_linears.items()))
    relation_module = cast(torch.nn.Linear, module)
    arity = relation_module.in_features // hparam_config.embedding_size
    argument_indices = torch.arange(arity, dtype=torch.long)
    node_embeddings = torch.arange(arity * hparam_config.embedding_size, dtype=torch.float).view(arity, hparam_config.embedding_size) + 1.0
    with torch.no_grad():
        relation_module.weight.zero_()
        relation_module.bias.zero_()

    output_messages, output_indices = linear_messages(node_embeddings, {relation_name: argument_indices})

    assert output_messages.shape == (arity, hparam_config.embedding_size)
    assert torch.equal(output_indices, argument_indices)
    assert torch.allclose(output_messages, torch.zeros_like(output_messages))
    assert not torch.allclose(output_messages, node_embeddings)


def test_predicate_linear_messages_ternarizes_and_records_quantization():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        embedding_size=3,
        ternarize_messages=True,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    linear_messages = PredicateLinearMessages(hparam_config, input_spec)
    relation_name, module = next(iter(linear_messages._relation_linears.items()))
    relation_module = cast(torch.nn.Linear, module)
    arity = relation_module.in_features // hparam_config.embedding_size
    argument_indices = torch.arange(arity, dtype=torch.long)
    node_embeddings = torch.randn(arity, hparam_config.embedding_size)

    linear_messages.begin_quantization_recording(layer_index=7)
    output_messages, _ = linear_messages(node_embeddings, {relation_name: argument_indices})
    records = linear_messages.quantization_records()

    assert len(records) == 1
    assert records[0].kind == 'ternary_message'
    assert records[0].layer_index == 7
    assert records[0].thresholds == (-1.0, 1.0)
    assert torch.equal(records[0].values, output_messages)
    assert set(torch.unique(output_messages).tolist()).issubset({-1.0, 0.0, 1.0})


def test_linear_updates_matches_underlying_linear_layer():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=3)
    linear_updates = LinearUpdates(hparam_config)
    node_embeddings = torch.randn(5, hparam_config.embedding_size)
    aggregated_messages = torch.randn(5, hparam_config.embedding_size)

    output = linear_updates(node_embeddings, aggregated_messages)
    expected = linear_updates._update(torch.cat((aggregated_messages, node_embeddings), 1))

    assert torch.allclose(output, expected)


@pytest.mark.parametrize(("hidden_size", "expected_hidden_size"), [(None, 5), (2, 2)])
def test_mlp_hidden_size(hidden_size: int | None, expected_hidden_size: int):
    mlp = (
        MLP(input_size=5, output_size=3)
        if hidden_size is None
        else MLP(input_size=5, output_size=3, hidden_size=hidden_size)
    )
    output = mlp(torch.randn(4, 5))

    assert mlp.input_size == 5
    assert mlp.hidden_size == expected_hidden_size
    assert mlp.output_size == 3
    assert mlp._inner.in_features == 5
    assert mlp._inner.out_features == expected_hidden_size
    assert mlp._outer.in_features == expected_hidden_size
    assert mlp._outer.out_features == 3
    assert output.shape == (4, 3)


def test_sparse_mlp_eval_mask_has_topk_active_inputs():
    sparse_mlp = SparseMLP(input_size=5, output_size=3, k=2)
    sparse_mlp.eval()
    gate = sparse_mlp._gate(sparse_mlp._outer_log_alpha)
    assert gate.shape == (3, 5)
    assert torch.equal(gate.sum(dim=1), torch.full((3,), 2.0))
    assert set(torch.unique(gate).tolist()).issubset({0.0, 1.0})


def test_sparse_mlp_k_zero_output_depends_only_on_bias():
    sparse_mlp = SparseMLP(input_size=4, output_size=2, k=0)
    input = torch.randn(6, 4)
    output = sparse_mlp(input)
    expected = sparse_mlp._outer.bias.view(1, -1).expand_as(output)
    assert torch.allclose(output, expected)


def test_sparse_mlp_k_larger_than_input_activates_all_inputs():
    sparse_mlp = SparseMLP(input_size=4, output_size=2, k=10)
    sparse_mlp.eval()
    gate = sparse_mlp._gate(sparse_mlp._outer_log_alpha)
    assert torch.equal(gate, torch.ones_like(gate))


@pytest.mark.parametrize("gate_mode", ["gumbel_topk", "deterministic_topk"])
def test_sparse_mlp_training_gate_is_hard(gate_mode: Literal["gumbel_topk", "deterministic_topk"]):
    sparse_mlp = SparseMLP(input_size=5, output_size=3, k=2, gate_mode=gate_mode)
    sparse_mlp.train()
    gate = sparse_mlp._gate(sparse_mlp._outer_log_alpha)
    assert torch.equal(gate.sum(dim=1), torch.full((3,), 2.0))
    assert set(torch.unique(gate.detach()).tolist()).issubset({0.0, 1.0})


def test_sparse_mlp_gumbel_topk_explores_tied_logits():
    sparse_mlp = SparseMLP(input_size=4, output_size=1, k=2, gate_mode="gumbel_topk")
    sparse_mlp.train()
    masks = {
        tuple(sparse_mlp._gate(sparse_mlp._outer_log_alpha).detach().view(-1).tolist())
        for _ in range(20)
    }
    assert len(masks) > 1


def test_sparse_mlp_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SparseMLP(input_size=4, output_size=2, k=-1)
    with pytest.raises(ValueError):
        SparseMLP(input_size=4, output_size=2, k=1, gate_mode="soft")  # type: ignore[arg-type]


def test_sparse_mlp_topk_margin_penalty_zero_when_gap_is_large():
    sparse_mlp = SparseMLP(input_size=4, output_size=2, k=2)
    with torch.no_grad():
        sparse_mlp._outer_log_alpha.copy_(
            torch.tensor([
                [4.0, 3.0, 0.0, -1.0],
                [5.0, 2.0, 0.5, -2.0],
            ])
        )
    penalty = sparse_mlp.topk_margin_penalty(margin=1.0)
    assert penalty.shape == ()
    assert penalty.item() == 0.0


def test_sparse_mlp_topk_margin_penalty_positive_when_boundary_is_close():
    sparse_mlp = SparseMLP(input_size=4, output_size=2, k=2)
    with torch.no_grad():
        sparse_mlp._outer_log_alpha.copy_(
            torch.tensor([
                [4.0, 3.0, 2.5, -1.0],
                [5.0, 2.0, 1.8, -2.0],
            ])
        )
    penalty = sparse_mlp.topk_margin_penalty(margin=1.0)
    assert torch.allclose(penalty, torch.tensor(0.65))


def test_sparse_mlp_topk_margin_penalty_has_gate_gradients():
    sparse_mlp = SparseMLP(input_size=4, output_size=1, k=2)
    with torch.no_grad():
        sparse_mlp._outer_log_alpha.copy_(torch.tensor([[4.0, 3.0, 2.5, -1.0]]))
    penalty = sparse_mlp.topk_margin_penalty(margin=1.0)
    penalty.backward()
    assert sparse_mlp._outer_log_alpha.grad is not None
    assert sparse_mlp._outer_log_alpha.grad.abs().sum() > 0


def test_sparse_mlp_topk_margin_penalty_zero_without_ranking_boundary():
    k_zero = SparseMLP(input_size=4, output_size=2, k=0)
    all_active = SparseMLP(input_size=4, output_size=2, k=4)
    more_than_all_active = SparseMLP(input_size=4, output_size=2, k=10)

    assert k_zero.topk_margin_penalty(margin=1.0).item() == 0.0
    assert all_active.topk_margin_penalty(margin=1.0).item() == 0.0
    assert more_than_all_active.topk_margin_penalty(margin=1.0).item() == 0.0


def test_sparse_mlp_topk_margin_penalty_rejects_negative_margin():
    sparse_mlp = SparseMLP(input_size=4, output_size=2, k=2)
    with pytest.raises(ValueError):
        sparse_mlp.topk_margin_penalty(margin=-1.0)


def test_sparse_mlp_messages_shape_and_no_residual_bypass():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        embedding_size=3,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    sparse_messages = SparseMLPMessages(hparam_config, input_spec, k=0, gate_mode="deterministic_topk")
    relation_name, module = next(iter(sparse_messages._relation_mlps.items()))
    relation_module = cast(SparseMLP, module)
    arity = relation_module.input_size // hparam_config.embedding_size
    argument_indices = torch.arange(arity, dtype=torch.long)
    node_embeddings = torch.arange(arity * hparam_config.embedding_size, dtype=torch.float).view(arity, hparam_config.embedding_size) + 1.0

    with torch.no_grad():
        relation_module._outer.bias.zero_()
    output_messages, output_indices = sparse_messages(node_embeddings, {relation_name: argument_indices})

    assert output_messages.shape == (arity, hparam_config.embedding_size)
    assert torch.equal(output_indices, argument_indices)
    assert torch.allclose(output_messages, torch.zeros_like(output_messages))
    assert not torch.allclose(output_messages, node_embeddings)


def test_sparse_mlp_messages_rejects_removed_linear_argument():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain)
    input_spec = (StateEncoder(), GoalEncoder())
    with pytest.raises(TypeError):
        SparseMLPMessages(hparam_config, input_spec, k=1, linear=True)  # type: ignore[call-arg]


def test_sparse_mlp_messages_topk_margin_penalty_averages_relations():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=2)
    input_spec = (StateEncoder(), GoalEncoder())
    sparse_messages = SparseMLPMessages(hparam_config, input_spec, k=1, gate_mode="deterministic_topk")

    expected_penalties: list[torch.Tensor] = []
    for index, module in enumerate(sparse_messages._relation_mlps.values()):
        relation_module = cast(SparseMLP, module)
        with torch.no_grad():
            relation_module._outer_log_alpha.zero_()
            relation_module._outer_log_alpha[:, 0] = 1.0 + index
            relation_module._outer_log_alpha[:, 1] = 0.5
        expected_penalties.append(relation_module.topk_margin_penalty(margin=1.0))

    penalty = sparse_messages.topk_margin_penalty(margin=1.0)
    assert torch.allclose(penalty, torch.stack(expected_penalties).mean())


def test_sparse_mlp_messages_topk_margin_penalty_empty_module_is_zero():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=2)
    sparse_messages = SparseMLPMessages(hparam_config, input_spec=(), k=1)
    penalty = sparse_messages.topk_margin_penalty(margin=1.0)
    assert penalty.shape == ()
    assert penalty.item() == 0.0


def test_sparse_mlp_updates_topk_margin_penalty_delegates_to_update_network():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=2)
    sparse_updates = SparseMLPUpdates(hparam_config, k=1, gate_mode="deterministic_topk")
    with torch.no_grad():
        sparse_updates._update._outer_log_alpha.zero_()
        sparse_updates._update._outer_log_alpha[:, 0] = 1.0
        sparse_updates._update._outer_log_alpha[:, 1] = 0.25

    assert torch.allclose(
        sparse_updates.topk_margin_penalty(margin=1.0),
        sparse_updates._update.topk_margin_penalty(margin=1.0),
    )


def _make_decodable_model(domain: mm.Domain) -> RelationalGraphNeuralNetwork:
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=4,
        embedding_size=8,
        normalize_updates=True,
        channelwise_normalization=True,
        residual_updates=False,
        or_residual_updates=True,
        binarize_updates=True,
        ternarize_messages=True,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    output_spec = [('value', ObjectsScalarDecoder(hparam_config)), ('embeddings', ObjectsEmbeddingDecoder())]
    module_config = ModuleConfig(
        aggregation_function=HardMaximumAggregation(),
        message_function=SenderOnlyMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    return RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_decodable_configuration(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = _make_decodable_model(domain)
    input = [(problem.initial_state, problem.goal)]
    output = model.forward(input)
    value = output.readout('value')
    assert isinstance(value, torch.Tensor)
    assert value.numel() == 1
    assert not torch.isnan(value).any()
    embeddings = output.readout('embeddings')
    for instance_embeddings in embeddings:
        unique_values = torch.unique(instance_embeddings)
        assert all(v in (0.0, 1.0) for v in unique_values.tolist()), 'Embeddings must be binary.'


@pytest.mark.parametrize("domain_name", [('blocks'), ('gripper')])
def test_decodable_configuration_deterministic(domain_name: str):
    domain_path = DATA_DIR / domain_name / 'domain.pddl'
    problem_path = DATA_DIR / domain_name / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = _make_decodable_model(domain)
    input = [(problem.initial_state, problem.goal)]
    embeddings_1 = model.forward(input).readout('embeddings')
    embeddings_2 = model.forward(input).readout('embeddings')
    for instance_embeddings_1, instance_embeddings_2 in zip(embeddings_1, embeddings_2):
        assert torch.equal(instance_embeddings_1, instance_embeddings_2), 'Forward passes must be deterministic.'


def test_quantization_records_on_decodable_configuration():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    model = _make_decodable_model(domain)
    assert model.quantization_records() == ()

    hook_records = []
    def hook_function(x: ForwardState):
        records = x.quantization_records()
        assert len(records) > 0
        assert all(record.layer_index == x.get_layer_index() for record in records)
        hook_records.append(records)
    model.add_hook(hook_function)

    input = [(problem.initial_state, problem.goal)]
    output = model.forward(input)
    final_records = output.quantization_records()
    model_records = model.quantization_records()
    assert len(final_records) == len(model_records)
    assert all(output_record is model_record for output_record, model_record in zip(final_records, model_records))
    assert len(final_records) == sum(len(records) for records in hook_records)

    kinds = {record.kind for record in final_records}
    assert kinds == {'binary_update', 'ternary_message'}
    for record in final_records:
        assert record.logits.dtype.is_floating_point
        assert record.logits.requires_grad
        assert record.values.dtype.is_floating_point
        if record.kind == 'binary_update':
            assert record.thresholds == (0.0,)
            assert set(torch.unique(record.values).tolist()).issubset({0.0, 1.0})
        else:
            assert record.thresholds == (-1.0, 1.0)
            assert set(torch.unique(record.values).tolist()).issubset({-1.0, 0.0, 1.0})


def test_quantization_records_empty_without_quantization():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    hparam_config = HyperparameterConfig(
        domain=domain,
        num_layers=2,
        embedding_size=4,
        binarize_updates=False,
        ternarize_messages=False,
    )
    input_spec = (StateEncoder(), GoalEncoder())
    output_spec = [('value', ObjectsScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config)
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)  # type: ignore
    input = [(problem.initial_state, problem.goal)]
    output = model.forward(input)
    assert output.quantization_records() == ()
    assert model.quantization_records() == ()


def test_boundary_margin_penalty():
    binary_logits = torch.tensor([-0.10, 0.40], requires_grad=True)
    ternary_logits = torch.tensor([-1.10, 0.00, 1.40], requires_grad=True)
    records = (
        QuantizationRecord(
            kind='binary_update',
            layer_index=0,
            logits=binary_logits,
            thresholds=(0.0,),
            values=(binary_logits > 0).float(),
        ),
        QuantizationRecord(
            kind='ternary_message',
            layer_index=0,
            logits=ternary_logits,
            thresholds=(-1.0, 1.0),
            values=(ternary_logits > 1.0).float() - (ternary_logits < -1.0).float(),
        ),
    )
    penalty = boundary_margin_penalty(records, margin=0.25)
    assert penalty.shape == ()
    assert torch.isfinite(penalty)
    assert penalty > 0
    penalty.backward()
    assert binary_logits.grad is not None
    assert ternary_logits.grad is not None
    assert binary_logits.grad.abs().sum() > 0
    assert ternary_logits.grad.abs().sum() > 0

    empty_penalty = boundary_margin_penalty((), margin=0.25, device=binary_logits.device, dtype=binary_logits.dtype)
    assert empty_penalty.shape == ()
    assert empty_penalty.device == binary_logits.device
    assert empty_penalty.dtype == binary_logits.dtype
    assert empty_penalty.item() == 0.0


def test_invalid_hparam_combinations():
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    domain = mm.Domain.from_file(domain_path)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, or_residual_updates=True, residual_updates=False, binarize_updates=False)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, or_residual_updates=True, residual_updates=True, binarize_updates=True)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, channelwise_normalization=True, normalize_updates=False)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, residual_updates=True, binarize_updates=True)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, message_hidden_size=0)
    with pytest.raises(ValueError):
        HyperparameterConfig(domain=domain, message_hidden_size=-1)


def _reference_messages(message_function: PredicateMLPMessages, node_embeddings: torch.Tensor,
                        relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The messages computed one relation at a time with its own MLP, as (messages, indices) sorted by index."""
    messages = []
    indices = []
    for relation_name, argument_indices in relations.items():
        if argument_indices.numel() > 0:
            mlp = message_function._relation_mlps[relation_name]
            arguments = node_embeddings.index_select(0, argument_indices).view(-1, mlp.input_size)
            messages.append((arguments + mlp(arguments)).view(-1, node_embeddings.shape[1]))
            indices.append(argument_indices)
    return torch.cat(messages), torch.cat(indices)


@pytest.mark.parametrize("dom", ['blocks', 'gripper'])
def test_predicate_mlp_messages_match_one_mlp_per_relation(dom: str):
    domain = mm.Domain.from_file(DATA_DIR / dom / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / dom / 'problem.pddl')
    hparam_config = HyperparameterConfig(domain=domain, num_layers=1, embedding_size=8)
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    message_function = PredicateMLPMessages(hparam_config, input_spec)
    states = [problem.initial_state] + [successor for _, successor in problem.initial_state.successor_states()]
    encoding = get_input_from_encoders([(state, list(state.applicable_actions()), problem.goal) for state in states],
                                       input_spec, torch.device('cpu'))
    relations = encoding.flattened_relations
    torch.manual_seed(0)
    node_embeddings = torch.randn(encoding.node_count, 8, requires_grad=True)
    messages, indices = message_function.forward(node_embeddings, relations)
    expected_messages, expected_indices = _reference_messages(message_function, node_embeddings, relations)
    # The relations may come in another order; the messages per receiving node do not.
    sums = torch.zeros_like(node_embeddings).index_add(0, indices, messages)
    expected_sums = torch.zeros_like(node_embeddings).index_add(0, expected_indices, expected_messages)
    assert torch.equal(indices.sort().values, expected_indices.sort().values)
    assert torch.allclose(sums, expected_sums, atol=1e-5)
    # Relations without atoms in the batch leave their MLPs unused.
    inputs = [node_embeddings, *message_function.parameters()]
    gradients = torch.autograd.grad(sums.square().sum(), inputs, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected_sums.square().sum(), inputs, allow_unused=True)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        assert (gradient is None) == (expected_gradient is None)
        if gradient is not None:
            assert torch.allclose(gradient, expected_gradient, atol=1e-4)


def test_sum_readout_sums_each_group():
    torch.manual_seed(0)
    readout = SumReadout(4, 3)
    node_embeddings = torch.randn(9, 4)
    node_sizes = torch.tensor([2, 0, 4, 3])
    expected = torch.stack([node_embeddings[0:2].sum(0), torch.zeros(4), node_embeddings[2:6].sum(0),
                            node_embeddings[6:9].sum(0)])
    assert torch.allclose(readout(node_embeddings, node_sizes), readout._value(expected), atol=1e-6)


def _q_value_model(domain: mm.Domain, num_layers: int, aggregation: type = SumAggregation, messages: type = PredicateMLPMessages,
                   updates: type = MLPUpdates, model_type: type = RelationalGraphNeuralNetwork,
                   **config: Any) -> RelationalGraphNeuralNetwork:
    torch.manual_seed(0)
    hparam_config = HyperparameterConfig(domain=domain, num_layers=num_layers, embedding_size=8, **config)
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    module_config = ModuleConfig(aggregation(), messages(hparam_config, input_spec), updates(hparam_config))
    return model_type(hparam_config, module_config, input_spec, [('q', ActionScalarDecoder(hparam_config))])


def _eager_and_replayed_q_values(monkeypatch: pytest.MonkeyPatch, model: RelationalGraphNeuralNetwork,
                                 inputs: list[tuple]) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    """The Q-values without gradients, run layer by layer and as far as possible replayed, and the number of
    replays."""
    replays = 0
    replay_layers = RelationalLayerStackModule._replay_layers

    def counting_replay_layers(self: RelationalLayerStackModule, *args: Any) -> torch.Tensor:
        nonlocal replays
        replays += 1
        return replay_layers(self, *args)

    with monkeypatch.context() as patch:
        patch.setattr(RelationalLayerStackModule, '_replay_layers', counting_replay_layers)
        with torch.no_grad():
            replayed = model.forward(inputs).readout('q')
    with monkeypatch.context() as patch:
        patch.setattr(rgnn_model, 'MIN_REPLAYED_LAYERS', model._hparam_config.num_layers + 1)
        with torch.no_grad():
            eager = model.forward(inputs).readout('q')
    return eager, replayed, replays


def _q_value_inputs(dom: str) -> tuple[mm.Domain, list[tuple]]:
    domain = mm.Domain.from_file(DATA_DIR / dom / 'domain.pddl')
    problem = mm.Problem.from_file(domain, DATA_DIR / dom / 'problem.pddl')
    states = [problem.initial_state] + [successor for _, successor in problem.initial_state.successor_states()]
    return domain, [(state, list(state.applicable_actions()), problem.goal) for state in states]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("dom", ['blocks', 'gripper'])
@pytest.mark.parametrize("config", [
    {},
    {'global_readout': False},
    {'normalize_updates': False, 'residual_updates': True},
    {'channelwise_normalization': True},
])
def test_replayed_layers_give_the_outputs_of_running_every_layer(monkeypatch: pytest.MonkeyPatch, dom: str, config: dict[str, Any]):
    domain, inputs = _q_value_inputs(dom)
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS + 2, **config).to('cuda')
    replays = 0
    replay_layers = RelationalLayerStackModule._replay_layers

    def counting_replay_layers(self: RelationalLayerStackModule, input: Any, node_embeddings: torch.Tensor) -> torch.Tensor:
        nonlocal replays
        replays += 1
        return replay_layers(self, input, node_embeddings)

    monkeypatch.setattr(RelationalLayerStackModule, '_replay_layers', counting_replay_layers)
    # The first call of the model replays, before any layer ran without a recording.
    with torch.no_grad():
        replayed = model.forward(inputs).readout('q')
    monkeypatch.setattr(rgnn_model, 'MIN_REPLAYED_LAYERS', model._hparam_config.num_layers + 1)
    with torch.no_grad():
        expected = model.forward(inputs).readout('q')
    assert replays == 1
    for replayed_values, expected_values in zip(replayed, expected):
        assert torch.allclose(replayed_values, expected_values, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("messages", [PredicateMLPMessages, PredicateLinearMessages])
@pytest.mark.parametrize("aggregation", [SumAggregation, MeanAggregation, HardMaximumAggregation, SmoothMaximumAggregation])
@pytest.mark.parametrize("updates", [MLPUpdates, LinearUpdates])
def test_functions_that_support_replay_give_the_outputs_of_running_every_layer(monkeypatch: pytest.MonkeyPatch, messages: type,
                                                                                aggregation: type, updates: type):
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, aggregation, messages, updates).to('cuda')
    eager, replayed, replays = _eager_and_replayed_q_values(monkeypatch, model, inputs)
    assert replays == 1
    for replayed_values, eager_values in zip(replayed, eager):
        assert torch.allclose(replayed_values, eager_values, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("messages", [AttentionMessages, SparseMLPMessages, SenderOnlyMLPMessages])
def test_functions_without_replay_support_run_every_layer(monkeypatch: pytest.MonkeyPatch, messages: type):
    domain, inputs = _q_value_inputs('blocks')
    if messages is SparseMLPMessages:
        messages = lambda config, spec: SparseMLPMessages(config, spec, k=2)  # noqa: E731
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, messages=messages).to('cuda').eval()
    eager, replayed, replays = _eager_and_replayed_q_values(monkeypatch, model, inputs)
    assert replays == 0
    for replayed_values, eager_values in zip(replayed, eager):
        assert torch.allclose(replayed_values, eager_values)


class _SynchronizingMessages(PredicateMLPMessages):
    """Claims to support replay, but synchronizes with the host, which a recording cannot contain."""
    supports_replay = True

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if node_embeddings.sum().item() > float('inf'):
            raise AssertionError("unreachable")
        return super().forward(node_embeddings, relations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_a_failed_recording_reports_its_error_and_leaves_the_thread_usable():
    domain, inputs = _q_value_inputs('blocks')
    failing = _q_value_model(domain, MIN_REPLAYED_LAYERS, messages=_SynchronizingMessages).to('cuda')
    with pytest.raises(RuntimeError, match="capturing"):
        with torch.no_grad():
            failing.forward(inputs).readout('q')
    working = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    with torch.no_grad():
        q_values = working.forward(inputs).readout('q')
    assert all(torch.isfinite(values).all() for values in q_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_layers_run_one_by_one_with_hooks_or_few_layers(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('blocks')

    def no_replay(self: RelationalLayerStackModule, input: Any, node_embeddings: torch.Tensor) -> torch.Tensor:
        raise AssertionError("layers must not be replayed")

    monkeypatch.setattr(RelationalLayerStackModule, '_replay_layers', no_replay)
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS + 2).to('cuda')
    layers: list[int] = []
    model.add_hook(lambda state: layers.append(state.get_layer_index()))
    with torch.no_grad():
        model.forward(inputs).readout('q')
    assert len(layers) == MIN_REPLAYED_LAYERS + 2
    shallow_model = _q_value_model(domain, MIN_REPLAYED_LAYERS - 1).to('cuda')
    with torch.no_grad():
        shallow_model.forward(inputs).readout('q')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_replaying_layers_reuses_the_memory_of_earlier_recordings():
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS + 2).to('cuda')
    with torch.no_grad():
        for count in range(1, len(inputs) + 1):
            model.forward(inputs[:count]).readout('q')
        reserved = torch.cuda.memory_reserved()
        for _ in range(3):
            for count in range(1, len(inputs) + 1):
                model.forward(inputs[:count]).readout('q')
    assert torch.cuda.memory_reserved() == reserved


def _loss_and_gradients(model: RelationalGraphNeuralNetwork, *batches: list[tuple]) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    """A loss of the Q-values of the batches, each from a forward pass of its own, and the gradients of the parameters."""
    model.zero_grad(set_to_none=True)
    loss = sum(values.square().sum() for batch in batches for values in model.forward(batch).readout('q'))
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    return loss.detach(), [parameter.grad for parameter in model.parameters()]


def _eager_and_recorded_gradients(monkeypatch: pytest.MonkeyPatch, model: RelationalGraphNeuralNetwork, *batches: list[tuple]
                                  ) -> tuple[tuple[torch.Tensor, list[torch.Tensor | None]],
                                             tuple[torch.Tensor, list[torch.Tensor | None]], int]:
    """The loss and gradients running every layer and as far as possible from training recordings, and the number of
    recordings."""
    recordings = 0

    class CountingRecording(rgnn_model._TrainingRecording):
        def __init__(self, *args: Any):
            nonlocal recordings
            recordings += 1
            super().__init__(*args)

    with monkeypatch.context() as patch:
        patch.setattr(rgnn_model, '_TrainingRecording', CountingRecording)
        recorded = _loss_and_gradients(model, *batches)
    with monkeypatch.context() as patch:
        patch.setattr(rgnn_model, 'REPLAY_TRAINING', False)
        eager = _loss_and_gradients(model, *batches)
    return eager, recorded, recordings


def _assert_same_gradients(eager: tuple[torch.Tensor, list[torch.Tensor | None]],
                           recorded: tuple[torch.Tensor, list[torch.Tensor | None]]) -> None:
    assert torch.allclose(recorded[0], eager[0], rtol=1e-5)
    for recorded_gradient, eager_gradient in zip(recorded[1], eager[1]):
        assert (recorded_gradient is None) == (eager_gradient is None)
        if eager_gradient is not None:
            # Relative to the scale of the gradient: the atomic additions of index_add_ make even two runs of every
            # layer differ slightly.
            assert (recorded_gradient - eager_gradient).abs().max() <= 1e-4 * eager_gradient.abs().max() + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("messages", [PredicateMLPMessages, PredicateLinearMessages])
@pytest.mark.parametrize("aggregation", [SumAggregation, MeanAggregation, HardMaximumAggregation, SmoothMaximumAggregation])
@pytest.mark.parametrize("updates", [MLPUpdates, LinearUpdates])
def test_recorded_training_gives_the_gradients_of_running_every_layer(monkeypatch: pytest.MonkeyPatch, messages: type,
                                                                     aggregation: type, updates: type):
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, aggregation, messages, updates).to('cuda')
    eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 1
    _assert_same_gradients(eager, recorded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("dom", ['blocks', 'gripper'])
@pytest.mark.parametrize("config", [
    {},
    {'global_readout': True},
    {'normalize_updates': False, 'residual_updates': True},
    {'channelwise_normalization': True, 'residual_updates': False},
])
def test_recorded_training_supports_the_layer_configurations(monkeypatch: pytest.MonkeyPatch, dom: str, config: dict[str, Any]):
    domain, inputs = _q_value_inputs(dom)
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS + 2, **config).to('cuda')
    eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 1
    _assert_same_gradients(eager, recorded)


class _EvaluatingModel(RelationalGraphNeuralNetwork):
    """Runs the layers without gradients on another batch after each forward pass, recording into the same memory."""

    def forward(self, x: list[tuple]) -> ForwardState:  # type: ignore
        forward_state = super().forward(x)
        with torch.no_grad():
            super().forward(x[:1] + x).readout('q')
        return forward_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_recordings_pending_a_backward_pass_keep_their_gradients(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, model_type=_EvaluatingModel).to('cuda')
    # Each recording may reuse the memory of the ones before, whose backward passes replay after it.
    eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs[:2], inputs, inputs[1:], inputs[:1])
    assert recordings == 4
    _assert_same_gradients(eager, recorded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_recorded_training_reuses_the_memory_of_earlier_recordings():
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS + 2).to('cuda')
    for count in range(1, len(inputs) + 1):
        _loss_and_gradients(model, inputs[:count], inputs[count - 1:])
    reserved = torch.cuda.memory_reserved()
    for _ in range(3):
        for count in range(1, len(inputs) + 1):
            _loss_and_gradients(model, inputs[:count], inputs[count - 1:])
    assert torch.cuda.memory_reserved() == reserved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_a_recording_without_backward_pass_frees_its_memory():
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    _loss_and_gradients(model, inputs)
    allocated = torch.cuda.memory_allocated()
    q_values = model.forward(inputs).readout('q')
    assert torch.cuda.memory_allocated() > allocated
    del q_values
    assert torch.cuda.memory_allocated() == allocated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_recorded_layers_support_one_backward_pass():
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    loss = sum(values.sum() for values in model.forward(inputs).readout('q'))
    assert isinstance(loss, torch.Tensor)
    loss.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="one backward pass"):
        loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("config", [{'ternarize_messages': True}, {'binarize_updates': True, 'residual_updates': False}])
def test_quantized_layers_train_without_recordings(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]):
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, **config).to('cuda')
    eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 0
    _assert_same_gradients(eager, recorded)
    assert len(model.quantization_records()) > 0


class _GradientOnlyMessages(PredicateMLPMessages):
    """Reads a parameter only when gradients are enabled, so the layer without gradients does not show it."""
    supports_replay = True

    def __init__(self, hparam_config: HyperparameterConfig, input_spec: tuple[Encoder, ...]):
        super().__init__(hparam_config, input_spec)
        self._scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        messages, indices = super().forward(node_embeddings, relations)
        if torch.is_grad_enabled():
            messages = messages * self._scale
        return messages, indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_layers_that_read_tensors_only_with_gradients_train_without_recordings(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, messages=_GradientOnlyMessages).to('cuda')
    with pytest.warns(UserWarning, match="only with gradients enabled"):
        eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 1
    _assert_same_gradients(eager, recorded)
    # Not attempted again.
    eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 0


class _SynchronizingGradient(torch.autograd.Function):
    """The identity, whose backward pass synchronizes with the host, which a recording cannot contain."""

    @staticmethod
    def forward(ctx: Any, messages: torch.Tensor) -> torch.Tensor:
        return messages.clone()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.sum().item() > float('inf'):
            raise AssertionError("unreachable")
        return gradient


class _SynchronizingBackwardMessages(PredicateMLPMessages):
    supports_replay = True

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        messages, indices = super().forward(node_embeddings, relations)
        return _SynchronizingGradient.apply(messages), indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_layers_whose_backward_pass_cannot_be_recorded_train_without_recordings(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS, messages=_SynchronizingBackwardMessages).to('cuda')
    with pytest.warns(UserWarning, match="recording its backward pass failed"):
        eager, recorded, recordings = _eager_and_recorded_gradients(monkeypatch, model, inputs)
    assert recordings == 1
    _assert_same_gradients(eager, recorded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
@pytest.mark.parametrize("failing_messages", [_GradientOnlyMessages, _SynchronizingMessages])
def test_recordings_work_after_a_failed_first_one(failing_messages: type):
    domain, inputs = _q_value_inputs('blocks')
    failing = _q_value_model(domain, MIN_REPLAYED_LAYERS, messages=failing_messages).to('cuda')
    working = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    errors: list[BaseException] = []

    def record() -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    _loss_and_gradients(failing, inputs)
                except RuntimeError:
                    pass
            # Frees the failed recording, which pytest.raises would keep through the traceback.
            gc.collect()
            with torch.no_grad():
                working.forward(inputs).readout('q')
            _loss_and_gradients(working, inputs)
        except BaseException as error:
            errors.append(error)

    # A new thread has replay resources of its own, which no recording used before.
    thread = threading.Thread(target=record)
    thread.start()
    thread.join()
    assert not errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_recorded_training_accumulates_the_gradients_of_separate_backward_passes(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')

    def accumulated_gradients() -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        model.zero_grad(set_to_none=True)
        for batch in (inputs[:2], inputs, inputs[1:]):
            loss = sum(values.square().sum() for values in model.forward(batch).readout('q'))
            assert isinstance(loss, torch.Tensor)
            loss.backward()
        return loss.detach(), [parameter.grad for parameter in model.parameters()]

    recorded = accumulated_gradients()
    monkeypatch.setattr(rgnn_model, 'REPLAY_TRAINING', False)
    _assert_same_gradients(accumulated_gradients(), recorded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_training_without_recordings_supports_repeated_and_higher_order_backward_passes(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    monkeypatch.setattr(rgnn_model, 'REPLAY_TRAINING', False)
    loss = sum(values.square().sum() for values in model.forward(inputs).readout('q'))
    assert isinstance(loss, torch.Tensor)
    loss.backward(retain_graph=True)
    loss.backward(retain_graph=True)
    gradients = torch.autograd.grad(loss, list(model.parameters()), create_graph=True, allow_unused=True)
    sum(gradient.square().sum() for gradient in gradients if gradient is not None).backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_layers_without_tensors_that_require_gradients_record_no_backward_pass(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('blocks')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda').requires_grad_(False)
    with torch.no_grad():
        expected = model.forward(inputs).readout('q')

    def no_backward_pass(*args: Any) -> None:
        raise AssertionError("nothing to backpropagate to")

    monkeypatch.setattr(rgnn_model._TrainingRecording, '_record_backward_pass', no_backward_pass)
    q_values = model.forward(inputs).readout('q')
    for values, expected_values in zip(q_values, expected):
        assert not values.requires_grad
        assert torch.allclose(values, expected_values, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_recordings_replayed_on_several_streams_keep_their_gradients(monkeypatch: pytest.MonkeyPatch):
    domain, inputs = _q_value_inputs('gripper')
    model = _q_value_model(domain, MIN_REPLAYED_LAYERS).to('cuda')
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    def gradients_on_streams() -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        model.zero_grad(set_to_none=True)
        losses = []
        for stream, batch in zip(streams, (inputs, inputs[1:])):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                losses.append(sum(values.square().sum() for values in model.forward(batch).readout('q')))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        loss = losses[0] + losses[1]
        loss.backward()
        return loss.detach(), [parameter.grad for parameter in model.parameters()]

    recorded = gradients_on_streams()
    monkeypatch.setattr(rgnn_model, 'REPLAY_TRAINING', False)
    _assert_same_gradients(gradients_on_streams(), recorded)


def test_tensors_in_named_tuples_and_dict_subclasses_are_mapped():
    Pair = namedtuple('Pair', ['first', 'second'])
    mapped = rgnn_model._map_tensors(lambda tensor: tensor + 1, (
        Pair(torch.zeros(1), [torch.ones(1)]),
        OrderedDict(x=torch.ones(1)),
        defaultdict(list, y=torch.ones(1)),
        torch.max(torch.ones(2, 2), 0),
        5,
    ))
    pair, ordered, default, maximum, number = mapped
    assert isinstance(pair, Pair) and pair.first.item() == 1 and pair.second[0].item() == 2
    assert type(ordered) is OrderedDict and ordered['x'].item() == 2
    assert type(default) is defaultdict and default.default_factory is list and default['y'].item() == 2
    assert type(maximum) is type(torch.max(torch.ones(2, 2), 0)) and maximum.values.tolist() == [2, 2]
    assert number == 5
