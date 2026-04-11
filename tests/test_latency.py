import pymimir as mm

from pathlib import Path
from pymimir_rgnn import ActionScalarDecoder, GoalEncoder, GroundActionsEncoder, HyperparameterConfig, MeanAggregation, MLPUpdates, ModuleConfig, PredicateMLPMessages, RelationalGraphNeuralNetwork, StateEncoder, measure_forward_readout_latency


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


def test_measure_forward_readout_latency() -> None:
    domain_path = DATA_DIR / 'blocks' / 'domain.pddl'
    problem_path = DATA_DIR / 'blocks' / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    hparam_config = HyperparameterConfig(domain=domain, num_layers=2, embedding_size=4)
    input_spec = (StateEncoder(), GroundActionsEncoder(), GoalEncoder())
    output_spec = [('q_values', ActionScalarDecoder(hparam_config))]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation(),
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=MLPUpdates(hparam_config),
    )
    model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)
    state = problem.get_initial_state()
    actions = state.generate_applicable_actions()
    goal = problem.get_goal_condition()
    latency = measure_forward_readout_latency(
        model,
        [(state, actions, goal)],
        'q_values',
        iterations=2,
        warmup_iterations=1,
    )
    assert latency.device_type == model.get_device().type
    assert latency.readout_names == ('q_values',)
    assert latency.total.num_iterations == 2
    assert latency.encode.num_iterations == 2
    assert latency.compute.num_iterations == 2
    assert latency.readout.num_iterations == 2
