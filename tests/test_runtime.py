import pymimir as mm
import pytest
import torch

from pathlib import Path
from pymimir_rgnn import ActionScalarDecoder, GoalEncoder, GroundActionsEncoder, HyperparameterConfig, MeanAggregation, MLPUpdates, ModuleConfig, PredicateMLPMessages, RelationalGraphNeuralNetwork, StateEncoder, is_tf32_enabled, set_tf32_enabled


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


def _build_model() -> tuple[RelationalGraphNeuralNetwork, list[tuple]]:
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
    return model, [(state, actions, goal)]


def test_enable_torch_compile_uses_expected_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _build_model()
    compile_calls: list[tuple[str, bool]] = []

    def fake_compile(fn: object, *, mode: str, dynamic: bool):
        compile_calls.append((mode, dynamic))
        return fn

    monkeypatch.setattr(torch, 'compile', fake_compile)

    training_mode = model.enable_torch_compile('training')
    inference_mode = model.enable_torch_compile('inference', dynamic=True)

    assert training_mode == 'default'
    assert inference_mode == 'reduce-overhead'
    assert model.get_torch_compile_mode('training') == 'default'
    assert model.get_torch_compile_mode('inference') == 'reduce-overhead'
    assert compile_calls == [('default', False), ('reduce-overhead', True)]


def test_disable_torch_compile_clears_runtime_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _build_model()

    def fake_compile(fn: object, *, mode: str, dynamic: bool):
        return fn

    monkeypatch.setattr(torch, 'compile', fake_compile)

    model.enable_torch_compile('training', compile_mode='max-autotune')
    assert model.get_torch_compile_mode('training') == 'max-autotune'

    model.disable_torch_compile('training')
    assert model.get_torch_compile_mode('training') is None


def test_forward_with_hooks_uses_eager_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model, input_data = _build_model()

    def fake_compile(fn: object, *, mode: str, dynamic: bool):
        def compiled(_: object) -> torch.Tensor:
            raise AssertionError('Compiled path should not run when hooks are active.')
        return compiled

    monkeypatch.setattr(torch, 'compile', fake_compile)

    model.enable_torch_compile('inference')
    hook_layers: list[int] = []
    model.add_hook(lambda state: hook_layers.append(state.get_layer_index()))
    model.eval()

    readout = model.forward(input_data).readout('q_values')
    assert len(readout[0]) > 0
    assert len(hook_layers) == model.get_hparam_config().num_layers


def test_tf32_helpers_toggle_state() -> None:
    original_precision = torch.get_float32_matmul_precision()
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    try:
        set_tf32_enabled(True)
        assert is_tf32_enabled()

        set_tf32_enabled(False)
        assert not is_tf32_enabled()
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
