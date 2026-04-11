from __future__ import annotations

import argparse
import json
import sys
import torch
import pymimir as mm

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymimir_rgnn as rgnn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark the model.forward(...).readout(...) latency path.',
    )
    parser.add_argument('--domain-path', type=Path, required=True, help='Path to the PDDL domain file.')
    parser.add_argument('--problem-path', type=Path, required=True, help='Path to the PDDL problem file.')
    parser.add_argument(
        '--aggregation',
        choices=('mean', 'sum', 'hardmax', 'smoothmax'),
        default='mean',
        help='Aggregation function to benchmark.',
    )
    parser.add_argument('--embedding-size', type=int, default=32, help='Node embedding size.')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of message-passing layers.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of identical instances in the batch.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of measured iterations.')
    parser.add_argument('--warmup-iterations', type=int, default=20, help='Number of warmup iterations.')
    parser.add_argument('--trials', type=int, default=3, help='Number of repeated benchmark trials.')
    parser.add_argument(
        '--mode',
        choices=('inference', 'training'),
        default='inference',
        help='Benchmark inference or training-style forward passes.',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='Device used for the model benchmark.',
    )
    parser.add_argument(
        '--readout',
        dest='readouts',
        action='append',
        choices=('q_values', 'value'),
        help='Readout to benchmark. Repeat to benchmark multiple readouts. Defaults to q_values.',
    )
    parser.add_argument('--global-readout', action='store_true', help='Enable the global readout module.')
    parser.add_argument('--normalize-updates', dest='normalize_updates', action='store_true', default=True, help='Enable LayerNorm on updates.')
    parser.add_argument('--no-normalize-updates', dest='normalize_updates', action='store_false', help='Disable LayerNorm on updates.')
    parser.add_argument('--residual-updates', action='store_true', default=True, help='Enable residual updates.')
    parser.add_argument('--no-residual-updates', dest='residual_updates', action='store_false', help='Disable residual updates.')
    parser.add_argument('--binarize-updates', action='store_true', help='Enable binarized updates.')
    parser.add_argument('--cuda-events', dest='use_cuda_events', action='store_true', default=None, help='Use CUDA event timing for compute and readout sections when running on CUDA.')
    parser.add_argument('--no-cuda-events', dest='use_cuda_events', action='store_false', help='Disable CUDA event timing and use synchronized wall-clock timing for all sections.')
    return parser.parse_args()


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == 'cpu':
        return torch.device('cpu')
    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but no CUDA device is available.')
        return torch.device('cuda')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_aggregation(name: str) -> rgnn.AggregationFunction:
    if name == 'mean':
        return rgnn.MeanAggregation()
    if name == 'sum':
        return rgnn.SumAggregation()
    if name == 'hardmax':
        return rgnn.HardMaximumAggregation()
    if name == 'smoothmax':
        return rgnn.SmoothMaximumAggregation()
    raise AssertionError(f'Unsupported aggregation {name!r}.')


def _build_output_spec(
    hparam_config: rgnn.HyperparameterConfig,
    requested_readouts: tuple[str, ...],
) -> list[tuple[str, rgnn.Decoder]]:
    output_spec: list[tuple[str, rgnn.Decoder]] = []
    if 'q_values' in requested_readouts:
        output_spec.append(('q_values', rgnn.ActionScalarDecoder(hparam_config)))
    if 'value' in requested_readouts:
        output_spec.append(('value', rgnn.ObjectsScalarDecoder(hparam_config)))
    assert len(output_spec) > 0, 'At least one readout must be configured.'
    return output_spec


def main() -> int:
    args = _parse_args()
    readouts = tuple(args.readouts) if args.readouts else ('q_values',)
    device = _resolve_device(args.device)

    domain = mm.Domain(args.domain_path)
    problem = mm.Problem(domain, args.problem_path)
    state = problem.get_initial_state()
    goal = problem.get_goal_condition()
    actions = state.generate_applicable_actions()
    input_spec = (rgnn.StateEncoder(), rgnn.GroundActionsEncoder(), rgnn.GoalEncoder())
    hparam_config = rgnn.HyperparameterConfig(
        domain=domain,
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
        normalize_updates=args.normalize_updates,
        global_readout=args.global_readout,
        residual_updates=args.residual_updates,
        binarize_updates=args.binarize_updates,
    )
    output_spec = _build_output_spec(hparam_config, readouts)
    module_config = rgnn.ModuleConfig(
        aggregation_function=_build_aggregation(args.aggregation),
        message_function=rgnn.PredicateMLPMessages(hparam_config, input_spec),
        update_function=rgnn.MLPUpdates(hparam_config),
    )
    model = rgnn.RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec).to(device)
    input_data = [(state, actions, goal)] * args.batch_size
    latency = rgnn.measure_forward_readout_latency(
        model,
        input_data,
        readouts,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        trials=args.trials,
        mode=args.mode,
        use_cuda_events=args.use_cuda_events,
    )

    print(json.dumps(latency.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
