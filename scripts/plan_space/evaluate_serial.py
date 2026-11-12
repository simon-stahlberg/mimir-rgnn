"""Evaluate a trained serial plan-space policy on a folder of problems.

Loads a checkpoint written by ``train_serial.py`` and runs the search on every
problem of ``--problems`` (default ``<domain-dir>/test``) with a round budget and
a time limit. Every solved plan is validated by applying its actions from the
initial state and checking the goal; a plan that fails this counts as unsolved
and is reported. Writes ``results.json`` with a summary and the per-problem
results, and prints the per-problem table, the summary and the coverage by
problem size (number of objects). ``--restarts`` makes several sampled attempts
per problem within the time limit.

Example:
    python scripts/plan_space/evaluate_serial.py --checkpoint runs/gripper_serial/best.pt --domain-dir data/gripper \
        --max-rounds 60 --time-limit 15
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_table, load_domain, load_problems  # noqa: E402

from pymimir_rgnn import SerialPolicy, SerialSearch, is_valid_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--domain-dir', type=Path, required=True)
    parser.add_argument('--problems', type=Path, default=None, help='folder of problems; default <domain-dir>/test')
    parser.add_argument('--output', type=Path, default=None, help='results file; default results.json next to the checkpoint')
    parser.add_argument('--max-rounds', type=int, default=100)
    parser.add_argument('--time-limit', type=float, default=60.0, help='seconds per problem')
    parser.add_argument('--mode', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--restarts', type=int, default=1,
                        help='independent attempts per problem within the time limit (seeds seed, seed+1, ...); solved if any attempt is')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Coverage, rounds, plan length and cycling over per-problem results."""
    solved = [result for result in results if result['solved']]
    mean = lambda values: statistics.fmean(values) if values else None  # noqa: E731
    return {
        'problems': len(results),
        'solved': len(solved),
        'coverage': len(solved) / max(len(results), 1),
        'invalid_plans': sum(1 for result in results if result['valid'] is False),
        'mean_rounds': mean([result['rounds'] for result in results]),
        'mean_rounds_solved': mean([result['rounds'] for result in solved]),
        'mean_plan_length': mean([result['plan_length'] for result in solved]),
        'mean_cycling_events': mean([result['cycling_events'] for result in results]),
        'mean_wall_time': mean([result['wall_time'] for result in results]),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    domain = load_domain(args.domain_dir)
    policy, _ = SerialPolicy.load(domain, args.checkpoint, torch.device(args.device))
    problems_dir = args.domain_dir / 'test' if args.problems is None else args.problems

    search = SerialSearch()
    results: dict[str, dict[str, Any]] = {}
    for name, problem in load_problems(domain, problems_dir):
        outcome = search.run_with_restarts(
            policy,
            problem,
            restarts=args.restarts,
            max_rounds=args.max_rounds,
            mode=args.mode,
            temperature=args.temperature,
            seed=args.seed,
            time_limit=args.time_limit,
        )
        valid = None if outcome.plan is None else is_valid_plan(problem, outcome.plan)
        if valid is False:
            print(f'warning: the plan found for {name} is not a valid plan; counting the problem as unsolved')
        results[name] = outcome.to_dict() | {'solved': valid is True, 'valid': valid, 'objects': len(problem.all_objects)}
    summary = summarize(list(results.values()))

    columns = ['objects', 'solved', 'attempts', 'rounds', 'plan_length', 'steps_final', 'cycling_events', 'wall_time']
    print(format_table([[name, *(result[column] for column in columns)] for name, result in results.items()], ['problem', *columns]))
    print()
    print(format_table([[key, value] for key, value in summary.items()], ['metric', 'value']))
    print()
    size_columns = ['problems', 'solved', 'coverage', 'mean_rounds_solved', 'mean_plan_length']
    by_size = {
        size: summarize([result for result in results.values() if result['objects'] == size])
        for size in sorted({result['objects'] for result in results.values()})
    }
    print(format_table([[size, *(group[column] for column in size_columns)] for size, group in by_size.items()], ['objects', *size_columns]))

    output = args.checkpoint.with_name('results.json') if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        json.dump({
            'checkpoint': str(args.checkpoint),
            'problems': str(problems_dir),
            'settings': {
                'max_rounds': args.max_rounds, 'time_limit': args.time_limit, 'mode': args.mode,
                'temperature': args.temperature, 'restarts': args.restarts, 'seed': args.seed,
            },
            'summary': summary,
            'coverage_by_objects': by_size,
            'problems_results': results,
        }, handle, indent=1)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
