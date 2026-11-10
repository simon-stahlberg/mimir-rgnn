"""Evaluate a trained plan-space policy on a folder of problems.

Loads a checkpoint written by ``train.py``, runs the search on every problem of
``--problems`` (default ``<domain-dir>/test``) with a round budget and a time
limit, writes ``results.json`` and prints a table with coverage, rounds, plan
length vs the reference, duplicates and cycling. ``--restarts`` makes several
sampled attempts per problem within the time limit.

Example:
    python scripts/plan_space/evaluate.py --checkpoint runs/blocks/best.pt --domain-dir data/blocks --max-rounds 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_table, load_domain, load_problems, reference_plans, summarize_outcomes  # noqa: E402

from pymimir_rgnn import PlanSpacePolicy, PlanSpaceSearch  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--domain-dir', type=Path, required=True)
    parser.add_argument('--problems', type=Path, default=None, help='folder of problems; default <domain-dir>/test')
    parser.add_argument('--output', type=Path, default=None, help='results file; default next to the checkpoint')
    parser.add_argument('--max-rounds', type=int, default=100)
    parser.add_argument('--time-limit', type=float, default=60.0, help='seconds per problem')
    parser.add_argument('--mode', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tombstone-capacity', type=int, default=None, help='default: the training setting')
    parser.add_argument('--tombstone-lifespan', type=int, default=None, help='default: the training setting')
    parser.add_argument('--no-reference', action='store_true', help='skip reference plans (no plan-length ratio)')
    parser.add_argument('--reference-timeout', type=float, default=60.0)
    parser.add_argument('--restarts', type=int, default=1,
                        help='independent attempts per problem within the time limit (seeds seed, seed+1, ...); solved if any attempt is')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    domain = load_domain(args.domain_dir)
    policy, extras = PlanSpacePolicy.load(domain, args.checkpoint, torch.device(args.device))
    training_args = extras.get('args', {})
    tombstone_capacity = training_args.get('tombstone_capacity', 0) if args.tombstone_capacity is None else args.tombstone_capacity
    tombstone_lifespan = training_args.get('tombstone_lifespan', 0) if args.tombstone_lifespan is None else args.tombstone_lifespan
    assert args.restarts >= 1, 'At least one attempt per problem is required.'

    problems_dir = args.domain_dir / 'test' if args.problems is None else args.problems
    problems = load_problems(domain, problems_dir)
    reference_lengths: dict[str, int] = {}
    if not args.no_reference:
        references = reference_plans(problems, args.domain_dir / 'references.json', timeout_seconds=args.reference_timeout)
        reference_lengths = {name: len(actions) for name, actions in references.items()}

    search = PlanSpaceSearch()
    outcomes = []
    for name, problem in problems:
        outcome = search.run_with_restarts(
            policy,
            problem,
            restarts=args.restarts,
            max_rounds=args.max_rounds,
            mode=args.mode,
            temperature=args.temperature,
            tombstone_capacity=tombstone_capacity,
            tombstone_lifespan=tombstone_lifespan,
            recurrent=policy.config.recurrent,
            seed=args.seed,
            time_limit=args.time_limit,
        )
        outcomes.append((name, outcome))
    summary = summarize_outcomes(outcomes, reference_lengths)

    rows = [
        [name, outcome.solved, outcome.attempts, outcome.rounds, outcome.plan_length, reference_lengths.get(name), outcome.steps_final,
         outcome.duplicate_creations, outcome.cycling_events, outcome.wall_time]
        for name, outcome in outcomes
    ]
    print(format_table(rows, ['problem', 'solved', 'attempts', 'rounds', 'length', 'reference', 'steps', 'duplicates', 'cycling', 'seconds']))
    print()
    print(format_table([[key, value] for key, value in summary.items()], ['metric', 'value']))

    output = args.checkpoint.with_name('results.json') if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        json.dump({
            'checkpoint': str(args.checkpoint),
            'problems': str(problems_dir),
            'settings': {
                'max_rounds': args.max_rounds, 'time_limit': args.time_limit, 'mode': args.mode,
                'temperature': args.temperature, 'tombstone_capacity': tombstone_capacity,
                'tombstone_lifespan': tombstone_lifespan, 'recurrent': policy.config.recurrent, 'seed': args.seed,
                'restarts': args.restarts,
            },
            'summary': summary,
            'problems_results': {
                name: outcome.to_dict() | {'reference_length': reference_lengths.get(name)} for name, outcome in outcomes
            },
        }, handle, indent=1)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
