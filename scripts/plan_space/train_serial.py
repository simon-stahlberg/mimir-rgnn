"""Train a serial plan-space policy by hindsight expert iteration, bootstrapped from the edit space.

Expects ``<domain-dir>/domain.pddl`` with ``train/`` and ``validation/`` folders
(see ``generate_instances.py``). No planner is involved: the references of the
bootstrap come from ``pymimir.serial_search``, a search over single repairs of the
serial plan, with ``--bootstrap-timeout`` seconds per training problem (0 disables
the bootstrap).

Every iteration collects samples on the training problems
(``pymimir_rgnn.serial_collect``), appends them to a FIFO buffer and trains on the
buffer. A problem with a reference is in bootstrap mode until the policy has
solved it ``--bootstrap-phase-out`` times: it gets ``--bootstrap-oracle-rollouts``
teacher-forced oracle rollouts and the policy's own episodes (one greedy and
``--rollouts-per-problem`` sampled ones) are labelled against the reference when unsolved and against their own final plan when solved.
Every other problem takes the hindsight path: a solved episode is labelled
against its own final plan, an unsolved one against the executable part of its
final plan and the sub-goal that part achieves. ``--bootstrap-iterations``
prepends warm-up iterations in which only the oracle rollouts run.

Validation runs the greedy search on the validation problems. One line per
iteration goes to the console and one record to ``<output>/metrics.jsonl``;
``latest.pt`` is the last iteration and ``best.pt`` the one with the best
validation coverage (ties go to the latest).

Example:
    python scripts/plan_space/train_serial.py --domain-dir data/gripper --output runs/gripper_serial \
        --bootstrap-iterations 2 --iterations 15 --layers 8 --update gru
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args

import pymimir as mm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_domain, load_problems  # noqa: E402

from pymimir_rgnn import (  # noqa: E402
    Pooling,
    SerialPolicy,
    SerialPolicyConfig,
    SerialSearch,
    SerialTrainer,
    serial_collect,
    serial_references,
)


def parse_contexts(value: str) -> tuple[str, ...]:
    """Comma-separated context names; ``none`` (or an empty string) is no context."""
    names = tuple(name.strip() for name in value.split(',') if name.strip())
    return () if names == ('none',) else names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-from', type=Path, default=None,
                        help='warm start from a checkpoint; the model flags are then taken from the checkpoint')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--bootstrap-iterations', type=int, default=0,
                        help='warm-up iterations of pure oracle-rollout collection before the mixed iterations')
    parser.add_argument('--bootstrap-timeout', type=float, default=10.0,
                        help='seconds per reference search of the bootstrap; 0 disables the bootstrap')
    parser.add_argument('--bootstrap-phase-out', type=int, default=2,
                        help='a problem leaves bootstrap mode once the policy has solved it this many times during collection')
    parser.add_argument('--bootstrap-oracle-rollouts', type=int, default=1, help='oracle rollouts per bootstrap problem and iteration')
    parser.add_argument('--follow-fraction', type=float, default=0.5,
                        help='fraction of non-NoOp oracle choices followed per round in the oracle rollouts')
    parser.add_argument('--problems-per-iteration', type=int, default=None,
                        help='collect on a random subset of this many training problems per iteration (default: all)')
    parser.add_argument('--rollouts-per-problem', type=int, default=3,
                        help='sampled rollouts per training problem and iteration, next to one greedy rollout')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature of the rollouts')
    parser.add_argument('--epochs-per-iteration', type=int, default=2)
    parser.add_argument('--buffer-cap', type=int, default=20000, help='maximum number of retained samples (oldest dropped first)')
    parser.add_argument('--max-rounds', type=int, default=50, help='round budget of rollouts and validation')
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--embedding-size', type=int, default=32)
    parser.add_argument('--aggregation', choices=['mean', 'sum', 'max', 'smooth_max'], default='mean')
    parser.add_argument('--update', choices=['mlp', 'gru'], default='mlp')
    parser.add_argument('--repair-contexts', type=parse_contexts, default=('objects', 'steps', 'errors', 'siblings'),
                        help='comma-separated pooled contexts of the repair head: objects, steps, errors, siblings; none for no '
                             'context (default: objects,steps,errors,siblings)')
    parser.add_argument('--delete-contexts', type=parse_contexts, default=('objects', 'steps', 'errors'),
                        help='comma-separated pooled contexts of the delete head: objects, steps, errors; none for no context '
                             '(default: objects,steps,errors)')
    parser.add_argument('--context-pooling', choices=get_args(Pooling), default='maxmean', help='pooling of the head contexts')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--delete-loss-weight', type=float, default=1.0)
    parser.add_argument('--validation-time-limit', type=float, default=None, help='seconds per validation problem')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser


def build_policy(domain: mm.Domain, args: argparse.Namespace, parser: argparse.ArgumentParser) -> SerialPolicy:
    device = torch.device(args.device)
    if args.init_from is not None:
        policy, _ = SerialPolicy.load(domain, args.init_from, device)
        print(f'warm start from {args.init_from}: {asdict(policy.config)}')
        return policy
    try:
        config = SerialPolicyConfig(
            embedding_size=args.embedding_size,
            num_layers=args.layers,
            aggregation=args.aggregation,
            update=args.update,
            repair_contexts=args.repair_contexts,
            delete_contexts=args.delete_contexts,
            context_pooling=args.context_pooling,
        )
    except ValueError as error:
        parser.error(str(error))
    return SerialPolicy(domain, config).to(device)


def validate(policy: SerialPolicy, problems: list[mm.Problem], args: argparse.Namespace) -> dict[str, Any]:
    """Coverage and rounds of the greedy search on the validation problems."""
    search = SerialSearch()
    outcomes = [search.run(policy, problem, max_rounds=args.max_rounds, time_limit=args.validation_time_limit) for problem in problems]
    solved = [outcome for outcome in outcomes if outcome.solved]
    return {
        'problems': len(outcomes),
        'solved': len(solved),
        'coverage': len(solved) / max(len(outcomes), 1),
        'mean_rounds': sum(outcome.rounds for outcome in outcomes) / max(len(outcomes), 1),
        'mean_rounds_solved': sum(outcome.rounds for outcome in solved) / len(solved) if solved else None,
        # Every step of a solution is one action of the plan.
        'mean_plan_length': sum(outcome.steps_final for outcome in solved) / len(solved) if solved else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    domain = load_domain(args.domain_dir)
    train_problems = [problem for _, problem in load_problems(domain, args.domain_dir / 'train')]
    validation_problems = [problem for _, problem in load_problems(domain, args.domain_dir / 'validation')]
    print(f'{len(train_problems)} training and {len(validation_problems)} validation problems')

    policy = build_policy(domain, args, parser)
    trainer = SerialTrainer(
        policy,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        delete_loss_weight=args.delete_loss_weight,
        buffer_cap=args.buffer_cap,
        seed=args.seed,
    )

    start = time.perf_counter()
    references = serial_references(train_problems, args.bootstrap_timeout)
    print(f'bootstrap: references for {len(references)}/{len(train_problems)} training problems within '
          f'{args.bootstrap_timeout:g}s each ({time.perf_counter() - start:.1f}s); phase-out after '
          f'{args.bootstrap_phase_out} solved episodes', flush=True)
    solved_counts: dict[mm.Problem, int] = {}
    best_solutions: dict[mm.Problem, mm.SerialPlan] = {}

    saved_args = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    best_coverage = -1.0
    with (args.output / 'metrics.jsonl').open('a') as metrics_file:
        for iteration in range(1, args.bootstrap_iterations + args.iterations + 1):
            iteration_start = time.perf_counter()
            warmup = iteration <= args.bootstrap_iterations
            problems = train_problems
            if args.problems_per_iteration is not None and args.problems_per_iteration < len(problems):
                problems = rng.sample(problems, args.problems_per_iteration)
            samples, collect = serial_collect(
                policy,
                problems,
                references=references,
                solved_counts=solved_counts,
                best_solutions=best_solutions,
                phase_out=args.bootstrap_phase_out,
                oracle_rollouts=args.bootstrap_oracle_rollouts,
                follow_fraction=args.follow_fraction,
                rollouts_per_problem=args.rollouts_per_problem,
                temperature=args.temperature,
                max_rounds=args.max_rounds,
                rng=rng,
                policy_episodes=not warmup,
            )
            trainer.add(samples)
            train = None
            if trainer.buffer:
                for _ in range(args.epochs_per_iteration):
                    train = trainer.train_epoch()
            validation = validate(policy, validation_problems, args)
            wall_time = time.perf_counter() - iteration_start

            solved_fraction = collect.solved_episodes / max(collect.episodes, 1)
            record = {
                'iteration': iteration,
                'warmup': warmup,
                'problems': len(problems),
                'references': len(references),
                'buffer': len(trainer.buffer),
                'best_solutions': len(best_solutions),
                'collect': asdict(collect) | {'solved_fraction': solved_fraction},
                'train': None if train is None else asdict(train),
                'validation': validation,
                'wall_time': wall_time,
            }
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            goal_fraction = float('nan') if collect.mean_goal_fraction is None else collect.mean_goal_fraction
            mean_rounds = float('nan') if collect.mean_rounds is None else collect.mean_rounds
            train_text = 'no samples' if train is None else (
                f'loss {train.loss:.3f} repair_acc {train.repair_accuracy:.3f} delete_acc {train.delete_accuracy:.3f}'
            )
            print(
                f'iteration {iteration:3d}{" (warm-up)" if warmup else ""} | '
                f'collect episodes {collect.episodes} solved {solved_fraction:.2f} goal_fraction {goal_fraction:.2f} '
                f'rounds {mean_rounds:.1f} skipped {collect.skipped_episodes} | '
                f'bootstrap remaining {collect.bootstrap_remaining}/{len(references)} '
                f'samples {collect.bootstrap_samples}+{collect.hindsight_samples} | buffer {len(trainer.buffer)} | {train_text} | '
                f'validation coverage {validation["coverage"]:.2f} rounds {validation["mean_rounds"]:.1f} | {wall_time:.1f}s',
                flush=True,
            )
            extras = {'iteration': iteration, 'args': saved_args, 'validation': validation, 'collect': record['collect']}
            policy.save(args.output / 'latest.pt', extras)
            if validation['coverage'] >= best_coverage:
                best_coverage = validation['coverage']
                policy.save(args.output / 'best.pt', extras)
    print(f'best validation coverage {best_coverage:.2f}')


if __name__ == '__main__':
    main()
