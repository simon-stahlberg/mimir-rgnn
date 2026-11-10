"""Train a plan-space policy by hindsight expert iteration, without a planner.

Expects ``<domain-dir>/domain.pddl`` with ``train/`` and ``validation/`` folders
(see ``generate_instances.py``); no reference plans are needed or computed. Every
iteration rolls the current policy out on the training problems (sampled rollouts
plus one greedy rollout per problem), relabels each episode with the plan it
achieved (``pymimir_rgnn.hindsight_collect``: the final plan of a solved episode,
or the largest flaw-free sub-plan of an unsolved one together with the sub-goal it
achieves), labels the visited plans with the oracle against that plan, appends the
samples to a FIFO buffer and trains on the buffer. Validation runs the greedy
search on the validation problems; the plan-length ratio is reported only when
``<domain-dir>/references.json`` already holds a reference for a problem. One line
per iteration goes to ``<output>/metrics.jsonl``; ``best.pt`` tracks the best
validation (coverage, then ratio, then rounds) and ``latest.pt`` the last iteration.

Example:
    python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei --iterations 12
    python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei --init-from runs/blocks/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pymimir as mm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_table, is_better, load_domain, load_problems, summarize_outcomes  # noqa: E402

from pymimir_rgnn import (  # noqa: E402
    ImitationTrainer,
    PlanSpacePolicy,
    PlanSpaceSearch,
    PolicyConfig,
    ReferenceCache,
    Sample,
    TrainingConfig,
    hindsight_collect,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-from', type=Path, default=None,
                        help='warm start from a checkpoint; the model flags are then taken from the checkpoint')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--rollouts-per-problem', type=int, default=3, help='sampled rollouts per training problem and iteration')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature of the rollouts')
    parser.add_argument('--no-greedy-rollout', action='store_true', help='skip the additional greedy rollout per problem')
    parser.add_argument('--epochs-per-iteration', type=int, default=2)
    parser.add_argument('--buffer-cap', type=int, default=20000, help='maximum number of retained samples (oldest dropped first)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--embedding-size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--aggregation', choices=['mean', 'sum', 'max', 'smooth_max'], default='mean')
    parser.add_argument('--update', choices=['mlp', 'gru'], default='mlp')
    parser.add_argument('--recurrent', action='store_true', help='carry node embeddings across rounds (NodeStateStore)')
    parser.add_argument('--tombstone-capacity', type=int, default=0)
    parser.add_argument('--tombstone-lifespan', type=int, default=0)
    parser.add_argument('--max-rounds', type=int, default=50, help='round budget of rollouts and validation')
    parser.add_argument('--include-noop-targets', type=int, default=1, help='1: flaws whose oracle target is NoOp enter the loss')
    parser.add_argument('--delete-loss-weight', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--validation-time-limit', type=float, default=None, help='seconds per validation problem')
    return parser


def cached_reference_lengths(problems: list[tuple[str, mm.Problem]], cache_path: Path) -> dict[str, int]:
    """Reference lengths already in the cache; nothing is computed, so this never needs a planner."""
    if not cache_path.exists():
        return {}
    cache = ReferenceCache(cache_path)
    lengths: dict[str, int] = {}
    for name, problem in problems:
        actions = cache.get(name, problem)
        if actions is not None:
            lengths[name] = len(actions)
    return lengths


def validate(
    policy: PlanSpacePolicy,
    problems: list[tuple[str, mm.Problem]],
    reference_lengths: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    search = PlanSpaceSearch()
    outcomes = [
        (name, search.run(
            policy,
            problem,
            max_rounds=args.max_rounds,
            tombstone_capacity=args.tombstone_capacity,
            tombstone_lifespan=args.tombstone_lifespan,
            recurrent=policy.config.recurrent,
            time_limit=args.validation_time_limit,
        ))
        for name, problem in problems
    ]
    return summarize_outcomes(outcomes, reference_lengths)


def build_policy(domain: mm.Domain, args: argparse.Namespace, device: torch.device) -> PlanSpacePolicy:
    if args.init_from is not None:
        policy, _ = PlanSpacePolicy.load(domain, args.init_from, device)
        return policy
    config = PolicyConfig(
        embedding_size=args.embedding_size,
        num_layers=args.layers,
        aggregation=args.aggregation,
        update=args.update,
        recurrent=args.recurrent,
    )
    return PlanSpacePolicy(domain, config).to(device)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    domain = load_domain(args.domain_dir)
    train_problems = load_problems(domain, args.domain_dir / 'train')
    validation_problems = load_problems(domain, args.domain_dir / 'validation')
    validation_lengths = cached_reference_lengths(validation_problems, args.domain_dir / 'references.json')
    print(f'{len(train_problems)} training and {len(validation_problems)} validation problems; '
          f'cached reference lengths for {len(validation_lengths)} validation problems')

    policy = build_policy(domain, args, device)
    if args.init_from is not None:
        print(f'warm start from {args.init_from}: {policy.config.to_dict()}')
    trainer = ImitationTrainer(policy, TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        include_noop_targets=bool(args.include_noop_targets),
        delete_loss_weight=args.delete_loss_weight,
        recurrent=policy.config.recurrent,
    ), seed=args.seed)

    buffer: list[Sample] = []
    best_solutions: dict[int, mm.PartialOrderPlan] = {}
    best: dict[str, Any] | None = None
    trajectory_id = 0
    saved_args = vars(args) | {'domain_dir': str(args.domain_dir), 'output': str(args.output),
                               'init_from': None if args.init_from is None else str(args.init_from)}
    metrics_path = args.output / 'metrics.jsonl'
    with metrics_path.open('a') as metrics_file:
        for iteration in range(1, args.iterations + 1):
            new_samples, collect = hindsight_collect(
                policy,
                [problem for _, problem in train_problems],
                rollouts_per_problem=args.rollouts_per_problem,
                temperature=args.temperature,
                max_rounds=args.max_rounds,
                rng=rng,
                tombstone_capacity=args.tombstone_capacity,
                tombstone_lifespan=args.tombstone_lifespan,
                best_solutions=best_solutions,
                include_greedy=not args.no_greedy_rollout,
                trajectory_id_start=trajectory_id,
                recurrent=policy.config.recurrent,
            )
            trajectory_id += collect.episodes
            buffer.extend(new_samples)
            if len(buffer) > args.buffer_cap:
                buffer = buffer[-args.buffer_cap:]

            start = time.perf_counter()
            train_metrics = None
            for _ in range(args.epochs_per_iteration):
                if buffer:
                    train_metrics = trainer.train_epoch(buffer)
            train_time = time.perf_counter() - start
            validation = validate(policy, validation_problems, validation_lengths, args)

            record = {
                'iteration': iteration,
                'buffer': len(buffer),
                'best_solutions': len(best_solutions),
                'collect': collect.to_dict(),
                'train': None if train_metrics is None else train_metrics.to_dict() | {'epochs_wall_time': train_time},
                'validation': validation,
            }
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            solved_fraction = collect.solved_episodes / max(collect.episodes, 1)
            goal_fraction = collect.mean_goal_fraction if collect.mean_goal_fraction is not None else float('nan')
            ratio = validation['mean_length_ratio'] if validation['mean_length_ratio'] is not None else float('nan')
            train_text = 'no samples' if train_metrics is None else (
                f'loss {train_metrics.loss:.3f} resolver_acc {train_metrics.resolver_accuracy:.3f} '
                f'delete_acc {train_metrics.delete_accuracy:.3f} ({train_time:.1f}s)'
            )
            print(
                f'iteration {iteration:3d} | collect solved {solved_fraction:.2f} goal_fraction {goal_fraction:.2f} '
                f'skipped {collect.skipped_episodes} samples {collect.samples} rounds {collect.mean_rounds if collect.mean_rounds is not None else float("nan"):.1f} '
                f'({collect.wall_time:.1f}s) | buffer {len(buffer)} | {train_text} | '
                f'validation coverage {validation["coverage"]:.2f} rounds {validation["mean_rounds"]:.1f} length_ratio {ratio:.2f} '
                f'cycling {validation["total_cycling_events"]}',
                flush=True,
            )
            extras = {'iteration': iteration, 'args': saved_args, 'validation': validation, 'collect': collect.to_dict()}
            policy.save(args.output / 'latest.pt', extras)
            if is_better(validation, best):
                best = validation
                policy.save(args.output / 'best.pt', extras)

    if best is not None:
        print('best validation:')
        print(format_table([[key, value] for key, value in best.items()], ['metric', 'value']))


if __name__ == '__main__':
    main()
