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

From scratch, random rollouts rarely complete a causal chain and the relabeled
goals stay trivial. ``--bootstrap`` accepts planner dependence for the start:
reference plans of the training problems are computed with a short budget
(``--bootstrap-timeout``; cached in ``<domain-dir>/references.json``) and the
problems the planner solved in time, whose reference the oracle can reproduce,
are collected in bootstrap mode (oracle rollouts against the reference plus
DAgger-style labelling of the policy's own episodes,
``pymimir_rgnn.bootstrap_collect``) until the policy has solved them
``--bootstrap-phase-out`` times, after which they take the hindsight path.
``--bootstrap-iterations`` prepends pure oracle-rollout warm-up iterations.

Example:
    python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei --iterations 12
    python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei --init-from runs/blocks/best.pt
    python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei --bootstrap --bootstrap-iterations 2
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
    bootstrap_collect,
    hindsight_collect,
    oracle_reaches_reference,
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
    parser.add_argument('--problems-per-iteration', type=int, default=None,
                        help='collect on a random subset of this many training problems per iteration (default: all)')
    parser.add_argument('--bootstrap', action='store_true',
                        help='bootstrap from planner references of the training problems the planner solves in time')
    parser.add_argument('--bootstrap-timeout', type=float, default=10.0, help='seconds per reference-plan search of the bootstrap')
    parser.add_argument('--bootstrap-phase-out', type=int, default=2,
                        help='a problem leaves bootstrap mode once the policy has solved it this many times during collection')
    parser.add_argument('--bootstrap-oracle-rollouts', type=int, default=1, help='oracle rollouts per bootstrap problem and iteration')
    parser.add_argument('--follow-fraction', type=float, default=0.5,
                        help='fraction of non-NoOp oracle choices followed per round in the bootstrap oracle rollouts')
    parser.add_argument('--bootstrap-iterations', type=int, default=0,
                        help='warm-up iterations of pure oracle-rollout collection before the mixed iterations')
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


def bootstrap_references(
    problems: list[tuple[str, mm.Problem]],
    cache_path: Path,
    timeout_seconds: float,
    max_rounds: int,
) -> tuple[dict[int, mm.PartialOrderPlan], int]:
    """Planner references of the problems solved within ``timeout_seconds`` (or already cached), keyed by ``id(problem)``.

    The budget applies to each of the two searches of ``compute_reference_plan`` (BFS, then GBFS with FF); a cached
    reference computed with a longer budget by another script is used as well. References the oracle cannot
    reproduce within ``max_rounds`` (``oracle_reaches_reference``) are left out; their number is returned as well.
    """
    cache = ReferenceCache(cache_path)
    references: dict[int, mm.PartialOrderPlan] = {}
    unusable = 0
    for name, problem in problems:
        actions = cache.get_or_compute(name, problem, timeout_seconds=timeout_seconds)
        if actions is None:
            continue
        reference = mm.PartialOrderPlan.from_sequential_plan(problem, actions)
        if not oracle_reaches_reference(problem, reference, max_rounds=max_rounds):
            unusable += 1
            continue
        references[id(problem)] = reference
    return references, unusable


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

    references: dict[int, mm.PartialOrderPlan] = {}
    solved_counts: dict[int, int] = {}
    if args.bootstrap:
        start = time.perf_counter()
        references, unusable = bootstrap_references(
            train_problems, args.domain_dir / 'references.json', args.bootstrap_timeout, args.max_rounds,
        )
        print(f'bootstrap: references for {len(references)}/{len(train_problems)} training problems '
              f'within {args.bootstrap_timeout:g}s each ({time.perf_counter() - start:.1f}s; {unusable} more found but not '
              f'reproducible by the oracle within {args.max_rounds} rounds); phase-out after {args.bootstrap_phase_out} solved episodes',
              flush=True)
    warmup_iterations = args.bootstrap_iterations if args.bootstrap else 0

    buffer: list[Sample] = []
    best_solutions: dict[int, mm.PartialOrderPlan] = {}
    best: dict[str, Any] | None = None
    trajectory_id = 0
    saved_args = vars(args) | {'domain_dir': str(args.domain_dir), 'output': str(args.output),
                               'init_from': None if args.init_from is None else str(args.init_from)}
    metrics_path = args.output / 'metrics.jsonl'
    with metrics_path.open('a') as metrics_file:
        for iteration in range(1, warmup_iterations + args.iterations + 1):
            warmup = iteration <= warmup_iterations
            problems = [problem for _, problem in train_problems]
            if args.problems_per_iteration is not None and args.problems_per_iteration < len(problems):
                problems = rng.sample(problems, args.problems_per_iteration)
            bootstrap = None
            if args.bootstrap:
                new_samples, collect, bootstrap = bootstrap_collect(
                    policy,
                    problems,
                    references=references,
                    solved_counts=solved_counts,
                    phase_out=args.bootstrap_phase_out,
                    rollouts_per_problem=args.rollouts_per_problem,
                    temperature=args.temperature,
                    max_rounds=args.max_rounds,
                    rng=rng,
                    oracle_rollouts=args.bootstrap_oracle_rollouts,
                    follow_fraction=args.follow_fraction,
                    policy_episodes=not warmup,
                    tombstone_capacity=args.tombstone_capacity,
                    tombstone_lifespan=args.tombstone_lifespan,
                    best_solutions=best_solutions,
                    include_greedy=not args.no_greedy_rollout,
                    trajectory_id_start=trajectory_id,
                    recurrent=policy.config.recurrent,
                )
                trajectory_id += bootstrap.oracle_episodes
            else:
                new_samples, collect = hindsight_collect(
                    policy,
                    problems,
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
                'warmup': warmup,
                'problems': len(problems),
                'buffer': len(buffer),
                'best_solutions': len(best_solutions),
                'collect': collect.to_dict(),
                'bootstrap': None if bootstrap is None else bootstrap.to_dict() | {'references': len(references)},
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
            bootstrap_text = '' if bootstrap is None else (
                f'bootstrap {bootstrap.bootstrap_problems}/{len(problems)} problems (remaining {bootstrap.remaining}/{len(references)}) '
                f'oracle_episodes {bootstrap.oracle_episodes} samples {bootstrap.bootstrap_samples}+{bootstrap.hindsight_samples} | '
            )
            print(
                f'iteration {iteration:3d}{" (warm-up)" if warmup else ""} | {bootstrap_text}'
                f'collect solved {solved_fraction:.2f} goal_fraction {goal_fraction:.2f} '
                f'skipped {collect.skipped_episodes} samples {collect.samples} rounds {collect.mean_rounds if collect.mean_rounds is not None else float("nan"):.1f} '
                f'({collect.wall_time:.1f}s) | buffer {len(buffer)} | {train_text} | '
                f'validation coverage {validation["coverage"]:.2f} rounds {validation["mean_rounds"]:.1f} length_ratio {ratio:.2f} '
                f'cycling {validation["total_cycling_events"]}',
                flush=True,
            )
            extras = {'iteration': iteration, 'args': saved_args, 'validation': validation, 'collect': collect.to_dict(),
                      'bootstrap': record['bootstrap']}
            policy.save(args.output / 'latest.pt', extras)
            if is_better(validation, best):
                best = validation
                policy.save(args.output / 'best.pt', extras)

    if best is not None:
        print('best validation:')
        print(format_table([[key, value] for key, value in best.items()], ['metric', 'value']))


if __name__ == '__main__':
    main()
