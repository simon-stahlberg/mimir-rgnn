"""Train a plan-space policy by imitating the oracle on a folder of instances.

Expects ``<domain-dir>/domain.pddl`` with ``train/`` and ``validation/`` folders
(see ``generate_instances.py``). Reference plans are cached in
``<domain-dir>/references.json``. Every epoch validates by running the greedy
search on the validation problems and appends a line to ``<output>/metrics.jsonl``;
``best.pt`` tracks the best validation (coverage, then plan-length ratio, then
rounds) and ``latest.pt`` the last epoch.

Example:
    python scripts/plan_space/train.py --domain-dir data/blocks --output runs/blocks --epochs 20
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

from common import format_table, is_better, load_domain, load_problems, reference_plans, summarize_outcomes  # noqa: E402

from pymimir_rgnn import (  # noqa: E402
    ImitationTrainer,
    PlanSpacePolicy,
    PlanSpaceSearch,
    PolicyConfig,
    Sample,
    TrainingConfig,
    oracle_rollout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--embedding-size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--aggregation', choices=['mean', 'sum', 'max', 'smooth_max'], default='mean')
    parser.add_argument('--update', choices=['mlp', 'gru'], default='mlp')
    parser.add_argument('--recurrent', action='store_true', help='carry node embeddings across rounds (NodeStateStore)')
    parser.add_argument('--tombstone-capacity', type=int, default=0)
    parser.add_argument('--tombstone-lifespan', type=int, default=0)
    parser.add_argument('--follow-fraction', type=float, default=0.5,
                        help='fraction of non-NoOp oracle choices followed per round in the diversifying rollouts')
    parser.add_argument('--rollouts-per-problem', type=int, default=4,
                        help='oracle rollouts per training problem; the first follows the oracle completely')
    parser.add_argument('--dagger', action='store_true', help='after each epoch, label the plans visited by the greedy policy')
    parser.add_argument('--dataset-cap', type=int, default=20000, help='maximum number of DAgger samples kept')
    parser.add_argument('--max-rounds', type=int, default=50, help='round budget of rollouts, DAgger and validation')
    parser.add_argument('--include-noop-targets', type=int, default=1, help='1: flaws whose oracle target is NoOp enter the loss')
    parser.add_argument('--delete-loss-weight', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--validation-time-limit', type=float, default=None, help='seconds per validation problem')
    parser.add_argument('--reference-timeout', type=float, default=60.0, help='seconds per reference-plan search')
    return parser


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
            recurrent=args.recurrent,
            time_limit=args.validation_time_limit,
        ))
        for name, problem in problems
    ]
    return summarize_outcomes(outcomes, reference_lengths)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    domain = load_domain(args.domain_dir)
    train_problems = load_problems(domain, args.domain_dir / 'train')
    validation_problems = load_problems(domain, args.domain_dir / 'validation')
    cache_path = args.domain_dir / 'references.json'
    start = time.perf_counter()
    train_references = reference_plans(train_problems, cache_path, timeout_seconds=args.reference_timeout)
    validation_references = reference_plans(validation_problems, cache_path, timeout_seconds=args.reference_timeout)
    print(f'reference plans for {len(train_references)} training and {len(validation_references)} validation problems '
          f'({time.perf_counter() - start:.1f}s)')
    train_problems = [(name, problem) for name, problem in train_problems if name in train_references]
    references = {
        name: mm.PartialOrderPlan.from_sequential_plan(problem, train_references[name])
        for name, problem in train_problems
    }
    validation_lengths = {name: len(actions) for name, actions in validation_references.items()}

    start = time.perf_counter()
    oracle_samples: list[Sample] = []
    trajectory_id = 0
    for name, problem in train_problems:
        for rollout in range(args.rollouts_per_problem):
            oracle_samples.extend(oracle_rollout(
                problem,
                references[name],
                follow_fraction=1.0 if rollout == 0 else args.follow_fraction,
                rng=rng,
                max_rounds=args.max_rounds,
                tombstone_capacity=args.tombstone_capacity,
                tombstone_lifespan=args.tombstone_lifespan,
                trajectory_id=trajectory_id,
            ))
            trajectory_id += 1
    print(f'{len(oracle_samples)} oracle samples from {trajectory_id} rollouts ({time.perf_counter() - start:.1f}s)')

    config = PolicyConfig(
        embedding_size=args.embedding_size,
        num_layers=args.layers,
        aggregation=args.aggregation,
        update=args.update,
        recurrent=args.recurrent,
    )
    policy = PlanSpacePolicy(domain, config).to(device)
    trainer = ImitationTrainer(policy, TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        include_noop_targets=bool(args.include_noop_targets),
        delete_loss_weight=args.delete_loss_weight,
        recurrent=args.recurrent,
    ), seed=args.seed)

    dagger_samples: list[Sample] = []
    best: dict[str, Any] | None = None
    metrics_path = args.output / 'metrics.jsonl'
    with metrics_path.open('a') as metrics_file:
        for epoch in range(1, args.epochs + 1):
            samples = oracle_samples + dagger_samples
            train_metrics = trainer.train_epoch(samples)
            validation = validate(policy, validation_problems, validation_lengths, args)
            record = {
                'epoch': epoch,
                'samples': len(samples),
                'dagger_samples': len(dagger_samples),
                'train': train_metrics.to_dict(),
                'validation': validation,
            }
            metrics_file.write(json.dumps(record) + '\n')
            metrics_file.flush()
            print(
                f'epoch {epoch:3d} | loss {train_metrics.loss:.3f} resolver_acc {train_metrics.resolver_accuracy:.3f} '
                f'delete_acc {train_metrics.delete_accuracy:.3f} ({train_metrics.wall_time:.1f}s, {len(samples)} samples) | '
                f'validation coverage {validation["coverage"]:.2f} rounds {validation["mean_rounds"]:.1f} '
                f'length_ratio {validation["mean_length_ratio"] if validation["mean_length_ratio"] is not None else float("nan"):.2f} '
                f'duplicates {validation["total_duplicate_creations"]} cycling {validation["total_cycling_events"]}'
            )
            extras = {'epoch': epoch, 'args': vars(args) | {'domain_dir': str(args.domain_dir), 'output': str(args.output)}, 'validation': validation}
            policy.save(args.output / 'latest.pt', extras)
            if is_better(validation, best):
                best = validation
                policy.save(args.output / 'best.pt', extras)

            if args.dagger and epoch < args.epochs:
                start = time.perf_counter()
                new_samples = trainer.dagger_collect(
                    [(problem, references[name]) for name, problem in train_problems],
                    max_rounds=args.max_rounds,
                    tombstone_capacity=args.tombstone_capacity,
                    tombstone_lifespan=args.tombstone_lifespan,
                    trajectory_id_start=trajectory_id,
                )
                trajectory_id += len(train_problems)
                dagger_samples.extend(new_samples)
                if len(dagger_samples) > args.dataset_cap:
                    dagger_samples = dagger_samples[-args.dataset_cap:]
                print(f'  dagger: {len(new_samples)} new samples, {len(dagger_samples)} kept ({time.perf_counter() - start:.1f}s)')

    if best is not None:
        print('best validation:')
        print(format_table([[key, value] for key, value in best.items()], ['metric', 'value']))


if __name__ == '__main__':
    main()
