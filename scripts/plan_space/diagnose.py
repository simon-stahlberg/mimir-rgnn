"""Per-head agreement of a trained plan-space policy with the oracle.

For every problem folder given, the policy is scored on two kinds of partial
plans: the plans it visits itself during greedy search (``policy`` rows) and the
plans of a pure oracle rollout (``oracle`` rows). For each, the script reports
the resolver head's accuracy against the oracle target (overall, on flaws whose
target is not NoOp, and per flaw kind) and the delete head's precision, recall
and rate, plus the mean delete logit on steps the oracle keeps. Comparing the
rows of an in-distribution folder with those of a larger-instance folder tells
whether the heads stop transferring (accuracy drops) or whether errors merely
compound through the search (accuracy holds while coverage drops).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pymimir as mm  # noqa: E402

from common import format_table, load_domain, load_problems, reference_plans  # noqa: E402

from pymimir_rgnn import PlanSpacePolicy, PlanSpaceSearch, oracle_rollout  # noqa: E402

KIND_NAMES = {mm.FlawKind.OPEN_CONDITION: 'open', mm.FlawKind.THREAT: 'threat', mm.FlawKind.UNBOUND_PARAMETER: 'bind'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--domain-dir', type=Path, required=True, help='folder with domain.pddl (references are cached per problem folder parent)')
    parser.add_argument('--problems', type=Path, nargs='+', required=True, help='one or more folders of problems, e.g. validation and a larger test set')
    parser.add_argument('--max-rounds', type=int, default=100)
    parser.add_argument('--time-limit', type=float, default=30.0)
    parser.add_argument('--output', type=Path, default=None, help='JSON file for the per-folder statistics')
    parser.add_argument('--reference-timeout', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


class HeadStats:
    """Accumulates agreement counts for the resolver and delete heads."""

    def __init__(self) -> None:
        self.resolver_total = 0
        self.resolver_correct = 0
        self.resolver_active_total = 0
        self.resolver_active_correct = 0
        self.by_kind_total: dict[str, int] = defaultdict(int)
        self.by_kind_correct: dict[str, int] = defaultdict(int)
        self.predicted_noop_when_active = 0
        self.delete_tp = 0
        self.delete_fp = 0
        self.delete_fn = 0
        self.delete_tn = 0
        self.keep_logits: list[float] = []
        self.plans = 0

    def add(self, flaws: mm.FlawSet, targets: mm.OracleTargets, resolver_log_probs: list[torch.Tensor], delete_logits: torch.Tensor, step_ids: list[int]) -> None:
        self.plans += 1
        for flaw, target, log_probs in zip(flaws.flaws, targets.resolver_index_per_flaw, resolver_log_probs):
            predicted = int(log_probs.argmax())
            kind = KIND_NAMES[flaw.kind]
            self.resolver_total += 1
            self.by_kind_total[kind] += 1
            if predicted == target:
                self.resolver_correct += 1
                self.by_kind_correct[kind] += 1
            if target != 0:
                self.resolver_active_total += 1
                self.resolver_active_correct += int(predicted == target)
                self.predicted_noop_when_active += int(predicted == 0)
        to_delete = set(targets.step_ids_to_delete)
        for step_id, logit in zip(step_ids, delete_logits.tolist()):
            predicted = logit > 0.0
            actual = step_id in to_delete
            if predicted and actual:
                self.delete_tp += 1
            elif predicted and not actual:
                self.delete_fp += 1
            elif actual:
                self.delete_fn += 1
            else:
                self.delete_tn += 1
            if not actual:
                self.keep_logits.append(logit)

    def summary(self) -> dict[str, Any]:
        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        steps = self.delete_tp + self.delete_fp + self.delete_fn + self.delete_tn
        return {
            'plans': self.plans,
            'flaws': self.resolver_total,
            'resolver_acc': ratio(self.resolver_correct, self.resolver_total),
            'resolver_acc_active': ratio(self.resolver_active_correct, self.resolver_active_total),
            'noop_when_active': ratio(self.predicted_noop_when_active, self.resolver_active_total),
            **{f'acc_{kind}': ratio(self.by_kind_correct[kind], self.by_kind_total[kind]) for kind in ('open', 'threat', 'bind')},
            'steps': steps,
            'delete_rate_policy': ratio(self.delete_tp + self.delete_fp, steps),
            'delete_rate_oracle': ratio(self.delete_tp + self.delete_fn, steps),
            'delete_precision': ratio(self.delete_tp, self.delete_tp + self.delete_fp),
            'delete_recall': ratio(self.delete_tp, self.delete_tp + self.delete_fn),
            'mean_keep_logit': statistics.fmean(self.keep_logits) if self.keep_logits else None,
        }


def step_ids_of(plan: mm.PartialOrderPlan) -> list[int]:
    return [step.id for step in plan.steps if step.kind == mm.StepKind.STEP]


def score_plans(policy: PlanSpacePolicy, problem: mm.Problem, reference: mm.PartialOrderPlan, plans: list[mm.PartialOrderPlan], stats: HeadStats) -> None:
    for plan in plans:
        flaws = plan.compute_flaws()
        if not flaws.flaws:
            continue
        targets = mm.oracle_targets(plan, flaws, reference)
        with torch.no_grad():
            output = policy.evaluate([(problem.initial_state, problem.goal, plan, flaws)])[0]
        stats.add(flaws, targets, output.resolver_log_probs, output.delete_logits, step_ids_of(plan))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    domain = load_domain(args.domain_dir)
    policy, extras = PlanSpacePolicy.load(domain, args.checkpoint, device=args.device)
    tombstone_capacity = int(extras.get('tombstone_capacity', 0))
    tombstone_lifespan = int(extras.get('tombstone_lifespan', 0))
    search = PlanSpaceSearch()
    results: dict[str, Any] = {}
    rows = []
    for folder in args.problems:
        problems = load_problems(domain, folder)
        references = reference_plans(problems, folder.parent / 'references.json', timeout_seconds=args.reference_timeout)
        policy_stats = HeadStats()
        oracle_stats = HeadStats()
        solved = 0
        for name, problem in problems:
            if name not in references:
                continue
            reference = mm.PartialOrderPlan.from_sequential_plan(problem, references[name])
            outcome = search.run(
                policy, problem, max_rounds=args.max_rounds, mode='greedy', tombstone_capacity=tombstone_capacity,
                tombstone_lifespan=tombstone_lifespan, recurrent=policy.config.recurrent, seed=args.seed,
                time_limit=args.time_limit, record_visited=True,
            )
            solved += int(outcome.solved)
            score_plans(policy, problem, reference, outcome.visited, policy_stats)
            samples = oracle_rollout(problem, reference, follow_fraction=1.0, rng=random.Random(args.seed), max_rounds=args.max_rounds,
                                     tombstone_capacity=tombstone_capacity, tombstone_lifespan=tombstone_lifespan)
            score_plans(policy, problem, reference, [sample.plan_snapshot for sample in samples], oracle_stats)
        key = str(folder)
        results[key] = {'coverage': solved / max(len(problems), 1), 'policy_plans': policy_stats.summary(), 'oracle_plans': oracle_stats.summary()}
        for source, summary in (('policy', policy_stats.summary()), ('oracle', oracle_stats.summary())):
            rows.append([folder.parent.name + '/' + folder.name, source, results[key]['coverage'], summary['plans'], summary['resolver_acc'], summary['resolver_acc_active'],
                         summary['noop_when_active'], summary['acc_open'], summary['acc_threat'], summary['acc_bind'],
                         summary['delete_rate_policy'], summary['delete_rate_oracle'], summary['delete_precision'], summary['delete_recall'], summary['mean_keep_logit']])
    print(format_table(rows, ['problems', 'plans from', 'coverage', 'plans', 'res acc', 'res acc active', 'noop|active', 'acc open', 'acc threat', 'acc bind',
                              'del rate pol', 'del rate orc', 'del prec', 'del recall', 'keep logit']))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
