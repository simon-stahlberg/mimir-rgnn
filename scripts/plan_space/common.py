"""Shared helpers of the plan-space scripts: loading data folders, reference plans and result summaries."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymimir as mm

from pymimir_rgnn import ReferenceCache, SearchOutcome


def load_domain(domain_dir: Path) -> mm.Domain:
    return mm.Domain.from_file(domain_dir / 'domain.pddl')


def problem_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob('*.pddl') if path.name != 'domain.pddl')


def load_problems(domain: mm.Domain, folder: Path, generator: str = 'grounded') -> list[tuple[str, mm.Problem]]:
    """Load every problem of ``folder`` as ``("<folder name>/<file name>", problem)``, grounded by default so the FF heuristic works.

    The name is the key of the reference cache, so it includes the split folder: the splits reuse file names.
    """
    return [
        (f'{folder.name}/{path.name}', mm.Problem.from_file(domain, path, generator=generator))  # type: ignore[arg-type]
        for path in problem_files(folder)
    ]


def reference_plans(
    problems: Sequence[tuple[str, mm.Problem]],
    cache_path: Path,
    *,
    max_expanded_states: int = 100_000,
    timeout_seconds: float | None = 60.0,
) -> dict[str, list[mm.GroundAction]]:
    """Reference plans of the problems from the JSON cache, computing and caching the missing ones."""
    cache = ReferenceCache(cache_path)
    references: dict[str, list[mm.GroundAction]] = {}
    for name, problem in problems:
        actions = cache.get_or_compute(name, problem, max_expanded_states=max_expanded_states, timeout_seconds=timeout_seconds)
        if actions is None:
            print(f'warning: no reference plan found for {name}; skipping it')
            continue
        references[name] = actions
    return references


def summarize_outcomes(
    outcomes: Sequence[tuple[str, SearchOutcome]],
    reference_lengths: dict[str, int],
) -> dict[str, Any]:
    """Aggregate coverage, rounds, plan length vs the reference, duplicates and cycling over problems."""
    solved = [outcome for _, outcome in outcomes if outcome.solved]
    ratios = [
        outcome.plan_length / reference_lengths[name]
        for name, outcome in outcomes
        if outcome.solved and outcome.plan_length is not None and reference_lengths.get(name)
    ]
    mean = lambda values: statistics.fmean(values) if values else None  # noqa: E731
    return {
        'problems': len(outcomes),
        'solved': len(solved),
        'coverage': len(solved) / max(len(outcomes), 1),
        'mean_rounds': mean([outcome.rounds for _, outcome in outcomes]),
        'mean_rounds_solved': mean([outcome.rounds for outcome in solved]),
        'mean_plan_length': mean([outcome.plan_length for outcome in solved if outcome.plan_length is not None]),
        'mean_reference_length': mean([reference_lengths[name] for name, _ in outcomes if name in reference_lengths]),
        'mean_length_ratio': mean(ratios),
        'mean_duplicate_creations': mean([outcome.duplicate_creations for _, outcome in outcomes]),
        'mean_cycling_events': mean([outcome.cycling_events for _, outcome in outcomes]),
        'total_duplicate_creations': sum(outcome.duplicate_creations for _, outcome in outcomes),
        'total_cycling_events': sum(outcome.cycling_events for _, outcome in outcomes),
        'mean_wall_time': mean([outcome.wall_time for _, outcome in outcomes]),
    }


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Whether a validation summary beats or ties the best one so far: coverage, then plan-length ratio, then rounds.

    Ties go to the candidate, so when validation cannot separate checkpoints (for example every
    epoch solves nothing on a hard validation set) the latest, most trained checkpoint is kept.
    """
    if best is None:
        return True
    key = lambda summary: (  # noqa: E731
        -summary['coverage'],
        summary['mean_length_ratio'] if summary['mean_length_ratio'] is not None else float('inf'),
        summary['mean_rounds'] if summary['mean_rounds'] is not None else float('inf'),
    )
    return key(candidate) <= key(best)


def format_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    """A fixed-width text table."""
    def cell(value: Any) -> str:
        if value is None:
            return '-'
        if isinstance(value, float):
            return f'{value:.2f}'
        return str(value)

    text = [[cell(value) for value in row] for row in rows]
    widths = [max(len(header[column]), *(len(row[column]) for row in text)) if text else len(header[column]) for column in range(len(header))]
    lines = [
        '  '.join(name.ljust(width) for name, width in zip(header, widths)),
        '  '.join('-' * width for width in widths),
    ]
    lines.extend('  '.join(value.ljust(width) for value, width in zip(row, widths)) for row in text)
    return '\n'.join(lines)
