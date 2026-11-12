"""Shared helpers of the plan-space scripts: loading data folders and printing tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymimir as mm


def load_domain(domain_dir: Path) -> mm.Domain:
    return mm.Domain.from_file(domain_dir / 'domain.pddl')


def problem_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob('*.pddl') if path.name != 'domain.pddl')


def load_problems(domain: mm.Domain, folder: Path, generator: str = 'grounded') -> list[tuple[str, mm.Problem]]:
    """Load every problem of ``folder`` as ``("<folder name>/<file name>", problem)``."""
    return [
        (f'{folder.name}/{path.name}', mm.Problem.from_file(domain, path, generator=generator))  # type: ignore[arg-type]
        for path in problem_files(folder)
    ]


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
