"""Generate random, solvable Blocksworld or Gripper instances for the plan-space experiments.

Writes ``<output>/domain.pddl`` (the fixture domain of ``tests/data``) and the
folders ``train/``, ``validation/`` and ``test/`` with ``pNNN.pddl`` files.
Instances are solvable by construction: Blocksworld goals are reachable tower
configurations and Gripper goals are room assignments.

Example:
    python scripts/plan_space/generate_instances.py --domain blocks --output data/blocks --min-size 3 --max-size 5 \
        --train 20 --validation 5 --test 5 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / 'tests' / 'data'


def _towers(blocks: list[str], rng: random.Random) -> list[list[str]]:
    """Partition the blocks into random towers, bottom first."""
    shuffled = list(blocks)
    rng.shuffle(shuffled)
    towers: list[list[str]] = []
    for block in shuffled:
        if towers and rng.random() < 0.6:
            rng.choice(towers).append(block)
        else:
            towers.append([block])
    return towers


def _tower_atoms(towers: list[list[str]]) -> list[str]:
    atoms: list[str] = []
    for tower in towers:
        atoms.append(f'(on-table {tower[0]})')
        for below, above in zip(tower, tower[1:]):
            atoms.append(f'(on {above} {below})')
        atoms.append(f'(clear {tower[-1]})')
    return atoms


def blocks_instance(name: str, size: int, rng: random.Random) -> str:
    blocks = [f'b{index}' for index in range(1, size + 1)]
    initial = _towers(blocks, rng)
    goal = _towers(blocks, rng)
    while _tower_atoms(goal) == _tower_atoms(initial):
        goal = _towers(blocks, rng)
    init_atoms = ['(arm-empty)', *_tower_atoms(initial)]
    goal_atoms = _tower_atoms(goal)
    return '\n'.join([
        f'(define (problem {name})',
        '  (:domain blocksworld)',
        f'  (:objects {" ".join(blocks)})',
        '  (:init',
        *(f'    {atom}' for atom in init_atoms),
        '  )',
        '  (:goal (and',
        *(f'    {atom}' for atom in goal_atoms),
        '  ))',
        ')',
        '',
    ])


def gripper_instance(name: str, size: int, rng: random.Random) -> str:
    rooms = ['rooma', 'roomb']
    balls = [f'ball{index}' for index in range(1, size + 1)]
    initial_rooms = [rng.choice(rooms) for _ in balls]
    goal_rooms = [rng.choice(rooms) for _ in balls]
    while goal_rooms == initial_rooms:
        goal_rooms = [rng.choice(rooms) for _ in balls]
    init_atoms = [
        '(room rooma)', '(room roomb)', '(gripper left)', '(gripper right)', '(free left)', '(free right)',
        f'(at-robby {rng.choice(rooms)})',
        *(f'(ball {ball})' for ball in balls),
        *(f'(at {ball} {room})' for ball, room in zip(balls, initial_rooms)),
    ]
    goal_atoms = [f'(at {ball} {room})' for ball, room in zip(balls, goal_rooms)]
    return '\n'.join([
        f'(define (problem {name})',
        '  (:domain gripper-strips)',
        f'  (:objects left right {" ".join(balls)})',
        '  (:init',
        *(f'    {atom}' for atom in init_atoms),
        '  )',
        '  (:goal (and',
        *(f'    {atom}' for atom in goal_atoms),
        '  ))',
        ')',
        '',
    ])


GENERATORS = {'blocks': blocks_instance, 'gripper': gripper_instance}


def generate(domain: str, output: Path, counts: dict[str, int], min_size: int, max_size: int, seed: int) -> None:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'domain.pddl').write_text((FIXTURES / domain / 'domain.pddl').read_text())
    generator = GENERATORS[domain]
    for split, count in counts.items():
        folder = output / split
        folder.mkdir(exist_ok=True)
        for index in range(count):
            size = rng.randint(min_size, max_size)
            name = f'{domain}-{split}-{index:03d}-n{size}'
            (folder / f'p{index:03d}.pddl').write_text(generator(name, size, rng))
        print(f'wrote {count} {domain} instances of size {min_size}-{max_size} to {folder}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', choices=sorted(GENERATORS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, default=20)
    parser.add_argument('--validation', type=int, default=5)
    parser.add_argument('--test', type=int, default=5)
    parser.add_argument('--min-size', type=int, default=3, help='smallest number of blocks or balls')
    parser.add_argument('--max-size', type=int, default=5, help='largest number of blocks or balls')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    counts = {'train': args.train, 'validation': args.validation, 'test': args.test}
    generate(args.domain, args.output, counts, args.min_size, args.max_size, args.seed)


if __name__ == '__main__':
    main()
