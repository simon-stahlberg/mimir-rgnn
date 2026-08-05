from pathlib import Path

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import StateEncoder
from pymimir_rgnn.bases import EncodedLists, EncodedTensors, EncodingContext
from pymimir_rgnn.encoders import get_input_from_encoders
from pymimir_rgnn.utils import get_atom_name


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


class _UncachedStateEncoder(StateEncoder):
    """Reference implementation matching the original StateEncoder."""

    def encode(
        self,
        input_value: object,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, mm.State)
        for atom in input_value.get_atoms():
            relation_name = get_atom_name(atom, state, False, self.suffix)
            object_indices = [
                context.get_object_id(obj.get_index())
                for obj in atom.get_terms()
            ]
            if relation_name not in encoding.flattened_relations:
                encoding.flattened_relations[relation_name] = object_indices
            else:
                encoding.flattened_relations[relation_name].extend(object_indices)


def _assert_same_encoding(actual: EncodedTensors, expected: EncodedTensors) -> None:
    assert actual.node_count == expected.node_count
    assert actual.flattened_relations.keys() == expected.flattened_relations.keys()
    for relation_name in actual.flattened_relations:
        assert torch.equal(
            actual.flattened_relations[relation_name],
            expected.flattened_relations[relation_name],
        )
    for attribute_name in (
        "node_sizes",
        "object_indices",
        "object_sizes",
        "action_indices",
        "action_sizes",
        "virtual_indices",
        "virtual_sizes",
        "auxiliary_indices",
        "auxiliary_sizes",
    ):
        assert torch.equal(
            getattr(actual, attribute_name),
            getattr(expected, attribute_name),
        )


def test_state_encoder_static_cache_preserves_batched_relation_ids() -> None:
    domain = mm.Domain(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem(domain, DATA_DIR / "gripper" / "problem.pddl")
    initial_state = problem.get_initial_state()
    successor_state = initial_state.generate_applicable_actions()[0].apply(initial_state)
    model_input = [(initial_state,), (successor_state,)]

    expected = get_input_from_encoders(
        model_input,
        (_UncachedStateEncoder(suffix="_test"),),
        torch.device("cpu"),
    )
    actual = get_input_from_encoders(
        model_input,
        (StateEncoder(suffix="_test"),),
        torch.device("cpu"),
    )

    _assert_same_encoding(actual, expected)


def test_state_encoder_materializes_problem_static_atoms_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem(domain, DATA_DIR / "gripper" / "problem.pddl")
    initial_state = problem.get_initial_state()
    successor_state = initial_state.generate_applicable_actions()[0].apply(initial_state)
    original_get_initial_atoms = mm.Problem.get_initial_atoms
    num_static_materializations = 0

    def counting_get_initial_atoms(
        queried_problem: mm.Problem,
        *args: object,
        **kwargs: object,
    ) -> list[mm.GroundAtom]:
        nonlocal num_static_materializations
        if (
            queried_problem is problem
            and kwargs
            == {
                "ignore_static": False,
                "ignore_fluent": True,
                "ignore_derived": True,
            }
        ):
            num_static_materializations += 1
        return original_get_initial_atoms(queried_problem, *args, **kwargs)

    monkeypatch.setattr(mm.Problem, "get_initial_atoms", counting_get_initial_atoms)

    get_input_from_encoders(
        [(initial_state,)],
        (StateEncoder(),),
        torch.device("cpu"),
    )
    get_input_from_encoders(
        [(successor_state,)],
        (StateEncoder(suffix="_another_encoder"),),
        torch.device("cpu"),
    )

    assert num_static_materializations == 1


def test_state_encoder_static_cache_preserves_nullary_relations_and_suffix(
    tmp_path: Path,
) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        """
(define (domain nullary-static)
    (:requirements :strips)
    (:predicates
        (static-ready)
        (marked ?object)
    )
    (:action mark
        :parameters (?object)
        :precondition (static-ready)
        :effect (marked ?object)
    )
)
""".strip()
    )
    problem_path.write_text(
        """
(define (problem nullary-static-problem)
    (:domain nullary-static)
    (:objects item)
    (:init (static-ready))
    (:goal (marked item))
)
""".strip()
    )
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    initial_state = problem.get_initial_state()
    successor_state = initial_state.generate_applicable_actions()[0].apply(initial_state)
    model_input = [(initial_state,), (successor_state,)]

    expected = get_input_from_encoders(
        model_input,
        (_UncachedStateEncoder(suffix="_suffix"),),
        torch.device("cpu"),
    )
    actual = get_input_from_encoders(
        model_input,
        (StateEncoder(suffix="_suffix"),),
        torch.device("cpu"),
    )

    _assert_same_encoding(actual, expected)
    assert "relation_static-ready_suffix" in actual.flattened_relations
    assert actual.flattened_relations["relation_static-ready_suffix"].numel() == 0


def test_state_encoder_falls_back_for_custom_state_get_atoms_protocol() -> None:
    class CustomObject:
        def __init__(self, index: int) -> None:
            self._index = index

        def get_index(self) -> int:
            return self._index

    class CustomPredicate:
        def get_name(self) -> str:
            return "custom"

    class CustomAtom:
        def __init__(self, terms: list[CustomObject]) -> None:
            self._terms = terms

        def get_predicate(self) -> CustomPredicate:
            return CustomPredicate()

        def get_terms(self) -> list[CustomObject]:
            return self._terms

    class CustomDomain(mm.Domain):
        def get_constants(self) -> list[CustomObject]:  # type: ignore[override]
            return []

    class CustomProblem(mm.Problem):
        def __init__(self, objects: list[CustomObject]) -> None:
            self._objects = objects
            self._domain = CustomDomain.__new__(CustomDomain)

        def get_objects(self) -> list[CustomObject]:  # type: ignore[override]
            return self._objects

        def get_domain(self) -> CustomDomain:  # type: ignore[override]
            return self._domain

    class CustomState(mm.State):
        def __init__(self, problem: CustomProblem, atoms: list[CustomAtom]) -> None:
            self._custom_problem = problem
            self._custom_atoms = atoms
            self.num_get_atoms_calls = 0

        def get_problem(self) -> CustomProblem:  # type: ignore[override]
            return self._custom_problem

        def get_atoms(self) -> list[CustomAtom]:  # type: ignore[override]
            self.num_get_atoms_calls += 1
            return self._custom_atoms

    objects = [CustomObject(0), CustomObject(1)]
    problem = CustomProblem(objects)
    state = CustomState(problem, [CustomAtom(objects)])

    encoded = get_input_from_encoders(
        [(state,), (state,)],
        (StateEncoder(suffix="_fallback"),),
        torch.device("cpu"),
    )

    assert state.num_get_atoms_calls == 2
    assert torch.equal(
        encoded.flattened_relations["relation_custom_fallback"],
        torch.tensor([0, 1, 2, 3], dtype=torch.int),
    )
