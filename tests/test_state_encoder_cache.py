from pathlib import Path

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    ExpressiveGoalEncoder,
    ExpressiveStateEncoder,
    GoalEncoder,
    StateEncoder,
    TransitionEffectsEncoder,
)
from pymimir_rgnn.bases import EncodedLists, EncodedTensors, EncodingContext
import pymimir_rgnn.encoders as encoders_module
from pymimir_rgnn.encoders import (
    _STATE_STATIC_RELATION_CACHE,
    VirtualNodeEncoder,
    get_input_from_encoders,
)
from pymimir_rgnn.utils import get_atom_name


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


class _UncachedStateEncoder(StateEncoder):
    """Reference implementation without static relation preprocessing."""

    def encode(
        self,
        input_value: object,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, mm.State)
        domain = input_value.problem.domain
        if domain.uses_typing:
            for type_name in domain.type_hierarchy:
                relation_name = f"type_relation_{type_name}{self.suffix}"
                type_ids = [
                    context.get_object_id(obj)
                    for obj in input_value.problem.all_objects
                    if domain.is_type_compatible(obj.type_name, type_name)
                ]
                if relation_name in encoding.flattened_relations:
                    encoding.flattened_relations[relation_name].extend(type_ids)
                else:
                    encoding.flattened_relations[relation_name] = type_ids

        if domain.uses_equality:
            equality_ids = [
                context.get_object_id(obj)
                for obj in input_value.problem.all_objects
                for _ in range(2)
            ]
            encoding.flattened_relations.setdefault(
                f"relation_={self.suffix}", []
            ).extend(equality_ids)

        for atom in input_value.atoms:
            if atom.predicate.name == "=":
                continue
            relation_name = get_atom_name(atom, state, False, self.suffix)
            object_indices = [
                context.get_object_id(obj)
                for obj in atom.objects
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
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    initial_state = problem.initial_state
    successor_state = initial_state.applicable_actions()[0].apply(initial_state)
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
    assert torch.equal(
        actual.flattened_relations["relation_room_test"],
        torch.tensor([0, 1, 6, 7], dtype=torch.int),
    )


def test_state_encoder_materializes_problem_static_atoms_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    initial_state = problem.initial_state
    successor_state = initial_state.applicable_actions()[0].apply(initial_state)
    original_cache_static_relation_groups = (
        encoders_module._cache_static_relation_groups
    )
    num_static_preprocessings = 0

    def counting_cache_static_relation_groups(
        queried_problem: mm.Problem,
        static_atoms: tuple[mm.GroundAtom, ...],
        context: EncodingContext,
    ):
        nonlocal num_static_preprocessings
        if queried_problem is problem:
            num_static_preprocessings += 1
        return original_cache_static_relation_groups(
            queried_problem,
            static_atoms,
            context,
        )

    _STATE_STATIC_RELATION_CACHE.clear()
    monkeypatch.setattr(
        encoders_module,
        "_cache_static_relation_groups",
        counting_cache_static_relation_groups,
    )

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

    assert num_static_preprocessings == 1


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
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    initial_state = problem.initial_state
    successor_state = initial_state.applicable_actions()[0].apply(initial_state)
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


def test_encoding_context_uses_canonical_all_object_order() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")

    context = EncodingContext(problem, 0)

    assert [obj.name for obj in context.object_to_id] == [
        "rooma",
        "roomb",
        "left",
        "right",
        "ball1",
        "ball2",
    ]


def test_state_encoder_gripper_golden_uses_namespaced_types() -> None:
    problem = mm.Problem.from_files(
        DATA_DIR / "gripper" / "domain.pddl",
        DATA_DIR / "gripper" / "problem.pddl",
    )

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (StateEncoder(),),
        torch.device("cpu"),
    )

    expected_relations = {
        "relation_room": [0, 1],
        "relation_gripper": [2, 3],
        "relation_ball": [4, 5],
        "relation_free": [2, 3],
        "relation_at": [4, 0, 5, 0],
        "relation_at-robby": [0],
    }
    assert encoded.flattened_relations.keys() == expected_relations.keys()
    for relation_name, expected_ids in expected_relations.items():
        assert torch.equal(
            encoded.flattened_relations[relation_name],
            torch.tensor(expected_ids, dtype=torch.int),
        )


def test_virtual_node_encoder_links_only_declared_objects() -> None:
    problem = mm.Problem.from_files(
        DATA_DIR / "gripper" / "domain.pddl",
        DATA_DIR / "gripper" / "problem.pddl",
    )

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (VirtualNodeEncoder(),),
        torch.device("cpu"),
    )

    assert encoded.virtual_indices.tolist() == [6]
    assert encoded.flattened_relations["virtual_node_link"].tolist() == [
        6, 2,
        6, 3,
        6, 4,
        6, 5,
    ]


def test_transition_effects_encoder_accepts_new_ground_literal_values() -> None:
    problem = mm.Problem.from_files(
        DATA_DIR / "gripper" / "domain.pddl",
        DATA_DIR / "gripper" / "problem.pddl",
    )
    state = problem.initial_state
    move = problem.action("move", "rooma", "roomb")
    effect_literals = [
        *(problem.ground_literal(atom) for atom in move.effect.add_atoms),
        *(
            problem.ground_literal(atom, positive=False)
            for atom in move.effect.delete_atoms
        ),
    ]
    transition_input = ([effect_literals], (), problem.goal)

    encoded = get_input_from_encoders(
        [(state, transition_input)],
        (StateEncoder(), TransitionEffectsEncoder()),
        torch.device("cpu"),
    )

    assert encoded.action_indices.tolist() == [6]
    assert encoded.flattened_relations["at-robby_pos"].tolist() == [6, 1]
    assert encoded.flattened_relations["at-robby_neg"].tolist() == [6, 0]


@pytest.mark.parametrize(
    ("encoder", "prefix"),
    [
        (StateEncoder(suffix="_typed"), ""),
        (ExpressiveStateEncoder(suffix="_typed"), "expressive_"),
    ],
    ids=["state", "expressive-state"],
)
@pytest.mark.parametrize("requirements", [":strips :typing", ":adl"])
def test_typed_domains_encode_constants_subtypes_and_real_named_predicates(
    encoder: StateEncoder | ExpressiveStateEncoder,
    prefix: str,
    requirements: str,
) -> None:
    domain = mm.Domain.from_pddl(
        f"""
(define (domain typed-demo)
    (:requirements {requirements})
    (:types
        vehicle - object
        car truck - vehicle
        electric - car
    )
    (:constants depot-car - car)
    (:predicates
        (object ?item - object)
        (number ?item - object)
        (marked ?item - object)
    )
)
""".strip()
    )
    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem typed-demo-problem)
    (:domain typed-demo)
    (:objects
        ev - electric
        truck1 - truck
    )
    (:init
        (object ev)
        (number truck1)
    )
    (:goal (marked ev))
)
""".strip(),
    )

    relation_schema = dict(encoder.get_relations(domain))
    assert f"{prefix}relation_object_typed" in relation_schema
    assert f"{prefix}relation_number_typed" in relation_schema
    for type_name in ("object", "vehicle", "car", "truck", "electric"):
        assert f"{prefix}type_relation_{type_name}_typed" in relation_schema

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (encoder,),
        torch.device("cpu"),
    )

    expected_type_ids = {
        "object": [0, 1, 2],
        "vehicle": [0, 1, 2],
        "car": [0, 1],
        "truck": [2],
        "electric": [1],
    }
    for type_name, expected_ids in expected_type_ids.items():
        assert encoded.flattened_relations[
            f"{prefix}type_relation_{type_name}_typed"
        ].tolist() == expected_ids
    assert encoded.flattened_relations[
        f"{prefix}relation_object_typed"
    ].tolist() == [1]
    assert encoded.flattened_relations[
        f"{prefix}relation_number_typed"
    ].tolist() == [2]


def test_types_are_only_part_of_state_relation_schemas() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain typed-schema)
    (:requirements :strips :typing)
    (:types item)
    (:predicates (marked ?item - item))
)
""".strip()
    )

    state_relations = StateEncoder().get_relations(domain)
    expressive_state_relations = ExpressiveStateEncoder().get_relations(domain)
    assert ("type_relation_object", 1) in state_relations
    assert ("type_relation_item", 1) in state_relations
    assert ("expressive_type_relation_object", 1) in expressive_state_relations
    assert ("expressive_type_relation_item", 1) in expressive_state_relations

    for encoder in (
        GoalEncoder(),
        ExpressiveGoalEncoder(),
        TransitionEffectsEncoder(),
    ):
        assert all(
            "type_relation_" not in relation_name
            for relation_name, _ in encoder.get_relations(domain)
        )


@pytest.mark.parametrize(
    ("encoder", "derived_relation_name"),
    [
        (StateEncoder(), "relation_reachable"),
        (ExpressiveStateEncoder(), "expressive_relation_reachable"),
    ],
    ids=["state", "expressive-state"],
)
def test_derived_domains_encode_truth_across_transitions(
    tmp_path: Path,
    encoder: StateEncoder | ExpressiveStateEncoder,
    derived_relation_name: str,
) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        """
(define (domain derived-demo)
    (:requirements :strips :derived-predicates)
    (:predicates
        (base ?object)
        (reachable ?object)
    )
    (:derived (reachable ?object) (base ?object))
    (:action clear
        :parameters (?object)
        :precondition (base ?object)
        :effect (not (base ?object))
    )
)
""".strip()
    )
    problem_path.write_text(
        """
(define (problem derived-demo-problem)
    (:domain derived-demo)
    (:objects item)
    (:init (base item))
    (:goal (reachable item))
)
""".strip()
    )
    problem = mm.Problem.from_files(domain_path, problem_path)
    state = problem.initial_state
    successor = state.applicable_actions()[0].apply(state)

    assert state.contains(problem.fact("reachable", "item"))
    assert not successor.contains(problem.fact("reachable", "item"))

    initial_encoding = get_input_from_encoders(
        [(state,)],
        (encoder,),
        torch.device("cpu"),
    )
    successor_encoding = get_input_from_encoders(
        [(successor,)],
        (encoder,),
        torch.device("cpu"),
    )

    assert initial_encoding.flattened_relations[
        derived_relation_name
    ].tolist() == [0]
    assert derived_relation_name not in successor_encoding.flattened_relations


@pytest.mark.parametrize("requirements", [":strips :equality", ":adl"])
def test_state_encoders_reconstruct_semantic_reflexive_equality(
    tmp_path: Path,
    requirements: str,
) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        """
(define (domain equality-demo)
    (:requirements {requirements})
    (:predicates (marked ?object))
    (:action mark
        :parameters (?object)
        :precondition (= ?object ?object)
        :effect (marked ?object)
    )
)
""".strip().format(requirements=requirements)
    )
    problem_path.write_text(
        """
(define (problem equality-demo-problem)
    (:domain equality-demo)
    (:objects first second)
    (:init)
    (:goal (marked first))
)
""".strip()
    )
    problem = mm.Problem.from_files(domain_path, problem_path)
    state = problem.initial_state
    assert problem.domain.uses_equality
    assert ("relation_=", 2) in StateEncoder().get_relations(problem.domain)
    assert (
        "expressive_relation_=",
        4,
    ) in ExpressiveStateEncoder().get_relations(problem.domain)

    regular = get_input_from_encoders(
        [(state,)],
        (StateEncoder(),),
        torch.device("cpu"),
    )
    expressive = get_input_from_encoders(
        [(state,)],
        (ExpressiveStateEncoder(),),
        torch.device("cpu"),
    )

    assert torch.equal(
        regular.flattened_relations["relation_="],
        torch.tensor([0, 0, 1, 1], dtype=torch.int),
    )
    assert torch.equal(
        expressive.flattened_relations["expressive_relation_="],
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int),
    )


def test_state_encoders_do_not_add_equality_without_semantic_support() -> None:
    problem = mm.Problem.from_pddl(
        mm.Domain.from_pddl(
            """
(define (domain no-equality)
    (:requirements :strips)
    (:predicates (marked ?object))
)
""".strip()
        ),
        """
(define (problem no-equality-problem)
    (:domain no-equality)
    (:objects item)
    (:init)
    (:goal (marked item))
)
""".strip(),
    )

    for encoder in (StateEncoder(), ExpressiveStateEncoder()):
        encoded = get_input_from_encoders(
            [(problem.initial_state,)],
            (encoder,),
            torch.device("cpu"),
        )
        assert all(
            not relation_name.endswith("relation_=")
            for relation_name in encoded.flattened_relations
        )
