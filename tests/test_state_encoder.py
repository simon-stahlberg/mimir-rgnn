from collections import Counter
from pathlib import Path

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    EncoderRelation,
    EncoderRelationKind,
    ExpressiveGoalEncoder,
    ExpressiveStateEncoder,
    GoalEncoder,
    StateEncoder,
    TransitionEffectsEncoder,
)
from pymimir_rgnn.bases import EncodingContext
from pymimir_rgnn.encoders import (
    VirtualNodeEncoder,
    get_input_from_encoders,
)


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


def _relation_rows(
    relation: torch.Tensor,
    arity: int,
) -> list[tuple[int, ...]]:
    assert arity > 0
    assert relation.numel() % arity == 0
    return sorted(
        tuple(int(term_id) for term_id in row)
        for row in relation.reshape(-1, arity).tolist()
    )


def test_state_encoder_expands_nullary_relations_with_suffix_and_offsets(
    tmp_path: Path,
) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        """
(define (domain nullary-static)
    (:requirements :strips)
    (:constants shared)
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
    (:objects first second)
    (:init (static-ready))
    (:goal (marked first))
)
""".strip()
    )
    domain = mm.Domain.from_file(domain_path)
    problem = mm.Problem.from_file(domain, problem_path)
    other_problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem other-nullary-static-problem)
    (:domain nullary-static)
    (:objects only)
    (:init (static-ready))
    (:goal (marked only))
)
""".strip(),
    )
    model_input = [(problem.initial_state,), (other_problem.initial_state,)]

    actual = get_input_from_encoders(
        model_input,
        (StateEncoder(suffix="_suffix"),),
        torch.device("cpu"),
    )

    relation_name = "relation_static-ready_suffix"
    relation_schema = dict(
        StateEncoder(suffix="_suffix").get_relations(domain)
    )
    assert relation_schema[relation_name] == 1
    assert relation_schema["constant_relation_shared_suffix"] == 1
    assert actual.node_sizes.tolist() == [3, 2]
    assert sorted(actual.flattened_relations[relation_name].tolist()) == list(
        range(5)
    )
    assert actual.flattened_relations[
        "constant_relation_shared_suffix"
    ].tolist() == [0, 3]
    assert actual._native_relation_values is not None
    for native_relation_name in (
        relation_name,
        "constant_relation_shared_suffix",
    ):
        assert (
            actual.flattened_relations[native_relation_name]
            .untyped_storage()
            .data_ptr()
            == actual._native_relation_values.untyped_storage().data_ptr()
        )


def test_nullary_predicate_relation_schemas_match_encoder_contracts() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain nullary-schema)
    (:requirements :strips)
    (:predicates
        (ready)
        (linked ?left ?right)
    )
)
""".strip()
    )

    state_schema = dict(StateEncoder(suffix="_suffix").get_relations(domain))
    goal_schema = dict(GoalEncoder(suffix="_suffix").get_relations(domain))
    effect_schema = dict(
        TransitionEffectsEncoder(suffix="_suffix").get_relations(domain)
    )
    expressive_state_schema = dict(
        ExpressiveStateEncoder(suffix="_suffix").get_relations(domain)
    )
    expressive_goal_schema = dict(
        ExpressiveGoalEncoder(suffix="_suffix").get_relations(domain)
    )

    assert state_schema["relation_ready_suffix"] == 1
    assert state_schema["relation_linked_suffix"] == 2
    assert state_schema["type_relation_object_suffix"] == 1
    assert EncoderRelation(
        "type_relation_object_suffix",
        1,
        EncoderRelationKind.STATE_TYPE,
        "object",
    ) in StateEncoder(suffix="_suffix").get_relation_descriptors(domain)
    assert all(
        descriptor.kind is not EncoderRelationKind.STATE_CONSTANT
        for descriptor in StateEncoder(
            suffix="_suffix"
        ).get_relation_descriptors(domain)
    )
    assert goal_schema["relation_ready_suffix_goal_true"] == 1
    assert goal_schema["relation_ready_suffix_goal_false"] == 1
    assert goal_schema["relation_linked_suffix_goal_true"] == 2
    assert goal_schema["relation_linked_suffix_goal_false"] == 2
    for polarity in ("pos", "neg"):
        assert effect_schema[f"ready_suffix_{polarity}"] == 1
        assert effect_schema[f"ready_suffix_{polarity}_goal"] == 1
        assert effect_schema[f"linked_suffix_{polarity}"] == 3
        assert effect_schema[f"linked_suffix_{polarity}_goal"] == 3
    assert expressive_state_schema["expressive_relation_ready_suffix"] == 1
    assert expressive_state_schema["expressive_relation_linked_suffix"] == 4
    assert expressive_state_schema["expressive_type_relation_object_suffix"] == 1
    assert expressive_goal_schema[
        "expressive_relation_ready_suffix_goal_true"
    ] == 1
    assert expressive_goal_schema[
        "expressive_relation_ready_suffix_goal_false"
    ] == 1
    assert expressive_goal_schema[
        "expressive_relation_linked_suffix_goal_true"
    ] == 4
    assert expressive_goal_schema[
        "expressive_relation_linked_suffix_goal_false"
    ] == 4


def test_standard_encoder_relation_descriptors_expose_semantics() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain descriptor-schema)
    (:requirements :strips :typing)
    (:types item)
    (:constants anchor - item)
    (:predicates
        (ready)
        (linked ?left - item ?right - object)
    )
)
""".strip()
    )
    suffix = "_descriptor"

    state = StateEncoder(suffix=suffix)
    assert Counter(state.get_relation_descriptors(domain)) == Counter(
        {
            EncoderRelation(
                "relation_ready_descriptor",
                1,
                EncoderRelationKind.STATE_PREDICATE,
                "ready",
            ),
            EncoderRelation(
                "relation_linked_descriptor",
                2,
                EncoderRelationKind.STATE_PREDICATE,
                "linked",
            ),
            EncoderRelation(
                "type_relation_object_descriptor",
                1,
                EncoderRelationKind.STATE_TYPE,
                "object",
            ),
            EncoderRelation(
                "type_relation_item_descriptor",
                1,
                EncoderRelationKind.STATE_TYPE,
                "item",
            ),
            EncoderRelation(
                "constant_relation_anchor_descriptor",
                1,
                EncoderRelationKind.STATE_CONSTANT,
                "anchor",
            ),
        }
    )

    goal = GoalEncoder(suffix=suffix)
    assert Counter(goal.get_relation_descriptors(domain)) == Counter(
        {
            EncoderRelation(
                f"relation_{predicate}{suffix}_goal_{truth}",
                arity,
                kind,
                predicate,
            )
            for predicate, arity in (("ready", 1), ("linked", 2))
            for truth, kind in (
                ("false", EncoderRelationKind.GOAL_FALSE),
                ("true", EncoderRelationKind.GOAL_TRUE),
            )
        }
    )

    effects = TransitionEffectsEncoder(suffix=suffix)
    assert Counter(effects.get_relation_descriptors(domain)) == Counter(
        {
            *(
                EncoderRelation(
                    f"{predicate}{suffix}_{polarity}{goal_suffix}",
                    arity + 1,
                    kind,
                    predicate,
                )
                for predicate, arity in (("ready", 0), ("linked", 2))
                for polarity, goal_suffix, kind in (
                    ("pos", "", EncoderRelationKind.EFFECT_POSITIVE),
                    ("neg", "", EncoderRelationKind.EFFECT_NEGATIVE),
                    (
                        "pos",
                        "_goal",
                        EncoderRelationKind.EFFECT_POSITIVE_GOAL,
                    ),
                    (
                        "neg",
                        "_goal",
                        EncoderRelationKind.EFFECT_NEGATIVE_GOAL,
                    ),
                )
            ),
            EncoderRelation(
                "effect_relation_descriptor",
                2,
                EncoderRelationKind.EFFECT_LINK,
                None,
            ),
        }
    )

    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem descriptor-instance)
    (:domain descriptor-schema)
    (:objects left - item right - object)
    (:init
        (ready)
        (linked left right)
    )
    (:goal (ready))
)
""".strip(),
    )
    encoded_state = get_input_from_encoders(
        [(problem.initial_state,)],
        (state,),
        torch.device("cpu"),
    )
    assert Counter(encoded_state.flattened_relations.keys()) == Counter(
        descriptor.name for descriptor in state.get_relation_descriptors(domain)
    )

    for encoder in (state, goal, effects):
        assert encoder.get_relations(domain) == [
            (descriptor.name, descriptor.arity)
            for descriptor in encoder.get_relation_descriptors(domain)
        ]

    descriptor = state.get_relation_descriptors(domain)[0]
    with pytest.raises(AttributeError):
        descriptor.name = "changed"  # type: ignore[misc]


def test_state_constant_relations_have_a_distinct_namespace() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain constant-namespace)
    (:requirements :strips :typing)
    (:types shared)
    (:constants shared - shared)
    (:predicates (shared ?object - shared))
)
""".strip()
    )

    descriptors = StateEncoder().get_relation_descriptors(domain)

    assert {
        descriptor.name
        for descriptor in descriptors
        if descriptor.source_name == "shared"
    } == {
        "relation_shared",
        "type_relation_shared",
        "constant_relation_shared",
    }
    assert {
        descriptor.kind
        for descriptor in descriptors
        if descriptor.source_name == "shared"
    } == {
        EncoderRelationKind.STATE_PREDICATE,
        EncoderRelationKind.STATE_TYPE,
        EncoderRelationKind.STATE_CONSTANT,
    }


@pytest.mark.parametrize(
    ("encoder", "relation_name"),
    (
        (StateEncoder(suffix="_untyped"), "type_relation_object_untyped"),
        (
            ExpressiveStateEncoder(suffix="_untyped"),
            "expressive_type_relation_object_untyped",
        ),
    ),
    ids=("state", "expressive-state"),
)
def test_untyped_state_encoders_mark_every_object_with_the_root_type(
    encoder: StateEncoder | ExpressiveStateEncoder,
    relation_name: str,
) -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain untyped-object-root)
    (:requirements :strips)
    (:constants shared)
    (:predicates (marked ?object))
)
""".strip()
    )
    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem untyped-object-root-problem)
    (:domain untyped-object-root)
    (:objects local)
    (:init)
    (:goal (marked local))
)
""".strip(),
    )

    assert not domain.uses_typing
    assert dict(encoder.get_relations(domain))[relation_name] == 1

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (encoder,),
        torch.device("cpu"),
    )

    assert encoded.flattened_relations[relation_name].tolist() == [0, 1]
    assert encoded._native_relation_values is not None
    assert (
        encoded.flattened_relations[relation_name].untyped_storage().data_ptr()
        == encoded._native_relation_values.untyped_storage().data_ptr()
    )


def test_predicate_encoders_apply_their_nullary_row_contracts() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain nullary-all-encoders)
    (:requirements :strips)
    (:constants shared)
    (:predicates (ready))
    (:action clear
        :parameters ()
        :precondition (ready)
        :effect (not (ready))
    )
)
""".strip()
    )
    first_problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem first-nullary-problem)
    (:domain nullary-all-encoders)
    (:objects first second)
    (:init (ready))
    (:goal (ready))
)
""".strip(),
    )
    second_problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem second-nullary-problem)
    (:domain nullary-all-encoders)
    (:objects third)
    (:init (ready))
    (:goal (ready))
)
""".strip(),
    )
    first_state = first_problem.initial_state
    first_successor = first_state.applicable_actions()[0].apply(first_state)
    second_initial_state = second_problem.initial_state
    second_state = second_initial_state.applicable_actions()[0].apply(
        second_initial_state
    )
    first_transition = (
        [first_successor],
        (),
        first_problem.goal,
    )
    second_transition = (
        [second_initial_state],
        (),
        second_problem.goal,
    )
    specification = (
        StateEncoder(suffix="_suffix"),
        GoalEncoder(suffix="_suffix"),
        ExpressiveStateEncoder(suffix="_suffix"),
        ExpressiveGoalEncoder(suffix="_suffix"),
        TransitionEffectsEncoder(suffix="_suffix"),
    )

    encoded = get_input_from_encoders(
        [
            (
                first_state,
                first_problem.goal,
                first_state,
                first_problem.goal,
                first_transition,
            ),
            (
                second_state,
                second_problem.goal,
                second_state,
                second_problem.goal,
                second_transition,
            ),
        ],
        specification,
        torch.device("cpu"),
    )

    first_objects = encoded.object_indices[:3].tolist()
    second_objects = encoded.object_indices[3:].tolist()
    first_transition_id, second_transition_id = encoded.action_indices.tolist()
    assert encoded.object_sizes.tolist() == [3, 2]
    assert sorted(
        encoded.flattened_relations["relation_ready_suffix"].tolist()
    ) == first_objects
    assert sorted(
        encoded.flattened_relations[
            "relation_ready_suffix_goal_true"
        ].tolist()
    ) == first_objects
    assert sorted(
        encoded.flattened_relations[
            "relation_ready_suffix_goal_false"
        ].tolist()
    ) == second_objects
    assert sorted(
        encoded.flattened_relations[
            "expressive_relation_ready_suffix"
        ].tolist()
    ) == first_objects
    assert sorted(
        encoded.flattened_relations[
            "expressive_relation_ready_suffix_goal_true"
        ].tolist()
    ) == first_objects
    assert sorted(
        encoded.flattened_relations[
            "expressive_relation_ready_suffix_goal_false"
        ].tolist()
    ) == second_objects
    assert _relation_rows(
        encoded.flattened_relations["ready_suffix_neg"],
        1,
    ) == [(first_transition_id,)]
    assert _relation_rows(
        encoded.flattened_relations["ready_suffix_neg_goal"],
        1,
    ) == [(first_transition_id,)]
    assert _relation_rows(
        encoded.flattened_relations["ready_suffix_pos"],
        1,
    ) == [(second_transition_id,)]
    assert _relation_rows(
        encoded.flattened_relations["ready_suffix_pos_goal"],
        1,
    ) == [(second_transition_id,)]

    assert encoded._native_relation_values is not None
    for relation_name in (
        "relation_ready_suffix",
        "relation_ready_suffix_goal_true",
        "relation_ready_suffix_goal_false",
        "expressive_relation_ready_suffix",
        "expressive_relation_ready_suffix_goal_true",
        "expressive_relation_ready_suffix_goal_false",
        "ready_suffix_neg",
        "ready_suffix_neg_goal",
        "ready_suffix_pos",
        "ready_suffix_pos_goal",
    ):
        assert (
            encoded.flattened_relations[relation_name]
            .untyped_storage()
            .data_ptr()
            == encoded._native_relation_values.untyped_storage().data_ptr()
        )


def test_nullary_transition_effect_is_present_without_objects() -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain empty-nullary-transition)
    (:requirements :strips)
    (:predicates (ready))
    (:action clear
        :parameters ()
        :precondition (ready)
        :effect (not (ready))
    )
)
""".strip()
    )
    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem empty-nullary-transition-problem)
    (:domain empty-nullary-transition)
    (:init (ready))
    (:goal (ready))
)
""".strip(),
    )
    state = problem.initial_state
    successor = state.applicable_actions()[0].apply(state)

    encoded = get_input_from_encoders(
        [(state, ([successor], (), problem.goal))],
        (StateEncoder(), TransitionEffectsEncoder()),
        torch.device("cpu"),
    )

    assert encoded.object_indices.numel() == 0
    assert encoded.action_indices.tolist() == [0]
    assert encoded.flattened_relations["relation_ready"].numel() == 0
    assert encoded.flattened_relations["ready_neg"].tolist() == [0]
    assert encoded.flattened_relations["ready_neg_goal"].tolist() == [0]
    assert encoded._native_relation_values is not None
    assert (
        encoded.flattened_relations["ready_neg"].untyped_storage().data_ptr()
        == encoded._native_relation_values.untyped_storage().data_ptr()
    )


def test_encoding_context_uses_canonical_all_object_order() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")

    context = EncodingContext()
    context.begin_instance(problem)
    try:
        assert [
            context.get_object_id(obj)
            for obj in problem.all_objects
        ] == list(range(6))
    finally:
        context.end_instance()


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
        "constant_relation_rooma": [0],
        "constant_relation_roomb": [1],
        "type_relation_object": [0, 1, 2, 3, 4, 5],
        "relation_room": [0, 1],
        "relation_gripper": [2, 3],
        "relation_ball": [4, 5],
        "relation_free": [2, 3],
        "relation_at": [4, 0, 5, 0],
        "relation_at-robby": [0],
    }
    assert encoded.flattened_relations.keys() == expected_relations.keys()
    relation_arities = dict(StateEncoder().get_relations(problem.domain))
    for relation_name, expected_ids in expected_relations.items():
        arity = relation_arities[relation_name]
        assert _relation_rows(
            encoded.flattened_relations[relation_name],
            arity,
        ) == _relation_rows(
            torch.tensor(expected_ids, dtype=torch.int),
            arity,
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


def test_transition_effects_encoder_forwards_ordered_successors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = mm.Problem.from_files(
        DATA_DIR / "gripper" / "domain.pddl",
        DATA_DIR / "gripper" / "problem.pddl",
    )
    state = problem.initial_state
    actions = state.applicable_actions()
    successors = tuple(action.apply(state) for action in actions[:2])
    effect_relations = ((1, 0),)
    goal_condition = problem.goal
    calls: list[tuple[object, ...]] = []

    def record_transition_effects(*args: object, **kwargs: object) -> None:
        calls.append((*args, kwargs))

    monkeypatch.setattr(
        mm.learning,
        "encode_transition_effects",
        record_transition_effects,
    )

    get_input_from_encoders(
        [(state, (successors, effect_relations, goal_condition))],
        (StateEncoder(), TransitionEffectsEncoder(suffix="_ordered")),
        torch.device("cpu"),
    )

    assert len(calls) == 1
    (
        context,
        source,
        forwarded_successors,
        forwarded_relations,
        goal,
        kwargs,
    ) = calls[0]
    assert isinstance(context, EncodingContext)
    assert source is state
    assert forwarded_successors is successors
    assert forwarded_relations is effect_relations
    assert goal is goal_condition
    assert kwargs == {"suffix": "_ordered"}


def test_transition_effects_encoder_accepts_ordered_successor_states() -> None:
    problem = mm.Problem.from_files(
        DATA_DIR / "gripper" / "domain.pddl",
        DATA_DIR / "gripper" / "problem.pddl",
    )
    state = problem.initial_state
    unchanged_successor = problem.action("move", "rooma", "rooma").apply(state)
    move = problem.action("move", "rooma", "roomb")
    changed_successor = move.apply(state)
    transition_input = (
        [unchanged_successor, changed_successor],
        ((0, 1),),
        problem.goal,
    )

    encoded = get_input_from_encoders(
        [(state, transition_input)],
        (StateEncoder(), TransitionEffectsEncoder()),
        torch.device("cpu"),
    )

    assert encoded.action_indices.tolist() == [6, 7]
    assert encoded.flattened_relations["at-robby_pos"].tolist() == [7, 1]
    assert encoded.flattened_relations["at-robby_neg"].tolist() == [7, 0]
    assert encoded.flattened_relations["effect_relation"].tolist() == [6, 7]


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
        assert sorted(encoded.flattened_relations[
            f"{prefix}type_relation_{type_name}_typed"
        ].tolist()) == sorted(expected_ids)
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

    assert _relation_rows(
        regular.flattened_relations["relation_="],
        2,
    ) == [(0, 0), (1, 1)]
    assert _relation_rows(
        expressive.flattened_relations["expressive_relation_="],
        4,
    ) == [(0, 0, 0, 0), (1, 1, 1, 1)]


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
