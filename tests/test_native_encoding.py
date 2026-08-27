import gc
import io
from pathlib import Path
from typing import Any

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    Encoder,
    ExpressiveGoalEncoder,
    ExpressiveStateEncoder,
    GoalEncoder,
    GroundActionsEncoder,
    StateEncoder,
    TransitionEffectsEncoder,
)
import pymimir_rgnn.encoders as encoders_module
from pymimir_rgnn.bases import EncodedLists, EncodingContext
from pymimir_rgnn.encoders import (
    VirtualNodeEncoder,
    get_input_from_encoders,
)


DATA_DIR = Path(__file__).parent / "data"
CPU = torch.device("cpu")


class _CustomAuxiliaryEncoder(Encoder):
    """Small legacy-contract encoder used to verify native/custom mixing."""

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [("custom_auxiliary", 1)]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        auxiliary_id = context.new_or_existing_auxiliary_id(
            ("custom", input_value)
        )
        encoding.flattened_relations.setdefault(
            "custom_auxiliary",
            [],
        ).append(auxiliary_id)


class _CustomCollisionEncoder(Encoder):
    """Write into a relation also populated by the native state encoder."""

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [("relation_room", 1)]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, int)
        encoding.flattened_relations.setdefault("relation_room", []).append(
            input_value
        )


def _relation_rows(relation: torch.Tensor, arity: int) -> list[tuple[int, ...]]:
    if arity == 0:
        assert relation.numel() == 0
        return []
    assert relation.numel() % arity == 0
    return sorted(
        tuple(int(term_id) for term_id in row)
        for row in relation.reshape(-1, arity).tolist()
    )


def _gripper_batch() -> tuple[mm.Domain, list[tuple]]:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    first_problem = mm.Problem.from_file(
        domain,
        DATA_DIR / "gripper" / "problem.pddl",
    )
    second_problem = mm.Problem.from_file(
        domain,
        DATA_DIR / "gripper" / "problem.pddl",
    )
    return domain, [
        (
            first_problem.initial_state,
            first_problem.initial_state.applicable_actions(),
            first_problem.goal,
        ),
        (
            second_problem.initial_state,
            second_problem.initial_state.applicable_actions(),
            second_problem.goal,
        ),
    ]


def test_empty_input_preserves_the_existing_empty_encoding() -> None:
    encoded = get_input_from_encoders([], (StateEncoder(),), CPU)

    assert encoded.node_count == 0
    assert encoded.flattened_relations == {}
    assert encoded.node_sizes.numel() == 0
    assert encoded.object_indices.numel() == 0
    assert encoded.action_indices.numel() == 0


def test_standard_native_encoding_uses_expected_offsets() -> None:
    _, full_model_input = _gripper_batch()
    model_input = [
        (state, actions)
        for state, actions, _ in full_model_input
    ]
    input_specification = (
        StateEncoder(),
        GroundActionsEncoder(),
    )

    actual = get_input_from_encoders(model_input, input_specification, CPU)
    assert actual.node_count == 24
    assert actual.node_sizes.tolist() == [12, 12]
    assert actual.object_sizes.tolist() == [6, 6]
    assert actual.object_indices.tolist() == [*range(0, 6), *range(12, 18)]
    assert actual.action_sizes.tolist() == [6, 6]
    assert actual.action_indices.tolist() == [*range(6, 12), *range(18, 24)]
    assert _relation_rows(actual.flattened_relations["relation_room"], 1) == [
        (0,),
        (1,),
        (12,),
        (13,),
    ]
    assert _relation_rows(
        actual.flattened_relations["type_relation_object"],
        1,
    ) == [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (12,),
        (13,),
        (14,),
        (15,),
        (16,),
        (17,),
    ]
    assert _relation_rows(
        actual.flattened_relations["constant_relation_rooma"],
        1,
    ) == [(0,), (12,)]
    assert _relation_rows(
        actual.flattened_relations["constant_relation_roomb"],
        1,
    ) == [(1,), (13,)]


def test_multiple_standard_state_and_goal_encoders_merge_relations() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    input_specification = (
        StateEncoder(suffix="_before"),
        StateEncoder(suffix="_again"),
        GoalEncoder(suffix="_goal"),
        GoalEncoder(suffix="_goal"),
    )
    model_input = [(state, state, problem.goal, problem.goal)]

    actual = get_input_from_encoders(model_input, input_specification, CPU)
    assert "relation_room_before" in actual.flattened_relations
    assert "relation_room_again" in actual.flattened_relations
    assert actual.flattened_relations["relation_at_goal_goal_false"].numel() == 4


def test_one_batch_context_drives_every_native_encoder_in_specification_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    actions = state.applicable_actions()
    move = problem.action("move", "rooma", "roomb")
    successor = move.apply(state)
    transition_input = ([successor], (), problem.goal)

    original_context_type = mm.learning.EncodingContext
    original_begin_instance = original_context_type.begin_instance
    original_end_instance = original_context_type.end_instance
    original_to_relation_buffer = original_context_type.to_relation_buffer
    contexts: list[EncodingContext] = []
    lifecycle: list[tuple[str, mm.Problem | None]] = []
    materialized_contexts: list[EncodingContext] = []

    def begin_instance(context: EncodingContext, value: mm.Problem) -> None:
        lifecycle.append(("begin", value))
        original_begin_instance(context, value)

    def end_instance(context: EncodingContext) -> None:
        lifecycle.append(("end", None))
        original_end_instance(context)

    def to_relation_buffer(context: EncodingContext) -> Any:
        materialized_contexts.append(context)
        return original_to_relation_buffer(context)

    monkeypatch.setattr(original_context_type, "begin_instance", begin_instance)
    monkeypatch.setattr(original_context_type, "end_instance", end_instance)
    monkeypatch.setattr(
        original_context_type,
        "to_relation_buffer",
        to_relation_buffer,
    )

    def make_context() -> EncodingContext:
        context = original_context_type()
        contexts.append(context)
        return context

    monkeypatch.setattr(encoders_module, "EncodingContext", make_context)
    calls: list[tuple[str, EncodingContext]] = []
    function_names = (
        "encode_virtual_node",
        "encode_expressive_goal",
        "encode_action_list",
        "encode_state",
        "encode_transition_effects",
        "encode_expressive_state",
        "encode_goal",
    )

    for function_name in function_names:
        def record_call(
            *args: object,
            _function_name: str = function_name,
            **kwargs: object,
        ) -> None:
            calls.append((_function_name, args[0]))  # type: ignore[arg-type]

        monkeypatch.setattr(mm.learning, function_name, record_call)

    specification = (
        VirtualNodeEncoder(),
        ExpressiveGoalEncoder(),
        GroundActionsEncoder(),
        StateEncoder(),
        TransitionEffectsEncoder(),
        ExpressiveStateEncoder(),
        GoalEncoder(),
    )
    instance = (
        None,
        problem.goal,
        actions,
        state,
        transition_input,
        state,
        problem.goal,
    )

    encoded = get_input_from_encoders([instance, instance], specification, CPU)

    assert len(contexts) == 1
    assert lifecycle == [
        ("begin", problem),
        ("end", None),
        ("begin", problem),
        ("end", None),
    ]
    assert materialized_contexts == contexts
    assert calls == [
        (function_name, contexts[0])
        for _ in range(2)
        for function_name in function_names
    ]
    assert encoded.node_sizes.tolist() == [6, 6]
    assert encoded.object_indices.tolist() == [*range(6), *range(6, 12)]
    assert encoded.flattened_relations == {}


def test_native_relations_use_one_packed_buffer_and_shared_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    original_frombuffer = torch.frombuffer
    frombuffer_calls: list[tuple[object, torch.dtype, int, int, bool]] = []

    def frombuffer(
        buffer: Any,
        *,
        dtype: torch.dtype,
        count: int = -1,
        offset: int = 0,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        frombuffer_calls.append(
            (buffer, dtype, count, offset, requires_grad)
        )
        return original_frombuffer(
            buffer,
            dtype=dtype,
            count=count,
            offset=offset,
            requires_grad=requires_grad,
        )

    monkeypatch.setattr(encoders_module.torch, "frombuffer", frombuffer)

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (StateEncoder(),),
        CPU,
    )

    assert len(frombuffer_calls) == 1
    assert encoded._native_relation_values is not None

    packed_values = encoded._native_relation_values
    exported_values, dtype, count, offset, requires_grad = frombuffer_calls[0]
    assert dtype is torch.int32
    assert count == packed_values.numel()
    assert offset == 0
    assert not requires_grad
    assert isinstance(exported_values, memoryview)
    assert memoryview(exported_values).nbytes == packed_values.numel() * 4

    expected_offset = 0
    for relation in encoded.flattened_relations.values():
        assert relation.untyped_storage().data_ptr() == (
            packed_values.untyped_storage().data_ptr()
        )
        assert relation.storage_offset() == expected_offset
        expected_offset += relation.numel()
    assert expected_offset == packed_values.numel()


def test_nullary_broadcast_without_objects_skips_frombuffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_pddl(
        """
(define (domain nullary-only)
    (:requirements :strips)
    (:predicates (ready))
)
""".strip()
    )
    problem = mm.Problem.from_pddl(
        domain,
        """
(define (problem nullary-only-problem)
    (:domain nullary-only)
    (:init (ready))
    (:goal (ready))
)
""".strip(),
    )
    frombuffer_calls = 0

    def unexpected_frombuffer(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal frombuffer_calls
        frombuffer_calls += 1
        raise AssertionError("frombuffer must not receive an empty buffer")

    monkeypatch.setattr(
        encoders_module.torch,
        "frombuffer",
        unexpected_frombuffer,
    )

    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (StateEncoder(),),
        CPU,
    )

    assert dict(StateEncoder().get_relations(domain))["relation_ready"] == 1
    assert dict(GoalEncoder().get_relations(domain))[
        "relation_ready_goal_true"
    ] == 1
    assert dict(GoalEncoder().get_relations(domain))[
        "relation_ready_goal_false"
    ] == 1
    transition_schema = dict(TransitionEffectsEncoder().get_relations(domain))
    assert transition_schema["ready_pos"] == 1
    assert transition_schema["ready_neg"] == 1
    assert transition_schema["ready_pos_goal"] == 1
    assert transition_schema["ready_neg_goal"] == 1
    assert dict(ExpressiveStateEncoder().get_relations(domain))[
        "expressive_relation_ready"
    ] == 1
    assert dict(ExpressiveGoalEncoder().get_relations(domain))[
        "expressive_relation_ready_goal_true"
    ] == 1
    assert dict(ExpressiveGoalEncoder().get_relations(domain))[
        "expressive_relation_ready_goal_false"
    ] == 1
    assert frombuffer_calls == 0
    assert encoded.flattened_relations["relation_ready"].numel() == 0
    assert encoded._native_relation_values is not None
    assert encoded._native_relation_values.numel() == 0
    assert encoded.flattened_relations["relation_ready"]._base is (
        encoded._native_relation_values
    )


def test_relation_view_keeps_exported_buffer_alive_by_itself() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")

    def make_relation_view() -> torch.Tensor:
        encoded = get_input_from_encoders(
            [(problem.initial_state,)],
            (StateEncoder(),),
            CPU,
        )
        return encoded.flattened_relations["relation_at"]

    relation = make_relation_view()
    gc.collect()

    # No EncodedTensors, RelationBuffer, or explicit base tensor reference is
    # left here.  PyTorch's storage must retain the buffer exporter itself.
    assert relation.tolist() == [4, 0, 5, 0]


def test_packed_relation_encoding_is_torch_serializable() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    encoded = get_input_from_encoders(
        [(problem.initial_state,)],
        (StateEncoder(),),
        CPU,
    )

    destination = io.BytesIO()
    torch.save(encoded, destination)
    destination.seek(0)
    restored = torch.load(destination, weights_only=False)

    assert restored.flattened_relations["relation_at"].tolist() == [4, 0, 5, 0]
    assert restored._native_relation_values is not None
    assert (
        restored.flattened_relations["relation_at"].untyped_storage().data_ptr()
        == restored._native_relation_values.untyped_storage().data_ptr()
    )


def test_custom_relation_collision_preserves_custom_then_native_order() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state

    encoded = get_input_from_encoders(
        [(state, 90), (state, 91)],
        (StateEncoder(), _CustomCollisionEncoder()),
        CPU,
    )

    assert encoded.flattened_relations["relation_room"].tolist() == [
        90,
        91,
        0,
        1,
        6,
        7,
    ]
    assert encoded._native_relation_values is not None
    # A collision requires one concatenation, while untouched native
    # relations remain views into the packed allocation.
    assert encoded.flattened_relations["relation_free"].untyped_storage().data_ptr() == (
        encoded._native_relation_values.untyped_storage().data_ptr()
    )


def test_context_closes_before_tensor_conversion_and_preserves_custom_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    context_type = mm.learning.EncodingContext
    original_close = context_type.close
    original_relations_to_tensors = encoders_module.relations_to_tensors
    original_frombuffer = torch.frombuffer
    contexts: list[EncodingContext] = []
    events: list[tuple[str, EncodingContext | None]] = []

    def make_context() -> EncodingContext:
        context = context_type()
        contexts.append(context)
        return context

    def close(context: EncodingContext) -> None:
        events.append(("close", context))
        original_close(context)

    def convert_relations(
        relations: dict[str, list[int]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        events.append(("tensor", None))
        return original_relations_to_tensors(relations, device)

    def frombuffer(
        buffer: Any,
        *,
        dtype: torch.dtype,
        count: int = -1,
        offset: int = 0,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        assert contexts[0]._handle == 0
        events.append(("frombuffer", None))
        return original_frombuffer(
            buffer,
            dtype=dtype,
            count=count,
            offset=offset,
            requires_grad=requires_grad,
        )

    monkeypatch.setattr(context_type, "close", close)
    monkeypatch.setattr(encoders_module, "EncodingContext", make_context)
    monkeypatch.setattr(
        encoders_module,
        "relations_to_tensors",
        convert_relations,
    )
    monkeypatch.setattr(encoders_module.torch, "frombuffer", frombuffer)

    encoded = get_input_from_encoders(
        [(problem.initial_state, "shared-context")],
        (StateEncoder(), _CustomAuxiliaryEncoder()),
        CPU,
    )

    assert len(contexts) == 1
    assert events == [
        ("close", contexts[0]),
        ("tensor", None),
        ("frombuffer", None),
    ]
    assert encoded.flattened_relations["custom_auxiliary"].tolist() == [6]
    assert encoded.auxiliary_indices.tolist() == [6]


def test_context_closes_exactly_once_when_an_encoder_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    context_type = mm.learning.EncodingContext
    original_close = context_type.close
    contexts: list[EncodingContext] = []
    closed_contexts: list[EncodingContext] = []

    def make_context() -> EncodingContext:
        context = context_type()
        contexts.append(context)
        return context

    def close(context: EncodingContext) -> None:
        closed_contexts.append(context)
        original_close(context)

    def fail_encoding(*args: object, **kwargs: object) -> None:
        raise RuntimeError("native state encoding failed")

    monkeypatch.setattr(context_type, "close", close)
    monkeypatch.setattr(encoders_module, "EncodingContext", make_context)
    monkeypatch.setattr(mm.learning, "encode_state", fail_encoding)

    with pytest.raises(RuntimeError, match="native state encoding failed") as error:
        get_input_from_encoders(
            [(problem.initial_state,)],
            (StateEncoder(),),
            CPU,
        )

    # Retaining ExceptionInfo keeps the traceback frames alive. Cleanup must
    # therefore have happened through the context manager, not finalization.
    assert error.traceback is not None
    assert len(contexts) == 1
    assert closed_contexts == contexts


def test_goal_encoder_calls_native_with_each_instance_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    true_goal = problem.ground_condition(
        problem.ground_literal(problem.fact("at", "ball2", "rooma"))
    )
    original_encode_goal = mm.learning.encode_goal
    native_inputs: list[tuple[mm.State, mm.GroundConjunctiveCondition]] = []

    def recording_encode_goal(*args: object, **kwargs: object) -> None:
        native_inputs.append((args[1], args[2]))  # type: ignore[arg-type]
        original_encode_goal(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mm.learning, "encode_goal", recording_encode_goal)
    input_specification = (StateEncoder(), GoalEncoder())
    model_input = [
        (state, true_goal),
        (state, problem.goal),
    ]

    actual = get_input_from_encoders(model_input, input_specification, CPU)
    assert _relation_rows(
        actual.flattened_relations["relation_at_goal_true"],
        2,
    ) == [(5, 0)]
    assert _relation_rows(
        actual.flattened_relations["relation_at_goal_false"],
        2,
    ) == [(11, 7)]
    assert native_inputs == [(state, true_goal), (state, problem.goal)]


def test_state_encoder_calls_native_when_mixed_with_virtual_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, full_model_input = _gripper_batch()
    original_encode_state = mm.learning.encode_state
    native_calls = 0

    def counting_encode_state(*args: object, **kwargs: object) -> None:
        nonlocal native_calls
        native_calls += 1
        original_encode_state(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mm.learning, "encode_state", counting_encode_state)
    encoded = get_input_from_encoders(
        [
            (None, state)
            for state, _, _ in full_model_input
        ],
        (VirtualNodeEncoder(), StateEncoder()),
        CPU,
    )
    assert native_calls == 2
    assert encoded.node_sizes.tolist() == [7, 7]
    assert encoded.object_indices.tolist() == [*range(0, 6), *range(7, 13)]
    assert encoded.virtual_indices.tolist() == [6, 13]
    assert _relation_rows(encoded.flattened_relations["relation_room"], 1) == [
        (0,),
        (1,),
        (7,),
        (8,),
    ]


def test_repeated_ground_actions_mix_with_custom_encoder_in_specification_order(
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    actions = state.applicable_actions()
    encoded = get_input_from_encoders(
        [(actions, None, actions, state)],
        (
            GroundActionsEncoder(suffix="_first"),
            _CustomAuxiliaryEncoder(),
            GroundActionsEncoder(suffix="_second"),
            StateEncoder(),
        ),
        CPU,
    )
    assert encoded.action_sizes.tolist() == [12]
    assert encoded.action_indices.tolist() == [
        *range(6, 12),
        *range(13, 19),
    ]
    assert encoded.auxiliary_indices.tolist() == [12]
    assert encoded.flattened_relations["custom_auxiliary"].tolist() == [12]
    assert encoded.node_sizes.tolist() == [19]
    for suffix, expected_ids in (
        ("_first", set(range(6, 12))),
        ("_second", set(range(13, 19))),
    ):
        observed_ids: set[int] = set()
        for action_schema in domain.actions:
            relation_name = f"action_{action_schema.name}{suffix}"
            relation = encoded.flattened_relations.get(relation_name)
            if relation is not None:
                observed_ids.update(
                    row[0]
                    for row in _relation_rows(relation, action_schema.arity + 1)
                )
        assert observed_ids == expected_ids


def test_transition_and_native_actions_allocate_in_encoder_order() -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    actions = state.applicable_actions()
    move = problem.action("move", "rooma", "roomb")
    successor = move.apply(state)
    transition_input = ([successor], (), problem.goal)

    encoded = get_input_from_encoders(
        [(transition_input, state, actions)],
        (TransitionEffectsEncoder(), StateEncoder(), GroundActionsEncoder()),
        CPU,
    )

    assert encoded.action_indices.tolist() == [*range(6, 13)]
    assert encoded.flattened_relations["at-robby_pos"].tolist() == [6, 1]
    assert encoded.flattened_relations["at-robby_neg"].tolist() == [6, 0]
    native_action_ids = {
        row[0]
        for action_schema in domain.actions
        if f"action_{action_schema.name}" in encoded.flattened_relations
        for row in _relation_rows(
            encoded.flattened_relations[f"action_{action_schema.name}"],
            action_schema.arity + 1,
        )
    }
    assert native_action_ids == set(range(7, 13))


def test_each_native_encoder_propagates_native_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = mm.Domain.from_file(DATA_DIR / "gripper" / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / "gripper" / "problem.pddl")
    state = problem.initial_state
    actions = state.applicable_actions()
    move = problem.action("move", "rooma", "roomb")
    successor = move.apply(state)
    transition_input = ([successor], (), problem.goal)

    cases = (
        ("encode_state", (StateEncoder(),), (state,)),
        (
            "encode_goal",
            (StateEncoder(), GoalEncoder()),
            (state, problem.goal),
        ),
        (
            "encode_action_list",
            (StateEncoder(), GroundActionsEncoder()),
            (state, actions),
        ),
        (
            "encode_transition_effects",
            (StateEncoder(), TransitionEffectsEncoder()),
            (state, transition_input),
        ),
        (
            "encode_virtual_node",
            (StateEncoder(), VirtualNodeEncoder()),
            (state, None),
        ),
        (
            "encode_expressive_state",
            (ExpressiveStateEncoder(),),
            (state,),
        ),
        (
            "encode_expressive_goal",
            (StateEncoder(), ExpressiveGoalEncoder()),
            (state, problem.goal),
        ),
    )
    for function_name, specification, instance in cases:
        def fail_encoding(*args: object, **kwargs: object) -> None:
            raise RuntimeError(f"{function_name} failed")

        with monkeypatch.context() as scoped:
            scoped.setattr(mm.learning, function_name, fail_encoding)
            with pytest.raises(RuntimeError, match=f"{function_name} failed"):
                get_input_from_encoders([instance], specification, CPU)
