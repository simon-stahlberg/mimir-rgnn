"""Tests of the generic model support for plan-space (partial-order plan) search.

A stub encoder stands in for the native partial-plan encoder: per instance it
allocates auxiliary "flaw" nodes with several "resolver" nodes each, writes the
relation ``resolves(resolver, flaw)`` with unary marks, and lists the flaw nodes
under ``extras``.
"""

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    Encoder,
    ForwardState,
    GRUUpdates,
    GlobalScalarDecoder,
    GoalEncoder,
    GroupedLogProbs,
    GroupedSoftmaxDecoder,
    HyperparameterConfig,
    MLPUpdates,
    MeanAggregation,
    ModuleConfig,
    NodeScalarDecoder,
    NodeStateStore,
    PredicateMLPMessages,
    RelationalGraphNeuralNetwork,
    StateEncoder,
    SumAggregation,
)
from pymimir_rgnn.bases import EncodedLists, EncodingContext
from pymimir_rgnn.encoders import get_input_from_encoders
from pymimir_rgnn.model import MIN_REPLAYED_LAYERS, RelationalLayerStackModule


DATA_DIR = Path(__file__).parent / "data"
CPU = torch.device("cpu")

# A partial-plan stub: every flaw name maps to its number of resolvers.
FlawSpec = list[tuple[str, int]]


class _PartialPlanStubEncoder(Encoder):
    """Allocates flaw and resolver auxiliary nodes and the relation between them."""

    RESOLVES = "resolves"
    FLAW_MARK = "flaw_mark"
    RESOLVER_MARK = "resolver_mark"
    FLAW_NODES = "flaw_nodes"

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [(self.RESOLVES, 2), (self.FLAW_MARK, 1), (self.RESOLVER_MARK, 1)]

    def encode(self, input_value: Any, state: mm.State, encoding: EncodedLists, context: EncodingContext) -> None:
        flaws: FlawSpec = input_value
        flaw_ids: list[int] = []
        for flaw_name, resolver_count in flaws:
            flaw_id = context.new_or_existing_auxiliary_id(("flaw", flaw_name))
            flaw_ids.append(flaw_id)
            encoding.flattened_relations.setdefault(self.FLAW_MARK, []).append(flaw_id)
            for resolver_index in range(resolver_count):
                resolver_id = context.new_or_existing_auxiliary_id(("resolver", flaw_name, resolver_index))
                encoding.flattened_relations.setdefault(self.RESOLVER_MARK, []).append(resolver_id)
                encoding.flattened_relations.setdefault(self.RESOLVES, []).extend((resolver_id, flaw_id))
        encoding.extras.setdefault(self.FLAW_NODES, []).append(torch.tensor(flaw_ids, dtype=torch.long))


def _node_keys_and_types(problem: mm.Problem, instance_key: Hashable, flaws: FlawSpec) -> tuple[list[tuple[Hashable, Hashable] | None], list[str]]:
    """Keys and types aligned with the node order of (StateEncoder, GoalEncoder, stub) for one instance."""
    keys: list[tuple[Hashable, Hashable] | None] = [None] * len(problem.all_objects)
    types = ["object"] * len(problem.all_objects)
    for flaw_name, resolver_count in flaws:
        keys.append((instance_key, ("flaw", flaw_name)))
        types.append("flaw")
        for resolver_index in range(resolver_count):
            keys.append((instance_key, ("resolver", flaw_name, resolver_index)))
            types.append("resolver")
    return keys, types


def _load(domain_name: str) -> tuple[mm.Domain, mm.Problem]:
    domain = mm.Domain.from_file(DATA_DIR / domain_name / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / domain_name / "problem.pddl")
    return domain, problem


def _input_spec() -> tuple[Encoder, ...]:
    return (StateEncoder(), GoalEncoder(), _PartialPlanStubEncoder())


def _model(
    domain: mm.Domain,
    num_layers: int = 2,
    embedding_size: int = 8,
    update_function: type = MLPUpdates,
    aggregation_function: Any = None,
    **config: Any,
) -> RelationalGraphNeuralNetwork:
    hparam_config = HyperparameterConfig(domain=domain, num_layers=num_layers, embedding_size=embedding_size, **config)
    input_spec = _input_spec()
    output_spec = [
        ("resolver_choice", GroupedSoftmaxDecoder(hparam_config, _PartialPlanStubEncoder.RESOLVES)),
        ("flaw_values", NodeScalarDecoder(hparam_config, _PartialPlanStubEncoder.FLAW_NODES)),
        ("plan_value", GlobalScalarDecoder(hparam_config)),
        ("plan_value_all_nodes", GlobalScalarDecoder(hparam_config, nodes="all")),
    ]
    module_config = ModuleConfig(
        aggregation_function=MeanAggregation() if aggregation_function is None else aggregation_function,
        message_function=PredicateMLPMessages(hparam_config, input_spec),
        update_function=update_function(hparam_config),
    )
    return RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)


FIRST_FLAWS: FlawSpec = [("open-a", 3), ("open-b", 2)]
SECOND_FLAWS: FlawSpec = [("open-c", 4), ("threat-d", 1), ("open-e", 2)]


def _batch(problem: mm.Problem) -> list[tuple]:
    state = problem.initial_state
    return [(state, problem.goal, FIRST_FLAWS), (state, problem.goal, SECOND_FLAWS)]


def test_extras_are_carried_from_lists_to_tensors() -> None:
    _, problem = _load("blocks")
    encoded = get_input_from_encoders(_batch(problem), _input_spec(), CPU)
    flaw_nodes = encoded.extras[_PartialPlanStubEncoder.FLAW_NODES]
    assert len(flaw_nodes) == 2
    object_count = len(problem.all_objects)
    # Flaws and their resolvers are allocated in order after the objects of each instance.
    assert flaw_nodes[0].tolist() == [object_count, object_count + 4]
    first_size = int(encoded.node_sizes[0])
    assert first_size == object_count + 2 + 3 + 2
    assert flaw_nodes[1].tolist() == [first_size + object_count, first_size + object_count + 5, first_size + object_count + 7]
    assert encoded.auxiliary_sizes.tolist() == [7, 10]


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("include_group_embedding", [True, False])
def test_grouped_log_probs_normalize_per_group_and_split_by_instance(domain_name: str, include_group_embedding: bool) -> None:
    domain, problem = _load(domain_name)
    model = _model(domain)
    if not include_group_embedding:
        model._readouts["resolver_choice"] = GroupedSoftmaxDecoder(  # type: ignore[index]
            model.get_hparam_config(), _PartialPlanStubEncoder.RESOLVES, include_group_embedding=False)
    output = model.forward(_batch(problem))
    grouped = output.readout("resolver_choice")
    assert isinstance(grouped, list) and len(grouped) == 2
    encoding = output.encoding
    boundaries = [0, *encoding.node_sizes.cumsum(0).tolist()]
    for instance_index, (result, flaws) in enumerate(zip(grouped, (FIRST_FLAWS, SECOND_FLAWS))):
        assert isinstance(result, GroupedLogProbs)
        expected_rows = sum(count for _, count in flaws)
        assert result.member_indices.shape == (expected_rows,)
        assert result.group_indices.shape == (expected_rows,)
        assert result.log_probs.shape == (expected_rows,)
        assert result.member_indices.dtype == torch.long and result.group_indices.dtype == torch.long
        low, high = boundaries[instance_index], boundaries[instance_index + 1]
        assert all(low <= node_id < high for node_id in result.member_indices.tolist())
        assert all(low <= node_id < high for node_id in result.group_indices.tolist())
        assert len(set(result.group_indices.tolist())) == len(flaws)
        # Original relation row order: the resolvers of a flaw are consecutive and the flaws follow the spec.
        group_order = list(dict.fromkeys(result.group_indices.tolist()))
        assert group_order == sorted(group_order)
        sums = torch.zeros(encoding.node_count).index_add(0, result.group_indices, result.log_probs.exp())
        assert torch.allclose(sums[result.group_indices], torch.ones(expected_rows), atol=1e-5)
        chosen = result.argmax_per_group()
        sampled = result.sample_per_group(torch.Generator().manual_seed(0))
        assert set(chosen) == set(result.group_indices.tolist()) == set(sampled)
        members_of = {int(g): set() for g in result.group_indices.tolist()}  # type: ignore[var-annotated]
        for member, group in zip(result.member_indices.tolist(), result.group_indices.tolist()):
            members_of[group].add(member)
        for group, member in chosen.items():
            assert member in members_of[group]
        for group, member in sampled.items():
            assert member in members_of[group]


def test_grouped_log_probs_are_stable_for_large_scores() -> None:
    domain, problem = _load("blocks")
    model = _model(domain)
    encoded = get_input_from_encoders(_batch(problem), _input_spec(), CPU)
    decoder = GroupedSoftmaxDecoder(model.get_hparam_config(), _PartialPlanStubEncoder.RESOLVES)
    embeddings = torch.randn(encoded.node_count, model.get_hparam_config().embedding_size) * 1e3
    for result in decoder(embeddings, encoded):
        assert torch.isfinite(result.log_probs).all()
        sums = torch.zeros(encoded.node_count).index_add(0, result.group_indices, result.log_probs.exp())
        assert torch.allclose(sums[result.group_indices], torch.ones_like(result.log_probs), atol=1e-5)


def test_grouped_decoder_without_relation_rows_returns_empty_results() -> None:
    domain, problem = _load("blocks")
    model = _model(domain)
    state = problem.initial_state
    grouped = model.forward([(state, problem.goal, []), (state, problem.goal, [])]).readout("resolver_choice")
    assert len(grouped) == 2
    for result in grouped:
        assert result.member_indices.numel() == 0 and result.log_probs.numel() == 0
        assert result.argmax_per_group() == {}


def test_grouped_decoder_single_instance() -> None:
    domain, problem = _load("gripper")
    model = _model(domain)
    grouped = model.forward([(problem.initial_state, problem.goal, SECOND_FLAWS)]).readout("resolver_choice")
    assert len(grouped) == 1
    assert grouped[0].log_probs.shape == (7,)


def test_node_and_global_scalar_decoder_shapes() -> None:
    domain, problem = _load("blocks")
    model = _model(domain)
    output = model.forward(_batch(problem))
    flaw_values = output.readout("flaw_values")
    assert isinstance(flaw_values, list) and len(flaw_values) == 2
    assert flaw_values[0].shape == (len(FIRST_FLAWS),)
    assert flaw_values[1].shape == (len(SECOND_FLAWS),)
    for name in ("plan_value", "plan_value_all_nodes"):
        value = output.readout(name)
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    single = model.forward([(problem.initial_state, problem.goal, FIRST_FLAWS)])
    assert single.readout("flaw_values")[0].shape == (len(FIRST_FLAWS),)
    assert single.readout("plan_value").shape == (1,)


def test_node_scalar_decoder_accepts_lists_and_empty_instances() -> None:
    domain, problem = _load("blocks")
    model = _model(domain)
    state = problem.initial_state
    encoded = get_input_from_encoders([(state, problem.goal, FIRST_FLAWS), (state, problem.goal, [])], _input_spec(), CPU)
    encoded.extras[_PartialPlanStubEncoder.FLAW_NODES] = [
        encoded.extras[_PartialPlanStubEncoder.FLAW_NODES][0].tolist(),
        [],
    ]
    values = model.forward_encoded(encoded).readout("flaw_values")
    assert values[0].shape == (len(FIRST_FLAWS),)
    assert values[1].shape == (0,)


def test_forward_state_exposes_node_embeddings_and_encoding() -> None:
    domain, problem = _load("blocks")
    model = _model(domain, embedding_size=6)
    output = model.forward(_batch(problem))
    assert isinstance(output, ForwardState)
    assert output.node_embeddings.shape == (output.encoding.node_count, 6)
    assert output.encoding.node_sizes.numel() == 2
    seen: list[tuple[int, int]] = []
    model.add_hook(lambda state: seen.append((state.get_layer_index(), state.node_embeddings.shape[0])))
    model.forward(_batch(problem))
    model.clear_hooks()
    assert seen == [(0, output.encoding.node_count), (1, output.encoding.node_count)]


def test_initial_embeddings_change_the_output_and_receive_gradients() -> None:
    domain, problem = _load("blocks")
    torch.manual_seed(0)
    model = _model(domain)
    batch = _batch(problem)
    encoded = model.encode(batch)
    embedding_size = model.get_hparam_config().embedding_size
    initial = torch.randn(encoded.node_count, embedding_size, requires_grad=True)

    zero_output = model.forward(batch)
    carried_output = model.forward(batch, initial)
    assert not torch.allclose(zero_output.readout("plan_value"), carried_output.readout("plan_value"))
    assert not torch.allclose(zero_output.node_embeddings, carried_output.node_embeddings)
    explicit_zero_output = model.forward(batch, torch.zeros(encoded.node_count, embedding_size))
    assert torch.allclose(zero_output.readout("plan_value"), explicit_zero_output.readout("plan_value"))

    carried_output.readout("plan_value").sum().backward()
    assert initial.grad is not None
    assert initial.grad.shape == initial.shape
    assert initial.grad.abs().sum() > 0

    curried = model.curry_forward(batch)
    assert torch.allclose(curried().readout("plan_value"), zero_output.readout("plan_value"))
    assert torch.allclose(curried(initial.detach()).readout("plan_value"), carried_output.readout("plan_value").detach())
    with pytest.raises(AssertionError):
        model.forward(batch, torch.zeros(encoded.node_count + 1, embedding_size))


@pytest.mark.parametrize("num_layers", [1, MIN_REPLAYED_LAYERS + 1])
@pytest.mark.parametrize("grad", [True, False])
def test_gru_updates_run_with_and_without_replay_conditions(num_layers: int, grad: bool) -> None:
    assert GRUUpdates.supports_replay
    domain, problem = _load("gripper")
    model = _model(domain, num_layers=num_layers, update_function=GRUUpdates, residual_updates=False)
    with torch.set_grad_enabled(grad):
        output = model.forward(_batch(problem))
        value = output.readout("plan_value")
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert value.requires_grad == grad
    if grad:
        value.sum().backward()
        gru = model.get_module_config().update_function._update  # type: ignore[attr-defined]
        assert isinstance(gru, torch.nn.GRUCell)
        assert gru.weight_ih.grad is not None and gru.weight_ih.grad.abs().sum() > 0


def test_gru_updates_match_a_gru_cell_step() -> None:
    domain, _ = _load("blocks")
    hparam_config = HyperparameterConfig(domain=domain, embedding_size=5)
    updates = GRUUpdates(hparam_config)
    hidden = torch.randn(7, 5)
    messages = torch.randn(7, 5)
    assert torch.allclose(updates(hidden, messages), updates._update(messages, hidden))
    assert updates(torch.empty(0, 5), torch.empty(0, 5)).shape == (0, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="replaying layers needs CUDA")
def test_gru_updates_and_initial_embeddings_replay_like_running_every_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    domain, problem = _load("gripper")
    model = _model(domain, num_layers=MIN_REPLAYED_LAYERS + 2, update_function=GRUUpdates, residual_updates=False).to("cuda")
    batch = _batch(problem)
    encoded = model.encode(batch)
    initial = torch.randn(encoded.node_count, model.get_hparam_config().embedding_size, device="cuda")
    replays = 0
    replay_layers = RelationalLayerStackModule._replay_layers

    def counting_replay_layers(self: RelationalLayerStackModule, input: Any, node_embeddings: torch.Tensor) -> torch.Tensor:
        nonlocal replays
        replays += 1
        return replay_layers(self, input, node_embeddings)

    monkeypatch.setattr(RelationalLayerStackModule, "_replay_layers", counting_replay_layers)
    with torch.no_grad():
        replayed = model.forward_encoded(encoded, initial).readout("plan_value")
    eager = model.forward_encoded(encoded, initial).readout("plan_value")
    assert replays == 1
    assert torch.allclose(replayed, eager.detach(), atol=1e-5)


def test_node_state_store_round_trips_keys_and_uses_type_vectors_for_new_keys() -> None:
    domain, problem = _load("blocks")
    torch.manual_seed(1)
    model = _model(domain, embedding_size=8, update_function=GRUUpdates, residual_updates=False)
    store = NodeStateStore(8, ["object", "flaw", "resolver"])
    state = problem.initial_state

    first_batch = [(state, problem.goal, FIRST_FLAWS)]
    first_encoded = model.encode(first_batch)
    first_keys, first_types = _node_keys_and_types(problem, "plan-1", FIRST_FLAWS)
    initial = store.build_initial(first_encoded, [first_keys], [first_types])
    assert initial.shape == (first_encoded.node_count, 8)
    # Nothing is stored yet: every node starts from the vector of its type.
    type_vectors = store._type_vectors
    for node_id, type_name in enumerate(first_types):
        assert torch.equal(initial[node_id], type_vectors[store.type_names.index(type_name)])
    assert initial.requires_grad  # The type vectors are learned.

    first_output = model.forward_encoded(first_encoded, initial)
    store.update(first_encoded, [first_keys], first_output.node_embeddings)
    assert len(store) == len([key for key in first_keys if key is not None])
    stored = store.get(("plan-1", ("flaw", "open-a")))
    assert stored is not None and not stored.requires_grad
    object_count = len(problem.all_objects)
    assert torch.equal(stored, first_output.node_embeddings[object_count].detach())

    # The next forward pass sees the first instance with one more flaw, and a new instance.
    grown_flaws: FlawSpec = [*FIRST_FLAWS, ("open-new", 2)]
    second_batch = [(state, problem.goal, grown_flaws), (state, problem.goal, SECOND_FLAWS)]
    second_encoded = model.encode(second_batch)
    grown_keys, grown_types = _node_keys_and_types(problem, "plan-1", grown_flaws)
    other_keys, other_types = _node_keys_and_types(problem, "plan-2", SECOND_FLAWS)
    initial = store.build_initial(second_encoded, [grown_keys, other_keys], [grown_types, other_types])
    assert initial.shape == (second_encoded.node_count, 8)
    first_size = int(second_encoded.node_sizes[0])
    for node_id, (key, type_name) in enumerate(zip(grown_keys + other_keys, grown_types + other_types)):
        if key is not None and key in store:
            assert torch.equal(initial[node_id], first_output.node_embeddings[node_id].detach())
        else:
            assert torch.equal(initial[node_id], type_vectors[store.type_names.index(type_name)])
    assert ("plan-1", ("flaw", "open-new")) not in store
    assert ("plan-2", ("flaw", "open-c")) not in store
    # The old flaws of the first instance keep their stored rows; the second instance only sees type vectors.
    assert torch.equal(initial[object_count], stored)
    assert torch.equal(initial[first_size + object_count], type_vectors[store.type_names.index("flaw")])

    second_output = model.forward_encoded(second_encoded, initial)
    store.update(second_encoded, [grown_keys, other_keys], second_output.node_embeddings)
    assert ("plan-2", ("flaw", "open-c")) in store
    store.forget("plan-2")
    assert ("plan-2", ("flaw", "open-c")) not in store
    assert ("plan-1", ("flaw", "open-a")) in store
    store.clear()
    assert len(store) == 0


def test_node_state_store_keeps_graph_when_not_detached() -> None:
    domain, problem = _load("blocks")
    model = _model(domain, embedding_size=4)
    store = NodeStateStore(4, ["object", "flaw", "resolver"], detach=False)
    batch = [(problem.initial_state, problem.goal, FIRST_FLAWS)]
    encoded = model.encode(batch)
    keys, types = _node_keys_and_types(problem, "plan", FIRST_FLAWS)
    output = model.forward_encoded(encoded, store.build_initial(encoded, [keys], [types]))
    store.update(encoded, [keys], output.node_embeddings)
    second_initial = store.build_initial(encoded, [keys], [types])
    assert second_initial.requires_grad
    model.forward_encoded(encoded, second_initial).readout("plan_value").sum().backward()
    assert store._type_vectors.grad is not None
    assert store._type_vectors.grad.abs().sum() > 0


def test_node_state_store_rejects_misaligned_keys() -> None:
    domain, problem = _load("blocks")
    model = _model(domain, embedding_size=4)
    store = NodeStateStore(4, ["object", "flaw", "resolver"])
    encoded = model.encode(_batch(problem))
    keys, types = _node_keys_and_types(problem, "plan", FIRST_FLAWS)
    with pytest.raises(AssertionError):
        store.build_initial(encoded, [keys], [types])
    with pytest.raises(AssertionError):
        store.build_initial(encoded, [keys, keys[:-1]], [types, types[:-1]])
