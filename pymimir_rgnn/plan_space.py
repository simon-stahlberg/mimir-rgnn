"""Plan-space local search with a neural policy over partial-order plans.

A partial-order causal-link plan is edited round by round, ANYCSP style: every
flaw of the plan picks one of its resolvers (index 0 is the NoOp resolver) and
every step gets a delete decision. The plan is a solution once no flaws remain.
This module provides the encoder that turns a plan and its flaws into graph
relations, a policy module with the two heads, a search driver, the oracle
rollout that generates imitation data from reference plans, and the trainer.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymimir as mm
import torch
import torch.nn as nn
from pymimir.heuristics import Heuristic

from .aggregations import HardMaximumAggregation, MeanAggregation, SmoothMaximumAggregation, SumAggregation
from .bases import AggregationFunction, EncodedLists, EncodedTensors, Encoder, EncodingContext, UpdateFunction
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import GroupedLogProbs, GroupedSoftmaxDecoder, NodeScalarDecoder
from .encoders import GoalEncoder, StateEncoder
from .messages import PredicateMLPMessages
from .model import RelationalGraphNeuralNetwork
from .recurrence import NodeKey, NodeStateStore
from .updates import GRUUpdates, MLPUpdates


# Node types of a plan-space graph; every type has a learned initial vector in the recurrent variant. 'other' covers
# nodes allocated by encoders that are not known to the plan encoder, such as a virtual node.
PLAN_NODE_TYPES: tuple[str, ...] = (
    'object', 'step', 'param', 'link', 'ghost', 'ghost_param', 'flaw', 'resolver', 'tombstone', 'other',
)

PlanInput = tuple[mm.PartialOrderPlan, mm.FlawSet]
PolicyItem = tuple[mm.State, mm.GroundConjunctiveCondition, mm.PartialOrderPlan, mm.FlawSet]
StepSignature = tuple[str, tuple[str, ...]]


def _step_signature(step: mm.PlanStep) -> StepSignature | None:
    """The (schema, binding) of a fully bound Step-kind step, or None for any other step."""
    if step.kind is not mm.StepKind.STEP or not step.is_fully_bound or step.schema is None:
        return None
    return step.schema.name, tuple(str(value) for value in step.binding)


class PartialPlanEncoder(Encoder):
    """Encoder for a partial-order plan together with its flaws and resolvers.

    The input value of an instance is the pair ``(plan, flaws)`` with ``flaws``
    computed from ``plan``. The relations come from the native
    :func:`pymimir.encode_partial_plan`; this encoder records under
    ``encoding.extras``, one entry per instance:

    - ``plan_layouts``: the :class:`pymimir.PlanEncodingLayout` with the node ids,
    - ``plan_node_keys``: per node of the instance, a key for persistent nodes
      (``('object', i)``, ``('step', id)``, ``('param', step_id, index)``,
      ``('link', id)``) and None for ghosts, flaws, resolvers and tombstones,
    - ``plan_node_types``: per node of the instance, one of :data:`PLAN_NODE_TYPES`,
    - ``plan_step_nodes``: LongTensor of the node ids of the Step-kind steps, in
      the order of ``plan.steps``, for :class:`NodeScalarDecoder`.

    The keys and types cover the nodes allocated up to and including this
    encoder, so it must be the last node-allocating encoder of the input spec.
    """

    LAYOUTS = 'plan_layouts'
    NODE_KEYS = 'plan_node_keys'
    NODE_TYPES = 'plan_node_types'
    STEP_NODES = 'plan_step_nodes'

    def __init__(self, suffix: str = '') -> None:
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    @property
    def resolves_relation(self) -> str:
        """The name of the ``(resolver, flaw)`` relation the resolver head scores."""
        return f'plan_resolves{self.suffix}'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return mm.partial_plan_relations(domain, self.suffix)

    def encode(self, input_value: Any, state: mm.State, encoding: EncodedLists, context: EncodingContext) -> None:
        assert isinstance(input_value, tuple) and len(input_value) == 2, (
            f'PartialPlanEncoder expected a (plan, flaws) pair, got {type(input_value)}'
        )
        plan, flaws = input_value
        assert isinstance(plan, mm.PartialOrderPlan), f'PartialPlanEncoder expected a PartialOrderPlan, got {type(plan)}'
        assert isinstance(flaws, mm.FlawSet), f'PartialPlanEncoder expected a FlawSet, got {type(flaws)}'
        assert flaws.plan is plan, 'The flaws must be computed from the plan of the same instance.'
        assert plan.problem is state.problem, 'The plan must belong to the problem of the state.'
        layout = mm.encode_partial_plan(context, plan, flaws, self.suffix)

        offset = context.id_offset
        node_count = context.get_node_count()
        keys: list[Hashable | None] = [None] * node_count
        types: list[str] = ['other'] * node_count

        def put(node_id: int, key: Hashable | None, type_name: str) -> None:
            local = node_id - offset
            assert 0 <= local < node_count, f'Node id {node_id} is outside the current instance.'
            keys[local] = key
            types[local] = type_name

        object_to_id = context.object_to_id
        for index, value in enumerate(plan.problem.all_objects):
            put(object_to_id[value], ('object', index), 'object')
        steps = plan.steps
        assert len(steps) == len(layout.step_node_ids), 'The layout must list every step of the plan.'
        for step, node_id, parameter_ids in zip(steps, layout.step_node_ids, layout.parameter_node_ids):
            put(node_id, ('step', step.id), 'step')
            for parameter_index, parameter_id in enumerate(parameter_ids):
                put(parameter_id, ('param', step.id, parameter_index), 'param')
        links = plan.links
        assert len(links) == len(layout.link_node_ids), 'The layout must list every link of the plan.'
        for link, node_id in zip(links, layout.link_node_ids):
            put(node_id, ('link', link.id), 'link')
        for node_id, parameter_ids in zip(layout.ghost_node_ids, layout.ghost_parameter_node_ids):
            put(node_id, None, 'ghost')
            for parameter_id in parameter_ids:
                put(parameter_id, None, 'ghost_param')
        for node_id, resolver_ids in zip(layout.flaw_node_ids, layout.resolver_node_ids):
            put(node_id, None, 'flaw')
            for resolver_id in resolver_ids:
                put(resolver_id, None, 'resolver')
        for node_id in layout.tombstone_node_ids:
            put(node_id, None, 'tombstone')

        step_nodes = [node_id for step, node_id in zip(steps, layout.step_node_ids) if step.kind is mm.StepKind.STEP]
        encoding.extras.setdefault(self.LAYOUTS, []).append(layout)
        encoding.extras.setdefault(self.NODE_KEYS, []).append(keys)
        encoding.extras.setdefault(self.NODE_TYPES, []).append(types)
        encoding.extras.setdefault(self.STEP_NODES, []).append(torch.tensor(step_nodes, dtype=torch.long))


@dataclass
class PolicyOutput:
    """The two heads of the policy for one instance.

    ``resolver_log_probs[i]`` is aligned with ``flaws.flaws[i].resolvers`` (index 0
    is NoOp), and ``delete_logits`` with the Step-kind steps of ``plan.steps``.
    """

    resolver_log_probs: list[torch.Tensor]
    delete_logits: torch.Tensor

    def greedy_choices(self) -> list[int]:
        """The most probable resolver index of every flaw."""
        return [int(log_probs.argmax()) if log_probs.numel() > 0 else 0 for log_probs in self.resolver_log_probs]

    def greedy_deletes(self) -> list[bool]:
        """Whether every Step-kind step is deleted, i.e. sigmoid(logit) > 0.5."""
        return [bool(value) for value in (self.delete_logits > 0).tolist()]

    def sample_choices(self, generator: torch.Generator, temperature: float = 1.0) -> list[int]:
        """Sample a resolver index per flaw from the tempered distribution."""
        choices: list[int] = []
        for log_probs in self.resolver_log_probs:
            if log_probs.numel() == 0:
                choices.append(0)
                continue
            probs = torch.softmax(log_probs.detach().cpu() / temperature, dim=0)
            choices.append(int(torch.multinomial(probs, 1, generator=generator)))
        return choices

    def sample_deletes(self, generator: torch.Generator, temperature: float = 1.0) -> list[bool]:
        """Sample a delete decision per Step-kind step from the tempered sigmoid."""
        probs = torch.sigmoid(self.delete_logits.detach().cpu() / temperature)
        draws = torch.rand(probs.shape, generator=generator)
        return [bool(value) for value in (draws < probs).tolist()]


@dataclass
class PolicyConfig:
    """Architecture flags of a :class:`PlanSpacePolicy`; serialized into checkpoints."""

    embedding_size: int = 32
    num_layers: int = 10
    aggregation: Literal['mean', 'sum', 'max', 'smooth_max'] = 'mean'
    update: Literal['mlp', 'gru'] = 'mlp'
    recurrent: bool = False
    global_readout: bool = False
    normalize_updates: bool = True
    # None picks the natural default: residual updates for the MLP, none for the GRU, whose gates already carry the
    # embedding across layers.
    residual_updates: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(values: dict[str, Any]) -> PolicyConfig:
        return PolicyConfig(**values)


def _aggregation(name: str) -> AggregationFunction:
    if name == 'mean':
        return MeanAggregation()
    if name == 'sum':
        return SumAggregation()
    if name == 'max':
        return HardMaximumAggregation()
    if name == 'smooth_max':
        return SmoothMaximumAggregation()
    raise ValueError(f'Unknown aggregation {name!r}.')


def _update(name: str, hparams: HyperparameterConfig) -> UpdateFunction:
    if name == 'mlp':
        return MLPUpdates(hparams)
    if name == 'gru':
        return GRUUpdates(hparams)
    raise ValueError(f'Unknown update function {name!r}.')


class PlanSpacePolicy(nn.Module):
    """R-GNN policy over (state, goal, partial plan) with a resolver head and a delete head.

    With ``config.recurrent`` the policy owns a :class:`NodeStateStore` and
    :meth:`evaluate` with ``carried=True`` starts persistent plan nodes from the
    embeddings of the previous round of the same instance (detached).
    """

    def __init__(self, domain: mm.Domain, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = PolicyConfig() if config is None else config
        self.domain = domain
        self.plan_encoder = PartialPlanEncoder()
        input_spec: tuple[Encoder, ...] = (StateEncoder(), GoalEncoder(), self.plan_encoder)
        residual = self.config.residual_updates
        if residual is None:
            residual = self.config.update != 'gru'
        hparams = HyperparameterConfig(
            domain=domain,
            embedding_size=self.config.embedding_size,
            num_layers=self.config.num_layers,
            global_readout=self.config.global_readout,
            normalize_updates=self.config.normalize_updates,
            residual_updates=residual,
        )
        output_spec = [
            ('resolver_log_probs', GroupedSoftmaxDecoder(hparams, self.plan_encoder.resolves_relation)),
            ('delete_logits', NodeScalarDecoder(hparams, PartialPlanEncoder.STEP_NODES)),
        ]
        module_config = ModuleConfig(
            aggregation_function=_aggregation(self.config.aggregation),
            message_function=PredicateMLPMessages(hparams, input_spec),
            update_function=_update(self.config.update, hparams),
        )
        self.model = RelationalGraphNeuralNetwork(hparams, module_config, input_spec, output_spec)
        self.memory: NodeStateStore | None = None
        if self.config.recurrent:
            self.memory = NodeStateStore(self.config.embedding_size, PLAN_NODE_TYPES)

    @property
    def device(self) -> torch.device:
        return self.model.get_device()

    def forget(self, instance_key: Hashable) -> None:
        """Drop the carried embeddings of one instance."""
        if self.memory is not None:
            self.memory.forget(instance_key)

    def clear_memory(self) -> None:
        """Drop every carried embedding."""
        if self.memory is not None:
            self.memory.clear()

    @staticmethod
    def _node_keys(encoded: EncodedTensors, instance_keys: Sequence[Hashable]) -> list[list[NodeKey | None]]:
        per_instance: list[list[Hashable | None]] = encoded.extras[PartialPlanEncoder.NODE_KEYS]
        assert len(per_instance) == len(instance_keys), 'One instance key per item is required.'
        return [
            [None if key is None else (instance_key, key) for key in keys]
            for instance_key, keys in zip(instance_keys, per_instance)
        ]

    @staticmethod
    def _align_resolver_log_probs(grouped: GroupedLogProbs, layout: mm.PlanEncodingLayout, node_count: int) -> list[torch.Tensor]:
        """Reorder the rows of the grouped softmax to the flaws' resolver lists."""
        sizes = [len(resolver_ids) for resolver_ids in layout.resolver_node_ids]
        if not sizes:
            return []
        device = grouped.log_probs.device
        all_ids = torch.tensor([node_id for ids in layout.resolver_node_ids for node_id in ids], dtype=torch.long, device=device)
        row_of_node = torch.full((node_count,), -1, dtype=torch.long, device=device)
        row_of_node[grouped.member_indices] = torch.arange(grouped.member_indices.numel(), device=device)
        rows = row_of_node.index_select(0, all_ids)
        assert bool((rows >= 0).all()), 'Every resolver node must be scored by the resolver head.'
        return list(grouped.log_probs.index_select(0, rows).split(sizes))

    def evaluate(
        self,
        items: Sequence[PolicyItem],
        carried: bool = False,
        instance_keys: Sequence[Hashable] | None = None,
    ) -> list[PolicyOutput]:
        """Run the policy on a batch of (state, goal, plan, flaws) items.

        Args:
            items: The instances; ``flaws`` must be computed from ``plan``.
            carried: Whether to start from and store carried embeddings (recurrent policies only).
            instance_keys: The key of every item in the state store; defaults to ``id(plan)``.

        Returns:
            One :class:`PolicyOutput` per item.
        """
        batch: list[tuple] = [(state, goal, (plan, flaws)) for state, goal, plan, flaws in items]
        encoded = self.model.encode(batch)
        initial: torch.Tensor | None = None
        node_keys: list[list[NodeKey | None]] | None = None
        if carried:
            assert self.memory is not None, 'Carried embeddings need a recurrent policy.'
            keys = [id(plan) for _, _, plan, _ in items] if instance_keys is None else list(instance_keys)
            node_keys = self._node_keys(encoded, keys)
            initial = self.memory.build_initial(encoded, node_keys, encoded.extras[PartialPlanEncoder.NODE_TYPES])
        output = self.model.forward_encoded(encoded, initial)
        if carried:
            assert self.memory is not None and node_keys is not None
            self.memory.update(encoded, node_keys, output.node_embeddings)
        grouped: list[GroupedLogProbs] = output.readout('resolver_log_probs')
        deletes: list[torch.Tensor] = output.readout('delete_logits')
        layouts: list[mm.PlanEncodingLayout] = encoded.extras[PartialPlanEncoder.LAYOUTS]
        assert len(grouped) == len(deletes) == len(layouts) == len(items)
        return [
            PolicyOutput(self._align_resolver_log_probs(group, layout, encoded.node_count), delete_logits)
            for group, delete_logits, layout in zip(grouped, deletes, layouts)
        ]

    def save(self, path: Path | str, extras: dict[str, Any] | None = None) -> None:
        """Save the weights, the architecture config and optional extras (e.g. the optimizer state)."""
        torch.save({
            'config': self.config.to_dict(),
            'domain_name': self.domain.name,
            'state_dict': self.state_dict(),
            'extras': {} if extras is None else extras,
        }, path)

    @staticmethod
    def load(domain: mm.Domain, path: Path | str, device: torch.device | str = 'cpu') -> tuple[PlanSpacePolicy, dict[str, Any]]:
        """Load a policy saved with :meth:`save` for ``domain`` onto ``device``."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        policy = PlanSpacePolicy(domain, PolicyConfig.from_dict(checkpoint['config']))
        policy.load_state_dict(checkpoint['state_dict'])
        return policy.to(device), checkpoint['extras']


@dataclass
class SearchOutcome:
    """What one run of :class:`PlanSpaceSearch` did."""

    solved: bool
    rounds: int
    plan: list[mm.GroundAction] | None
    steps_final: int
    applied_total: int
    dropped_total: int
    created_total: int
    deleted_total: int
    # Created steps whose (schema, binding) equals an existing fully bound step at creation time.
    duplicate_creations: int
    # A (schema, binding) adopted again after a step with it was deleted earlier in the run.
    cycling_events: int
    wall_time: float
    visited: list[mm.PartialOrderPlan] = field(default_factory=list)
    # A clone of the plan after the last round (solved or not), e.g. for ``prune_to_achieved_goal``.
    final_plan: mm.PartialOrderPlan | None = None
    # How many independent attempts (restarts) this outcome aggregates.
    attempts: int = 1

    @property
    def plan_length(self) -> int | None:
        return None if self.plan is None else len(self.plan)

    def to_dict(self) -> dict[str, Any]:
        return {
            'solved': self.solved,
            'rounds': self.rounds,
            'plan': None if self.plan is None else [str(action) for action in self.plan],
            'plan_length': self.plan_length,
            'steps_final': self.steps_final,
            'applied_total': self.applied_total,
            'dropped_total': self.dropped_total,
            'created_total': self.created_total,
            'deleted_total': self.deleted_total,
            'duplicate_creations': self.duplicate_creations,
            'cycling_events': self.cycling_events,
            'wall_time': self.wall_time,
            'attempts': self.attempts,
        }


class PlanSpaceSearch:
    """Round-based local search in plan space driven by a :class:`PlanSpacePolicy`."""

    def run(
        self,
        policy: PlanSpacePolicy,
        problem: mm.Problem,
        *,
        max_rounds: int,
        mode: Literal['greedy', 'sample'] = 'greedy',
        temperature: float = 1.0,
        tombstone_capacity: int = 0,
        tombstone_lifespan: int = 0,
        recurrent: bool = False,
        seed: int | None = None,
        time_limit: float | None = None,
        record_visited: bool = False,
    ) -> SearchOutcome:
        """Edit the empty plan of ``problem`` until it is a solution or the budget is spent.

        Args:
            policy: The policy deciding the edits; evaluated without gradients.
            problem: The problem to solve.
            max_rounds: The maximum number of edit rounds.
            mode: ``'greedy'`` takes the argmax resolver and deletes where sigmoid(logit) > 0.5;
                ``'sample'`` samples both from the tempered distributions.
            temperature: The temperature of ``'sample'`` mode.
            tombstone_capacity: How many deleted steps the plan remembers as tombstones.
            tombstone_lifespan: For how many rounds a tombstone is kept.
            recurrent: Whether the policy carries node embeddings across rounds.
            seed: The seed of ``'sample'`` mode.
            time_limit: Wall-clock budget in seconds; checked before every round.
            record_visited: Whether to keep a clone of the plan before every round (for DAgger).
        """
        assert mode in ('greedy', 'sample'), f'Unknown mode {mode!r}.'
        assert temperature > 0.0, 'The temperature must be positive.'
        start = time.perf_counter()
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
        plan = mm.PartialOrderPlan.empty(problem, tombstone_capacity, tombstone_lifespan)
        state = problem.initial_state
        goal = plan.goal
        instance_key: Hashable = ('search', id(plan))
        if recurrent:
            policy.forget(instance_key)

        applied = dropped = created = deleted = duplicates = cycling = 0
        deleted_signatures: set[StepSignature] = set()
        bound_seen: set[int] = set()
        visited: list[mm.PartialOrderPlan] = []
        rounds = 0
        was_training = policy.training
        policy.eval()
        try:
            while rounds < max_rounds and not plan.is_solution():
                if time_limit is not None and time.perf_counter() - start > time_limit:
                    break
                flaws = plan.compute_flaws()
                if record_visited:
                    visited.append(plan.clone())
                with torch.no_grad():
                    output = policy.evaluate([(state, goal, plan, flaws)], carried=recurrent, instance_keys=[instance_key])[0]
                if mode == 'greedy':
                    chosen = output.greedy_choices()
                    delete_flags = output.greedy_deletes()
                else:
                    chosen = output.sample_choices(generator, temperature)
                    delete_flags = output.sample_deletes(generator, temperature)
                before = {step.id: step for step in plan.steps}
                step_ids = [step.id for step in before.values() if step.kind is mm.StepKind.STEP]
                assert len(step_ids) == len(delete_flags), 'One delete decision per Step-kind step is required.'
                deletes = [step_id for step_id, flag in zip(step_ids, delete_flags) if flag]

                result = plan.apply(flaws, chosen, deletes)
                applied += result.applied_count
                dropped += len(result.dropped_flaw_indices)
                created += len(result.created_step_ids)
                deleted += len(result.deleted_step_ids)
                deleted_ids = set(result.deleted_step_ids)
                for step_id in deleted_ids:
                    signature = _step_signature(before[step_id])
                    if signature is not None:
                        deleted_signatures.add(signature)
                    bound_seen.discard(step_id)
                existing = {
                    signature
                    for step_id, step in before.items()
                    if step_id not in deleted_ids and (signature := _step_signature(step)) is not None
                }
                created_ids = set(result.created_step_ids)
                for step in plan.steps:
                    signature = _step_signature(step)
                    if signature is None or step.id in bound_seen:
                        continue
                    bound_seen.add(step.id)
                    if step.id in created_ids and signature in existing:
                        duplicates += 1
                    if signature in deleted_signatures:
                        cycling += 1
                rounds += 1
        finally:
            policy.train(was_training)
            if recurrent:
                policy.forget(instance_key)

        solved = plan.is_solution()
        return SearchOutcome(
            solved=solved,
            rounds=rounds,
            plan=plan.linearize() if solved else None,
            steps_final=sum(1 for step in plan.steps if step.kind is mm.StepKind.STEP),
            applied_total=applied,
            dropped_total=dropped,
            created_total=created,
            deleted_total=deleted,
            duplicate_creations=duplicates,
            cycling_events=cycling,
            wall_time=time.perf_counter() - start,
            visited=visited,
            final_plan=plan.clone(),
        )

    def run_with_restarts(
        self,
        policy: PlanSpacePolicy,
        problem: mm.Problem,
        *,
        restarts: int,
        time_limit: float | None = None,
        seed: int | None = None,
        **options: Any,
    ) -> SearchOutcome:
        """Up to ``restarts`` independent :meth:`run` attempts (seeds ``seed``, ``seed+1``, ...) sharing ``time_limit``.

        Stops at the first solved attempt. The outcome carries the plan of the solving attempt (or the final plan of
        the last one), sums the rounds and the edit counters over the attempts, and records ``attempts``.
        """
        assert restarts >= 1, 'At least one attempt is required.'
        start = time.perf_counter()
        base_seed = 0 if seed is None else seed
        outcomes: list[SearchOutcome] = []
        for attempt in range(restarts):
            remaining = None if time_limit is None else time_limit - (time.perf_counter() - start)
            if remaining is not None and remaining <= 0.0 and outcomes:
                break
            outcome = self.run(policy, problem, seed=base_seed + attempt, time_limit=remaining, **options)
            outcomes.append(outcome)
            if outcome.solved:
                break
        last = outcomes[-1]
        return SearchOutcome(
            solved=last.solved,
            rounds=sum(outcome.rounds for outcome in outcomes),
            plan=last.plan,
            steps_final=last.steps_final,
            applied_total=sum(outcome.applied_total for outcome in outcomes),
            dropped_total=sum(outcome.dropped_total for outcome in outcomes),
            created_total=sum(outcome.created_total for outcome in outcomes),
            deleted_total=sum(outcome.deleted_total for outcome in outcomes),
            duplicate_creations=sum(outcome.duplicate_creations for outcome in outcomes),
            cycling_events=sum(outcome.cycling_events for outcome in outcomes),
            wall_time=time.perf_counter() - start,
            visited=last.visited,
            final_plan=last.final_plan,
            attempts=len(outcomes),
        )


@dataclass
class Sample:
    """One imitation sample: a plan snapshot and the oracle's targets for its flaws.

    The flaws are recomputed from ``plan_snapshot`` when training; ``compute_flaws``
    is deterministic, so ``targets.resolver_index_per_flaw`` stays aligned. The goal
    the policy sees is ``plan_snapshot.goal``, which hindsight relabeling may have
    reduced to a subset of the problem's goal.
    """

    problem: mm.Problem
    plan_snapshot: mm.PartialOrderPlan
    targets: mm.OracleTargets
    trajectory_id: int = 0
    round_index: int = 0


def oracle_rollout(
    problem: mm.Problem,
    reference: mm.PartialOrderPlan,
    *,
    follow_fraction: float = 1.0,
    rng: random.Random | None = None,
    max_rounds: int = 100,
    tombstone_capacity: int = 0,
    tombstone_lifespan: int = 0,
    trajectory_id: int = 0,
) -> list[Sample]:
    """Roll out the oracle from the empty plan and record a sample before every round.

    Each round follows the oracle for a random subset of the flaws with a non-NoOp
    target (fraction ``follow_fraction``, at least one) and chooses NoOp for the
    rest; the oracle's deletions are always applied. The recorded targets are the
    full oracle targets, so partial following only diversifies the visited plans.
    """
    assert 0.0 <= follow_fraction <= 1.0, 'follow_fraction must be in [0, 1].'
    assert reference.problem is problem, 'The reference plan must belong to the problem.'
    rng = random.Random(0) if rng is None else rng
    plan = mm.PartialOrderPlan.empty(problem, tombstone_capacity, tombstone_lifespan)
    samples: list[Sample] = []
    for round_index in range(max_rounds):
        if plan.is_solution():
            break
        flaws = plan.compute_flaws()
        targets = mm.oracle_targets(plan, flaws, reference)
        assert len(targets.resolver_index_per_flaw) == len(flaws.flaws)
        samples.append(Sample(problem, plan.clone(), targets, trajectory_id, round_index))
        chosen = list(targets.resolver_index_per_flaw)
        if follow_fraction < 1.0:
            active = [index for index, choice in enumerate(chosen) if choice != 0]
            if active:
                keep_count = max(1, round(follow_fraction * len(active)))
                kept = set(rng.sample(active, keep_count))
                chosen = [choice if index in kept else 0 for index, choice in enumerate(chosen)]
        plan.apply(flaws, chosen, targets.step_ids_to_delete)
    return samples


def oracle_reaches_reference(problem: mm.Problem, reference: mm.PartialOrderPlan, *, max_rounds: int) -> bool:
    """Whether following the oracle completely from the empty plan reaches a solution within ``max_rounds``.

    The oracle matches steps of the current plan to steps of the reference heuristically, and on references with
    repeated ground actions (typical of non-optimal planner plans) the matching can flip between the copies and the
    rollout cycles without ever solving the problem. Such references yield useless imitation data; use this to filter them.
    """
    assert reference.problem is problem, 'The reference plan must belong to the problem.'
    plan = mm.PartialOrderPlan.empty(problem)
    for _ in range(max_rounds):
        if plan.is_solution():
            return True
        flaws = plan.compute_flaws()
        targets = mm.oracle_targets(plan, flaws, reference)
        plan.apply(flaws, targets.resolver_index_per_flaw, targets.step_ids_to_delete)
    return plan.is_solution()


def compute_reference_plan(
    problem: mm.Problem,
    *,
    max_expanded_states: int = 100_000,
    timeout_seconds: float | None = None,
) -> list[mm.GroundAction] | None:
    """A reference plan: optimal from BFS when the problem is small, else from GBFS with the FF heuristic."""
    result = mm.bfs(problem, max_expanded_states=max_expanded_states, timeout_seconds=timeout_seconds)
    if result.solution is None:
        heuristic: Heuristic
        try:
            heuristic = mm.FFHeuristic(problem)
        except ValueError:
            # The grounded FF heuristic needs a grounded problem; the lifted one works for any problem.
            heuristic = mm.LiftedFFHeuristic(problem)
        result = mm.gbfs(problem, heuristic, timeout_seconds=timeout_seconds)
    if result.solution is None:
        return None
    return list(result.solution.plan)


def plan_to_strings(actions: Sequence[mm.GroundAction]) -> list[str]:
    """Serialize ground actions as ``"name arg1 arg2"`` strings."""
    return [' '.join([action.schema.name, *(str(value) for value in action.arguments)]) for action in actions]


def plan_from_strings(problem: mm.Problem, values: Sequence[str]) -> list[mm.GroundAction]:
    """Rebuild the ground actions of :func:`plan_to_strings` for ``problem``."""
    actions: list[mm.GroundAction] = []
    for value in values:
        name, *arguments = value.split()
        actions.append(problem.action(name, *arguments))
    return actions


class ReferenceCache:
    """JSON cache of reference plans keyed by a problem name, stored next to the data."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._plans: dict[str, list[str]] = {}
        if self.path.exists():
            with self.path.open() as handle:
                self._plans = json.load(handle)

    def get(self, key: str, problem: mm.Problem) -> list[mm.GroundAction] | None:
        values = self._plans.get(key)
        return None if values is None else plan_from_strings(problem, values)

    def put(self, key: str, actions: Sequence[mm.GroundAction]) -> None:
        self._plans[key] = plan_to_strings(actions)

    def save(self) -> None:
        # Write to a temporary file and rename so concurrent runs sharing the cache never read a partial file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        with temporary.open('w') as handle:
            json.dump(self._plans, handle, indent=1, sort_keys=True)
        temporary.replace(self.path)

    def get_or_compute(self, key: str, problem: mm.Problem, **options: Any) -> list[mm.GroundAction] | None:
        actions = self.get(key, problem)
        if actions is None:
            actions = compute_reference_plan(problem, **options)
            if actions is not None:
                self.put(key, actions)
                self.save()
        return actions


@dataclass
class TrainingConfig:
    """Optimization settings of :class:`ImitationTrainer`."""

    lr: float = 1e-3
    batch_size: int = 16
    grad_clip: float = 1.0
    # Whether flaws whose oracle target is NoOp contribute to the resolver loss.
    include_noop_targets: bool = True
    delete_loss_weight: float = 1.0
    # Trajectories are processed round by round with carried, detached embeddings (needs a recurrent policy).
    recurrent: bool = False


@dataclass
class EpochMetrics:
    """Averaged losses and accuracies of one pass over the samples."""

    loss: float
    resolver_loss: float
    delete_loss: float
    resolver_accuracy: float
    delete_accuracy: float
    sample_count: int
    flaw_count: int
    step_count: int
    wall_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _LossStats:
    resolver_nll: torch.Tensor
    delete_bce: torch.Tensor
    resolver_correct: int
    flaw_count: int
    delete_correct: int
    step_count: int


class ImitationTrainer:
    """Supervised imitation of oracle targets with Adam and gradient clipping."""

    def __init__(self, policy: PlanSpacePolicy, config: TrainingConfig | None = None, seed: int = 0) -> None:
        self.policy = policy
        self.config = TrainingConfig() if config is None else config
        if self.config.recurrent:
            assert policy.memory is not None, 'Recurrent training needs a policy with recurrent=True.'
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr)
        self.rng = random.Random(seed)

    @staticmethod
    def items_of(samples: Sequence[Sample]) -> list[tuple[PolicyItem, mm.OracleTargets]]:
        """Recompute the flaws of every snapshot and pair the policy input with its targets."""
        items: list[tuple[PolicyItem, mm.OracleTargets]] = []
        for sample in samples:
            plan = sample.plan_snapshot
            flaws = plan.compute_flaws()
            assert len(flaws.flaws) == len(sample.targets.resolver_index_per_flaw), (
                'The flaws recomputed from the snapshot must match the recorded oracle targets.'
            )
            items.append(((sample.problem.initial_state, plan.goal, plan, flaws), sample.targets))
        return items

    def compute_loss(self, outputs: Sequence[PolicyOutput], items: Sequence[tuple[PolicyItem, mm.OracleTargets]]) -> tuple[torch.Tensor, _LossStats]:
        """The imitation loss of a batch: mean resolver NLL plus weighted mean delete BCE."""
        device = self.policy.device
        target_log_probs: list[torch.Tensor] = []
        resolver_correct = 0
        delete_logits: list[torch.Tensor] = []
        delete_targets: list[torch.Tensor] = []
        for output, ((_, _, plan, flaws), targets) in zip(outputs, items):
            assert len(output.resolver_log_probs) == len(flaws.flaws) == len(targets.resolver_index_per_flaw)
            for log_probs, flaw, target in zip(output.resolver_log_probs, flaws.flaws, targets.resolver_index_per_flaw):
                assert log_probs.numel() == len(flaw.resolvers), 'Resolver counts must match.'
                if target == 0 and not self.config.include_noop_targets:
                    continue
                target_log_probs.append(log_probs[target])
                resolver_correct += int(int(log_probs.argmax()) == target)
            step_ids = [step.id for step in plan.steps if step.kind is mm.StepKind.STEP]
            assert output.delete_logits.numel() == len(step_ids), 'Delete logits must cover the Step-kind steps.'
            to_delete = set(targets.step_ids_to_delete)
            delete_logits.append(output.delete_logits)
            delete_targets.append(torch.tensor([1.0 if step_id in to_delete else 0.0 for step_id in step_ids], device=device))

        zero = torch.zeros((), device=device)
        if target_log_probs:
            resolver_nll = -torch.stack(target_log_probs).mean()
        else:
            resolver_nll = zero
        logits = torch.cat(delete_logits) if delete_logits else torch.empty(0, device=device)
        labels = torch.cat(delete_targets) if delete_targets else torch.empty(0, device=device)
        if logits.numel() > 0:
            delete_bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            delete_correct = int(((logits > 0).float() == labels).sum())
        else:
            delete_bce = zero
            delete_correct = 0
        loss = resolver_nll + self.config.delete_loss_weight * delete_bce
        stats = _LossStats(resolver_nll.detach(), delete_bce.detach(), resolver_correct, len(target_log_probs), delete_correct, int(logits.numel()))
        return loss, stats

    def _step(self, samples: Sequence[Sample], carried: bool, instance_keys: Sequence[Hashable] | None) -> _LossStats:
        items = self.items_of(samples)
        self.optimizer.zero_grad()
        outputs = self.policy.evaluate([item for item, _ in items], carried=carried, instance_keys=instance_keys)
        loss, stats = self.compute_loss(outputs, items)
        if loss.requires_grad:
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return stats

    def _summarize(self, stats: Sequence[_LossStats], sample_count: int, start: float) -> EpochMetrics:
        flaw_count = sum(stat.flaw_count for stat in stats)
        step_count = sum(stat.step_count for stat in stats)
        # Weight the batch means by their sizes so that the epoch numbers are per-flaw and per-step averages.
        resolver_loss = sum(float(stat.resolver_nll) * stat.flaw_count for stat in stats) / max(flaw_count, 1)
        delete_loss = sum(float(stat.delete_bce) * stat.step_count for stat in stats) / max(step_count, 1)
        return EpochMetrics(
            loss=resolver_loss + self.config.delete_loss_weight * delete_loss,
            resolver_loss=resolver_loss,
            delete_loss=delete_loss,
            resolver_accuracy=sum(stat.resolver_correct for stat in stats) / max(flaw_count, 1),
            delete_accuracy=sum(stat.delete_correct for stat in stats) / max(step_count, 1),
            sample_count=sample_count,
            flaw_count=flaw_count,
            step_count=step_count,
            wall_time=time.perf_counter() - start,
        )

    def train_epoch(self, samples: Sequence[Sample]) -> EpochMetrics:
        """One pass over the samples in shuffled mini-batches (or trajectory by trajectory when recurrent)."""
        start = time.perf_counter()
        self.policy.train()
        stats: list[_LossStats] = []
        if self.config.recurrent:
            trajectories: dict[int, list[Sample]] = {}
            for sample in samples:
                trajectories.setdefault(sample.trajectory_id, []).append(sample)
            ordered = [sorted(group, key=lambda sample: sample.round_index) for group in trajectories.values()]
            self.rng.shuffle(ordered)
            for offset in range(0, len(ordered), self.config.batch_size):
                chunk = ordered[offset:offset + self.config.batch_size]
                self.policy.clear_memory()
                for round_index in range(max(len(trajectory) for trajectory in chunk)):
                    active = [trajectory for trajectory in chunk if round_index < len(trajectory)]
                    batch = [trajectory[round_index] for trajectory in active]
                    keys: list[Hashable] = [('train', trajectory[0].trajectory_id) for trajectory in active]
                    stats.append(self._step(batch, True, keys))
            self.policy.clear_memory()
        else:
            shuffled = list(samples)
            self.rng.shuffle(shuffled)
            for offset in range(0, len(shuffled), self.config.batch_size):
                stats.append(self._step(shuffled[offset:offset + self.config.batch_size], False, None))
        return self._summarize(stats, len(samples), start)

    def evaluate(self, samples: Sequence[Sample]) -> EpochMetrics:
        """Losses and accuracies over the samples without updating the weights."""
        start = time.perf_counter()
        was_training = self.policy.training
        self.policy.eval()
        stats: list[_LossStats] = []
        with torch.no_grad():
            if self.config.recurrent:
                trajectories: dict[int, list[Sample]] = {}
                for sample in samples:
                    trajectories.setdefault(sample.trajectory_id, []).append(sample)
                for trajectory_id, group in trajectories.items():
                    self.policy.forget(('eval', trajectory_id))
                    for sample in sorted(group, key=lambda sample: sample.round_index):
                        items = self.items_of([sample])
                        outputs = self.policy.evaluate([items[0][0]], carried=True, instance_keys=[('eval', trajectory_id)])
                        stats.append(self.compute_loss(outputs, items)[1])
                    self.policy.forget(('eval', trajectory_id))
            else:
                ordered = list(samples)
                for offset in range(0, len(ordered), self.config.batch_size):
                    items = self.items_of(ordered[offset:offset + self.config.batch_size])
                    outputs = self.policy.evaluate([item for item, _ in items])
                    stats.append(self.compute_loss(outputs, items)[1])
        self.policy.train(was_training)
        return self._summarize(stats, len(samples), start)

    def dagger_collect(
        self,
        problems: Sequence[tuple[mm.Problem, mm.PartialOrderPlan]],
        *,
        max_rounds: int,
        tombstone_capacity: int = 0,
        tombstone_lifespan: int = 0,
        trajectory_id_start: int = 0,
    ) -> list[Sample]:
        """Roll out the current policy greedily and label the visited plans with the oracle (DAgger).

        The snapshots keep the goal of the search plan (the problem's goal), which
        :meth:`items_of` reads back from ``plan_snapshot.goal``.
        """
        search = PlanSpaceSearch()
        samples: list[Sample] = []
        trajectory_id = trajectory_id_start
        for problem, reference in problems:
            outcome = search.run(
                self.policy,
                problem,
                max_rounds=max_rounds,
                tombstone_capacity=tombstone_capacity,
                tombstone_lifespan=tombstone_lifespan,
                recurrent=self.config.recurrent,
                record_visited=True,
            )
            for round_index, snapshot in enumerate(outcome.visited):
                flaws = snapshot.compute_flaws()
                targets = mm.oracle_targets(snapshot, flaws, reference)
                samples.append(Sample(problem, snapshot, targets, trajectory_id, round_index))
            trajectory_id += 1
        return samples

    def hindsight_collect(
        self,
        problems: Sequence[mm.Problem],
        *,
        rollouts_per_problem: int,
        temperature: float,
        max_rounds: int,
        rng: random.Random,
        tombstone_capacity: int = 0,
        tombstone_lifespan: int = 0,
        best_solutions: dict[int, mm.PartialOrderPlan] | None = None,
        include_greedy: bool = True,
        trajectory_id_start: int = 0,
    ) -> tuple[list[Sample], CollectStats]:
        """Planner-free data from the current policy's own rollouts; see :func:`hindsight_collect`."""
        return hindsight_collect(
            self.policy,
            problems,
            rollouts_per_problem=rollouts_per_problem,
            temperature=temperature,
            max_rounds=max_rounds,
            rng=rng,
            tombstone_capacity=tombstone_capacity,
            tombstone_lifespan=tombstone_lifespan,
            best_solutions=best_solutions,
            include_greedy=include_greedy,
            trajectory_id_start=trajectory_id_start,
            recurrent=self.config.recurrent,
        )


@dataclass
class CollectStats:
    """What one call of :func:`hindsight_collect` produced."""

    episodes: int
    solved_episodes: int
    # Unsolved episodes whose pruned plan achieved no goal atom; they yield no samples.
    skipped_episodes: int
    # Mean of |relabeled goal| / |problem goal| over the non-skipped episodes (1.0 for solved ones).
    mean_goal_fraction: float | None
    samples: int
    mean_rounds: float | None
    wall_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapStats:
    """What one call of :func:`bootstrap_collect` did on top of its :class:`CollectStats`."""

    # Problems with a reference that were collected in bootstrap mode in this call.
    bootstrap_problems: int
    # Problems with a reference that are still in bootstrap mode after this call (solved fewer than ``phase_out`` times).
    remaining: int
    oracle_episodes: int
    # Samples from the oracle rollouts and the reference-labelled policy episodes of bootstrap problems.
    bootstrap_samples: int
    # Samples from the hindsight path (problems out of bootstrap mode or without a reference).
    hindsight_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _same_goal(a: mm.GroundConjunctiveCondition, b: mm.GroundConjunctiveCondition) -> bool:
    return [str(literal) for literal in a.literals] == [str(literal) for literal in b.literals]


def _plan_step_count(plan: mm.PartialOrderPlan) -> int:
    return sum(1 for step in plan.steps if step.kind is mm.StepKind.STEP)


def _record_solution(best_solutions: dict[int, mm.PartialOrderPlan] | None, problem: mm.Problem, plan: mm.PartialOrderPlan) -> None:
    """Store ``plan`` as the best solution of ``problem`` when it has fewer steps than the stored one."""
    if best_solutions is None:
        return
    stored = best_solutions.get(id(problem))
    if stored is None or _plan_step_count(plan) < _plan_step_count(stored):
        best_solutions[id(problem)] = plan


def _label_visited(
    problem: mm.Problem,
    visited: Sequence[mm.PartialOrderPlan],
    reference: mm.PartialOrderPlan,
    goal: mm.GroundConjunctiveCondition | None,
    trajectory_id: int,
) -> list[Sample]:
    """Oracle targets towards ``reference`` for every visited plan, relabeled to ``goal`` when one is given."""
    samples: list[Sample] = []
    for round_index, snapshot in enumerate(visited):
        view = snapshot if goal is None or _same_goal(snapshot.goal, goal) else snapshot.with_goal(goal)
        flaws = view.compute_flaws()
        if not flaws.flaws:
            continue
        targets = mm.oracle_targets(view, flaws, reference)
        samples.append(Sample(problem, view, targets, trajectory_id, round_index))
    return samples


def _label_in_hindsight(problem: mm.Problem, outcome: SearchOutcome, trajectory_id: int) -> tuple[list[Sample], float] | None:
    """The samples of one episode labelled with the plan it achieved and the achieved goal fraction; None if it achieved nothing."""
    assert outcome.final_plan is not None
    goal_size = len(problem.goal.literals)
    if outcome.solved:
        return _label_visited(problem, outcome.visited, outcome.final_plan, None, trajectory_id), 1.0
    reference = outcome.final_plan.prune_to_achieved_goal()
    goal = reference.goal
    if not goal.literals:
        return None
    return _label_visited(problem, outcome.visited, reference, goal, trajectory_id), len(goal.literals) / max(goal_size, 1)


def _collect(
    policy: PlanSpacePolicy,
    problems: Sequence[mm.Problem],
    *,
    rollouts_per_problem: int,
    temperature: float,
    max_rounds: int,
    rng: random.Random,
    tombstone_capacity: int,
    tombstone_lifespan: int,
    best_solutions: dict[int, mm.PartialOrderPlan] | None,
    include_greedy: bool,
    trajectory_id_start: int,
    recurrent: bool,
    references: Mapping[int, mm.PartialOrderPlan] | None,
    solved_counts: dict[int, int] | None,
    phase_out: int,
    oracle_rollouts: int,
    follow_fraction: float,
    policy_episodes: bool,
) -> tuple[list[Sample], CollectStats, BootstrapStats]:
    assert rollouts_per_problem >= 0, 'rollouts_per_problem must be non-negative.'
    assert oracle_rollouts >= 0, 'oracle_rollouts must be non-negative.'
    start = time.perf_counter()
    search = PlanSpaceSearch()
    samples: list[Sample] = []
    trajectory_id = trajectory_id_start
    episodes = solved_episodes = skipped = 0
    bootstrap_problems = oracle_episodes = bootstrap_samples = hindsight_samples = 0
    goal_fractions: list[float] = []
    rounds: list[int] = []
    modes: list[Literal['greedy', 'sample']] = ['sample'] * rollouts_per_problem
    if include_greedy:
        modes.insert(0, 'greedy')
    for problem in problems:
        key = id(problem)
        reference = None if references is None else references.get(key)
        bootstrap = reference is not None and (solved_counts is None or solved_counts.get(key, 0) < phase_out)
        if bootstrap:
            assert reference is not None
            bootstrap_problems += 1
            for _ in range(oracle_rollouts):
                new_samples = oracle_rollout(
                    problem,
                    reference,
                    follow_fraction=follow_fraction,
                    rng=rng,
                    max_rounds=max_rounds,
                    tombstone_capacity=tombstone_capacity,
                    tombstone_lifespan=tombstone_lifespan,
                    trajectory_id=trajectory_id,
                )
                samples.extend(new_samples)
                bootstrap_samples += len(new_samples)
                oracle_episodes += 1
                trajectory_id += 1
        if not policy_episodes:
            continue
        for mode in modes:
            outcome = search.run(
                policy,
                problem,
                max_rounds=max_rounds,
                mode=mode,
                temperature=temperature,
                tombstone_capacity=tombstone_capacity,
                tombstone_lifespan=tombstone_lifespan,
                recurrent=recurrent,
                seed=rng.randrange(2**31),
                record_visited=True,
            )
            episodes += 1
            rounds.append(outcome.rounds)
            assert outcome.final_plan is not None
            if outcome.solved:
                solved_episodes += 1
                _record_solution(best_solutions, problem, outcome.final_plan)
                if solved_counts is not None:
                    solved_counts[key] = solved_counts.get(key, 0) + 1
            if bootstrap:
                assert reference is not None
                new_samples = _label_visited(problem, outcome.visited, reference, None, trajectory_id)
                bootstrap_samples += len(new_samples)
            else:
                labelled = _label_in_hindsight(problem, outcome, trajectory_id)
                if labelled is None:
                    skipped += 1
                    trajectory_id += 1
                    continue
                new_samples, goal_fraction = labelled
                goal_fractions.append(goal_fraction)
                hindsight_samples += len(new_samples)
            samples.extend(new_samples)
            trajectory_id += 1
    stats = CollectStats(
        episodes=episodes,
        solved_episodes=solved_episodes,
        skipped_episodes=skipped,
        mean_goal_fraction=sum(goal_fractions) / len(goal_fractions) if goal_fractions else None,
        samples=len(samples),
        mean_rounds=sum(rounds) / len(rounds) if rounds else None,
        wall_time=time.perf_counter() - start,
    )
    remaining = 0
    if references is not None:
        remaining = sum(1 for key in references if solved_counts is None or solved_counts.get(key, 0) < phase_out)
    bootstrap_stats = BootstrapStats(
        bootstrap_problems=bootstrap_problems,
        remaining=remaining,
        oracle_episodes=oracle_episodes,
        bootstrap_samples=bootstrap_samples,
        hindsight_samples=hindsight_samples,
    )
    return samples, stats, bootstrap_stats


def hindsight_collect(
    policy: PlanSpacePolicy,
    problems: Sequence[mm.Problem],
    *,
    rollouts_per_problem: int,
    temperature: float,
    max_rounds: int,
    rng: random.Random,
    tombstone_capacity: int = 0,
    tombstone_lifespan: int = 0,
    best_solutions: dict[int, mm.PartialOrderPlan] | None = None,
    include_greedy: bool = True,
    trajectory_id_start: int = 0,
    recurrent: bool = False,
) -> tuple[list[Sample], CollectStats]:
    """Hindsight expert iteration: label the policy's own rollouts with the plan they achieved.

    Every episode runs :meth:`PlanSpaceSearch.run` in ``'sample'`` mode (plus one
    greedy episode per problem when ``include_greedy``) and records the visited
    plans. A solved episode uses its final plan as the reference and keeps the
    problem's goal; ``best_solutions[id(problem)]`` is updated when the plan has
    fewer steps than the stored one. An unsolved episode is pruned to its largest
    flaw-free sub-plan (:meth:`pymimir.PartialOrderPlan.prune_to_achieved_goal`),
    which becomes the reference for the sub-goal it achieves; the visited plans are
    relabeled to that sub-goal with ``with_goal`` before the oracle targets are
    computed. Episodes whose pruned plan achieves nothing are skipped. No planner is
    involved, so the labels are consistent with what the policy itself can produce.
    """
    samples, stats, _ = _collect(
        policy,
        problems,
        rollouts_per_problem=rollouts_per_problem,
        temperature=temperature,
        max_rounds=max_rounds,
        rng=rng,
        tombstone_capacity=tombstone_capacity,
        tombstone_lifespan=tombstone_lifespan,
        best_solutions=best_solutions,
        include_greedy=include_greedy,
        trajectory_id_start=trajectory_id_start,
        recurrent=recurrent,
        references=None,
        solved_counts=None,
        phase_out=0,
        oracle_rollouts=0,
        follow_fraction=1.0,
        policy_episodes=True,
    )
    return samples, stats


def bootstrap_collect(
    policy: PlanSpacePolicy,
    problems: Sequence[mm.Problem],
    *,
    references: Mapping[int, mm.PartialOrderPlan],
    solved_counts: dict[int, int],
    phase_out: int,
    rollouts_per_problem: int,
    temperature: float,
    max_rounds: int,
    rng: random.Random,
    oracle_rollouts: int = 1,
    follow_fraction: float = 0.5,
    policy_episodes: bool = True,
    tombstone_capacity: int = 0,
    tombstone_lifespan: int = 0,
    best_solutions: dict[int, mm.PartialOrderPlan] | None = None,
    include_greedy: bool = True,
    trajectory_id_start: int = 0,
    recurrent: bool = False,
) -> tuple[list[Sample], CollectStats, BootstrapStats]:
    """Hindsight expert iteration bootstrapped from planner references that are phased out as the policy learns.

    ``references[id(problem)]`` is a planner solution of the problem. A problem with a
    reference is in *bootstrap mode* until ``solved_counts[id(problem)]`` reaches
    ``phase_out``; the counts are incremented here for every solved policy episode
    (sampled and greedy), so they accumulate across calls. For a problem in bootstrap
    mode this call runs ``oracle_rollouts`` teacher-forced :func:`oracle_rollout`
    episodes against the reference with ``follow_fraction`` and labels the policy's
    own episodes against the reference for the full goal (DAgger style) instead of
    pruning and relabeling them. Problems out of bootstrap mode, or without a
    reference, take the hindsight path of :func:`hindsight_collect`. Solved episodes
    always update ``best_solutions``. With ``policy_episodes=False`` only the oracle
    rollouts run (a pure bootstrap warm-up). The returned :class:`CollectStats`
    cover the policy episodes of both paths; ``mean_goal_fraction`` and
    ``skipped_episodes`` refer to the hindsight path only.
    """
    return _collect(
        policy,
        problems,
        rollouts_per_problem=rollouts_per_problem,
        temperature=temperature,
        max_rounds=max_rounds,
        rng=rng,
        tombstone_capacity=tombstone_capacity,
        tombstone_lifespan=tombstone_lifespan,
        best_solutions=best_solutions,
        include_greedy=include_greedy,
        trajectory_id_start=trajectory_id_start,
        recurrent=recurrent,
        references=references,
        solved_counts=solved_counts,
        phase_out=phase_out,
        oracle_rollouts=oracle_rollouts,
        follow_fraction=follow_fraction,
        policy_episodes=policy_episodes,
    )


__all__ = [
    'PLAN_NODE_TYPES',
    'BootstrapStats',
    'CollectStats',
    'EpochMetrics',
    'ImitationTrainer',
    'PartialPlanEncoder',
    'PlanSpacePolicy',
    'PlanSpaceSearch',
    'PolicyConfig',
    'PolicyOutput',
    'ReferenceCache',
    'Sample',
    'SearchOutcome',
    'TrainingConfig',
    'bootstrap_collect',
    'compute_reference_plan',
    'hindsight_collect',
    'oracle_reaches_reference',
    'oracle_rollout',
    'plan_from_strings',
    'plan_to_strings',
]
