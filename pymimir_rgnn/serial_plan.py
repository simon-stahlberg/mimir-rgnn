"""Plan-space local search with a neural policy over serial plans.

A serial plan (:class:`pymimir.SerialPlan`) is a sequence of steps. Its symbolic
simulator applies the effects of every step even when a precondition fails and
reports every error: unsatisfied preconditions, unmet goal literals and unbound
parameters. The plan is edited in parallel rounds: every error picks one of its
repairs (index 0 is NoOp) and every step gets a delete decision. The plan is a
solution once no errors remain.

This module holds the encoder, the policy with its two heads, the search driver,
the data collection for hindsight expert iteration with a bootstrap from
:func:`pymimir.serial_search` references, and the trainer.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import pymimir as mm
import torch
import torch.nn as nn

from .aggregations import HardMaximumAggregation, MeanAggregation, SmoothMaximumAggregation, SumAggregation
from .bases import AggregationFunction, EncodedLists, Encoder, EncodingContext, UpdateFunction
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import OBJECTS_CONTEXT, GroupedLogProbs, GroupedSoftmaxDecoder, NodeScalarDecoder
from .encoders import GoalEncoder, StateEncoder
from .messages import PredicateMLPMessages
from .model import RelationalGraphNeuralNetwork
from .modules import POOLINGS, Pooling
from .updates import GRUUpdates, MLPUpdates

SerialItem = tuple[mm.SerialPlan, mm.SerialDiagnosis]


class SerialPlanEncoder(Encoder):
    """Encoder for a serial plan together with its errors and repairs.

    The input value of an instance is the pair ``(plan, diagnosis)`` with
    ``diagnosis`` computed from ``plan``. The relations come from the native
    :func:`pymimir.encode_serial_plan`; this encoder records under
    ``encoding.extras``, one entry per instance:

    - ``serial_layouts``: the :class:`pymimir.SerialEncodingLayout` with the node ids,
    - ``serial_step_nodes``: LongTensor of the node ids of the steps, in the order of ``plan.steps``,
    - ``serial_error_nodes``: LongTensor of the node ids of the errors, in the order of ``diagnosis.errors``.
    """

    LAYOUTS = 'serial_layouts'
    STEP_NODES = 'serial_step_nodes'
    ERROR_NODES = 'serial_error_nodes'
    # The ``(repair, error)`` relation the repair head scores.
    REPAIRS_RELATION = 'serial_repairs'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return mm.serial_plan_relations(domain)

    def encode(self, input_value: Any, state: mm.State, encoding: EncodedLists, context: EncodingContext) -> None:
        assert isinstance(input_value, tuple) and len(input_value) == 2, (
            f'SerialPlanEncoder expected a (plan, diagnosis) pair, got {type(input_value)}'
        )
        plan, diagnosis = input_value
        assert isinstance(plan, mm.SerialPlan), f'SerialPlanEncoder expected a SerialPlan, got {type(plan)}'
        assert isinstance(diagnosis, mm.SerialDiagnosis), f'SerialPlanEncoder expected a SerialDiagnosis, got {type(diagnosis)}'
        assert diagnosis.plan is plan, 'The diagnosis must be computed from the plan of the same instance.'
        assert plan.problem is state.problem, 'The plan must belong to the problem of the state.'
        layout = mm.encode_serial_plan(context, plan, diagnosis)
        encoding.extras.setdefault(self.LAYOUTS, []).append(layout)
        encoding.extras.setdefault(self.STEP_NODES, []).append(torch.tensor(layout.step_node_ids, dtype=torch.long))
        encoding.extras.setdefault(self.ERROR_NODES, []).append(torch.tensor(layout.error_node_ids, dtype=torch.long))


# The node groups of an instance a head can pool as a context, by their name in a SerialPolicyConfig: the decoder's
# name of the group, i.e. 'objects' or the key of the encoding extras that lists its nodes.
HEAD_CONTEXTS: dict[str, str] = {
    'objects': OBJECTS_CONTEXT,
    'steps': SerialPlanEncoder.STEP_NODES,
    'errors': SerialPlanEncoder.ERROR_NODES,
}
# The repair head can additionally pool the repairs competing for the same error.
SIBLINGS_CONTEXT = 'siblings'


@dataclass
class SerialPolicyConfig:
    """Architecture flags of a :class:`SerialPolicy`; serialized into checkpoints.

    ``repair_contexts`` and ``delete_contexts`` name the node groups of the
    instance whose pooled embeddings the repair head and the delete head see next
    to the node they score: ``'objects'``, ``'steps'`` and ``'errors'``, and for
    the repair head also ``'siblings'`` (the repairs of the same error).
    ``context_pooling`` is how they are pooled: ``'sum'``, ``'mean'``, ``'max'``
    or ``'maxmean'``.
    """

    embedding_size: int = 32
    num_layers: int = 10
    aggregation: Literal['mean', 'sum', 'max', 'smooth_max'] = 'mean'
    update: Literal['mlp', 'gru'] = 'mlp'
    repair_contexts: tuple[str, ...] = ('objects', 'steps', 'errors', 'siblings')
    delete_contexts: tuple[str, ...] = ('objects', 'steps', 'errors')
    context_pooling: Pooling = 'maxmean'

    def __post_init__(self) -> None:
        # Checkpoints and command lines hand in lists; tuples keep configs comparable.
        self.repair_contexts = tuple(self.repair_contexts)
        self.delete_contexts = tuple(self.delete_contexts)
        for head, contexts, allowed in (
            ('repair', self.repair_contexts, (*HEAD_CONTEXTS, SIBLINGS_CONTEXT)),
            ('delete', self.delete_contexts, tuple(HEAD_CONTEXTS)),
        ):
            unknown = [name for name in contexts if name not in allowed]
            if unknown:
                raise ValueError(f'Unknown {head} contexts {unknown}; expected a subset of {allowed}.')
        if self.context_pooling not in POOLINGS:
            raise ValueError(f'context_pooling must be one of {POOLINGS}, got {self.context_pooling!r}.')


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


@dataclass
class SerialPolicyOutput:
    """The two heads of the policy for one instance.

    ``repair_log_probs[i]`` is aligned with ``diagnosis.errors[i].repairs`` (index 0
    is NoOp), and ``delete_logits`` with ``plan.steps``.
    """

    repair_log_probs: list[torch.Tensor]
    delete_logits: torch.Tensor


class SerialPolicy(nn.Module):
    """R-GNN policy over (initial state, goal, serial plan) with a repair head and a delete head.

    The state is the initial state of the plan's problem and the goal is
    ``plan.goal``, which hindsight relabeling may have reduced to a subset of the
    problem's goal.
    """

    def __init__(self, domain: mm.Domain, config: SerialPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = SerialPolicyConfig() if config is None else config
        self.domain = domain
        input_spec: tuple[Encoder, ...] = (StateEncoder(), GoalEncoder(), SerialPlanEncoder())
        hparams = HyperparameterConfig(
            domain=domain,
            embedding_size=self.config.embedding_size,
            num_layers=self.config.num_layers,
            # The gates of the GRU already carry the embedding across layers.
            residual_updates=self.config.update != 'gru',
        )
        output_spec = [
            ('repair_log_probs', GroupedSoftmaxDecoder(
                hparams,
                SerialPlanEncoder.REPAIRS_RELATION,
                contexts=[HEAD_CONTEXTS[name] for name in self.config.repair_contexts if name != SIBLINGS_CONTEXT],
                sibling_context=SIBLINGS_CONTEXT in self.config.repair_contexts,
                pooling=self.config.context_pooling,
            )),
            ('delete_logits', NodeScalarDecoder(
                hparams,
                SerialPlanEncoder.STEP_NODES,
                contexts=[HEAD_CONTEXTS[name] for name in self.config.delete_contexts],
                pooling=self.config.context_pooling,
            )),
        ]
        module_config = ModuleConfig(
            aggregation_function=_aggregation(self.config.aggregation),
            message_function=PredicateMLPMessages(hparams, input_spec),
            update_function=_update(self.config.update, hparams),
        )
        self.model = RelationalGraphNeuralNetwork(hparams, module_config, input_spec, output_spec)

    @property
    def device(self) -> torch.device:
        return self.model.get_device()

    @staticmethod
    def _align_repair_log_probs(grouped: GroupedLogProbs, layout: mm.SerialEncodingLayout, node_count: int) -> list[torch.Tensor]:
        """Reorder the rows of the grouped softmax to the errors' repair lists."""
        sizes = [len(repair_ids) for repair_ids in layout.repair_node_ids]
        if not sizes:
            return []
        device = grouped.log_probs.device
        repair_ids = torch.tensor([node_id for ids in layout.repair_node_ids for node_id in ids], dtype=torch.long, device=device)
        row_of_node = torch.full((node_count,), -1, dtype=torch.long, device=device)
        row_of_node[grouped.member_indices] = torch.arange(grouped.member_indices.numel(), device=device)
        rows = row_of_node.index_select(0, repair_ids)
        assert bool((rows >= 0).all()), 'Every repair node must be scored by the repair head.'
        return list(grouped.log_probs.index_select(0, rows).split(sizes))

    def evaluate(self, items: Sequence[SerialItem]) -> list[SerialPolicyOutput]:
        """Run the policy on a batch of ``(plan, diagnosis)`` items; ``diagnosis`` must be computed from ``plan``."""
        batch: list[tuple] = [(plan.problem.initial_state, plan.goal, (plan, diagnosis)) for plan, diagnosis in items]
        encoded = self.model.encode(batch)
        output = self.model.forward_encoded(encoded)
        grouped: list[GroupedLogProbs] = output.readout('repair_log_probs')
        deletes: list[torch.Tensor] = output.readout('delete_logits')
        layouts: list[mm.SerialEncodingLayout] = encoded.extras[SerialPlanEncoder.LAYOUTS]
        assert len(grouped) == len(deletes) == len(layouts) == len(items)
        return [
            SerialPolicyOutput(self._align_repair_log_probs(group, layout, encoded.node_count), delete_logits)
            for group, delete_logits, layout in zip(grouped, deletes, layouts)
        ]

    def save(self, path: Path | str, extras: dict[str, Any] | None = None) -> None:
        """Save the weights, the architecture config and optional extras (e.g. the training arguments)."""
        torch.save({
            'config': asdict(self.config),
            'domain_name': self.domain.name,
            'state_dict': self.state_dict(),
            'extras': {} if extras is None else extras,
        }, path)

    @staticmethod
    def load(domain: mm.Domain, path: Path | str, device: torch.device | str = 'cpu') -> tuple[SerialPolicy, dict[str, Any]]:
        """Load a policy saved with :meth:`save` for ``domain`` onto ``device``."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        policy = SerialPolicy(domain, SerialPolicyConfig(**checkpoint['config']))
        policy.load_state_dict(checkpoint['state_dict'])
        return policy.to(device), checkpoint['extras']


def is_valid_plan(problem: mm.Problem, actions: Sequence[mm.GroundAction]) -> bool:
    """Whether applying ``actions`` in order from the initial state is possible and reaches the goal of ``problem``."""
    state = problem.initial_state
    for action in actions:
        if not action.is_applicable(state):
            return False
        state = action.apply(state)
    return problem.goal.holds(state)


@dataclass
class SerialSearchOutcome:
    """What one run of :class:`SerialSearch` did."""

    solved: bool
    rounds: int
    # The ground actions of the plan when solved.
    plan: list[mm.GroundAction] | None
    steps_final: int
    applied_total: int
    dropped_total: int
    created_total: int
    deleted_total: int
    # Steps created with the schema, binding and purpose literal a deleted step of the run was created with.
    cycling_events: int
    wall_time: float
    # The plan after the last round (solved or not), e.g. for ``prune_to_achieved_goal``.
    final_plan: mm.SerialPlan
    # A clone of the plan before every round, when recorded.
    visited: list[mm.SerialPlan] = field(default_factory=list)
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
            'cycling_events': self.cycling_events,
            'wall_time': self.wall_time,
            'attempts': self.attempts,
        }


StepSignature = tuple[str, tuple[str, ...], str]


def _step_signature(step: mm.SerialStep) -> StepSignature:
    return step.schema.name, tuple(str(value) for value in step.binding), str(step.purpose)


class SerialSearch:
    """Round-based local search over serial plans driven by a :class:`SerialPolicy`."""

    def run(
        self,
        policy: SerialPolicy,
        problem: mm.Problem,
        *,
        max_rounds: int,
        mode: Literal['greedy', 'sample'] = 'greedy',
        temperature: float = 1.0,
        seed: int | None = None,
        time_limit: float | None = None,
        record_visited: bool = False,
    ) -> SerialSearchOutcome:
        """Edit the empty plan of ``problem`` until it is a solution or the budget is spent.

        Args:
            policy: The policy deciding the edits; evaluated without gradients.
            problem: The problem to solve.
            max_rounds: The maximum number of edit rounds.
            mode: ``'greedy'`` takes the most probable repair of every error and deletes the steps
                with a positive logit; ``'sample'`` samples both from the tempered distributions.
            temperature: The temperature of ``'sample'`` mode.
            seed: The seed of ``'sample'`` mode.
            time_limit: Wall-clock budget in seconds; checked before every round.
            record_visited: Whether to keep a clone of the plan before every round (for labelling).
        """
        assert mode in ('greedy', 'sample'), f'Unknown mode {mode!r}.'
        assert temperature > 0.0, 'The temperature must be positive.'
        start = time.perf_counter()
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
        plan = mm.SerialPlan.empty(problem)
        applied = dropped = created = deleted = cycling = 0
        # The signature every live step was created with, and those of the steps deleted so far.
        created_signatures: dict[int, StepSignature] = {}
        deleted_signatures: set[StepSignature] = set()
        visited: list[mm.SerialPlan] = []
        rounds = 0
        was_training = policy.training
        policy.eval()
        try:
            while rounds < max_rounds:
                if time_limit is not None and time.perf_counter() - start > time_limit:
                    break
                diagnosis = plan.diagnose()
                if diagnosis.is_solution:
                    break
                if record_visited:
                    visited.append(plan.clone())
                with torch.no_grad():
                    output = policy.evaluate([(plan, diagnosis)])[0]
                if mode == 'greedy':
                    chosen = [int(log_probs.argmax()) for log_probs in output.repair_log_probs]
                    delete_flags = (output.delete_logits > 0).tolist()
                else:
                    chosen = [
                        int(torch.multinomial(torch.softmax(log_probs.cpu() / temperature, dim=0), 1, generator=generator))
                        for log_probs in output.repair_log_probs
                    ]
                    delete_probabilities = torch.sigmoid(output.delete_logits.cpu() / temperature)
                    delete_flags = (torch.rand(delete_probabilities.shape, generator=generator) < delete_probabilities).tolist()
                steps = plan.steps
                assert len(steps) == len(delete_flags), 'One delete decision per step is required.'

                result = plan.apply(diagnosis, chosen, [step.id for step, flag in zip(steps, delete_flags) if flag])
                applied += result.applied_count
                dropped += len(result.dropped_error_indices)
                created += len(result.created_step_ids)
                deleted += len(result.deleted_step_ids)
                for step_id in result.deleted_step_ids:
                    deleted_signatures.add(created_signatures.pop(step_id))
                created_ids = set(result.created_step_ids)
                for step in plan.steps:
                    if step.id in created_ids:
                        signature = _step_signature(step)
                        created_signatures[step.id] = signature
                        cycling += signature in deleted_signatures
                rounds += 1
        finally:
            policy.train(was_training)

        solved = plan.is_solution()
        return SerialSearchOutcome(
            solved=solved,
            rounds=rounds,
            plan=plan.to_actions() if solved else None,
            steps_final=len(plan.steps),
            applied_total=applied,
            dropped_total=dropped,
            created_total=created,
            deleted_total=deleted,
            cycling_events=cycling,
            wall_time=time.perf_counter() - start,
            final_plan=plan,
            visited=visited,
        )

    def run_with_restarts(
        self,
        policy: SerialPolicy,
        problem: mm.Problem,
        *,
        restarts: int,
        time_limit: float | None = None,
        seed: int | None = None,
        **options: Any,
    ) -> SerialSearchOutcome:
        """Up to ``restarts`` independent :meth:`run` attempts (seeds ``seed``, ``seed+1``, ...) sharing ``time_limit``.

        Stops at the first solved attempt. The outcome is that of the last attempt with the rounds and the edit
        counters summed over the attempts, and records ``attempts``.
        """
        assert restarts >= 1, 'At least one attempt is required.'
        start = time.perf_counter()
        base_seed = 0 if seed is None else seed
        outcomes: list[SerialSearchOutcome] = []
        for attempt in range(restarts):
            remaining = None if time_limit is None else time_limit - (time.perf_counter() - start)
            if remaining is not None and remaining <= 0.0 and outcomes:
                break
            outcomes.append(self.run(policy, problem, seed=base_seed + attempt, time_limit=remaining, **options))
            if outcomes[-1].solved:
                break
        return replace(
            outcomes[-1],
            rounds=sum(outcome.rounds for outcome in outcomes),
            applied_total=sum(outcome.applied_total for outcome in outcomes),
            dropped_total=sum(outcome.dropped_total for outcome in outcomes),
            created_total=sum(outcome.created_total for outcome in outcomes),
            deleted_total=sum(outcome.deleted_total for outcome in outcomes),
            cycling_events=sum(outcome.cycling_events for outcome in outcomes),
            wall_time=time.perf_counter() - start,
            attempts=len(outcomes),
        )


@dataclass
class SerialSample:
    """One imitation sample: a plan snapshot and the oracle's targets for its errors and steps.

    The errors are diagnosed again from ``plan`` when training; ``diagnose`` is
    deterministic, so ``targets.repair_index_per_error`` stays aligned. The goal
    the policy sees is ``plan.goal``, which hindsight relabeling may have reduced
    to a subset of the problem's goal.
    """

    plan: mm.SerialPlan
    targets: mm.SerialOracleTargets


def serial_references(problems: Sequence[mm.Problem], timeout_seconds: float) -> dict[mm.Problem, mm.SerialPlan]:
    """Reference solutions from :func:`pymimir.serial_search` for the problems it solves within ``timeout_seconds`` each."""
    references: dict[mm.Problem, mm.SerialPlan] = {}
    for problem in problems:
        reference = mm.serial_search(problem, timeout_seconds=timeout_seconds)
        if reference is not None:
            references[problem] = reference
    return references


def serial_oracle_rollout(
    problem: mm.Problem,
    reference: mm.SerialPlan,
    *,
    follow_fraction: float,
    rng: random.Random,
    max_rounds: int,
) -> list[SerialSample]:
    """Roll out the oracle from the empty plan and record a sample before every round.

    Each round follows the oracle for a random subset of the errors with a non-NoOp
    target (fraction ``follow_fraction``, at least one) and chooses NoOp for the
    rest; the oracle's deletions are always applied. The recorded targets are the
    full oracle targets, so partial following only diversifies the visited plans.
    """
    assert 0.0 <= follow_fraction <= 1.0, 'follow_fraction must be in [0, 1].'
    plan = mm.SerialPlan.empty(problem)
    samples: list[SerialSample] = []
    for _ in range(max_rounds):
        diagnosis = plan.diagnose()
        if diagnosis.is_solution:
            break
        targets = mm.serial_oracle_targets(plan, diagnosis, reference)
        samples.append(SerialSample(plan.clone(), targets))
        chosen = list(targets.repair_index_per_error)
        active = [index for index, choice in enumerate(chosen) if choice != 0]
        if follow_fraction < 1.0 and active:
            followed = set(rng.sample(active, max(1, round(follow_fraction * len(active)))))
            chosen = [choice if index in followed else 0 for index, choice in enumerate(chosen)]
        plan.apply(diagnosis, chosen, targets.step_ids_to_delete)
    return samples


def _label_visited(visited: Sequence[mm.SerialPlan], reference: mm.SerialPlan) -> list[SerialSample]:
    """Oracle targets towards ``reference`` for every visited plan, judged against the goal of ``reference``."""
    goal = reference.goal
    samples: list[SerialSample] = []
    for snapshot in visited:
        plan = snapshot.with_goal(goal)
        diagnosis = plan.diagnose()
        # A snapshot that already achieves a relabeled sub-goal has no errors to decide on.
        if not diagnosis.is_solution:
            samples.append(SerialSample(plan, mm.serial_oracle_targets(plan, diagnosis, reference)))
    return samples


@dataclass
class SerialCollectStats:
    """What one call of :func:`serial_collect` produced."""

    # Episodes of the policy (greedy and sampled) and how many of them solved their problem.
    episodes: int
    solved_episodes: int
    # Unsolved hindsight episodes whose pruned plan achieved no goal literal; they yield no samples.
    skipped_episodes: int
    # Mean of |relabeled goal| / |problem goal| over the labelled hindsight episodes (1.0 for solved ones).
    mean_goal_fraction: float | None
    mean_rounds: float | None
    # Problems collected in bootstrap mode in this call, and their teacher-forced oracle rollouts.
    bootstrap_problems: int
    oracle_episodes: int
    # Problems with a reference that are still in bootstrap mode after this call.
    bootstrap_remaining: int
    # Samples labelled against a reference (oracle rollouts and policy episodes of bootstrap problems).
    bootstrap_samples: int
    # Samples labelled in hindsight against the plan the episode itself achieved.
    hindsight_samples: int
    wall_time: float


def serial_collect(
    policy: SerialPolicy,
    problems: Sequence[mm.Problem],
    *,
    references: Mapping[mm.Problem, mm.SerialPlan],
    solved_counts: dict[mm.Problem, int],
    best_solutions: dict[mm.Problem, mm.SerialPlan],
    phase_out: int,
    oracle_rollouts: int,
    follow_fraction: float,
    rollouts_per_problem: int,
    temperature: float,
    max_rounds: int,
    rng: random.Random,
    policy_episodes: bool = True,
) -> tuple[list[SerialSample], SerialCollectStats]:
    """Imitation samples for hindsight expert iteration, bootstrapped from references that are phased out.

    A problem is in *bootstrap mode* while it has a reference in ``references`` and
    ``solved_counts[problem]`` is below ``phase_out``; the counts are incremented
    here for every solved policy episode, so they accumulate across calls. For such
    a problem this runs ``oracle_rollouts`` teacher-forced :func:`serial_oracle_rollout`
    episodes and labels the policy's own unsolved episodes against the reference; a solved episode is its own reference.

    Every other problem takes the hindsight path, which needs no reference: a solved
    episode is labelled against its own final plan, an unsolved one against
    ``final_plan.prune_to_achieved_goal()`` with the visited plans relabeled to the
    sub-goal that plan achieves. Episodes that achieved no goal literal are skipped.
    Goal literals that hold in the initial state and still hold after the final plan
    count as achieved, so the pruned plan of an episode can have a goal but no steps.

    The policy's episodes of a problem are one greedy run and ``rollouts_per_problem``
    sampled runs; with ``policy_episodes=False`` only the oracle rollouts run (a
    warm-up). ``best_solutions`` keeps the shortest solution found per problem.
    """
    start = time.perf_counter()
    search = SerialSearch()
    samples: list[SerialSample] = []
    episodes = solved_episodes = skipped_episodes = 0
    bootstrap_problems = oracle_episodes = bootstrap_samples = 0
    goal_fractions: list[float] = []
    rounds: list[int] = []
    modes: list[Literal['greedy', 'sample']] = ['sample'] * rollouts_per_problem
    modes.insert(0, 'greedy')

    def in_bootstrap_mode(problem: mm.Problem) -> bool:
        return problem in references and solved_counts.get(problem, 0) < phase_out

    for problem in problems:
        # Decided once per problem, so that all its episodes of this call are labelled the same way.
        bootstrap = in_bootstrap_mode(problem)
        if bootstrap:
            bootstrap_problems += 1
            for _ in range(oracle_rollouts):
                new_samples = serial_oracle_rollout(
                    problem, references[problem], follow_fraction=follow_fraction, rng=rng, max_rounds=max_rounds,
                )
                samples += new_samples
                bootstrap_samples += len(new_samples)
                oracle_episodes += 1
        if not policy_episodes:
            continue
        for mode in modes:
            outcome = search.run(
                policy, problem, max_rounds=max_rounds, mode=mode, temperature=temperature,
                seed=rng.randrange(2**31), record_visited=True,
            )
            episodes += 1
            rounds.append(outcome.rounds)
            if outcome.solved:
                solved_episodes += 1
                solved_counts[problem] = solved_counts.get(problem, 0) + 1
                best = best_solutions.get(problem)
                if best is None or len(outcome.final_plan.steps) < len(best.steps):
                    best_solutions[problem] = outcome.final_plan
            if bootstrap and not outcome.solved:
                new_samples = _label_visited(outcome.visited, references[problem])
                samples += new_samples
                bootstrap_samples += len(new_samples)
                continue
            # A solved episode is its own reference in either mode: labelling it against the search's
            # reference would mark every valid choice that differs from the reference as a mistake.
            reference = outcome.final_plan if outcome.solved else outcome.final_plan.prune_to_achieved_goal()
            # A pruned plan without steps achieved nothing beyond the goal literals that already held initially.
            if not reference.steps:
                skipped_episodes += 1
                continue
            goal_fractions.append(len(reference.goal.literals) / len(problem.goal.literals))
            samples += _label_visited(outcome.visited, reference)

    stats = SerialCollectStats(
        episodes=episodes,
        solved_episodes=solved_episodes,
        skipped_episodes=skipped_episodes,
        mean_goal_fraction=sum(goal_fractions) / len(goal_fractions) if goal_fractions else None,
        mean_rounds=sum(rounds) / len(rounds) if rounds else None,
        bootstrap_problems=bootstrap_problems,
        oracle_episodes=oracle_episodes,
        bootstrap_remaining=sum(1 for problem in references if in_bootstrap_mode(problem)),
        bootstrap_samples=bootstrap_samples,
        hindsight_samples=len(samples) - bootstrap_samples,
        wall_time=time.perf_counter() - start,
    )
    return samples, stats


@dataclass
class SerialEpochMetrics:
    """Losses and accuracies of one pass over the buffer: batch means of the losses, per-decision accuracies."""

    loss: float
    repair_loss: float
    delete_loss: float
    repair_accuracy: float
    delete_accuracy: float
    samples: int
    wall_time: float


class SerialTrainer:
    """Supervised imitation of oracle targets on a FIFO buffer of samples, with Adam and gradient clipping.

    The loss of a batch is the mean negative log-likelihood of the oracle's repair
    per error plus ``delete_loss_weight`` times the mean binary cross-entropy of
    the delete logits per step.
    """

    def __init__(
        self,
        policy: SerialPolicy,
        *,
        lr: float = 1e-3,
        batch_size: int = 16,
        grad_clip: float = 1.0,
        delete_loss_weight: float = 1.0,
        buffer_cap: int = 20000,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.delete_loss_weight = delete_loss_weight
        self.buffer: deque[SerialSample] = deque(maxlen=buffer_cap)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.rng = random.Random(seed)

    def add(self, samples: Sequence[SerialSample]) -> None:
        """Append samples to the buffer; the oldest are dropped beyond its capacity."""
        self.buffer.extend(samples)

    def _losses(self, samples: Sequence[SerialSample]) -> tuple[torch.Tensor, torch.Tensor, list[bool], list[bool]]:
        """The repair loss and the delete loss of a batch, and whether each greedy repair and delete decision is the oracle's."""
        items = [(sample.plan, sample.plan.diagnose()) for sample in samples]
        outputs = self.policy.evaluate(items)
        target_log_probs: list[torch.Tensor] = []
        repair_hits: list[bool] = []
        delete_logits: list[torch.Tensor] = []
        delete_labels: list[float] = []
        for sample, (plan, diagnosis), output in zip(samples, items, outputs):
            targets = sample.targets.repair_index_per_error
            assert len(diagnosis.errors) == len(targets) == len(output.repair_log_probs), (
                'The errors diagnosed from the snapshot must match the recorded oracle targets.'
            )
            for error, target, log_probs in zip(diagnosis.errors, targets, output.repair_log_probs):
                assert target < len(error.repairs) == log_probs.numel(), 'The oracle target must be a repair of its error.'
                target_log_probs.append(log_probs[target])
                repair_hits.append(int(log_probs.argmax()) == target)
            to_delete = set(sample.targets.step_ids_to_delete)
            delete_logits.append(output.delete_logits)
            delete_labels += [float(step.id in to_delete) for step in plan.steps]

        repair_loss = -torch.stack(target_log_probs).mean()
        logits = torch.cat(delete_logits)
        labels = torch.tensor(delete_labels, device=logits.device)
        assert logits.numel() == labels.numel(), 'One delete logit per step is required.'
        # A batch of plans without steps has no delete decisions.
        delete_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) if delete_labels else logits.new_zeros(())
        delete_hits: list[bool] = ((logits > 0) == (labels > 0.5)).tolist()
        return repair_loss, delete_loss, repair_hits, delete_hits

    def train_epoch(self) -> SerialEpochMetrics:
        """One pass over the buffer in shuffled mini-batches."""
        assert self.buffer, 'The buffer is empty.'
        start = time.perf_counter()
        self.policy.train()
        samples = list(self.buffer)
        self.rng.shuffle(samples)
        repair_losses: list[float] = []
        delete_losses: list[float] = []
        repair_hits: list[bool] = []
        delete_hits: list[bool] = []
        for offset in range(0, len(samples), self.batch_size):
            repair_loss, delete_loss, batch_repair_hits, batch_delete_hits = self._losses(samples[offset:offset + self.batch_size])
            self.optimizer.zero_grad()
            (repair_loss + self.delete_loss_weight * delete_loss).backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()
            repair_losses.append(float(repair_loss))
            delete_losses.append(float(delete_loss))
            repair_hits += batch_repair_hits
            delete_hits += batch_delete_hits
        repair_loss_mean = sum(repair_losses) / len(repair_losses)
        delete_loss_mean = sum(delete_losses) / len(delete_losses)
        return SerialEpochMetrics(
            loss=repair_loss_mean + self.delete_loss_weight * delete_loss_mean,
            repair_loss=repair_loss_mean,
            delete_loss=delete_loss_mean,
            repair_accuracy=sum(repair_hits) / len(repair_hits),
            delete_accuracy=sum(delete_hits) / max(len(delete_hits), 1),
            samples=len(samples),
            wall_time=time.perf_counter() - start,
        )


__all__ = [
    'SerialCollectStats',
    'SerialEpochMetrics',
    'SerialPlanEncoder',
    'SerialPolicy',
    'SerialPolicyConfig',
    'SerialPolicyOutput',
    'SerialSample',
    'SerialSearch',
    'SerialSearchOutcome',
    'SerialTrainer',
    'is_valid_plan',
    'serial_collect',
    'serial_oracle_rollout',
    'serial_references',
]
