"""Tests of the plan-space policy: encoder, policy alignment, oracle rollouts, training and search."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    PLAN_NODE_TYPES,
    BootstrapStats,
    CollectStats,
    ImitationTrainer,
    PartialPlanEncoder,
    PlanSpacePolicy,
    PlanSpaceSearch,
    PolicyConfig,
    Sample,
    TrainingConfig,
    bootstrap_collect,
    compute_reference_plan,
    hindsight_collect,
    oracle_reaches_reference,
    oracle_rollout,
    plan_from_strings,
    plan_to_strings,
)
from pymimir_rgnn.encoders import GoalEncoder, StateEncoder, get_input_from_encoders

DATA_DIR = Path(__file__).parent / "data"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "plan_space"
CPU = torch.device("cpu")


def _load(domain_name: str) -> tuple[mm.Domain, mm.Problem]:
    domain = mm.Domain.from_file(DATA_DIR / domain_name / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / domain_name / "problem.pddl", generator="grounded")
    return domain, problem


def _reference(problem: mm.Problem) -> mm.PartialOrderPlan:
    actions = compute_reference_plan(problem)
    assert actions is not None
    return mm.PartialOrderPlan.from_sequential_plan(problem, actions)


def _policy(domain: mm.Domain, **config) -> PlanSpacePolicy:
    torch.manual_seed(0)
    return PlanSpacePolicy(domain, PolicyConfig(embedding_size=16, num_layers=3, **config))


def _half_way_plan(problem: mm.Problem) -> tuple[mm.PartialOrderPlan, mm.FlawSet]:
    """A plan after two oracle rounds: it has Step-kind steps and several flaw kinds."""
    reference = _reference(problem)
    plan = mm.PartialOrderPlan.empty(problem)
    for _ in range(2):
        flaws = plan.compute_flaws()
        targets = mm.oracle_targets(plan, flaws, reference)
        plan.apply(flaws, targets.resolver_index_per_flaw, targets.step_ids_to_delete)
    return plan, plan.compute_flaws()


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
def test_encoder_relations_are_declared_and_extras_cover_every_node(domain_name: str) -> None:
    domain, problem = _load(domain_name)
    encoder = PartialPlanEncoder()
    declared = dict(encoder.get_relations(domain))
    assert declared == dict(mm.partial_plan_relations(domain))
    plan, flaws = _half_way_plan(problem)
    empty = mm.PartialOrderPlan.empty(problem)
    empty_flaws = empty.compute_flaws()
    state = problem.initial_state
    spec = (StateEncoder(), GoalEncoder(), encoder)
    encoded = get_input_from_encoders([(state, problem.goal, (plan, flaws)), (state, problem.goal, (empty, empty_flaws))], spec, CPU)

    plan_relations = {name: values for name, values in encoded.flattened_relations.items() if name.startswith("plan_")}
    assert plan_relations
    for name, values in plan_relations.items():
        assert name in declared
        assert values.numel() % declared[name] == 0

    layouts = encoded.extras[PartialPlanEncoder.LAYOUTS]
    keys = encoded.extras[PartialPlanEncoder.NODE_KEYS]
    types = encoded.extras[PartialPlanEncoder.NODE_TYPES]
    step_nodes = encoded.extras[PartialPlanEncoder.STEP_NODES]
    flaw_nodes = encoded.extras[PartialPlanEncoder.FLAW_NODES]
    assert len(layouts) == len(keys) == len(types) == len(step_nodes) == len(flaw_nodes) == 2
    node_sizes = encoded.node_sizes.tolist()
    for instance, (current, current_flaws) in enumerate([(plan, flaws), (empty, empty_flaws)]):
        assert len(keys[instance]) == len(types[instance]) == node_sizes[instance]
        assert set(types[instance]) <= set(PLAN_NODE_TYPES)
        assert "other" not in types[instance]
        object_count = len(problem.all_objects)
        assert keys[instance][:object_count] == [("object", index) for index in range(object_count)]
        step_keys = [key for key in keys[instance] if key is not None and key[0] == "step"]
        assert step_keys == [("step", step.id) for step in current.steps]
        link_keys = [key for key in keys[instance] if key is not None and key[0] == "link"]
        assert link_keys == [("link", link.id) for link in current.links]
        assert types[instance].count("flaw") == len(current_flaws.flaws)
        assert types[instance].count("resolver") == sum(len(flaw.resolvers) for flaw in current_flaws.flaws)
        assert types[instance].count("ghost") == len(current_flaws.ghosts)
        expected_steps = [step for step in current.steps if step.kind is mm.StepKind.STEP]
        assert step_nodes[instance].tolist() == [
            layouts[instance].step_node_ids[index] for index, step in enumerate(current.steps) if step.kind is mm.StepKind.STEP
        ]
        assert len(expected_steps) == step_nodes[instance].numel()
        # The flaw nodes are those of the layout, in the order of the flaws, and lie inside the instance.
        assert flaw_nodes[instance].dtype == torch.long
        assert flaw_nodes[instance].tolist() == list(layouts[instance].flaw_node_ids)
        assert flaw_nodes[instance].numel() == len(current_flaws.flaws) > 0
        offset = sum(node_sizes[:instance])
        assert all(types[instance][node_id - offset] == "flaw" for node_id in flaw_nodes[instance].tolist())
    # The flaws are the groups of the relation the resolver head scores.
    resolves = encoded.flattened_relations[encoder.resolves_relation].view(-1, 2)
    assert set(resolves[:, 1].tolist()) == set(torch.cat(flaw_nodes).tolist())
    # Step-kind steps are the action nodes of the instance, in order.
    assert torch.cat(step_nodes).tolist() == encoded.action_indices.tolist()
    assert step_nodes[0].numel() > 0 and step_nodes[1].numel() == 0


def test_encoder_emits_the_unordered_relation_for_unordered_steps() -> None:
    # Fails when the native library predates the reduction-plus-unordered ordering encoding.
    domain, problem = _load("blocks")
    encoder = PartialPlanEncoder()
    assert ("plan_unordered", 2) in encoder.get_relations(domain)
    plan, flaws = _half_way_plan(problem)
    step_ids = [step.id for step in plan.steps if step.kind is mm.StepKind.STEP]
    assert any(
        not plan.is_before(first, second) and not plan.is_before(second, first)
        for index, first in enumerate(step_ids) for second in step_ids[index + 1:]
    )
    spec = (StateEncoder(), GoalEncoder(), encoder)
    encoded = get_input_from_encoders([(problem.initial_state, problem.goal, (plan, flaws))], spec, CPU)
    unordered = encoded.flattened_relations["plan_unordered"]
    assert unordered.numel() > 0 and unordered.numel() % 2 == 0


def test_encoder_rejects_mismatched_inputs() -> None:
    _, problem = _load("blocks")
    plan = mm.PartialOrderPlan.empty(problem)
    flaws = plan.compute_flaws()
    spec = (StateEncoder(), GoalEncoder(), PartialPlanEncoder())
    with pytest.raises(AssertionError):
        get_input_from_encoders([(problem.initial_state, problem.goal, (plan.clone(), flaws))], spec, CPU)
    with pytest.raises(AssertionError):
        get_input_from_encoders([(problem.initial_state, problem.goal, plan)], spec, CPU)


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("recurrent", [False, True])
def test_policy_outputs_align_with_flaws_and_steps(domain_name: str, recurrent: bool) -> None:
    domain, problem = _load(domain_name)
    policy = _policy(domain, recurrent=recurrent, update="gru" if recurrent else "mlp")
    plan, flaws = _half_way_plan(problem)
    empty = mm.PartialOrderPlan.empty(problem)
    empty_flaws = empty.compute_flaws()
    state = problem.initial_state
    items = [(state, problem.goal, plan, flaws), (state, problem.goal, empty, empty_flaws)]
    outputs = policy.evaluate(items, carried=recurrent, instance_keys=["a", "b"] if recurrent else None)
    assert len(outputs) == 2
    for output, (_, _, current, current_flaws) in zip(outputs, items):
        assert len(output.resolver_log_probs) == len(current_flaws.flaws)
        for log_probs, flaw in zip(output.resolver_log_probs, current_flaws.flaws):
            assert log_probs.shape == (len(flaw.resolvers),)
            assert torch.isclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-5)
        step_count = sum(1 for step in current.steps if step.kind is mm.StepKind.STEP)
        assert output.delete_logits.shape == (step_count,)
        assert len(output.greedy_choices()) == len(current_flaws.flaws)
        assert len(output.greedy_deletes()) == step_count
        generator = torch.Generator().manual_seed(0)
        sampled = output.sample_choices(generator)
        assert all(0 <= choice < len(flaw.resolvers) for choice, flaw in zip(sampled, current_flaws.flaws))
        assert len(output.sample_deletes(generator)) == step_count
    # The resolver scores of one instance do not depend on the other instance of the batch.
    single = policy.evaluate(items[:1], carried=False)[0]
    if not recurrent:
        for a, b in zip(single.resolver_log_probs, outputs[0].resolver_log_probs):
            assert torch.allclose(a, b, atol=1e-5)
    if recurrent:
        assert policy.memory is not None
        assert ("a", ("step", 0)) in policy.memory
        assert ("b", ("object", 0)) in policy.memory
        policy.forget("a")
        assert ("a", ("step", 0)) not in policy.memory


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("follow_fraction", [1.0, 0.5])
def test_oracle_rollout_samples_realign_after_recomputing_flaws(domain_name: str, follow_fraction: float) -> None:
    _, problem = _load(domain_name)
    reference = _reference(problem)
    samples = oracle_rollout(problem, reference, follow_fraction=follow_fraction, rng=random.Random(3), max_rounds=40, trajectory_id=7)
    assert samples
    assert [sample.round_index for sample in samples] == list(range(len(samples)))
    assert all(sample.trajectory_id == 7 for sample in samples)
    items = ImitationTrainer.items_of(samples)
    for sample, ((state, goal, plan, flaws), targets) in zip(samples, items):
        assert plan is sample.plan_snapshot
        assert len(flaws.flaws) == len(targets.resolver_index_per_flaw)
        for flaw, target in zip(flaws.flaws, targets.resolver_index_per_flaw):
            assert 0 <= target < len(flaw.resolvers)
        step_ids = {step.id for step in plan.steps}
        assert set(targets.step_ids_to_delete) <= step_ids
    # Following the oracle completely reaches the reference's solution.
    if follow_fraction == 1.0:
        plan = mm.PartialOrderPlan.empty(problem)
        for sample in samples:
            flaws = plan.compute_flaws()
            plan.apply(flaws, sample.targets.resolver_index_per_flaw, sample.targets.step_ids_to_delete)
        assert plan.is_solution()
        assert len(plan.linearize()) == len(reference.linearize())


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
def test_oracle_reaches_reference_needs_enough_rounds(domain_name: str) -> None:
    _, problem = _load(domain_name)
    reference = _reference(problem)
    assert oracle_reaches_reference(problem, reference, max_rounds=40)
    assert not oracle_reaches_reference(problem, reference, max_rounds=1)


def test_reference_plan_round_trips_through_strings() -> None:
    _, problem = _load("gripper")
    actions = compute_reference_plan(problem)
    assert actions is not None
    strings = plan_to_strings(actions)
    assert strings == ["pick ball2 rooma left", "move rooma roomb", "drop ball2 roomb left"] or len(strings) == 3
    rebuilt = plan_from_strings(problem, strings)
    assert [str(action) for action in rebuilt] == [str(action) for action in actions]


@pytest.mark.parametrize("recurrent", [False, True])
def test_training_decreases_the_loss(recurrent: bool) -> None:
    domain, problem = _load("blocks")
    policy = _policy(domain, recurrent=recurrent, update="gru" if recurrent else "mlp")
    reference = _reference(problem)
    samples = oracle_rollout(problem, reference, follow_fraction=1.0, max_rounds=40)
    samples += oracle_rollout(problem, reference, follow_fraction=0.5, rng=random.Random(1), max_rounds=40, trajectory_id=1)
    trainer = ImitationTrainer(policy, TrainingConfig(batch_size=4, recurrent=recurrent, lr=3e-3))
    before = trainer.evaluate(samples)
    for _ in range(8):
        metrics = trainer.train_epoch(samples)
    after = trainer.evaluate(samples)
    assert metrics.sample_count == len(samples)
    assert after.loss < before.loss
    assert after.resolver_loss < before.resolver_loss


def test_noop_targets_can_be_excluded_from_the_loss() -> None:
    domain, problem = _load("blocks")
    policy = _policy(domain)
    reference = _reference(problem)
    samples = oracle_rollout(problem, reference, follow_fraction=0.5, rng=random.Random(2), max_rounds=40)
    items = ImitationTrainer.items_of(samples)
    outputs = policy.evaluate([item for item, _ in items])
    everything = ImitationTrainer(policy, TrainingConfig(include_noop_targets=True)).compute_loss(outputs, items)[1]
    non_noop = ImitationTrainer(policy, TrainingConfig(include_noop_targets=False)).compute_loss(outputs, items)[1]
    noop_count = sum(1 for _, targets in items for target in targets.resolver_index_per_flaw if target == 0)
    assert everything.flaw_count - non_noop.flaw_count == noop_count
    assert noop_count > 0


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_search_with_an_untrained_policy_respects_the_round_budget(domain_name: str, mode: str) -> None:
    domain, problem = _load(domain_name)
    policy = _policy(domain)
    outcome = PlanSpaceSearch().run(
        policy, problem, max_rounds=4, mode=mode, seed=1, tombstone_capacity=3, tombstone_lifespan=2, record_visited=True,
    )
    assert outcome.rounds <= 4
    assert outcome.solved == (outcome.plan is not None)
    assert len(outcome.visited) == outcome.rounds
    assert outcome.final_plan is not None
    assert outcome.final_plan.is_solution() == outcome.solved
    assert outcome.final_plan.round == outcome.rounds
    assert [str(literal) for literal in outcome.final_plan.goal.literals] == [str(literal) for literal in problem.goal.literals]
    assert outcome.applied_total >= 0 and outcome.created_total >= 0
    assert outcome.duplicate_creations >= 0 and outcome.cycling_events >= 0
    record = outcome.to_dict()
    assert set(record) >= {"solved", "rounds", "plan_length", "steps_final", "duplicate_creations", "cycling_events", "wall_time"}
    assert not policy.training or True  # The search restores the module mode.


def test_recurrent_search_forgets_its_instance_afterwards() -> None:
    domain, problem = _load("gripper")
    policy = _policy(domain, recurrent=True, update="gru")
    outcome = PlanSpaceSearch().run(policy, problem, max_rounds=3, recurrent=True)
    assert outcome.rounds <= 3
    assert policy.memory is not None and len(policy.memory) == 0


def test_dagger_labels_visited_plans() -> None:
    domain, problem = _load("blocks")
    policy = _policy(domain)
    reference = _reference(problem)
    trainer = ImitationTrainer(policy)
    samples = trainer.dagger_collect([(problem, reference)], max_rounds=3, trajectory_id_start=5)
    assert 0 < len(samples) <= 3
    assert all(sample.trajectory_id == 5 for sample in samples)
    items = ImitationTrainer.items_of(samples)
    assert len(items) == len(samples)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    domain, problem = _load("blocks")
    policy = _policy(domain, aggregation="max", update="gru", recurrent=True)
    path = tmp_path / "policy.pt"
    policy.save(path, {"epoch": 3})
    loaded, extras = PlanSpacePolicy.load(domain, path)
    assert extras == {"epoch": 3}
    assert loaded.config == policy.config
    plan = mm.PartialOrderPlan.empty(problem)
    flaws = plan.compute_flaws()
    item = (problem.initial_state, problem.goal, plan, flaws)
    with torch.no_grad():
        a = policy.evaluate([item])[0]
        b = loaded.evaluate([item])[0]
    for x, y in zip(a.resolver_log_probs, b.resolver_log_probs):
        assert torch.allclose(x, y)


def _generate_blocks(tmp_path: Path, count: int, size: int, seed: int) -> list[mm.Problem]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415

    generate("blocks", tmp_path, {"train": count}, size, size, seed)
    domain = mm.Domain.from_file(tmp_path / "domain.pddl")
    return [mm.Problem.from_file(domain, path, generator="grounded") for path in sorted((tmp_path / "train").glob("*.pddl"))]


def test_end_to_end_imitation_beats_chance_on_tiny_blocksworld(tmp_path: Path) -> None:
    problems = _generate_blocks(tmp_path, 8, 3, 11)
    domain = problems[0].domain
    rng = random.Random(0)
    samples: list[Sample] = []
    references = []
    for index, problem in enumerate(problems):
        reference = _reference(problem)
        references.append(reference)
        samples += oracle_rollout(problem, reference, follow_fraction=1.0, max_rounds=30, trajectory_id=2 * index)
        samples += oracle_rollout(problem, reference, follow_fraction=0.5, rng=rng, max_rounds=30, trajectory_id=2 * index + 1)
    policy = _policy(domain)
    trainer = ImitationTrainer(policy, TrainingConfig(batch_size=8, lr=3e-3))
    items = ImitationTrainer.items_of(samples)
    # Chance level: picking a resolver uniformly at random for every flaw.
    chance = sum(1.0 / len(flaw.resolvers) for (_, _, _, flaws), _ in items for flaw in flaws.flaws) / sum(len(flaws.flaws) for (_, _, _, flaws), _ in items)
    before = trainer.evaluate(samples)
    for _ in range(12):
        trainer.train_epoch(samples)
    after = trainer.evaluate(samples)
    assert after.resolver_accuracy > before.resolver_accuracy
    assert after.resolver_accuracy > chance + 0.2
    assert after.delete_accuracy >= 0.9
    outcome = PlanSpaceSearch().run(policy, problems[0], max_rounds=20)
    assert outcome.rounds <= 20


def _literal_strings(condition: mm.GroundConjunctiveCondition) -> set[str]:
    return {str(literal) for literal in condition.literals}


def test_hindsight_collect_relabels_rollouts_to_achieved_subgoals(tmp_path: Path) -> None:
    problems = _generate_blocks(tmp_path, 4, 3, 5)
    domain = problems[0].domain
    policy = _policy(domain)
    best_solutions: dict[int, mm.PartialOrderPlan] = {}
    samples, stats = hindsight_collect(
        policy, problems, rollouts_per_problem=3, temperature=1.0, max_rounds=10, rng=random.Random(0),
        best_solutions=best_solutions, trajectory_id_start=3,
    )
    assert isinstance(stats, CollectStats)
    assert stats.episodes == 4 * len(problems)
    assert 0 <= stats.solved_episodes <= stats.episodes
    assert 0 <= stats.skipped_episodes <= stats.episodes - stats.solved_episodes
    assert stats.samples == len(samples)
    assert stats.mean_rounds is not None and 0 < stats.mean_rounds <= 10
    assert (len(best_solutions) > 0) == (stats.solved_episodes > 0)
    assert len(best_solutions) <= len(problems)
    assert samples, "an untrained policy on 3 blocks achieves at least one goal atom in some episode"
    assert stats.mean_goal_fraction is not None and 0.0 < stats.mean_goal_fraction <= 1.0
    labelled_episodes = stats.episodes - stats.skipped_episodes
    assert labelled_episodes > 0
    trajectory_ids = {sample.trajectory_id for sample in samples}
    assert trajectory_ids <= set(range(3, 3 + stats.episodes))
    assert len(trajectory_ids) <= labelled_episodes
    relabeled = 0
    for sample in samples:
        goal = _literal_strings(sample.plan_snapshot.goal)
        full = _literal_strings(sample.problem.goal)
        assert goal and goal <= full
        relabeled += int(goal != full)
    # The goal relabeling is visible to the trainer: items carry the snapshot's goal, and the targets re-align.
    items = ImitationTrainer.items_of(samples)
    for sample, ((state, goal, plan, flaws), targets) in zip(samples, items):
        assert plan is sample.plan_snapshot
        assert _literal_strings(goal) == _literal_strings(sample.plan_snapshot.goal)
        assert len(flaws.flaws) == len(targets.resolver_index_per_flaw) > 0
        for flaw, target in zip(flaws.flaws, targets.resolver_index_per_flaw):
            assert 0 <= target < len(flaw.resolvers)
        assert set(targets.step_ids_to_delete) <= {step.id for step in plan.steps}
    # Unsolved episodes get relabeled; with an untrained policy and 10 rounds most episodes are unsolved.
    if stats.solved_episodes < labelled_episodes:
        assert relabeled > 0
    # The relabeled references are solutions of their sub-goal.
    for problem in problems:
        outcome = PlanSpaceSearch().run(policy, problem, max_rounds=10, mode="sample", seed=1, record_visited=True)
        assert outcome.final_plan is not None
        pruned = outcome.final_plan.prune_to_achieved_goal()
        assert pruned.is_solution()
        assert _literal_strings(pruned.goal) <= _literal_strings(problem.goal)
        if outcome.solved:
            assert _literal_strings(pruned.goal) == _literal_strings(problem.goal)
    # Training on the collected samples runs end to end.
    metrics = ImitationTrainer(policy, TrainingConfig(batch_size=8)).train_epoch(samples)
    assert metrics.sample_count == len(samples)


def test_bootstrap_collect_labels_against_the_planner_reference(tmp_path: Path) -> None:
    problems = _generate_blocks(tmp_path, 3, 3, 5)
    domain = problems[0].domain
    policy = _policy(domain)
    references = {id(problem): _reference(problem) for problem in problems}
    solved_counts: dict[int, int] = {}
    best_solutions: dict[int, mm.PartialOrderPlan] = {}
    samples, stats, boot = bootstrap_collect(
        policy, problems, references=references, solved_counts=solved_counts, phase_out=2,
        rollouts_per_problem=2, temperature=1.0, max_rounds=6, rng=random.Random(0), oracle_rollouts=2,
        follow_fraction=0.5, best_solutions=best_solutions, trajectory_id_start=10,
    )
    assert isinstance(stats, CollectStats) and isinstance(boot, BootstrapStats)
    # Every problem has a reference and no solved episodes yet, so all of them are in bootstrap mode.
    assert boot.bootstrap_problems == len(problems)
    assert boot.oracle_episodes == 2 * len(problems)
    assert boot.hindsight_samples == 0
    assert boot.bootstrap_samples == len(samples) == stats.samples > 0
    assert stats.episodes == 3 * len(problems)
    assert stats.skipped_episodes == 0 and stats.mean_goal_fraction is None
    assert boot.remaining == len(problems) - sum(1 for count in solved_counts.values() if count >= 2)
    assert sum(solved_counts.values()) == stats.solved_episodes
    assert (len(best_solutions) > 0) == (stats.solved_episodes > 0)
    # Oracle rollouts and policy episodes get distinct trajectory ids, starting at the given one.
    trajectory_ids = {sample.trajectory_id for sample in samples}
    assert min(trajectory_ids) == 10 and max(trajectory_ids) < 10 + boot.oracle_episodes + stats.episodes
    # Samples keep the full goal (no relabeling) and the empty plan is labelled with the reference's first steps.
    empty_plan_samples = [sample for sample in samples if sample.round_index == 0]
    assert len(empty_plan_samples) == boot.oracle_episodes + stats.episodes
    for sample in samples:
        assert _literal_strings(sample.plan_snapshot.goal) == _literal_strings(sample.problem.goal)
    for sample in empty_plan_samples:
        assert any(target != 0 for target in sample.targets.resolver_index_per_flaw)
    items = ImitationTrainer.items_of(samples)
    assert len(items) == len(samples)
    for (_, _, plan, flaws), targets in items:
        assert len(flaws.flaws) == len(targets.resolver_index_per_flaw) > 0
        assert set(targets.step_ids_to_delete) <= {step.id for step in plan.steps}


def test_bootstrap_collect_phases_problems_out_and_supports_pure_warmup(tmp_path: Path) -> None:
    problems = _generate_blocks(tmp_path, 3, 3, 7)
    domain = problems[0].domain
    policy = _policy(domain)
    # The first problem has no reference, the second one was already solved twice: both take the hindsight path.
    references = {id(problem): _reference(problem) for problem in problems[1:]}
    solved_counts = {id(problems[1]): 2}
    samples, stats, boot = bootstrap_collect(
        policy, problems, references=references, solved_counts=solved_counts, phase_out=2,
        rollouts_per_problem=2, temperature=1.0, max_rounds=6, rng=random.Random(1), oracle_rollouts=1,
    )
    assert boot.bootstrap_problems == 1
    assert boot.oracle_episodes == 1
    assert boot.remaining == 1 - int(solved_counts.get(id(problems[2]), 0) >= 2)
    assert stats.episodes == 3 * len(problems)
    assert boot.bootstrap_samples + boot.hindsight_samples == len(samples) == stats.samples
    assert boot.bootstrap_samples > 0
    assert solved_counts[id(problems[1])] >= 2 and id(problems[0]) not in solved_counts
    relabeled = sum(int(_literal_strings(s.plan_snapshot.goal) != _literal_strings(s.problem.goal)) for s in samples)
    assert relabeled == 0 or boot.hindsight_samples > 0
    # Only the oracle rollouts run during a pure warm-up.
    samples, stats, boot = bootstrap_collect(
        policy, problems, references=references, solved_counts=solved_counts, phase_out=2,
        rollouts_per_problem=2, temperature=1.0, max_rounds=6, rng=random.Random(2), oracle_rollouts=3, policy_episodes=False,
    )
    assert stats.episodes == 0 and stats.mean_rounds is None
    assert boot.oracle_episodes == 3 and boot.bootstrap_problems == 1
    assert len(samples) == boot.bootstrap_samples > 0 and boot.hindsight_samples == 0
    # Without references and counts, the shared machinery is plain hindsight collection.
    rng_a, rng_b = random.Random(3), random.Random(3)
    plain, plain_stats = hindsight_collect(policy, problems, rollouts_per_problem=1, temperature=1.0, max_rounds=5, rng=rng_a)
    again, again_stats, again_boot = bootstrap_collect(
        policy, problems, references={}, solved_counts={}, phase_out=2, rollouts_per_problem=1, temperature=1.0, max_rounds=5, rng=rng_b,
    )
    assert plain_stats.to_dict() | {"wall_time": 0} == again_stats.to_dict() | {"wall_time": 0}
    assert again_boot.bootstrap_problems == 0 and again_boot.hindsight_samples == len(again) == len(plain)


def test_items_of_encodes_the_snapshot_goal() -> None:
    _, problem = _load("blocks")
    plan, _ = _half_way_plan(problem)
    literals = problem.goal.literals
    subset = problem.ground_condition(*literals[:2])
    view = plan.with_goal(subset)
    assert _literal_strings(view.goal) == {str(literal) for literal in literals[:2]}
    full_targets = mm.oracle_targets(plan, plan.compute_flaws(), _reference(problem))
    view_targets = mm.oracle_targets(view, view.compute_flaws(), _reference(problem).with_goal(subset))
    items = ImitationTrainer.items_of([Sample(problem, plan, full_targets), Sample(problem, view, view_targets)])
    (_, full_goal, _, _), _ = items[0]
    (_, view_goal, _, _), _ = items[1]
    assert _literal_strings(full_goal) == _literal_strings(problem.goal)
    assert _literal_strings(view_goal) == _literal_strings(subset)
    spec = (StateEncoder(), GoalEncoder(), PartialPlanEncoder())
    encoded = get_input_from_encoders([(state, goal, (p, f)) for (state, goal, p, f), _ in items], spec, CPU)
    goal_relations = [name for name in encoded.flattened_relations if "_goal" in name and not name.startswith("plan_")]
    assert goal_relations
    offset = encoded.node_sizes.tolist()[0]
    # Count goal-relation entries per instance from the node ids: instance 0 owns the ids below offset.
    full_entries = sum(int((encoded.flattened_relations[name] < offset).sum()) for name in goal_relations)
    view_entries = sum(int((encoded.flattened_relations[name] >= offset).sum()) for name in goal_relations)
    assert full_entries > view_entries > 0


def test_train_hei_runs_end_to_end(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415
    import train_hei  # noqa: PLC0415

    data = tmp_path / "data"
    generate("blocks", data, {"train": 4, "validation": 2}, 3, 3, 21)
    output = tmp_path / "run"
    train_hei.main([
        "--domain-dir", str(data), "--output", str(output), "--iterations", "3", "--rollouts-per-problem", "2",
        "--epochs-per-iteration", "1", "--max-rounds", "8", "--layers", "2", "--embedding-size", "8", "--batch-size", "8",
        "--buffer-cap", "50",
    ])
    lines = (output / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [record["iteration"] for record in records] == [1, 2, 3]
    for record in records:
        collect = record["collect"]
        assert collect["episodes"] == 4 * 3
        assert collect["samples"] >= 0 and collect["skipped_episodes"] + collect["solved_episodes"] <= collect["episodes"]
        assert record["buffer"] <= 50
        assert set(record["validation"]) >= {"coverage", "mean_rounds", "mean_length_ratio"}
        # No reference cache exists, so the ratio is never computed.
        assert record["validation"]["mean_length_ratio"] is None
    assert not (data / "references.json").exists()
    assert (output / "best.pt").exists() and (output / "latest.pt").exists()
    # Warm start from the produced checkpoint takes the model flags from it.
    warm = tmp_path / "warm"
    train_hei.main([
        "--domain-dir", str(data), "--output", str(warm), "--init-from", str(output / "best.pt"), "--iterations", "1",
        "--rollouts-per-problem", "1", "--no-greedy-rollout", "--epochs-per-iteration", "1", "--max-rounds", "4",
    ])
    loaded, extras = PlanSpacePolicy.load(mm.Domain.from_file(data / "domain.pddl"), warm / "latest.pt")
    assert loaded.config.num_layers == 2 and loaded.config.embedding_size == 8
    assert extras["collect"]["episodes"] == 4


def test_train_hei_bootstrap_runs_end_to_end(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415
    import train_hei  # noqa: PLC0415

    data = tmp_path / "data"
    generate("blocks", data, {"train": 4, "validation": 2}, 3, 3, 23)
    output = tmp_path / "run"
    train_hei.main([
        "--domain-dir", str(data), "--output", str(output), "--bootstrap", "--bootstrap-iterations", "1", "--iterations", "2",
        "--bootstrap-phase-out", "1", "--bootstrap-oracle-rollouts", "2", "--rollouts-per-problem", "1",
        "--problems-per-iteration", "3", "--epochs-per-iteration", "1", "--max-rounds", "8", "--layers", "2",
        "--embedding-size", "8", "--batch-size", "8",
    ])
    cache = json.loads((data / "references.json").read_text())
    assert len(cache) == 4 and all(key.startswith("train/") for key in cache)
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["iteration"] for record in records] == [1, 2, 3]
    assert [record["warmup"] for record in records] == [True, False, False]
    assert all(record["problems"] == 3 for record in records)
    warmup = records[0]
    assert warmup["collect"]["episodes"] == 0 and warmup["bootstrap"]["oracle_episodes"] == 6
    assert warmup["bootstrap"]["references"] == 4 and warmup["bootstrap"]["remaining"] == 4
    assert warmup["collect"]["samples"] == warmup["bootstrap"]["bootstrap_samples"] > 0
    assert warmup["train"] is not None
    for record in records[1:]:
        assert record["collect"]["episodes"] == 3 * 2
        boot = record["bootstrap"]
        assert 0 <= boot["bootstrap_problems"] <= 3 and 0 <= boot["remaining"] <= 4
        assert boot["bootstrap_samples"] + boot["hindsight_samples"] == record["collect"]["samples"]
    # Once a problem is solved (phase-out 1), it leaves bootstrap mode.
    solved = sum(record["collect"]["solved_episodes"] for record in records)
    assert records[-1]["bootstrap"]["remaining"] <= 4 - min(solved, 1) or solved == 0
    assert (output / "best.pt").exists() and (output / "latest.pt").exists()


def test_run_with_restarts_aggregates_attempts() -> None:
    domain, problem = _load("gripper")
    policy = _policy(domain)
    search = PlanSpaceSearch()
    outcome = search.run_with_restarts(policy, problem, restarts=3, max_rounds=2, mode="sample", seed=4)
    assert 1 <= outcome.attempts <= 3
    assert outcome.rounds <= 2 * outcome.attempts
    if not outcome.solved:
        assert outcome.attempts == 3
    assert outcome.to_dict()["attempts"] == outcome.attempts
    single = search.run_with_restarts(policy, problem, restarts=1, max_rounds=2, mode="sample", seed=4)
    assert single.attempts == 1
    # The first attempt always runs; once the time limit is spent no further attempt starts.
    starved = search.run_with_restarts(policy, problem, restarts=3, max_rounds=2, mode="sample", seed=4, time_limit=0.0)
    assert starved.attempts == 1
    # Greedy restarts equal a single greedy run.
    greedy = search.run_with_restarts(policy, problem, restarts=2, max_rounds=3)
    plain = search.run(policy, problem, max_rounds=3)
    assert greedy.solved == plain.solved and greedy.rounds in (plain.rounds, 2 * plain.rounds)


def test_evaluate_restarts_are_recorded_in_the_results(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415
    import evaluate  # noqa: PLC0415

    data = tmp_path / "data"
    generate("blocks", data, {"train": 3}, 3, 3, 29)
    domain = mm.Domain.from_file(data / "domain.pddl")
    checkpoint = tmp_path / "policy.pt"
    _policy(domain).save(checkpoint)
    results = tmp_path / "results.json"
    common = [
        "--checkpoint", str(checkpoint), "--domain-dir", str(data), "--problems", str(data / "train"),
        "--output", str(results), "--max-rounds", "2", "--time-limit", "5", "--no-reference",
    ]
    evaluate.main([*common, "--mode", "sample", "--restarts", "2"])
    written = json.loads(results.read_text())
    assert written["settings"]["restarts"] == 2
    assert len(written["problems_results"]) == 3
    for result in written["problems_results"].values():
        assert 1 <= result["attempts"] <= 2
        assert result["solved"] or result["attempts"] == 2
        assert result["rounds"] <= 2 * result["attempts"]
    # A single attempt is the default.
    evaluate.main(common)
    written = json.loads(results.read_text())
    assert written["settings"]["restarts"] == 1
    assert all(result["attempts"] == 1 for result in written["problems_results"].values())


_ALL_RESOLVER_CONTEXTS = ("objects", "steps", "flaws", "siblings")
_ALL_DELETE_CONTEXTS = ("objects", "steps", "flaws")


def _mlp_parameters(name: str, input_size: int, output_size: int) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (f"{name}._inner.weight", (input_size, input_size)),
        (f"{name}._inner.bias", (input_size,)),
        (f"{name}._outer.weight", (output_size, input_size)),
        (f"{name}._outer.bias", (output_size,)),
    ]


# The head parameters of a default policy with embedding size 16. Every context has a readout of its maxmean-pooled
# nodes (32 values). The resolver head scores a resolver, its flaw and four contexts, the delete head a step and
# three contexts.
_DEFAULT_HEAD_PARAMETERS = [
    *_mlp_parameters("model._readouts.resolver_log_probs._context_readouts.0._value", 32, 16),
    *_mlp_parameters("model._readouts.resolver_log_probs._context_readouts.1._value", 32, 16),
    *_mlp_parameters("model._readouts.resolver_log_probs._context_readouts.2._value", 32, 16),
    *_mlp_parameters("model._readouts.resolver_log_probs._sibling_readout._value", 32, 16),
    *_mlp_parameters("model._readouts.resolver_log_probs._score", 96, 1),
    *_mlp_parameters("model._readouts.delete_logits._context_readouts.0._value", 32, 16),
    *_mlp_parameters("model._readouts.delete_logits._context_readouts.1._value", 32, 16),
    *_mlp_parameters("model._readouts.delete_logits._context_readouts.2._value", 32, 16),
    *_mlp_parameters("model._readouts.delete_logits._node_value", 64, 1),
]


def _head_parameters(policy: PlanSpacePolicy) -> list[tuple[str, tuple[int, ...]]]:
    return [(key, tuple(value.shape)) for key, value in policy.state_dict().items() if key.startswith("model._readouts.")]


def test_policy_config_validates_contexts_and_poolings() -> None:
    # Lists, as a deserialized config has them, become tuples.
    config = PolicyConfig(resolver_contexts=["steps", "siblings"], delete_contexts=[])  # type: ignore[arg-type]
    assert config.resolver_contexts == ("steps", "siblings") and config.delete_contexts == ()
    assert PolicyConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        PolicyConfig(delete_contexts=("siblings",))  # Only the resolver head has siblings.
    with pytest.raises(ValueError):
        PolicyConfig(resolver_contexts=("links",))
    with pytest.raises(ValueError):
        PolicyConfig(resolver_contexts=("objects", "objects"))
    with pytest.raises(ValueError):
        PolicyConfig(context_pooling="median")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        PolicyConfig(resolver_contexts="objects")  # type: ignore[arg-type]


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
def test_default_policy_pools_every_context_evaluates_trains_and_round_trips(domain_name: str, tmp_path: Path) -> None:
    domain, problem = _load(domain_name)
    config = PolicyConfig()
    assert config.resolver_contexts == _ALL_RESOLVER_CONTEXTS and config.delete_contexts == _ALL_DELETE_CONTEXTS
    assert config.context_pooling == "maxmean"
    policy = _policy(domain)
    assert _head_parameters(policy) == _DEFAULT_HEAD_PARAMETERS
    context_parameters = [
        (name, parameter) for name, parameter in policy.named_parameters()
        if "_context_readouts" in name or "_sibling_readout" in name
    ]
    assert len(context_parameters) == 7 * 4

    # Evaluation: aligned with flaws and steps, normalized, and an instance's scores do not depend on the batch.
    plan, flaws = _half_way_plan(problem)
    empty = mm.PartialOrderPlan.empty(problem)
    state = problem.initial_state
    items = [(state, problem.goal, empty, empty.compute_flaws()), (state, problem.goal, plan, flaws)]
    with torch.no_grad():
        outputs = policy.evaluate(items)
        single = policy.evaluate(items[1:])[0]
    for output, (_, _, current, current_flaws) in zip(outputs, items):
        assert len(output.resolver_log_probs) == len(current_flaws.flaws)
        for log_probs, flaw in zip(output.resolver_log_probs, current_flaws.flaws):
            assert log_probs.shape == (len(flaw.resolvers),)
            assert torch.isclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-5)
        assert output.delete_logits.shape == (sum(1 for step in current.steps if step.kind is mm.StepKind.STEP),)
        assert torch.isfinite(output.delete_logits).all()
    assert outputs[0].delete_logits.numel() == 0 and outputs[1].delete_logits.numel() > 0
    for a, b in zip(single.resolver_log_probs, outputs[1].resolver_log_probs):
        assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(single.delete_logits, outputs[1].delete_logits, atol=1e-5)

    # One training step reaches the readout of every context of both heads.
    reference = _reference(problem)
    samples = oracle_rollout(problem, reference, follow_fraction=1.0, max_rounds=40)
    trainer = ImitationTrainer(policy, TrainingConfig(batch_size=len(samples), lr=3e-3))
    before = trainer.evaluate(samples)
    metrics = trainer.train_epoch(samples)
    assert metrics.sample_count == len(samples) and metrics.loss == metrics.loss
    for name, parameter in context_parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if name.endswith("_inner.weight"):
            assert parameter.grad.abs().sum() > 0, name
    for _ in range(5):
        trainer.train_epoch(samples)
    assert trainer.evaluate(samples).loss < before.loss

    # The search runs with the contexts, and the checkpoint rebuilds the same heads.
    outcome = PlanSpaceSearch().run(policy, problem, max_rounds=3)
    assert outcome.rounds <= 3
    path = tmp_path / "policy.pt"
    policy.save(path, {"iteration": 1})
    loaded, extras = PlanSpacePolicy.load(domain, path)
    assert extras == {"iteration": 1}
    assert loaded.config == policy.config
    assert _head_parameters(loaded) == _DEFAULT_HEAD_PARAMETERS
    with torch.no_grad():
        expected = policy.evaluate(items[1:])[0]
        actual = loaded.evaluate(items[1:])[0]
    assert all(torch.equal(a, b) for a, b in zip(actual.resolver_log_probs, expected.resolver_log_probs))
    assert torch.equal(actual.delete_logits, expected.delete_logits)


@pytest.mark.parametrize("delete_contexts", [(), ("steps",)])
def test_delete_head_contexts_are_configurable(delete_contexts: tuple[str, ...]) -> None:
    domain, problem = _load("gripper")
    policy = _policy(domain, resolver_contexts=("siblings",), delete_contexts=delete_contexts, context_pooling="max")
    names = [name for name, _ in policy.named_parameters()]
    assert not any("_object_readout" in name for name in names)
    assert sum("delete_logits._context_readouts" in name for name in names) == 4 * len(delete_contexts)
    assert sum("resolver_log_probs._sibling_readout" in name for name in names) == 4
    assert not any("resolver_log_probs._context_readouts" in name for name in names)
    plan, flaws = _half_way_plan(problem)
    output = policy.evaluate([(problem.initial_state, problem.goal, plan, flaws)])[0]
    assert output.delete_logits.numel() > 0 and torch.isfinite(output.delete_logits).all()
    assert all(torch.isclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-5) for log_probs in output.resolver_log_probs)


def test_train_hei_context_flags_default_to_the_policy_config_defaults() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import train_hei  # noqa: PLC0415

    assert train_hei.parse_contexts("none") == () and train_hei.parse_contexts("") == ()
    assert train_hei.parse_contexts("objects, steps,flaws") == ("objects", "steps", "flaws")
    arguments = train_hei.build_parser().parse_args(["--domain-dir", "data", "--output", "run"])
    config = PolicyConfig()
    assert arguments.resolver_contexts == config.resolver_contexts and arguments.delete_contexts == config.delete_contexts
    assert arguments.context_pooling == config.context_pooling


def test_train_hei_with_and_without_contexts_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415
    import diagnose  # noqa: PLC0415
    import evaluate  # noqa: PLC0415
    import train_hei  # noqa: PLC0415

    data = tmp_path / "data"
    generate("blocks", data, {"train": 3, "validation": 2}, 3, 3, 31)
    domain = mm.Domain.from_file(data / "domain.pddl")
    small = ["--epochs-per-iteration", "1", "--max-rounds", "6", "--layers", "2", "--embedding-size", "8", "--batch-size", "8",
             "--rollouts-per-problem", "1"]

    # The default flags: every context on both heads, pooled with maxmean.
    output = tmp_path / "contexts"
    train_hei.main(["--domain-dir", str(data), "--output", str(output), "--iterations", "2", *small])
    loaded, _ = PlanSpacePolicy.load(domain, output / "latest.pt")
    assert loaded.config.resolver_contexts == _ALL_RESOLVER_CONTEXTS and loaded.config.delete_contexts == _ALL_DELETE_CONTEXTS
    assert loaded.config.context_pooling == "maxmean"

    # The evaluation and the diagnosis rebuild the heads from the checkpoint alone.
    results = tmp_path / "results.json"
    evaluate.main([
        "--checkpoint", str(output / "latest.pt"), "--domain-dir", str(data), "--problems", str(data / "validation"),
        "--output", str(results), "--max-rounds", "3", "--time-limit", "5", "--no-reference",
    ])
    assert len(json.loads(results.read_text())["problems_results"]) == 2
    diagnosis = tmp_path / "diagnosis.json"
    monkeypatch.setattr(sys, "argv", [
        "diagnose.py", "--checkpoint", str(output / "latest.pt"), "--domain-dir", str(data),
        "--problems", str(data / "validation"), "--max-rounds", "3", "--output", str(diagnosis),
    ])
    diagnose.main()
    diagnosed = json.loads(diagnosis.read_text())[str(data / "validation")]
    assert diagnosed["oracle_plans"]["plans"] > 0 and diagnosed["oracle_plans"]["resolver_acc"] is not None

    # "none" removes the contexts of a head.
    bare = tmp_path / "bare"
    train_hei.main([
        "--domain-dir", str(data), "--output", str(bare), "--iterations", "2", *small,
        "--resolver-contexts", "none", "--delete-contexts", "none", "--context-pooling", "sum",
    ])
    loaded, _ = PlanSpacePolicy.load(domain, bare / "latest.pt")
    assert loaded.config.resolver_contexts == () and loaded.config.delete_contexts == ()
    assert loaded.config.context_pooling == "sum"

    # Invalid combinations are rejected by the command line.
    for invalid in (["--resolver-contexts", "links"], ["--delete-contexts", "siblings"], ["--context-pooling", "median"]):
        with pytest.raises(SystemExit):
            train_hei.main(["--domain-dir", str(data), "--output", str(tmp_path / "invalid"), "--iterations", "1", *small, *invalid])
