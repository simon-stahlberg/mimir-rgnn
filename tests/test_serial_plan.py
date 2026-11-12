"""Tests of the serial plan-space policy: encoder, policy alignment, oracle rollouts, hindsight relabeling, training and search."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pymimir as mm
import pytest
import torch

from pymimir_rgnn import (
    SerialPlanEncoder,
    SerialPolicy,
    SerialPolicyConfig,
    SerialSample,
    SerialSearch,
    SerialTrainer,
    is_valid_plan,
    serial_collect,
    serial_oracle_rollout,
    serial_references,
)
from pymimir_rgnn.encoders import GoalEncoder, StateEncoder, get_input_from_encoders

DATA_DIR = Path(__file__).parent / "data"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "plan_space"
CPU = torch.device("cpu")


def _load(domain_name: str) -> tuple[mm.Domain, mm.Problem]:
    domain = mm.Domain.from_file(DATA_DIR / domain_name / "domain.pddl")
    problem = mm.Problem.from_file(domain, DATA_DIR / domain_name / "problem.pddl", generator="grounded")
    return domain, problem


def _reference(problem: mm.Problem) -> mm.SerialPlan:
    reference = mm.serial_search(problem)
    assert reference is not None
    return reference


def _policy(domain: mm.Domain, **config) -> SerialPolicy:
    torch.manual_seed(0)
    return SerialPolicy(domain, SerialPolicyConfig(embedding_size=16, num_layers=3, **config))


def _oracle_rounds(plan: mm.SerialPlan, reference: mm.SerialPlan, rounds: int) -> None:
    for _ in range(rounds):
        diagnosis = plan.diagnose()
        targets = mm.serial_oracle_targets(plan, diagnosis, reference)
        plan.apply(diagnosis, targets.repair_index_per_error, targets.step_ids_to_delete)


def _half_way_plan(problem: mm.Problem) -> mm.SerialPlan:
    """A plan after two oracle rounds: it has steps and errors."""
    plan = mm.SerialPlan.empty(problem)
    _oracle_rounds(plan, _reference(problem), 2)
    assert plan.steps and not plan.is_solution()
    return plan


def _generate(tmp_path: Path, domain_name: str, splits: dict[str, int], size: int, seed: int) -> list[mm.Problem]:
    """Generate instances and return the problems of the ``train`` split."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_instances import generate  # noqa: PLC0415

    generate(domain_name, tmp_path, splits, size, size, seed)
    domain = mm.Domain.from_file(tmp_path / "domain.pddl")
    return [mm.Problem.from_file(domain, path, generator="grounded") for path in sorted((tmp_path / "train").glob("*.pddl"))]


def _literal_strings(condition: mm.GroundConjunctiveCondition) -> set[str]:
    return {str(literal) for literal in condition.literals}


def _assert_labels_line_up(sample: SerialSample) -> None:
    """The targets of a sample fit the errors and steps diagnosed again from its snapshot."""
    diagnosis = sample.plan.diagnose()
    assert len(diagnosis.errors) == len(sample.targets.repair_index_per_error) > 0
    for error, target in zip(diagnosis.errors, sample.targets.repair_index_per_error):
        assert 0 <= target < len(error.repairs)
    assert set(sample.targets.step_ids_to_delete) <= {step.id for step in sample.plan.steps}


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
def test_encoder_relations_are_declared_and_extras_list_steps_and_errors(domain_name: str) -> None:
    domain, problem = _load(domain_name)
    encoder = SerialPlanEncoder()
    declared = dict(encoder.get_relations(domain))
    assert declared == dict(mm.serial_plan_relations(domain))
    plans = [_half_way_plan(problem), mm.SerialPlan.empty(problem)]
    diagnoses = [plan.diagnose() for plan in plans]
    state = problem.initial_state
    spec = (StateEncoder(), GoalEncoder(), encoder)
    encoded = get_input_from_encoders([(state, plan.goal, (plan, diagnosis)) for plan, diagnosis in zip(plans, diagnoses)], spec, CPU)

    serial_relations = {name: values for name, values in encoded.flattened_relations.items() if name.startswith("serial_")}
    assert serial_relations
    for name, values in serial_relations.items():
        assert name in declared
        assert values.numel() % declared[name] == 0

    layouts = encoded.extras[SerialPlanEncoder.LAYOUTS]
    step_nodes = encoded.extras[SerialPlanEncoder.STEP_NODES]
    error_nodes = encoded.extras[SerialPlanEncoder.ERROR_NODES]
    assert len(layouts) == len(step_nodes) == len(error_nodes) == 2
    boundaries = [0, *encoded.node_sizes.cumsum(0).tolist()]
    for instance, (plan, diagnosis) in enumerate(zip(plans, diagnoses)):
        assert step_nodes[instance].dtype == error_nodes[instance].dtype == torch.long
        assert step_nodes[instance].tolist() == list(layouts[instance].step_node_ids)
        assert error_nodes[instance].tolist() == list(layouts[instance].error_node_ids)
        assert step_nodes[instance].numel() == len(plan.steps)
        assert error_nodes[instance].numel() == len(diagnosis.errors) > 0
        for node_id in [*step_nodes[instance].tolist(), *error_nodes[instance].tolist()]:
            assert boundaries[instance] <= node_id < boundaries[instance + 1]
    # The steps are the action nodes, and the errors are the groups of the relation the repair head scores.
    assert encoded.action_indices.tolist() == torch.cat(step_nodes).tolist()
    repairs = encoded.flattened_relations[SerialPlanEncoder.REPAIRS_RELATION].view(-1, 2)
    assert set(repairs[:, 1].tolist()) == set(torch.cat(error_nodes).tolist())


def test_encoder_rejects_mismatched_inputs() -> None:
    _, problem = _load("blocks")
    plan = mm.SerialPlan.empty(problem)
    spec = (StateEncoder(), GoalEncoder(), SerialPlanEncoder())
    with pytest.raises(AssertionError):
        get_input_from_encoders([(problem.initial_state, problem.goal, plan)], spec, CPU)
    with pytest.raises(AssertionError):
        get_input_from_encoders([(problem.initial_state, problem.goal, (plan, plan.clone().diagnose()))], spec, CPU)


def test_policy_config_validates_contexts_and_pooling() -> None:
    # Checkpoints hand in lists.
    assert SerialPolicyConfig(repair_contexts=["objects"]).repair_contexts == ("objects",)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SerialPolicyConfig(repair_contexts=("flaws",))
    with pytest.raises(ValueError):
        SerialPolicyConfig(delete_contexts=("siblings",))
    with pytest.raises(ValueError):
        SerialPolicyConfig(context_pooling="median")  # type: ignore[arg-type]


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("update", ["mlp", "gru"])
def test_policy_outputs_align_with_errors_and_steps(domain_name: str, update: str) -> None:
    domain, problem = _load(domain_name)
    policy = _policy(domain, update=update).eval()
    plans = [_half_way_plan(problem), mm.SerialPlan.empty(problem)]
    items = [(plan, plan.diagnose()) for plan in plans]
    with torch.no_grad():
        outputs = policy.evaluate(items)
        singles = [policy.evaluate([item])[0] for item in items]
    assert len(outputs) == 2
    for (plan, diagnosis), output, single in zip(items, outputs, singles):
        assert len(output.repair_log_probs) == len(diagnosis.errors)
        for error, log_probs, single_log_probs in zip(diagnosis.errors, output.repair_log_probs, single.repair_log_probs):
            assert log_probs.shape == (len(error.repairs),)
            assert torch.allclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-5)
            # Batching does not change which repair a log-probability belongs to.
            assert torch.allclose(log_probs, single_log_probs, atol=1e-5)
        assert output.delete_logits.shape == (len(plan.steps),)
        assert torch.allclose(output.delete_logits, single.delete_logits, atol=1e-5)


def test_default_policy_heads_pool_every_context() -> None:
    domain, _ = _load("gripper")
    config = SerialPolicy(domain).config
    assert config.repair_contexts == ("objects", "steps", "errors", "siblings")
    assert config.delete_contexts == ("objects", "steps", "errors")
    assert config.context_pooling == "maxmean"


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("follow_fraction", [1.0, 0.5])
def test_oracle_rollout_samples_line_up(domain_name: str, follow_fraction: float) -> None:
    _, problem = _load(domain_name)
    reference = _reference(problem)
    samples = serial_oracle_rollout(problem, reference, follow_fraction=follow_fraction, rng=random.Random(0), max_rounds=30)
    assert 0 < len(samples) < 30, "the oracle solves the fixtures well within the round budget"
    assert samples[0].plan.steps == ()
    for sample in samples:
        assert sample.plan.problem is problem
        assert _literal_strings(sample.plan.goal) == _literal_strings(problem.goal)
        _assert_labels_line_up(sample)
    # Applying the targets of the last sample completes the plan, and the result is a valid plan.
    last = samples[-1].plan.clone()
    _oracle_rounds(last, reference, 1)
    assert last.is_solution()
    assert is_valid_plan(problem, last.to_actions())
    if follow_fraction == 1.0:
        assert [str(action) for action in last.to_actions()] == [str(action) for action in reference.to_actions()]


def test_training_decreases_the_loss() -> None:
    domain, problem = _load("blocks")
    reference = _reference(problem)
    rng = random.Random(0)
    samples: list[SerialSample] = []
    for follow_fraction in (1.0, 0.5, 0.5):
        samples += serial_oracle_rollout(problem, reference, follow_fraction=follow_fraction, rng=rng, max_rounds=30)
    policy = _policy(domain)
    trainer = SerialTrainer(policy, batch_size=4, lr=3e-3)
    trainer.add(samples)
    first = trainer.train_epoch()
    for _ in range(15):
        last = trainer.train_epoch()
    assert first.samples == last.samples == len(samples)
    assert last.loss < first.loss
    assert last.repair_loss < first.repair_loss
    assert last.repair_accuracy >= first.repair_accuracy
    assert 0.0 <= last.delete_accuracy <= 1.0


def test_trainer_buffer_drops_the_oldest_samples() -> None:
    domain, problem = _load("gripper")
    samples = serial_oracle_rollout(problem, _reference(problem), follow_fraction=1.0, rng=random.Random(0), max_rounds=30)
    assert len(samples) > 2
    trainer = SerialTrainer(_policy(domain), buffer_cap=2)
    trainer.add(samples)
    assert list(trainer.buffer) == samples[-2:]
    assert trainer.train_epoch().samples == 2


@pytest.mark.parametrize("domain_name", ["blocks", "gripper"])
@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_search_with_an_untrained_policy_respects_the_round_budget(domain_name: str, mode: str) -> None:
    domain, problem = _load(domain_name)
    policy = _policy(domain)
    outcome = SerialSearch().run(policy, problem, max_rounds=5, mode=mode, seed=3, record_visited=True)  # type: ignore[arg-type]
    assert outcome.rounds <= 5
    assert len(outcome.visited) == outcome.rounds
    assert outcome.steps_final == len(outcome.final_plan.steps)
    assert outcome.solved == outcome.final_plan.is_solution()
    assert (outcome.plan is None) == (not outcome.solved)
    assert outcome.created_total - outcome.deleted_total == outcome.steps_final
    assert 0 <= outcome.cycling_events <= outcome.created_total
    assert policy.training, "the search restores the training mode of the policy"
    assert set(outcome.to_dict()) >= {"solved", "rounds", "plan_length", "steps_final", "cycling_events", "wall_time", "attempts"}
    # The time limit is checked before every round.
    assert SerialSearch().run(policy, problem, max_rounds=5, time_limit=0.0).rounds == 0


def test_search_with_a_trained_policy_finds_a_valid_plan() -> None:
    domain, problem = _load("gripper")
    reference = _reference(problem)
    rng = random.Random(0)
    policy = _policy(domain)
    trainer = SerialTrainer(policy, batch_size=8, lr=3e-3)
    for follow_fraction in (1.0, 0.5, 0.5, 0.5):
        trainer.add(serial_oracle_rollout(problem, reference, follow_fraction=follow_fraction, rng=rng, max_rounds=30))
    for _ in range(40):
        trainer.train_epoch()
    outcome = SerialSearch().run(policy, problem, max_rounds=10)
    assert outcome.solved and outcome.plan is not None
    assert outcome.rounds <= 10
    assert outcome.plan_length == outcome.steps_final == len(reference.steps)
    assert is_valid_plan(problem, outcome.plan)
    assert not is_valid_plan(problem, outcome.plan[:-1])
    assert not is_valid_plan(problem, outcome.plan[1:])


def test_run_with_restarts_aggregates_attempts() -> None:
    domain, problem = _load("gripper")
    policy = _policy(domain)
    search = SerialSearch()
    outcome = search.run_with_restarts(policy, problem, restarts=3, max_rounds=2, mode="sample", seed=4)
    assert 1 <= outcome.attempts <= 3
    assert outcome.rounds <= 2 * outcome.attempts
    assert outcome.solved or outcome.attempts == 3
    assert outcome.to_dict()["attempts"] == outcome.attempts
    # The first attempt always runs; once the time limit is spent no further attempt starts.
    starved = search.run_with_restarts(policy, problem, restarts=3, max_rounds=2, mode="sample", seed=4, time_limit=0.0)
    assert starved.attempts == 1


def test_hindsight_relabels_an_unsolved_episode_to_the_goal_it_achieved(tmp_path: Path) -> None:
    problems = _generate(tmp_path, "gripper", {"train": 1}, 3, 7)
    problem = problems[0]
    goal = _literal_strings(problem.goal)
    reference = _reference(problem)
    # An unsolved episode: a few oracle rounds, recorded like SerialSearch records its visited plans.
    plan = mm.SerialPlan.empty(problem)
    visited: list[mm.SerialPlan] = []
    pruned = plan.prune_to_achieved_goal()
    while not 0 < len(pruned.steps) < len(reference.steps):
        visited.append(plan.clone())
        _oracle_rounds(plan, reference, 1)
        assert not plan.is_solution(), "some round of the oracle achieves a part of the goal before all of it"
        pruned = plan.prune_to_achieved_goal()
    achieved = _literal_strings(pruned.goal)
    assert achieved and achieved < goal
    assert pruned.is_solution()

    # The oracle accepts the pruned plan as the reference of the snapshots relabeled to the achieved goal.
    labelled = 0
    for snapshot in visited:
        view = snapshot.with_goal(pruned.goal)
        assert [step.id for step in view.steps] == [step.id for step in snapshot.steps]
        diagnosis = view.diagnose()
        if diagnosis.is_solution:
            continue
        sample = SerialSample(view, mm.serial_oracle_targets(view, diagnosis, pruned))
        _assert_labels_line_up(sample)
        labelled += 1
    assert labelled > 0
    # Following the oracle towards the pruned plan from the empty plan achieves exactly the relabeled goal.
    relabeled = mm.SerialPlan.empty(problem, goal=pruned.goal)
    _oracle_rounds(relabeled, pruned, 30)
    assert relabeled.is_solution()
    assert len(relabeled.steps) == len(pruned.steps)


def test_collect_labels_the_policys_episodes_in_hindsight(tmp_path: Path) -> None:
    problems = _generate(tmp_path, "blocks", {"train": 4}, 3, 5)
    policy = _policy(problems[0].domain)
    solved_counts: dict[mm.Problem, int] = {}
    best_solutions: dict[mm.Problem, mm.SerialPlan] = {}
    samples, stats = serial_collect(
        policy, problems, references={}, solved_counts=solved_counts, best_solutions=best_solutions, phase_out=2,
        oracle_rollouts=1, follow_fraction=0.5, rollouts_per_problem=3, temperature=1.0, max_rounds=8, rng=random.Random(0),
    )
    # Without references every problem takes the hindsight path: one greedy and three sampled episodes each.
    assert stats.episodes == 4 * len(problems)
    assert stats.bootstrap_problems == stats.oracle_episodes == stats.bootstrap_samples == stats.bootstrap_remaining == 0
    assert stats.hindsight_samples == len(samples)
    assert 0 <= stats.solved_episodes <= stats.episodes
    # Skipped are the episodes whose reference has no steps, which includes solved episodes of an already satisfied goal.
    assert 0 <= stats.skipped_episodes <= stats.episodes
    assert stats.mean_rounds is not None and 0 < stats.mean_rounds <= 8
    assert sum(solved_counts.values()) == stats.solved_episodes
    assert (len(best_solutions) > 0) == (stats.solved_episodes > 0)
    assert samples, "an untrained policy keeps at least one goal literal of a 3-block problem in some episode"
    assert stats.mean_goal_fraction is not None and 0.0 < stats.mean_goal_fraction <= 1.0
    relabeled = 0
    for sample in samples:
        goal = _literal_strings(sample.plan.goal)
        full = _literal_strings(sample.plan.problem.goal)
        assert goal and goal <= full
        relabeled += int(goal != full)
        _assert_labels_line_up(sample)
    # Unsolved episodes are relabeled to a part of the goal; an untrained policy leaves most episodes unsolved.
    if stats.solved_episodes == 0:
        assert relabeled == len(samples)
    # The trainer diagnoses the snapshots again and asserts that the labels line up.
    trainer = SerialTrainer(policy, batch_size=8)
    trainer.add(samples)
    assert trainer.train_epoch().samples == len(samples)


def test_collect_bootstraps_from_references_and_phases_them_out(tmp_path: Path) -> None:
    problems = _generate(tmp_path, "blocks", {"train": 3}, 3, 5)
    policy = _policy(problems[0].domain)
    references = serial_references(problems, timeout_seconds=10.0)
    assert set(references) == set(problems)
    assert serial_references(problems, timeout_seconds=0.0) == {}
    options = dict(
        references=references, phase_out=2, oracle_rollouts=2, follow_fraction=0.5, rollouts_per_problem=1, temperature=1.0,
        max_rounds=6, rng=random.Random(0),
    )

    # A warm-up collects only the teacher-forced oracle rollouts.
    samples, stats = serial_collect(policy, problems, solved_counts={}, best_solutions={}, policy_episodes=False, **options)  # type: ignore[arg-type]
    assert stats.episodes == 0 and stats.mean_rounds is None and stats.mean_goal_fraction is None
    assert stats.bootstrap_problems == stats.bootstrap_remaining == len(problems)
    assert stats.oracle_episodes == 2 * len(problems)
    assert stats.bootstrap_samples == len(samples) > 0 and stats.hindsight_samples == 0

    # Bootstrap problems also label the policy's own episodes against the reference, for the full goal.
    solved_counts: dict[mm.Problem, int] = {problems[0]: 2}
    samples, stats = serial_collect(policy, problems, solved_counts=solved_counts, best_solutions={}, **options)  # type: ignore[arg-type]
    assert stats.episodes == 2 * len(problems)
    # The first problem was solved often enough before: it left bootstrap mode and takes the hindsight path.
    assert stats.bootstrap_problems == len(problems) - 1
    assert stats.oracle_episodes == 2 * (len(problems) - 1)
    assert stats.bootstrap_samples + stats.hindsight_samples == len(samples)
    assert stats.bootstrap_remaining == sum(1 for problem in problems if solved_counts.get(problem, 0) < 2)
    for sample in samples:
        _assert_labels_line_up(sample)
        if sample.plan.problem is not problems[0]:
            assert _literal_strings(sample.plan.goal) == _literal_strings(sample.plan.problem.goal)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    domain, problem = _load("blocks")
    policy = _policy(domain, update="gru", repair_contexts=("objects", "siblings"), delete_contexts=(), context_pooling="max")
    path = tmp_path / "policy.pt"
    policy.save(path, {"note": 7})
    loaded, extras = SerialPolicy.load(domain, path)
    assert extras == {"note": 7}
    assert loaded.config == policy.config
    plan = _half_way_plan(problem)
    item = (plan, plan.diagnose())
    with torch.no_grad():
        a = policy.eval().evaluate([item])[0]
        b = loaded.eval().evaluate([item])[0]
    for x, y in zip(a.repair_log_probs, b.repair_log_probs):
        assert torch.allclose(x, y)
    assert torch.allclose(a.delete_logits, b.delete_logits)


def test_train_serial_and_evaluate_serial_run_end_to_end(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import evaluate_serial  # noqa: PLC0415
    import train_serial  # noqa: PLC0415

    data = tmp_path / "data"
    _generate(data, "blocks", {"train": 4, "validation": 2, "test": 3}, 3, 23)
    output = tmp_path / "run"
    train_serial.main([
        "--domain-dir", str(data), "--output", str(output), "--bootstrap-iterations", "1", "--iterations", "2",
        "--bootstrap-phase-out", "1", "--bootstrap-oracle-rollouts", "2", "--rollouts-per-problem", "1",
        "--problems-per-iteration", "3", "--epochs-per-iteration", "1", "--max-rounds", "8", "--layers", "2",
        "--embedding-size", "8", "--batch-size", "8", "--buffer-cap", "60", "--validation-time-limit", "5",
    ])
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["iteration"] for record in records] == [1, 2, 3]
    assert [record["warmup"] for record in records] == [True, False, False]
    assert all(record["problems"] == 3 and record["references"] == 4 and record["buffer"] <= 60 for record in records)
    warmup = records[0]
    assert warmup["collect"]["episodes"] == 0 and warmup["collect"]["oracle_episodes"] == 6
    assert warmup["collect"]["bootstrap_remaining"] == 4
    assert warmup["collect"]["bootstrap_samples"] > 0 and warmup["collect"]["hindsight_samples"] == 0
    for record in records:
        assert set(record["collect"]) >= {"solved_fraction", "mean_goal_fraction", "bootstrap_remaining"}
        assert set(record["train"]) >= {"loss", "repair_accuracy", "delete_accuracy"}
        assert set(record["validation"]) >= {"coverage", "mean_rounds"}
        assert record["validation"]["problems"] == 2
        assert record["wall_time"] > 0
    assert all(record["collect"]["episodes"] == 3 * 2 for record in records[1:])
    assert (output / "best.pt").exists() and (output / "latest.pt").exists()

    # Warm start from the produced checkpoint takes the model flags from it; without a bootstrap everything is hindsight.
    warm = tmp_path / "warm"
    train_serial.main([
        "--domain-dir", str(data), "--output", str(warm), "--init-from", str(output / "latest.pt"), "--iterations", "1",
        "--bootstrap-timeout", "0", "--rollouts-per-problem", "1", "--epochs-per-iteration", "1", "--max-rounds", "4",
    ])
    loaded, extras = SerialPolicy.load(mm.Domain.from_file(data / "domain.pddl"), warm / "latest.pt")
    assert loaded.config.num_layers == 2 and loaded.config.embedding_size == 8
    assert extras["collect"]["episodes"] == 4 * 2 and extras["collect"]["bootstrap_samples"] == 0

    results_path = tmp_path / "results.json"
    common = ["--checkpoint", str(output / "latest.pt"), "--domain-dir", str(data), "--output", str(results_path), "--max-rounds", "6"]
    evaluate_serial.main([*common, "--time-limit", "5"])
    results = json.loads(results_path.read_text())
    assert results["summary"]["problems"] == 3 and results["summary"]["invalid_plans"] == 0
    assert set(results["problems_results"]) == {"test/p000.pddl", "test/p001.pddl", "test/p002.pddl"}
    for result in results["problems_results"].values():
        assert result["objects"] == 3
        assert result["rounds"] <= 6 and result["attempts"] == 1
        assert result["solved"] == (result["valid"] is True) == (result["plan"] is not None)
    assert results["coverage_by_objects"]["3"]["problems"] == 3
    assert results["coverage_by_objects"]["3"]["coverage"] == results["summary"]["coverage"]
    # Sampled restarts on another folder share the time limit.
    evaluate_serial.main([*common, "--problems", str(data / "train"), "--mode", "sample", "--restarts", "2", "--time-limit", "5"])
    results = json.loads(results_path.read_text())
    assert results["settings"]["restarts"] == 2 and len(results["problems_results"]) == 4
    for result in results["problems_results"].values():
        assert result["solved"] or result["attempts"] == 2
