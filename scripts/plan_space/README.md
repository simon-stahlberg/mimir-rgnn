# Plan-space local search with a learned policy

These scripts test whether a neural policy can do ANYCSP-style local search in
plan space. The plan is a sequence of steps (`pymimir.SerialPlan`), each an
action schema with a partial binding. Its simulator applies the effects of every
step even when a precondition fails and reports every error (unsatisfied
precondition, unmet goal literal, unbound parameter). Every round each error
picks one of its repairs (index 0 is NoOp; the others insert an achiever schema
before the flagged step, bind a parameter or rebind one) and every step gets a
delete decision. All choices of a round are applied together, and the plan is a
solution when no errors remain. The policy is an R-GNN
(`pymimir_rgnn.SerialPolicy`) over a graph of the objects, the state, the goal
and the plan nodes (steps, parameters, candidates, errors, repairs), with a
grouped-softmax head over `serial_repairs(repair, error)` and a scalar delete
head over the steps. The code is `pymimir_rgnn/serial_plan.py` (`SerialPolicy`,
`SerialSearch`, `serial_collect`, `SerialTrainer`).

## Hypothesis

In a sequential planner the number of decisions grows with the plan length.
Here all errors are repaired in parallel every round, so the number of rounds
should scale with the *causal depth* of the plan rather than with its length.
Gripper is the test case: moving `n` balls needs about `4n` actions but the
causal chains (`pick -> move -> drop`) of the balls are independent, so the
rounds should stay roughly flat as `n` grows. Blocksworld is the sequential
control: towers impose long causal chains, so rounds should grow with the number
of blocks.

## Pipeline

1. **Instances**: `generate_instances.py` writes `domain.pddl` (the fixture
   domain of `tests/data`) and `train/`, `validation/`, `test/` folders of
   solvable random instances.

       python scripts/plan_space/generate_instances.py --domain gripper --output data/gripper \
           --min-size 2 --max-size 4 --train 20 --validation 5 --test 10 --seed 0

2. **Training**: `train_serial.py` trains by hindsight expert iteration. Every
   iteration rolls the current policy out on the training problems (one greedy
   episode and `--rollouts-per-problem` sampled ones at `--temperature`) and
   labels the visited plans with the oracle (`pymimir.serial_oracle_targets`)
   against a reference plan. A solved episode is its own reference. An unsolved
   episode is pruned to the part of the goal it achieved
   (`SerialPlan.prune_to_achieved_goal`) and its visited plans are relabelled
   to that sub-goal; episodes that achieve nothing are skipped. Samples go into
   a FIFO buffer (`--buffer-cap`) and the policy trains
   `--epochs-per-iteration` epochs on it.

   From scratch the policy achieves nothing, so training is bootstrapped with
   references from `pymimir.serial_search`, a search over single repairs, with
   `--bootstrap-timeout` seconds per training problem (0 disables the
   bootstrap). A problem with a reference gets `--bootstrap-oracle-rollouts`
   teacher-forced oracle rollouts and has the policy's unsolved episodes
   labelled against the reference, until the policy has solved it
   `--bootstrap-phase-out` times; after that, and for problems without a
   reference, unsolved episodes are labelled in hindsight.
   `--bootstrap-iterations` prepends warm-up iterations of oracle rollouts
   only. The heads pool `objects`, `steps`, `errors` and, for the repair head,
   `siblings` (`--repair-contexts`, `--delete-contexts`, `--context-pooling`).

       python scripts/plan_space/train_serial.py --domain-dir data/gripper --output runs/gripper_serial \
           --bootstrap-iterations 2 --iterations 15 --problems-per-iteration 30 --rollouts-per-problem 1 \
           --max-rounds 40 --layers 8 --embedding-size 32 --update gru

3. **Evaluation**: `evaluate_serial.py` runs the search with a round budget and
   a time limit on a folder of problems, validates every solved plan by
   applying its actions from the initial state, writes `results.json` (summary,
   coverage by number of objects, per-problem results) and prints the tables.
   `--restarts N` with `--mode sample` makes up to N independent attempts per
   problem that share the time limit. No reference plans are used; the plan
   length is reported as is. `cycling_events` counts the steps created with the
   schema, binding and purpose literal that a step deleted earlier in the run
   was created with.

       python scripts/plan_space/evaluate_serial.py --checkpoint runs/gripper_serial/latest.pt --domain-dir data/gripper \
           --max-rounds 60 --time-limit 15

## Cluster

`slurm/train.sbatch`, `slurm/evaluate.sbatch` and `slurm/submit_sweep.sh`
submit these scripts; they take their configuration from environment
variables (see the comments at the top of each file).
