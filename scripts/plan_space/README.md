# Plan-space local search with a learned policy

These scripts test whether a neural policy can do ANYCSP-style local search in
plan space. A partial-order causal-link plan (`pymimir.PartialOrderPlan`) is
edited round by round: every flaw (open condition, threat, unbound parameter)
picks one of its resolvers (index 0 is NoOp) and every step gets a delete
decision; the plan is a solution when no flaws remain. The policy is an R-GNN
(`pymimir_rgnn.PlanSpacePolicy`) over a graph of the objects, the state, the
goal and the plan nodes (steps, parameters, links, ghosts, flaws, resolvers,
tombstones) with a grouped-softmax head over `plan_resolves(resolver, flaw)` and
a scalar delete head over the Step-kind steps.

The step ordering is encoded as the transitive reduction of the ordering
constraints (`plan_before`) plus explicit `plan_unordered` edges between steps
that are not ordered either way. Checkpoints trained before this encoding (when
`plan_before` was the transitive closure and there was no `plan_unordered`) are
incompatible and must be retrained.

## Hypothesis

In a sequential planner the number of decisions grows with the plan length. In
plan space all flaws are resolved in parallel every round, so the number of
rounds should scale with the *causal depth* of the plan rather than with its
length. Gripper is the test case: moving `n` balls needs about `4n` actions but
the causal chains (`pick -> move -> drop`) of the balls are independent, so the
rounds should stay roughly flat as `n` grows. Blocksworld is the sequential
control: towers impose long causal chains, so rounds should grow with the number
of blocks. `evaluate.py` reports rounds and plan length against the reference
length for both domains.

## Pipeline

1. **Instances**: `generate_instances.py` writes `domain.pddl` (the fixture
   domain of `tests/data`) and `train/`, `validation/`, `test/` folders of
   solvable random instances.

       python scripts/plan_space/generate_instances.py --domain gripper --output data/gripper \
           --min-size 2 --max-size 4 --train 20 --validation 5 --test 10 --seed 0

2. **Training**: `train.py` computes reference plans (BFS when small, otherwise
   GBFS with the FF heuristic; cached in `<domain-dir>/references.json`), rolls
   out the oracle (`pymimir.oracle_targets`) from the empty plan with partial
   following (`--follow-fraction`) to diversify the visited plans, and trains by
   supervised imitation: mean NLL of the oracle's resolver per flaw plus BCE on
   the delete decisions. Every epoch runs the greedy search on the validation
   set and logs coverage, rounds, plan length vs the reference, duplicate
   creations and cycling events to `metrics.jsonl`; `best.pt` and `latest.pt`
   are checkpoints. `--dagger` labels the plans visited by the current policy
   with the oracle after each epoch and adds them to the data. `--recurrent`
   carries node embeddings across rounds with `NodeStateStore` (the trainer
   then processes trajectories round by round with detached embeddings).

       python scripts/plan_space/train.py --domain-dir data/gripper --output runs/gripper \
           --epochs 20 --layers 10 --embedding-size 32 --update mlp --aggregation mean

   **Hindsight expert iteration** (`train_hei.py`) is the planner-free
   alternative. Imitation against planner references suffers from label noise:
   the reference picks one of many symmetric solutions, so on the policy's own
   plans the oracle labels many steps "delete" and the policy learns to do
   nothing. HEI drops the planner: every iteration rolls the current policy out
   on the training problems (`--rollouts-per-problem` sampled rollouts at
   `--temperature`, plus one greedy rollout unless `--no-greedy-rollout`) and
   relabels each episode with the plan it achieved
   (`pymimir_rgnn.hindsight_collect`). A solved episode keeps the problem goal
   and uses its final plan as the reference; an unsolved episode is pruned to
   its largest flaw-free sub-plan (`PartialOrderPlan.prune_to_achieved_goal`),
   which becomes the reference for the sub-goal it achieves, and the visited
   plans are relabeled to that sub-goal (`PartialOrderPlan.with_goal`) before
   `oracle_targets` labels them. Plans carry their goal, so the `GoalEncoder`
   sees the relabeled sub-goal (`Sample.plan_snapshot.goal`). Episodes that
   achieve no goal atom are skipped. Samples go into a FIFO buffer
   (`--buffer-cap`), the policy trains `--epochs-per-iteration` epochs on it,
   and validation runs the greedy search. `--init-from` warm-starts from an
   imitation checkpoint (the model flags then come from the checkpoint).
   `--problems-per-iteration` collects on a random subset of the training
   problems each iteration.

   Data requirement: only `domain.pddl`, `train/` and `validation/` are needed;
   without `--bootstrap`, `references.json` is never computed. If a cache
   already exists, validation reports the plan-length ratio for the problems
   it covers, otherwise the ratio is skipped. The per-iteration line and
   `metrics.jsonl` report the solved fraction and the mean achieved-goal
   fraction of the collected episodes.

       python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei \
           --iterations 12 --rollouts-per-problem 3 --temperature 1.0 --epochs-per-iteration 2 \
           --layers 8 --embedding-size 32
       python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei_warm \
           --init-from runs/blocks/best.pt --iterations 12

   **Bootstrapped HEI.** From scratch, HEI collapses: random rollouts never
   complete a causal chain, the pruned goals are trivial and the policy learns
   to do nothing. `--bootstrap` accepts planner dependence for the start.
   Before the first iteration, reference plans of the training problems are
   computed with a short per-instance budget (`--bootstrap-timeout`, default
   10 s; cached in `references.json`); whatever the planner solves in time, and
   the oracle can reproduce (see below), is the bootstrap set, i.e. the easiest
   instances. A problem stays in *bootstrap mode* until the policy has solved
   it `--bootstrap-phase-out` times (default 2) during collection, counting
   sampled and greedy episodes across iterations. Collection for a problem in
   bootstrap mode (`pymimir_rgnn.bootstrap_collect`) runs
   `--bootstrap-oracle-rollouts` (default 1) teacher-forced oracle rollouts
   against the reference with `--follow-fraction` (default 0.5), and labels the
   policy's own episodes against the reference for the full goal (DAgger-style)
   instead of pruning and relabeling them; problems out of bootstrap mode, or
   without a reference, take the hindsight path. `--bootstrap-iterations M`
   prepends M warm-up iterations in which only the oracle rollouts run. The
   per-iteration line and `metrics.jsonl` (`bootstrap`) report the problems in
   bootstrap mode, the problems still to be phased out, and the samples from
   the bootstrap and the hindsight path.

       python scripts/plan_space/train_hei.py --domain-dir data/blocks --output runs/blocks_hei_boot \
           --bootstrap --bootstrap-timeout 10 --bootstrap-iterations 2 --iterations 10 \
           --rollouts-per-problem 2 --layers 8 --embedding-size 32

   **Unusable references.** The oracle (`pymimir.oracle_targets`) matches the
   steps of the current plan to the reference heuristically. On references
   with repeated ground actions (typical of non-optimal GBFS plans; in
   Blocksworld from about 8 blocks on) the matching can flip between the copies,
   the threat rule deletes the "wrong" step and the teacher-forced rollout cycles
   without ever solving the problem. `train.py` and the bootstrap therefore
   check every reference with `pymimir_rgnn.oracle_reaches_reference` (a full
   oracle rollout within `--max-rounds`) and drop the problems whose reference
   the oracle cannot reproduce; the log reports how many.

3. **Evaluation**: `evaluate.py` runs the search with a round budget and a time
   limit on a folder of problems, writes `results.json` and prints a table.
   `--restarts N` with `--mode sample` makes up to N independent attempts per
   problem (seeds `--seed`, `--seed + 1`, ...) that share the time limit; the
   problem counts as solved if any attempt solves it, and `attempts` reports
   how many were made (`PlanSpaceSearch.run_with_restarts`).

       python scripts/plan_space/evaluate.py --checkpoint runs/gripper/best.pt --domain-dir data/gripper \
           --max-rounds 100 --time-limit 60
       python scripts/plan_space/evaluate.py --checkpoint runs/gripper/best.pt --domain-dir data/gripper \
           --max-rounds 100 --time-limit 60 --mode sample --restarts 8

Metrics: `coverage`, `mean_rounds`, `mean_plan_length` / `mean_length_ratio`
(solved plans vs the reference), `duplicate_creations` (a created step whose
schema and binding equal an existing fully bound step), `cycling_events` (a
schema and binding adopted again after a step with it was deleted earlier in
the run) and `wall_time`.

## Cluster

`slurm/train.sbatch`, `slurm/evaluate.sbatch` and `slurm/submit_sweep.sh`
submit these scripts; they take their configuration from environment
variables (see the comments at the top of each file). `SCRIPT=scripts/plan_space/train_hei.py`
makes `train.sbatch` (and the sweep) run hindsight expert iteration instead of
imitation.
