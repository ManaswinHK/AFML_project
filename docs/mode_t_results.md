# Mode T (H4): Not Completed

**Status: H4 not tested.** Online training with EPyMARL never produced a method that earns reward on the planted coalition task, so there is no return to correlate with credit faithfulness. This follows the decision rule in `docs/next_steps_pc.md` Step 1: both 400k-step checks stayed at return 0, and so did the one 1M-step QMIX run. Steps 2–3 (the 24 full runs, evaluation and `analyse_h4.py`) were therefore not carried out.

## What was run

Code: `scripts/run_mode_t.py train` (one EPyMARL run per method, delay and seed on `coalition_layout`, b = 2, horizon 60), on the Ryzen 5 7600, CPU only (`use_cuda=False`). Delay 0, seed 0 throughout.

| run | steps | wall time | test evaluations | test return | training-episode return |
|---|---|---|---|---|---|
| MAPPO | 400,000 | 3.6 min | 19 | 0 at every evaluation | 0 |
| QMIX | 400,000 | 7 min | 19 | 0 at every evaluation | 0 |
| QMIX | 1,000,000 | 16 min | 19 | 0 at every evaluation | 0 |

- Evaluations occur every t_max / 20 steps (greedy, ε = 0). The maximum episode return is 10.
- No training or test episode completed the event (`events_completed_mean` = 0 throughout).
- The runs completed normally; training losses and gradients were finite (QMIX grad norm about 0.03 at the end). This is a failure to find reward, not a crash.
- Logs: `results/mode_t/check_400k/{qmix,mappo}_d0_s0/train.log` and `results/mode_t/check_1M/qmix_d0_s0/train.log`. Model checkpoints are not committed.

The wall times are far below the "about 1 hour" planned for the check. They are for these single-seed runs; the 24-run set was not timed.

## Why: reward is never seen

The task needs a lever pull by A, then B and C pushing together. A uniform-random policy on this configuration (d = 0, b = 2, horizon 60) earned reward in **0 of 20,000 episodes** (37 s, `whoearned.envs.gridworld` directly). By the rule of three, the per-episode success probability is below about 1.5 × 10⁻⁴.

With ε annealed from 1.0 to 0.05 over the first 50k steps, QMIX's exploration is close to uniform-random for as long as that matters, so over 1M steps (about 17,000 episodes) it would expect to see a handful of rewards at best, and in practice saw none. MAPPO's policy starts near uniform and has the same problem. With no reward, the value targets are all zero, and the logged MAPPO TD error is correspondingly tiny (about 0.0015–0.0044).

This is the same reason the roadmap notes for the 3k-step smoke test, and it does not depend on the credit-assignment question. It means that, as configured, Mode T cannot separate methods by return: every method would sit at 0.

## What this means for H4

H4 ("return is a weak proxy for credit faithfulness") needs methods whose return varies. Here return is constant at 0 for all of them, so the correlation is undefined and the "two methods with indistinguishable return" comparison is vacuous. **No verdict on H4.** The evaluation half of Mode T (`run_mode_t.py eval`, `whoearned.mode_t`) was therefore not exercised on real checkpoints; only its unit tests (`tests/test_mode_t.py`) were run.

## What would unblock it (not tried)

Each of these changes the benchmark rather than checking it, so none was done without a decision:

- **Seed the exploration** with scripted-team episodes in the replay buffer (as Mode P does), then train online. This keeps the sparse reward but removes the cold-start problem.
- **Shorten the route or relax coordination** (a smaller grid, or a looser finishing condition), so random play succeeds occasionally.
- **Reward shaping or a curriculum** (for example a partial reward for the lever pull). This changes the credit target, so ground truth would have to be recomputed.

## Environment note

EPyMARL's `sacred 0.8.7` does not import under Python 3.14, the interpreter used for the Mode P, H3 and H5 runs. Mode T was run in a separate Python 3.12.10 venv (`.venv312`, `requirements.txt` plus `setuptools<81`). Mode P results are unaffected.
