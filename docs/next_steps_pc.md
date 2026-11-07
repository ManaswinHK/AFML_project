# Next Steps on the PC (from 2026-11-07)

This is the ordered work left after H5, to run on the Ryzen 5 7600 + RTX 4060. Do the steps in order. **Commit and push after each step**, results included (`git add -f results/...`), so the laptop can pull and check them.

Before starting:

```bash
git pull
```

**State so far:** Mode P is complete; the uniform baseline is done (`docs/main_results.md` §3.3); the H5 coalition estimator is done (`docs/h5_results.md`). This leaves Mode T (H4), two sensitivity checks, bookkeeping, and the write-up. The plan and cuts are in Slide 4 of the progress slides.

---

## Step 1. Mode T learning and timing check (about 1 hour)

Mode T trains VDN, QMIX, COMA and MAPPO online with EPyMARL on `coalition_layout` (b = 2). The code is `scripts/run_mode_t.py` and `src/whoearned/mode_t.py`. It has only had a 3k-step smoke run, which earned return 0. The question is whether the methods learn at all.

```bash
.venv/Scripts/python -m pytest -q tests/test_mode_t.py
```

```bash
.venv/Scripts/python scripts/run_mode_t.py train --algs qmix,mappo --delays 0 --seeds 0 --t-max 400000 --workers 2
```

Then check `results/mode_t/runs/qmix_d0_s0/train.log` and `results/mode_t/runs/mappo_d0_s0/train.log`:
- **Wall time** per run.
- **Test return** over time (`test_return_mean`). The maximum episode return is 10, earned when the coalition completes the task.

**Decide:**

| outcome | action |
|---|---|
| Return clearly rises above 0 for at least one method | Go to Step 2. Choose `--t-max` so the full set fits in about 12 hours (see below). |
| Return stays at 0 at 400k | Try one 1M-step QMIX run. If it is still 0, Mode T is not completed: write this up in `docs/mode_t_results.md` (what was run, logs, why) and skip to Step 4. |

Budget: the full set is 24 runs on 8 workers, so 3 rounds. If one run takes *T* hours at 400k steps, the full set takes about 3*T* hours.

Push the logs of the check runs.

## Step 2. Mode T full runs (background, several hours)

The plan is 4 methods × d ∈ {0, 25} × 3 seeds = 24 runs. Delete the Step 1 check runs first if you change `--t-max`. The runner skips runs that already have a model directory, so check runs at a different `--t-max` would otherwise be kept.

```bash
.venv/Scripts/python scripts/run_mode_t.py train --t-max <chosen> --workers 8
```

Note:
- EPyMARL may put COMA and MAPPO on the GPU. If runs fail with CUDA/CUBLAS errors, lower `--workers`.
- Model checkpoints can be large. Do **not** commit `results/mode_t/runs/*/models/`. Commit only `train.log` files and the eval JSONs.

While this runs, do Step 4. It is CPU-light, so keep `--workers` below 8 if the machine is needed.

## Step 3. Mode T evaluation and H4 analysis

```bash
.venv/Scripts/python scripts/run_mode_t.py eval --workers 8
```

This writes one JSON per run to `results/mode_t/`. Each holds, per checkpoint:
- the return with ε = 0.2 behaviour;
- the greedy return;
- end-to-end credit error against leave-one-out (`summary`).

Write `scripts/analyse_h4.py`. H4 (problem statement v2, "Return is a weak proxy for credit faithfulness"):
- Within each method, the Spearman correlation between return and 1 − L1 across checkpoints and seeds, with bootstrap 95% CIs over seeds.
- Whether two methods with indistinguishable final return differ in L1: a paired comparison over seeds and delays.
- Mode T is reduced in scope (2 delays, 3 seeds), so report H4 as **descriptive**, not confirmatory, as Slide 4 states.
- Exclude checkpoints where no test episode earns reward (credit is undefined). Report how many were excluded.

Write the results to `docs/mode_t_results.md` with a return-vs-L1 figure per method.

## Step 4. Sensitivity check A: Shapley vs leave-one-out (re-analysis, no training)

Every Mode P cell JSON already stores both truths (`truth.shapley`, `truth.loo`), so this needs no new runs. Write `scripts/sensitivity_shapley_loo.py` covering `results/mode_p/main`, `h3` and `h5`:
1. **Truth disagreement:** mean L1 between Shapley and leave-one-out shares, per layout and delay. On the coalition layout with all members needed, the two coincide. They differ where the game is not pure unanimity, e.g. partial synergy and overlap.
2. **Ranking stability:** score every method against both targets and report whether the method ranking (Kendall τ) and the H5 verdicts change.

Write the results to `docs/sensitivity.md` §A.

## Step 5. Sensitivity check B: open- vs closed-loop replay (small new run)

The default truth replays the recorded actions with agents removed (open loop, `ground_truth_open_loop` in `src/whoearned/replay.py`). The alternative re-runs the scripted team with agents removed, so the remaining agents can react (`ground_truth_closed_loop`, same file).

For the main-grid test episodes (coalition layout), compute both truths per episode and report:
- mean L1 between the two, by delay and b;
- whether any Mode P verdict changes when scored against the closed-loop truth.

Write `scripts/sensitivity_loop.py` and `docs/sensitivity.md` §B. A no-op vs removal check is dropped unless time remains (Slide 4).

## Step 6. Bookkeeping

- `ROADMAP.md`:
  - tick H5 (`docs/h5_results.md`);
  - Step 8 (Mode T) and Step 9 (sensitivity) as done, or as done with Mode T "not completed".
- `docs/results_overview.md`: add rows for H5 (the coalition estimator), H4 and the sensitivity checks; remove them from the to-do list.
- **Naming clash.** The results overview and `docs/main_results.md` use "H5" for "end-to-end accuracy does not imply fidelity". The coalition estimator is also "H5" in the problem statement and `docs/h5_preregistration.md`. Rename the older one in the overview and main results, e.g. to "H2-fid" or "fidelity finding", and say so in a footnote, so the report has one H5.
- Problem statement status table: mark each hypothesis supported, not supported, or not completed.

## Step 7. Write-up

- **Report:** question; benchmark (planted coalitions, exact truth); Mode P results (H0–H3, uniform baseline); the coalition estimator (H5); Mode T (H4); sensitivity; limitations (one environment, scripted Mode P data, ECE needs removal replays, ≤ 6 agents).
- **Figures:**
  - L1 vs uniform by delay (main grid);
  - interaction mass by method (H3);
  - dividend error, ECE vs NA²Q (H5);
  - return vs L1 (H4).
- **Final presentation:** update the progress slides.

Keep laptop-vs-PC timings out of any cost comparison. Use the per-machine table in `docs/h5_results.md`.
