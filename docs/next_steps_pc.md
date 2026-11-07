# Next Steps on the PC (updated 2026-11-07)

The work is ordered. Do the steps in order and **commit and push after each one**, results included (`git add -f results/...`), so the laptop can pull and check them.

```bash
git pull
```

**State:** every experiment in the plan is done: Mode P, the uniform baseline, H3, H5, Mode T/H4, and sensitivity checks A and B. The previous instruction list (Mode T, sensitivity, bookkeeping, draft report) is complete; it is in git history (commit `afcbae9`).

One analysis is left: **H0 and H1b under the closed-loop truth.** Sensitivity B re-scored H1a, H2, H2′ and H-fid against the closed-loop truth, and H1a flipped. H0 and H1b were not re-tested, and the report lists that as a limitation. The closed-loop truth recovers the planted coalition in 100% of episodes, so H0 may flip as well.

Already checked on the laptop (2026-11-07):
- the full test suite passes (190 tests, including 3 new ones for this step);
- every analysis script and all four figures regenerate byte for byte from the committed results;
- the only exception is `results/mode_p/main/summary.md`, whose numbers are the same but whose format was stale; it was regenerated.

---

## Step 1. Run the H0/H1b re-test (background, about 1–2 hours)

The script is `scripts/sensitivity_loop_h0_h1b.py`; its docstring explains the method. For each of the 120 main-grid cells it:
- regenerates the per-step counterfactuals and scores them against the closed-loop leave-one-out target (H1b);
- computes a closed-loop no-op null, where absent agents stay on the grid, inert, and the team plans as if they were gone; it is compared with the closed-loop removal truth (H0);
- validates first: the regenerated per-step truth must reproduce the stored open-loop `semantic_gap`, and the regenerated closed-loop truth must reproduce the one stored in `results/mode_p/main_closed_loop/`. A cell that fails either check prints `FAILED` and is not written.

It updates the cell JSONs in `results/mode_p/main_closed_loop/` in place, keeping the open-loop values under `*_open_loop` keys. It is resumable: cells already done are skipped.

```bash
.venv/Scripts/python -m pytest -q tests/test_sensitivity_loop.py
```

```bash
.venv/Scripts/python scripts/sensitivity_loop_h0_h1b.py --workers 10
```

Cost: the original per-step truth took about 10 CPU-hours over the grid on the machine that ran it, and the closed-loop truths add about as much again. On 10 workers expect roughly 1–2 hours. The b = 4 cells are the slowest.

- If **any cell prints `FAILED`**, stop and push the log; do not write up results.
- Laptop preview (two cells, both validations passed; the cells were then reset so the PC recomputes them): in `d0_b4_s0`, where H0 failed under open loop, the closed-loop remove-vs-noop L1 is 0.000 (open loop 0.124) and realised = planted is 1.00 (open loop 0.67). Level-2 missed credit is unchanged (noop 1.107, baseline 1.135). So expect H0 to become supported; check this on the full grid rather than assuming it.
- When all 120 are done, the script writes `results/mode_p/sensitivity/loop_h0_h1b.md` and re-runs `analyse_hypotheses.py` on the closed-loop folder.

Then refresh the part 1 report, so its verdict comparison includes H0 and H1b:

```bash
.venv/Scripts/python scripts/sensitivity_loop.py --report-only
```

Commit:
- `scripts/`, `tests/`;
- `results/mode_p/main_closed_loop/` (cell JSONs and `hypotheses.md`);
- `results/mode_p/sensitivity/loop.md` and `loop_h0_h1b.md`.

## Step 2. Write the result into the documents

Read `results/mode_p/sensitivity/loop_h0_h1b.md`. For each of H0 and H1b, note the closed-loop verdict and which cells or conditions decide it. Then update:

- **`docs/sensitivity.md` §B:** add a subsection "H0 and H1b under the closed-loop truth" with the two tables from `loop_h0_h1b.md` and the verdicts. Describe the closed-loop no-op null in one sentence (absent agents stay on the grid, inert; the team plans around them).
- **`docs/main_results.md`:**
  - the headline paragraph "Sensitivity to the removal protocol" (near the top);
  - the H0 and H1b sections, each with a one-line closed-loop note.
- **`docs/report.md`:**
  - §6, the open- vs closed-loop bullet;
  - §8, remove or rewrite the limitation "H0 and H1b were not re-tested under the closed-loop truth".
- **`docs/results_overview.md`:** the sensitivity row(s) and any H0/H1b row.
- **Problem statement status table** (`who_earned_the_reward_problem_statement_v2.md`): add a closed-loop note to H0 and H1b if a verdict changes.

How to word it:
- The pre-registered verdicts are the open-loop ones and stay as they are; closed-loop results are a **sensitivity** result, as for H1a.
- If H0 becomes supported, say that the H0 failure is an artefact of open-loop replay (the docs already say this informally).
- If H1b's critical half still fails, connect it to the existing explanation (only 54% of "critical" episodes were critical; `docs/main_results.md` §3.1).

## Step 3. Final consistency check

```bash
.venv/Scripts/python -m pytest -q
```

```bash
.venv/Scripts/python scripts/make_figures.py
```

```bash
git status --short
```

The figures should not change; none of them depends on the closed-loop results. Search the docs for stale phrases:

```bash
git grep -n "not re-tested\|H5 result\|(H5)" docs
```

Remaining "H5" mentions must refer to the coalition estimator. The old fidelity finding is "H-fid".

---

## Optional (only with time to spare; not recommended before submission)

**Mode T for COMA and MAPPO: done (8 November 2026).** Behavioural cloning plus critic warm-up was implemented (`scripts/run_epymarl_bc.py`) and run on 20 runs. Both start at return about 10 and collapse online, so H4 could not be assessed for them (`docs/mode_t_results.md` §3b).

## Later: submission (not part of this list)

- Turn `docs/report.md` into the required format, and add related work and a conclusion.
- Update the presentation (`docs/progress_slides.md`, `docs/review_prep.md`) to the final results.
