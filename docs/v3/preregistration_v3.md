# Pre-registration v3: credit by intervention

**Status: DRAFT. Thresholds are filled from the exploratory pilots (`docs/v3/pilot_findings.md`) before the commit that freezes this file. No confirmatory cell (seeds 5–9) is run before that commit.** The commit hash that freezes it is recorded in `docs/v3/PROGRESS.md`.

Authors: team Aporia. Date: 11–12 Nov 2026. Supersedes nothing: it pre-registers new hypotheses on fresh seeds. The earlier pre-registrations (`docs/h5_preregistration.md`, `docs/preregistration_amendment.md`) and their results stand, as qualified in `docs/v3/errata.md`.

## 1. Question

Can **persistent interventions** (agents stopping mid-episode for good) make "who earned the reward" learnable by an ordinary value critic? Does a critic that respects the relabelling symmetry make it generalise across role assignments (and team sizes)? Does any of it change how fast a team learns?

Theory: `docs/v3/theory.md`. Code: `src/whoearned/ablation.py`, `src/whoearned/critics.py`, `src/whoearned/experiments.py`.

## 2. Design (fixed)

- **Seeds 5–9**, not used by any earlier run (errata E9). One data cell = (layout, d, b, enabler start, seed). Each cell has 200 scripted training episodes (ε = 0.2) and 30 test episodes, seeded exactly as in `DataCell`.
- **Truth**:
  - **exact ablation truth** under each null (closed loop, common random numbers; `oracle_ablation_values`);
  - closed-loop removal truth (`ground_truth_closed_loop`);
  - closed-loop no-op game;
  - open-loop truth (reported, not graded).

  Individual credit is graded against the **exact truth of the method's own null** (`fid_l1`). Coalition credit is graded against the closed-loop game of the method's own null (`div_own`). Tables and ECE are trained on removal replays and graded against the removal game.
- **Grids**:

| Grid | Conditions | Cells |
|---|---|---|
| `confirm_main` | coalition layout, d ∈ {0, 25, 50} × b ∈ {2, 4} × {slack, critical} | 12 × 5 = 60 |
| `confirm_h5` | overlap (b = 1) and partial (b = 2), d ∈ {0, 50} | 4 × 5 = 20 |
| `confirm_perm` | shuffled roles, d = 25, b ∈ {2, 4} | 2 × 5 = 10 |
| `confirm_size` | size transfer: train shuffled b = 2 → test b = 4, and b = 4 → b = 2 | 2 × 5 runs |
| `gate` (Phase 3) | lanes K ∈ {1, 2, 4, 8} × {team, team_vector, difference}, IPPO | 60 runs |

- **Methods** (registry names in `experiments.py`):
  - **`atd`**: MLP 2×128 on the global state; 200 training episodes + 200 stops; stratified sampler; no-op null; Monte-Carlo regression; 3,000 Adam steps.
  - **`atd_set`**: the same data, permutation-invariant set critic (`critics.SetCritic`).
  - **`atd_noint`**: the same MLP, no stops.
  - **`atd_remove`**: removal null.
  - **`atd_n{25,50,100,400,800}`**: data scaling.
  - **`table2_mix`, `table3_mix`**: presence-only Lasso on ECE's `mix` data, α = 0.025, untuned.
  - **`ece2_mix`, `ece3_mix`**.
  - **`uniform`**: equal split.
  - **`vdn`, `qmix`, `coma`, `mappo`**: EPyMARL, 1,500 updates on the 200 training episodes.
  - **`mappo_int`, `qmix_int`, `*_masked`**: control V2. The same EPyMARL learners trained on the training episodes **plus the same 200 stopped episodes** as `atd`. Read out (i) with their standard one-step read-out and (ii) at the stopped state (masked).

  No hyperparameter of any method is changed after this commit.

## 3. Hypotheses

THRESHOLDS: see §3 table (filled from the pilots with a margin; every threshold is looser than the pilot value in the direction that makes the hypothesis harder to fail by chance, i.e. it is a conservative test of the direction the pilot showed).

(Table to be filled.)

## 4. Analysis (fixed; code `scripts/v3/stats.py`, `scripts/v3/analyse_prereg.py`)

- Unit: one seed (cell). Its value is the mean over the 30 test episodes of the metric. Episodes whose credit or target is all zero are excluded from L1 (`valid = False`) and counted.
- **Threshold hypotheses**: one-sided one-sample t-test per condition (df = 4); Holm across the hypothesis's conditions; supported if every condition is supported.
- **Comparison hypotheses**: one-sided paired t-test of the difference against the stated margin; Holm across conditions; supported if every condition is supported.
- Exact Wilcoxon p is reported alongside (descriptive).
- Verdicts: SUPPORTED / PARTIAL (k of m conditions) / NOT SUPPORTED. Every hypothesis is reported whatever the result.
- Deviations from this plan are listed in a "Deviations" section of the results document, with reasons. They are never silently applied.
