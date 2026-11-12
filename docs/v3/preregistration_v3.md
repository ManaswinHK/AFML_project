# Pre-registration v3: credit by intervention

**Status: FROZEN** by the commit titled "Freeze pre-registration v3". No confirmatory Mode P cell (seeds 5–9) existed before it. Phase 3 (the headroom gate) has its own frozen pre-registration: `docs/v3/preregistration_v3_phase3.md`.

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

Thresholds come from the exploratory pilots (`docs/v3/pilot_findings.md`), with a margin. For the MLP critic they are the plan's own values (assessment §9), even where the pilot suggests the MLP may fail them: the test should be able to catch its instability. Code: `scripts/v3/analyse_prereg.py` (the `TAU` table there must equal this one).

Notation:
- **loo_cn**: L1 against the exact no-op truth (closed loop, common random numbers). For no-op-null methods this equals the method's own null.
- **main**: `confirm_main`, 12 conditions.
- **overlap/partial**: `confirm_h5`.
- **perm**: `confirm_perm`, 2 conditions.

| ID | Hypothesis | Conditions | Supported if (every condition, Holm) | Pilot value |
|---|---|---|---|---|
| **A. Learnability** | | | | |
| P2a | The MLP ablation critic learns removal credit | main + overlap | loo_cn of `atd` < 0.15 | 0.004–0.15 |
| P2a-set | So does the set critic | main + overlap | loo_cn of `atd_set` < 0.05 | 0.001–0.05 |
| P2b | `atd` beats the equal split | main + overlap | paired, uniform − atd > 0 | 0.8 vs 0.04 |
| P3 | Interventions are necessary | main + overlap | paired, atd_noint − atd > 0.30 | gap 0.44–1.4 |
| P4a | Per-step fidelity (vs exact κ) is high for the MLP | main | step r of `atd` > 0.85 | 0.69–1.00 |
| P4a-set | … and for the set critic | main | step r of `atd_set` > 0.85 | 0.86–1.00 |
| P4b | One-step read-outs are not faithful to their own per-step targets | main; each of VDN, QMIX, COMA, MAPPO | step r < 0.50 | ≤ 0.36 (stored) |
| **B. Coalition level, fixed layouts** | | | | |
| P5 | The presence-only table is not worse than ECE-3 | coalition (b = 2 slack, d ∈ {0, 50}), overlap, partial | paired, table3 − ece3 dividend error < 0.05 (removal game) | lower on 6/6 (round 2) |
| **C. Consequence** | | | | |
| P6a | Ablation credit keeps exactly the coalition | main | top-k of `atd` > 0.90 | 100% |
| P6b | QMIX and COMA usually do not | main; each | top-k < 0.50 | 40%, 9% (stored); QMIX 75% in one pilot cell |
| **D. Generalisation across roles** | | | | |
| P7a | A table cannot use role information | perm | paired, table3 not better than uniform by more than 0.05 | equal |
| P7b | The MLP critic can, partly | perm | paired, uniform − atd > 0.25 | 0.34, 0.27 |
| P8a | The set critic is near exact | perm | loo_cn of `atd_set` < 0.05 | 0.004, 0.009 |
| P8b | … and beats the MLP | perm | paired, atd − atd_set > 0.25 | 0.45, 0.86 |
| P8c-a | … recovers coalition structure | perm | dividend error (own null) of `atd_set` < 0.40 | 0.06, 0.33 |
| P8c-b | … better than ECE-3_mix | perm | paired, ece3 − set dividend error > 0 | 0.21 vs 0.06; 0.68 vs 0.33 |
| P8d | The set critic's advantage does not depend on role flags | perm | paired, atd_noroles − atd_set_noroles > 0.25 | 0.59, 0.88 |
| **E. Better data or better read-out (V2)** | | | | |
| P9a | MAPPO trained on stops: the masked read-out beats its standard read-out | main | paired, standard − masked > 0.50 | 1.5 |
| P9b | QMIX trained on stops: masked vs standard | main, d > 0 | **two-sided, descriptive** (the pilot contradicted the plan's direction) | 0.78 vs 0.70 |
| P9c | Data alone does not fix one-step read-outs | main | paired, mappo − mappo_int < 0.10 (no real gain) | 0.07, 0.0 |
| **F. Luck** | | | | |
| P10a | On the partial layout the structural read-out is as good as any expected-credit estimator can be | partial | paired, struct − floor < 0.10 | 0.25 vs 0.19 (tiny cell) |
| P10b | … and better than κ, which carries the luck | partial | paired, atd − atd_struct > 0 | 0.45 vs 0.25 |
| **G. Size transfer** | | | | |
| P11a | A set critic trained on one team size picks the right coalition at another | b = 2 → 4 and 4 → 2 | top-k > 0.90 | 100% |
| P11b | … and is clearly better than the equal split | same | paired, uniform − transfer > 0.30 | 0.54, 0.51 |

**Descriptive, reported without a test:**
- the semantic gap of a perfect one-step critic (missed enabler credit), as before;
- data scaling (`atd_n*`, `atd_set_n*`);
- the removal-null variants;
- the commitment timeline figure.

## 4. Analysis (fixed; code `scripts/v3/stats.py`, `scripts/v3/analyse_prereg.py`)

- Unit: one seed (cell). Its value is the mean over the 30 test episodes of the metric. Episodes whose credit or target is all zero are excluded from L1 (`valid = False`) and counted.
- **Threshold hypotheses**: one-sided one-sample t-test per condition (df = 4); Holm across the hypothesis's conditions; supported if every condition is supported.
- **Comparison hypotheses**: one-sided paired t-test of the difference against the stated margin; Holm across conditions; supported if every condition is supported.
- Exact Wilcoxon p is reported alongside (descriptive).
- Verdicts: SUPPORTED / PARTIAL (k of m conditions) / NOT SUPPORTED. Every hypothesis is reported whatever the result.
- Deviations from this plan are listed in a "Deviations" section of the results document, with reasons. They are never silently applied.
