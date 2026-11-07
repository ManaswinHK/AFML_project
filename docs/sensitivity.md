# Sensitivity Checks

Two checks on whether the Mode P findings depend on a protocol choice. Check A is a re-analysis of stored results. Check B re-scores the main grid against a closed-loop truth (H1a, H2, H2′, H-fid, and in part 2 also H0 and H1b).

> "H-fid" is the finding earlier called H5 (accuracy without fidelity); see the naming note in `docs/main_results.md`.

## A. Shapley vs leave-one-out as the credit target

**Code:** `scripts/sensitivity_shapley_loo.py`. **Output:** `results/mode_p/sensitivity/shapley_loo.md`.
It re-reads every Mode P cell (main 120, H3 30, H5 30, `h5_ref` 10) and runs no training.

### A1. The two targets barely differ where the game is a pure team effort

L1 between the normalised Shapley and leave-one-out shares of each test episode (0 = identical, 2 = disjoint):

| layout | enabler | bystanders | mean L1 | episodes identical |
|---|---|---|---|---|
| coalition | slack / critical | 0 | 0.000 | 1.00 |
| coalition | slack / critical | 2 | 0.004 / 0.008 | 0.98 / 0.96 |
| coalition | slack / critical | 4 | 0.022 / 0.027 | 0.88 / 0.85 |
| partial synergy | slack | 2 | 0.060 | 0.72 |
| overlap | slack | 1 | 0.104 | 0.15 |

- On the coalition layout every member is needed, so the two coincide except when a bystander's realised interference changes the outcome. That happens more with more bystanders.
- They separate only on the two layouts where the game is not a single unanimity event: partial synergy, and overlap, where B sits in two events.
- Even the largest gap, 0.104 on overlap, is small next to the errors being compared (individual L1 of 0.4 to 2.0).

### A2. Method rankings do not depend on the target

Per condition (layout, enabler, b, d), methods ranked by mean individual L1 against each target:

| layout | conditions | mean Kendall tau | same best method |
|---|---|---|---|
| coalition | 24 | 1.00 | 24/24 |
| overlap | 3 | 0.98 | 3/3 |
| partial | 3 | 0.98 | 3/3 |

The two rankings are identical in 27 of 30 conditions. In the other three (overlap d = 0 and 50, partial d = 50, each ranking 12 methods) tau is 0.97, 0.97 and 0.94, i.e. one or two discordant pairs out of 66. The lowest-L1 method is the same under both targets in all 30 conditions.

### A3. The H5b verdict is unchanged

H5b (ECE-k_mix beats uniform credit on individual L1) was graded against Shapley, as pre-registered. Against leave-one-out it gives the same result:

| target | conditions significant (Holm) | verdict | L1 gain over uniform |
|---|---|---|---|
| Shapley (pre-registered) | 8/8 | supported | 0.29–0.61 |
| leave-one-out | 8/8 | supported | 0.32–0.61 |

(Holm-adjusted p ≤ 0.0001 in every condition under both.)

**Conclusion for check A.** Choosing Shapley or leave-one-out does not change any ranking or verdict tested here. The main grid's leave-one-out target was a safe default, and the layouts where the two differ (overlap, partial synergy) are the ones scored against Shapley in H3 and H5.

**Limits.** Only the individual-L1 comparison is re-scored. Dividend error (H3, H5a) does not depend on this choice, because it compares Harsanyi dividends. Leakage and missed credit were not recomputed against the second target.

## B. Open-loop vs closed-loop removal as the ground truth

**Code:** `scripts/sensitivity_loop.py`. **Output:** `results/mode_p/sensitivity/loop.md`; the re-scored cells are in `results/mode_p/main_closed_loop/` (same file names as `main/`, closed-loop metrics in place of the open-loop ones).

**Protocols.**
- *Open loop* (the default): replay the recorded full-team actions with the removed agents absent. The remaining agents cannot react, so a removed agent can leave the others blocked or in the way.
- *Closed loop*: re-run the scripted team with those agents absent, so the others react (`ground_truth_closed_loop`).

For each of the 120 main-grid cells the 30 test episodes were regenerated and the closed-loop truth computed (11–15 s per cell). Method credit is the stored credit; only the target changes. Two checks passed: (i) scoring the stored credit against the stored open-loop target reproduces every stored per-episode metric to 1e-9; (ii) the closed-loop full-team run reproduces the return of all 3,600 test episodes.

### B1. The truths agree without bystanders and differ with them

Mean L1 between open- and closed-loop normalised leave-one-out shares per test episode (0 = identical, 2 = disjoint). The delay does not change it.

| bystanders | enabler | LOO L1 | Shapley L1 | episodes with identical LOO |
|---|---|---|---|---|
| 0 | slack / critical | 0.000 / 0.000 | 0.000 / 0.000 | 1.00 / 1.00 |
| 2 | slack / critical | 0.079 / 0.099 | 0.079 / 0.096 | 0.87 / 0.83 |
| 4 | slack / critical | 0.183 / 0.214 | 0.179 / 0.211 | 0.71 / 0.66 |

Overall: LOO L1 0.096, identical in 84% of episodes.

**The closed-loop truth always equals the planted structure** (realised = planted in 3,600 of 3,600 test episodes). The open-loop truth does so in 100% / 84% / 66% of episodes at b = 0 / 2 / 4. So the open-loop deviations described under H0 (`docs/main_results.md`) are an artefact of replaying a fixed action sequence with an agent missing, not of the environment: when the team reacts, the physical interference disappears.

### B2. Two verdicts change: H1a and H0

The pre-registered tests that score method credit against the removal target were re-run on the re-scored cells (`scripts/analyse_hypotheses.py results/mode_p/main_closed_loop`).

| test | open loop (reported) | closed loop |
|---|---|---|
| H0, ground truth is protocol-stable (§B3) | not supported (narrow) | **supported** |
| H1a, missed enabler credit grows with delay | not supported | **supported** |
| H2, leakage grows with bystanders | COMA, MAPPO supported; VDN, QMIX not | unchanged |
| H2′, COMA leaks less than VDN/QMIX at d = 0 | not supported (0/8) | unchanged |
| H-fid, end-to-end accuracy does not imply fidelity | supported | unchanged (cut-off L1 1.103 → 1.147) |

**H1a changes through a single condition.** QMIX with 4 bystanders and a critical enabler goes from ΔM = +0.491 (Holm p = 0.074) to +0.547 (Holm p = 0.030), which clears the 0.05 line. No other condition changes status: every other QMIX condition stays below the rule, VDN's conditions still do not count because its critic did not converge, and COMA and MAPPO keep the same sign and a similar size. H1a as reported in `docs/main_results.md` should therefore be described as **sensitive to the removal protocol**: not supported under the pre-registered open-loop truth, supported by one condition under the closed-loop truth, with the same direction in both. It is not a robust positive result.

### B3. H0 and H1b under the closed-loop truth (part 2)

**Code:** `scripts/sensitivity_loop_h0_h1b.py`. **Output:** `results/mode_p/sensitivity/loop_h0_h1b.md`; the cell JSONs in `results/mode_p/main_closed_loop/` now hold the closed-loop values, with the open-loop ones kept under `*_open_loop` keys. All 120 cells passed both validations (the regenerated per-step truth reproduces the stored open-loop semantic gap; the regenerated closed-loop truth reproduces the one stored in part 1).

- **H0 gets a closed-loop null.** Absent agents stay on the grid, inert, and the team plans as if they were gone; this is compared with the closed-loop removal truth.
- **H1b** regenerates the per-step counterfactuals and scores them against the closed-loop leave-one-out target.

**H0 becomes supported.** The rule (every cell: remove-vs-noop share L1 ≤ 0.15 and realised = planted ≥ 0.6) is met in all 120 cells: max mean L1 0.000, realised = planted 1.00 everywhere. Under open loop the worst cells were 0.133 / 0.138 (b = 2) and 0.180 / 0.173 (b = 4), with realised = planted down to 0.53. So the H0 failure is an artefact of open-loop replay, as §B1 suggested; the environment itself is protocol-stable once the team can react.

**H1b is unchanged (not supported).** Level-2 missed enabler credit, mean over seeds:

| notion | enabler | open loop | closed loop |
|---|---|---|---|
| noop | slack | 0.991 | 0.991 |
| noop | critical | 0.653–0.667 | 0.646–0.665 |
| baseline | slack | 0.964–0.970 | 0.966–0.970 |
| baseline | critical | 1.020–1.039 | 0.989–1.028 |

The slack half still holds in every cell and the critical half (≤ 0.3) still fails, by the same margin. This fits the existing explanation: only 54% of "critical" episodes are critical (`docs/main_results.md` §3.1), and that does not depend on how the removal truth is computed.

Verdict table for the closed-loop re-score (`results/mode_p/main_closed_loop/hypotheses.md`): H0 supported; H1a supported (one QMIX condition); H1b not supported; H2 COMA and MAPPO supported, VDN and QMIX not; H2′ not supported (0/8); H-fid supported. Only H0 and H1a change status.

**Not re-scored.** H3 and H5 (coalition estimator) are not part of the main grid and were not re-scored.

**Conclusion for check B.** The main-grid findings that rest on leakage (H2, H2′), on end-to-end versus step-level fidelity (H-fid) and on the slack mechanism (H1b) do not depend on the removal protocol. Two verdicts do: the borderline H1a, and H0, whose failure was an artefact of open-loop replay. The pre-registered verdicts remain the open-loop ones; the closed-loop results are a sensitivity result. The closed-loop truth is also the cleaner benchmark (it always recovers the planted coalition).
