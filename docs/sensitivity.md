# Sensitivity Checks

Two checks on whether the Mode P findings depend on a protocol choice. Check A is a re-analysis of stored results. Check B (open- vs closed-loop replay) is added below when it has run.

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
