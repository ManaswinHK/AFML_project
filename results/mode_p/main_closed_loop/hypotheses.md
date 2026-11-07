# Hypothesis tests: `main_closed_loop`

120 data cells. Primary test: one-sided paired t-test on per-seed means, Holm-corrected within each hypothesis (see docs/preregistration_amendment.md). Exact Wilcoxon p-values are shown for reference; with 5 seeds their floor is 1/32 = 0.031.

## H0: ground truth is protocol-stable

Rule: every cell has mean remove-vs-noop share L1 ≤ 0.15 and realised = planted ≥ 0.6.

- Cells: 120. Max mean L1: 0.000. Max per-episode L1: 0.000.
- Min realised = planted: 1.00.
- **Verdict: SUPPORTED**

## H1a: missed enabler credit grows with delay

Rule: for some method and (bystanders, enabler) condition whose critic converged at both delays (mean final loss at d=50 ≤ 2× that at d=0), M(d=50) − M(d=0) ≥ 0.2 with Holm-adjusted paired t p < 0.05 (Holm over all conditions tested here).

| alg | bystanders | enabler | converged | ΔM | t p | Wilcoxon p | Holm t p | meets |
|---|---|---|---|---|---|---|---|---|
| vdn | 0 | slack | no | +0.048 | 0.4545 | 0.5938 | 1.0000 |  |
| vdn | 0 | critical | no | +1.440 | 0.0007 | 0.0312 | 0.0163 |  |
| vdn | 2 | slack | no | -0.158 | 0.6561 | 0.7812 | 1.0000 |  |
| vdn | 2 | critical | no | +0.779 | 0.0028 | 0.0312 | 0.0594 |  |
| vdn | 4 | slack | no | +0.187 | 0.0534 | 0.0625 | 0.9078 |  |
| vdn | 4 | critical | no | +1.057 | 3.9e-05 | 0.0312 | 0.0009 |  |
| qmix | 0 | slack | yes | +0.261 | 0.2496 | 0.4062 | 1.0000 |  |
| qmix | 0 | critical | yes | +0.002 | 0.4899 | 0.5938 | 1.0000 |  |
| qmix | 2 | slack | yes | +0.320 | 0.0420 | 0.0625 | 0.8396 |  |
| qmix | 2 | critical | yes | +0.129 | 0.0487 | 0.0625 | 0.8772 |  |
| qmix | 4 | slack | no | +0.193 | 0.0429 | 0.0625 | 0.8396 |  |
| qmix | 4 | critical | yes | +0.547 | 0.0013 | 0.0312 | 0.0295 | ✓ |
| coma | 0 | slack | yes | +0.052 | 0.3575 | 0.4062 | 1.0000 |  |
| coma | 0 | critical | yes | +0.156 | 0.1713 | 0.2188 | 1.0000 |  |
| coma | 2 | slack | yes | +0.023 | 0.3083 | 0.3125 | 1.0000 |  |
| coma | 2 | critical | yes | +0.064 | 0.1874 | 0.3125 | 1.0000 |  |
| coma | 4 | slack | yes | +0.035 | 0.2864 | 0.2188 | 1.0000 |  |
| coma | 4 | critical | yes | +0.054 | 0.1176 | 0.1562 | 1.0000 |  |
| mappo | 0 | slack | yes | -0.351 | 0.9041 | 0.9688 | 1.0000 |  |
| mappo | 0 | critical | yes | -1.730 | 1.0000 | 1.0000 | 1.0000 |  |
| mappo | 2 | slack | yes | -0.329 | 0.8919 | 0.9062 | 1.0000 |  |
| mappo | 2 | critical | yes | -1.263 | 1.0000 | 1.0000 | 1.0000 |  |
| mappo | 4 | slack | yes | -0.188 | 0.8300 | 0.9062 | 1.0000 |  |
| mappo | 4 | critical | yes | -0.861 | 1.0000 | 1.0000 | 1.0000 |  |

**Verdict: SUPPORTED**

## H1b: slack, not delay, removes per-step enabler credit (level 2)

Rule: for each per-step notion, the seed-mean level-2 missed credit is ≥ 0.8 in every slack cell and ≤ 0.3 in every critical cell.

| notion | enabler | delay | bystanders | missed (mean over seeds) | meets |
|---|---|---|---|---|---|
| baseline | slack | 0 | 0 | 0.966 | ✓ |
| baseline | slack | 0 | 2 | 0.970 | ✓ |
| baseline | slack | 0 | 4 | 0.966 | ✓ |
| baseline | slack | 10 | 0 | 0.966 | ✓ |
| baseline | slack | 10 | 2 | 0.970 | ✓ |
| baseline | slack | 10 | 4 | 0.966 | ✓ |
| baseline | slack | 25 | 0 | 0.966 | ✓ |
| baseline | slack | 25 | 2 | 0.970 | ✓ |
| baseline | slack | 25 | 4 | 0.966 | ✓ |
| baseline | slack | 50 | 0 | 0.966 | ✓ |
| baseline | slack | 50 | 2 | 0.970 | ✓ |
| baseline | slack | 50 | 4 | 0.966 | ✓ |
| baseline | critical | 0 | 0 | 1.024 | ✗ |
| baseline | critical | 0 | 2 | 0.994 | ✗ |
| baseline | critical | 0 | 4 | 0.989 | ✗ |
| baseline | critical | 10 | 0 | 1.024 | ✗ |
| baseline | critical | 10 | 2 | 0.994 | ✗ |
| baseline | critical | 10 | 4 | 0.989 | ✗ |
| baseline | critical | 25 | 0 | 1.023 | ✗ |
| baseline | critical | 25 | 2 | 0.994 | ✗ |
| baseline | critical | 25 | 4 | 0.989 | ✗ |
| baseline | critical | 50 | 0 | 1.028 | ✗ |
| baseline | critical | 50 | 2 | 0.997 | ✗ |
| baseline | critical | 50 | 4 | 0.991 | ✗ |
| noop | slack | 0 | 0 | 0.991 | ✓ |
| noop | slack | 0 | 2 | 0.991 | ✓ |
| noop | slack | 0 | 4 | 0.991 | ✓ |
| noop | slack | 10 | 0 | 0.991 | ✓ |
| noop | slack | 10 | 2 | 0.991 | ✓ |
| noop | slack | 10 | 4 | 0.991 | ✓ |
| noop | slack | 25 | 0 | 0.991 | ✓ |
| noop | slack | 25 | 2 | 0.991 | ✓ |
| noop | slack | 25 | 4 | 0.991 | ✓ |
| noop | slack | 50 | 0 | 0.991 | ✓ |
| noop | slack | 50 | 2 | 0.991 | ✓ |
| noop | slack | 50 | 4 | 0.991 | ✓ |
| noop | critical | 0 | 0 | 0.663 | ✗ |
| noop | critical | 0 | 2 | 0.646 | ✗ |
| noop | critical | 0 | 4 | 0.649 | ✗ |
| noop | critical | 10 | 0 | 0.663 | ✗ |
| noop | critical | 10 | 2 | 0.646 | ✗ |
| noop | critical | 10 | 4 | 0.649 | ✗ |
| noop | critical | 25 | 0 | 0.663 | ✗ |
| noop | critical | 25 | 2 | 0.646 | ✗ |
| noop | critical | 25 | 4 | 0.649 | ✗ |
| noop | critical | 50 | 0 | 0.665 | ✗ |
| noop | critical | 50 | 2 | 0.648 | ✗ |
| noop | critical | 50 | 4 | 0.651 | ✗ |

- baseline: 12/24 cells meet → **NOT SUPPORTED**
- noop: 12/24 cells meet → **NOT SUPPORTED**

## H2: leakage grows with bystanders

Rule per (method, delay, enabler) condition: Λ(b=4) − Λ(b=2) ≥ 0.05 with Holm-adjusted paired t p < 0.05 (Holm over all conditions) and mean Λ(b=2) > 0.1. A method is SUPPORTED if at least 75% of its conditions meet the rule.

| alg | delay | enabler | Λ(b=2) | ΔΛ | t p | Wilcoxon p | Holm t p | meets |
|---|---|---|---|---|---|---|---|---|
| vdn | 0 | slack | 0.138 | +0.138 | 0.0019 | 0.0312 | 0.0290 | ✓ |
| vdn | 0 | critical | 0.169 | +0.038 | 0.1511 | 0.1562 | 0.3021 |  |
| vdn | 10 | slack | 0.506 | +0.110 | 0.1625 | 0.2188 | 0.3021 |  |
| vdn | 10 | critical | 0.360 | +0.196 | 0.0062 | 0.0312 | 0.0739 |  |
| vdn | 25 | slack | 0.565 | +0.105 | 0.0206 | 0.0312 | 0.1648 |  |
| vdn | 25 | critical | 0.534 | +0.152 | 0.0288 | 0.0312 | 0.1789 |  |
| vdn | 50 | slack | 0.592 | +0.152 | 0.0107 | 0.0312 | 0.1179 |  |
| vdn | 50 | critical | 0.568 | +0.144 | 0.0038 | 0.0312 | 0.0534 |  |
| qmix | 0 | slack | 0.072 | +0.048 | 0.0256 | 0.0312 | 0.1789 |  |
| qmix | 0 | critical | 0.061 | +0.029 | 0.0506 | 0.0312 | 0.1789 |  |
| qmix | 10 | slack | 0.294 | +0.229 | 0.0117 | 0.0312 | 0.1179 |  |
| qmix | 10 | critical | 0.193 | +0.221 | 0.0046 | 0.0312 | 0.0594 |  |
| qmix | 25 | slack | 0.382 | +0.261 | 0.0340 | 0.0625 | 0.1789 |  |
| qmix | 25 | critical | 0.298 | +0.335 | 0.0014 | 0.0312 | 0.0222 | ✓ |
| qmix | 50 | slack | 0.443 | +0.188 | 0.0418 | 0.0312 | 0.1789 |  |
| qmix | 50 | critical | 0.371 | +0.209 | 0.0139 | 0.0312 | 0.1247 |  |
| coma | 0 | slack | 0.421 | +0.119 | 0.0001 | 0.0312 | 0.0028 | ✓ |
| coma | 0 | critical | 0.392 | +0.173 | 1.4e-05 | 0.0312 | 0.0003 | ✓ |
| coma | 10 | slack | 0.423 | +0.196 | 7.6e-05 | 0.0312 | 0.0018 | ✓ |
| coma | 10 | critical | 0.449 | +0.184 | 0.0010 | 0.0312 | 0.0174 | ✓ |
| coma | 25 | slack | 0.434 | +0.206 | 0.0011 | 0.0312 | 0.0179 | ✓ |
| coma | 25 | critical | 0.455 | +0.176 | 0.0007 | 0.0312 | 0.0131 | ✓ |
| coma | 50 | slack | 0.469 | +0.167 | 4.3e-06 | 0.0312 | 0.0001 | ✓ |
| coma | 50 | critical | 0.404 | +0.198 | 9.5e-05 | 0.0312 | 0.0021 | ✓ |
| mappo | 0 | slack | 0.400 | +0.170 | 5.6e-08 | 0.0312 | 1.6e-06 | ✓ |
| mappo | 0 | critical | 0.398 | +0.172 | 2.6e-08 | 0.0312 | 7.9e-07 | ✓ |
| mappo | 10 | slack | 0.402 | +0.167 | 1.1e-08 | 0.0312 | 3.3e-07 | ✓ |
| mappo | 10 | critical | 0.400 | +0.170 | 8.6e-09 | 0.0312 | 2.8e-07 | ✓ |
| mappo | 25 | slack | 0.399 | +0.170 | 2.9e-08 | 0.0312 | 8.5e-07 | ✓ |
| mappo | 25 | critical | 0.401 | +0.168 | 1.1e-06 | 0.0312 | 3.0e-05 | ✓ |
| mappo | 50 | slack | 0.378 | +0.186 | 0.0005 | 0.0312 | 0.0101 | ✓ |
| mappo | 50 | critical | 0.387 | +0.192 | 8.2e-05 | 0.0312 | 0.0019 | ✓ |

- vdn: 1/8 conditions → **NOT SUPPORTED**
- qmix: 1/8 conditions → **NOT SUPPORTED**
- coma: 8/8 conditions → **SUPPORTED**
- mappo: 8/8 conditions → **SUPPORTED**

## H2′: COMA leaks less than VDN and QMIX at d = 0

Rule: in every (bystanders ∈ {2, 4}, enabler) condition at d=0, Λ_VDN > Λ_COMA and Λ_QMIX > Λ_COMA, each with Holm-adjusted paired t p < 0.05 (Holm over all comparisons).

| bystanders | enabler | other | Λ_other − Λ_COMA | t p | Wilcoxon p | Holm t p | meets |
|---|---|---|---|---|---|---|---|
| 2 | slack | vdn | -0.283 | 1.0000 | 1.0000 | 1.0000 |  |
| 2 | slack | qmix | -0.349 | 1.0000 | 1.0000 | 1.0000 |  |
| 2 | critical | vdn | -0.223 | 0.9995 | 1.0000 | 1.0000 |  |
| 2 | critical | qmix | -0.331 | 1.0000 | 1.0000 | 1.0000 |  |
| 4 | slack | vdn | -0.264 | 0.9998 | 1.0000 | 1.0000 |  |
| 4 | slack | qmix | -0.420 | 1.0000 | 1.0000 | 1.0000 |  |
| 4 | critical | vdn | -0.358 | 0.9999 | 1.0000 | 1.0000 |  |
| 4 | critical | qmix | -0.475 | 1.0000 | 1.0000 | 1.0000 |  |

**Verdict: NOT SUPPORTED** (0/8)

## H5: end-to-end accuracy does not imply fidelity

Rule: among all (method, delay, bystanders, enabler) cells, at least one in the best third by end-to-end L1 has level-1 step r < 0.3.

- Best-third cut-off: L1 ≤ 1.147 (32 of 96 cells).
- Of those, 30 have step r < 0.3. Lowest-L1 examples:

| alg | delay | bystanders | enabler | end-to-end L1 | step r |
|---|---|---|---|---|---|
| mappo | 50 | 0 | critical | 0.318 | 0.251 |
| qmix | 0 | 2 | slack | 0.403 | 0.136 |
| qmix | 10 | 0 | critical | 0.506 | 0.252 |
| vdn | 0 | 0 | slack | 0.559 | 0.246 |
| qmix | 0 | 4 | slack | 0.582 | 0.062 |
| qmix | 0 | 0 | slack | 0.599 | 0.042 |
| vdn | 10 | 0 | slack | 0.607 | 0.237 |
| vdn | 10 | 0 | critical | 0.621 | 0.364 |
| qmix | 50 | 0 | critical | 0.639 | 0.213 |
| qmix | 10 | 0 | slack | 0.660 | 0.218 |

**Verdict: SUPPORTED**

H3 (coalition level) needs NA²Q (Step 7). H4 is Mode T (Step 8).
