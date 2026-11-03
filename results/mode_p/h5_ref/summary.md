# Mode P summary: `h5_ref`

10 data cells; mean ± sd over seeds of per-seed means over test episodes. `enabler` = slack (A pulls before finishers are ready) or critical (A starts late).

## Level 2: semantic gap (a *perfect* per-step critic vs leave-one-out removal credit)

Method-independent: the per-step truth itself, summed over the episode, scored against leave-one-out.

| per-step notion | enabler | delay | bystanders | L1 | leakage | missed | mean shares (A, B, C, bystanders...) |
|---|---|---|---|---|---|---|---|
| baseline | slack | 0 | 2 | 1.144 ± 0.093 | 0.074 ± 0.037 | 0.969 ± 0.104 | 0.01, 0.09, 0.19, 0.00, 0.01 |
| baseline | slack | 50 | 2 | 1.144 ± 0.094 | 0.074 ± 0.037 | 0.970 ± 0.103 | 0.01, 0.09, 0.19, 0.00, 0.01 |
| noop | slack | 0 | 2 | 1.137 ± 0.039 | 0.034 ± 0.028 | 0.991 ± 0.075 | 0.01, 0.26, 0.36, -0.00, -0.00 |
| noop | slack | 50 | 2 | 1.139 ± 0.040 | 0.034 ± 0.028 | 0.991 ± 0.075 | 0.01, 0.26, 0.36, -0.00, -0.00 |

## End to end: read-out vs its matched removal target

`target`: leave-one-out (loo) or Shapley, as matched to each read-out (problem statement v2, Section 2.4).

| alg | target | enabler | delay | bystanders | seeds | L1 | Spearman | leakage | missed |
|---|---|---|---|---|---|---|---|---|---|
| na2q | shapley | slack | 0 | 2 | 5 | 0.603 ± 0.172 | 0.750 ± 0.138 | 0.115 ± 0.074 | 0.627 ± 0.276 |
| na2q | shapley | slack | 50 | 2 | 5 | 1.381 ± 0.480 | -0.038 ± 0.333 | 0.164 ± 0.081 | 1.709 ± 0.581 |
| stas | shapley | slack | 0 | 2 | 5 | 0.799 ± 0.263 | -0.023 ± 0.495 | 0.351 ± 0.071 | 0.557 ± 0.479 |
| stas | shapley | slack | 50 | 2 | 5 | 0.715 ± 0.077 | 0.026 ± 0.491 | 0.374 ± 0.076 | 0.339 ± 0.147 |

## Coalition level: mixer-game dividends vs true Harsanyi dividends (H3)

Dividend error is normalised (0 = exact, 2 = disjoint), over all orders and over orders ≤ 2. Recall and precision are Jaccard matches between coalitions holding > 10% of |dividends| and the true non-zero coalitions. Interaction mass = share of |dividends| on coalitions of 2+ agents (estimated vs true).

| alg | enabler | delay | bystanders | dividend error | order ≤ 2 | recall | precision | interaction mass | true |
|---|---|---|---|---|---|---|---|---|---|
| na2q | slack | 0 | 2 | 2.000 ± 0.000 | n/a | 0.320 ± 0.008 | 0.296 ± 0.027 | 0.005 ± 0.005 | 1.000 ± 0.000 |
| na2q | slack | 50 | 2 | 2.000 ± 0.000 | n/a | 0.320 ± 0.008 | 0.265 ± 0.040 | 0.000 ± 0.000 | 1.000 ± 0.000 |

## Data and ground truth (H0)

| cell | test return | realised = planted | A critical | clipped | GT remove-vs-noop L1 (mean / max) | minutes |
|---|---|---|---|---|---|---|
| d0 b2 slack s0 | 10.00 | 0.80 | 0.03 | 0.00 | 0.080 / 0.500 | 17.0 |
| d0 b2 slack s1 | 10.00 | 0.87 | 0.07 | 0.00 | 0.027 / 0.400 | 16.8 |
| d0 b2 slack s2 | 10.00 | 0.93 | 0.07 | 0.00 | 0.040 / 0.500 | 17.1 |
| d0 b2 slack s3 | 10.00 | 0.93 | 0.00 | 0.00 | 0.028 / 0.436 | 17.0 |
| d0 b2 slack s4 | 10.00 | 0.77 | 0.00 | 0.00 | 0.138 / 0.800 | 17.1 |
| d50 b2 slack s0 | 10.00 | 0.80 | 0.03 | 0.00 | 0.080 / 0.500 | 20.5 |
| d50 b2 slack s1 | 10.00 | 0.87 | 0.07 | 0.00 | 0.027 / 0.400 | 20.5 |
| d50 b2 slack s2 | 10.00 | 0.93 | 0.07 | 0.00 | 0.040 / 0.500 | 20.5 |
| d50 b2 slack s3 | 10.00 | 0.93 | 0.00 | 0.03 | 0.028 / 0.436 | 13.9 |
| d50 b2 slack s4 | 10.00 | 0.77 | 0.00 | 0.00 | 0.138 / 0.800 | 13.9 |

## Final critic loss (mean over seeds)

| alg | enabler | delay | bystanders | final loss |
|---|---|---|---|---|
| na2q | slack | 0 | 2 | 0.041 ± 0.003 |
| na2q | slack | 50 | 2 | 0.042 ± 0.002 |
| stas | slack | 0 | 2 | 9.070 ± 2.403 |
| stas | slack | 50 | 2 | 24.881 ± 21.541 |
