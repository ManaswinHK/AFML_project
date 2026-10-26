# POST-HOC: H1b split by realised enabler criticality

**Not pre-registered.** Exploratory analysis done after seeing the main-grid results (docs/main_results.md §3.1). It does not change the pre-registered H1b verdict.

120 cells. An episode is *critical* if the event completed in the first step its gate allowed (the enabler was the bottleneck). For each cell: mean level-2 missed over its critical and over its non-critical test episodes; then mean ± sd over seeds that have such episodes. Bounds reused from H1b: ≥ 0.8 expected without criticality, ≤ 0.3 with it.

| notion | setting | delay | bystanders | critical eps (total) | missed, critical eps | missed, non-critical eps |
|---|---|---|---|---|---|---|
| noop | slack | 0 | 0 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 0 | 2 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 0 | 4 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 10 | 0 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 10 | 2 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 10 | 4 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 25 | 0 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 25 | 2 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 25 | 4 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 50 | 0 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 50 | 2 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | slack | 50 | 4 | 5 (150) | 1.217 ± 2.065 (3 seeds) | 1.000 ± 0.015 (5 seeds) |
| noop | critical (12) | 0 | 0 | 80 (150) | 0.123 ± 0.107 (5 seeds) | 1.310 ± 0.175 (5 seeds) |
| noop | critical (12) | 0 | 2 | 81 (150) | 0.117 ± 0.100 (5 seeds) | 1.317 ± 0.180 (5 seeds) |
| noop | critical (12) | 0 | 4 | 81 (150) | 0.112 ± 0.105 (5 seeds) | 1.346 ± 0.194 (5 seeds) |
| noop | critical (12) | 10 | 0 | 80 (150) | 0.123 ± 0.107 (5 seeds) | 1.310 ± 0.175 (5 seeds) |
| noop | critical (12) | 10 | 2 | 81 (150) | 0.117 ± 0.100 (5 seeds) | 1.317 ± 0.180 (5 seeds) |
| noop | critical (12) | 10 | 4 | 81 (150) | 0.112 ± 0.105 (5 seeds) | 1.346 ± 0.194 (5 seeds) |
| noop | critical (12) | 25 | 0 | 80 (150) | 0.123 ± 0.107 (5 seeds) | 1.310 ± 0.175 (5 seeds) |
| noop | critical (12) | 25 | 2 | 81 (150) | 0.117 ± 0.100 (5 seeds) | 1.317 ± 0.180 (5 seeds) |
| noop | critical (12) | 25 | 4 | 81 (150) | 0.112 ± 0.105 (5 seeds) | 1.346 ± 0.194 (5 seeds) |
| noop | critical (12) | 50 | 0 | 80 (150) | 0.123 ± 0.107 (5 seeds) | 1.315 ± 0.178 (5 seeds) |
| noop | critical (12) | 50 | 2 | 81 (150) | 0.117 ± 0.100 (5 seeds) | 1.322 ± 0.185 (5 seeds) |
| noop | critical (12) | 50 | 4 | 81 (150) | 0.112 ± 0.105 (5 seeds) | 1.352 ± 0.194 (5 seeds) |
| baseline | slack | 0 | 0 | 5 (150) | 1.298 ± 2.138 (3 seeds) | 0.971 ± 0.045 (5 seeds) |
| baseline | slack | 0 | 2 | 5 (150) | 1.307 ± 2.154 (3 seeds) | 0.975 ± 0.045 (5 seeds) |
| baseline | slack | 0 | 4 | 5 (150) | 1.303 ± 2.157 (3 seeds) | 0.969 ± 0.051 (5 seeds) |
| baseline | slack | 10 | 0 | 5 (150) | 1.298 ± 2.138 (3 seeds) | 0.971 ± 0.045 (5 seeds) |
| baseline | slack | 10 | 2 | 5 (150) | 1.307 ± 2.154 (3 seeds) | 0.975 ± 0.045 (5 seeds) |
| baseline | slack | 10 | 4 | 5 (150) | 1.303 ± 2.157 (3 seeds) | 0.969 ± 0.051 (5 seeds) |
| baseline | slack | 25 | 0 | 5 (150) | 1.298 ± 2.138 (3 seeds) | 0.971 ± 0.045 (5 seeds) |
| baseline | slack | 25 | 2 | 5 (150) | 1.307 ± 2.154 (3 seeds) | 0.975 ± 0.045 (5 seeds) |
| baseline | slack | 25 | 4 | 5 (150) | 1.303 ± 2.157 (3 seeds) | 0.969 ± 0.051 (5 seeds) |
| baseline | slack | 50 | 0 | 5 (150) | 1.298 ± 2.138 (3 seeds) | 0.971 ± 0.045 (5 seeds) |
| baseline | slack | 50 | 2 | 5 (150) | 1.307 ± 2.154 (3 seeds) | 0.975 ± 0.045 (5 seeds) |
| baseline | slack | 50 | 4 | 5 (150) | 1.303 ± 2.157 (3 seeds) | 0.970 ± 0.051 (5 seeds) |
| baseline | critical (12) | 0 | 0 | 80 (150) | 0.524 ± 0.163 (5 seeds) | 1.610 ± 0.172 (5 seeds) |
| baseline | critical (12) | 0 | 2 | 81 (150) | 0.540 ± 0.170 (5 seeds) | 1.611 ± 0.091 (5 seeds) |
| baseline | critical (12) | 0 | 4 | 81 (150) | 0.563 ± 0.204 (5 seeds) | 1.619 ± 0.161 (5 seeds) |
| baseline | critical (12) | 10 | 0 | 80 (150) | 0.524 ± 0.163 (5 seeds) | 1.610 ± 0.172 (5 seeds) |
| baseline | critical (12) | 10 | 2 | 81 (150) | 0.540 ± 0.170 (5 seeds) | 1.611 ± 0.091 (5 seeds) |
| baseline | critical (12) | 10 | 4 | 81 (150) | 0.563 ± 0.204 (5 seeds) | 1.619 ± 0.161 (5 seeds) |
| baseline | critical (12) | 25 | 0 | 80 (150) | 0.524 ± 0.163 (5 seeds) | 1.609 ± 0.171 (5 seeds) |
| baseline | critical (12) | 25 | 2 | 81 (150) | 0.540 ± 0.171 (5 seeds) | 1.610 ± 0.091 (5 seeds) |
| baseline | critical (12) | 25 | 4 | 81 (150) | 0.563 ± 0.204 (5 seeds) | 1.618 ± 0.161 (5 seeds) |
| baseline | critical (12) | 50 | 0 | 80 (150) | 0.525 ± 0.164 (5 seeds) | 1.616 ± 0.162 (5 seeds) |
| baseline | critical (12) | 50 | 2 | 81 (150) | 0.541 ± 0.172 (5 seeds) | 1.615 ± 0.091 (5 seeds) |
| baseline | critical (12) | 50 | 4 | 81 (150) | 0.564 ± 0.205 (5 seeds) | 1.622 ± 0.153 (5 seeds) |

## Pooled over all cells and episodes

| notion | missed, critical eps | missed, non-critical eps |
|---|---|---|
| noop | 0.141 (n = 1028; median 0.000) | 1.100 (n = 2572; median 1.000) |
| baseline | 0.544 (n = 1028; median 0.000) | 1.178 (n = 2572; median 1.000) |
