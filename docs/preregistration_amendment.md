# Pre-registration Amendment (2026-10-25)

**Written after pilot 2 and after launching the main grid (commit `50ee30f`), but before looking at any main-grid result.**

## Problem with the adopted plan

`docs/pilot_findings.md` specified *Wilcoxon signed-rank on per-seed means, α = 0.05, Holm-corrected within each hypothesis*, with 5 seeds. With n = 5, the smallest possible one-sided exact Wilcoxon p-value is 1/32 = 0.031 (checked with scipy). After Holm correction over m ≥ 2 tests, the smallest adjusted p is ≥ 0.0625. **No pre-registered test could reach significance, whatever the data.** That would make the plan uninformative, not conservative.

## Amendment

1. **Primary test.** One-sided **paired t-test** on per-seed means (df = 4), Holm-corrected within each hypothesis. Exact Wilcoxon p-values are still reported next to every test.
2. **Effect sizes and conditions are unchanged** (ΔM ≥ 0.2; ΔΛ ≥ 0.05 and Λ(b=2) > 0.1; H0 bounds; H1b bounds; H5 rule).
3. **Verdict rules made explicit** (they were implicit in the adopted table):
   - **H1a:** supported if at least one method in at least one (bystanders, enabler) condition meets the rule, with the critic converged at both delays. Holm runs over all H1a conditions.
   - **H1b:** for each per-step notion, supported if every slack cell has seed-mean level-2 missed ≥ 0.8 and every critical cell ≤ 0.3.
   - **H2:** per method, supported if ≥ 75% of its (delay, enabler) conditions meet the rule. Holm runs over all H2 conditions.
   - **H2′:** supported if COMA leaks significantly less than both VDN and QMIX in every (b ∈ {2, 4}, enabler) condition at d=0.
4. **Known risk, stated in advance.** The critical-enabler setting puts A on the critical path in only about 50% of noisy episodes (measured before the main run), so H1b's "≤ 0.3 when critical" may not be met even if the slack mechanism is real. The threshold is kept as adopted.

The analysis code that implements this is `scripts/analyse_hypotheses.py`.
