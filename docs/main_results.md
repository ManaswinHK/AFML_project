# Mode P Main-Grid Results

**Run:** 120 data cells:
- delay d ∈ {0, 10, 25, 50};
- bystanders b ∈ {0, 2, 4};
- enabler ∈ {slack, critical (start 12)};
- seeds 0–4.

It ran on the Ryzen 5 7600 with 10 workers, about 20 CPU-hours in total. Raw per-cell JSON and the generated tables are in `results/mode_p/main/` (commit `c872308`):
- `summary.md` (all tables);
- `per_seed.csv`;
- `hypotheses.md` (the pre-registered tests).

**Protocol:** `docs/mode_p_protocol.md`. Thresholds are from `docs/pilot_findings.md`, as amended by `docs/preregistration_amendment.md`:
- one-sided paired t-test on per-seed means;
- Holm correction within each hypothesis;
- exact Wilcoxon p reported alongside (its floor with 5 seeds is 0.031).

All numbers are mean ± sd over 5 seeds of per-seed means over 30 test episodes.

Sections 1–2 are **confirmatory**: the pre-registered rules, applied exactly as written. Section 3 is **post-hoc and exploratory**, done after seeing the data, and is labelled as such throughout.

## 1. Pre-registered verdicts

| Hypothesis | Verdict | One-line reason |
|---|---|---|
| H0: ground truth is protocol-stable | **Not supported** (narrow) | Fails only in 4 of 24 data configurations, all with b = 4 |
| H1a: missed enabler credit grows with delay | **Not supported** | The only converged method with a trend (QMIX) misses Holm |
| H1b: slack, not delay, removes per-step enabler credit | **Not supported** | Slack half holds in 24/24 cells; critical half fails in 24/24 |
| H2: leakage grows with bystanders | **COMA supported** (8/8); VDN 1/8, QMIX 1/8; MAPPO 8/8 **is an artifact** (§2) | |
| H2′: COMA leaks less than VDN/QMIX at d = 0 | **Not supported, reversed** (0/8) | COMA leaks *more*, in every comparison |
| H5: end-to-end accuracy does not imply fidelity | **Supported** | 30 of the 32 best end-to-end cells have step r < 0.3 |

H3 (coalition level) needs NA²Q (Step 7). H4 needs Mode T (Step 8).

### H0: ground-truth stability

Rule: mean remove-vs-noop share L1 ≤ 0.15 **and** realised = planted ≥ 0.6, in every cell.

- 104 of 120 cells pass.
- **The 16 failures are 4 data configurations repeated across the 4 delays** (delay does not change trajectories): b = 4 critical seeds 1, 3, 4 and b = 4 slack seed 4.
  - Worst mean L1: 0.180.
  - Worst realised = planted: 0.53.
- **Mechanism:** this is the physical interference already seen in Step 3. With four bystanders on a 5-row grid, a noisy agent sometimes blocks or unblocks someone, so the realised coalition differs from the planted one.
- **Consequence:** every metric is graded against the **realised** ground truth (exact replay), not the planted structure. The failure lowers how far the planted labels can be trusted at b = 4, not the validity of the grading. It does mean that at b = 4 the "bystanders" are not always inert. The leakage metric already handles this, because it uses *realised* bystanders (agents in no non-zero dividend).

### H1a: delay increases missed enabler credit

Rule: for a method whose critic converged at both d = 0 and d = 50 (final loss at d = 50 ≤ 2× that at d = 0):

- ΔM = M(d=50) − M(d=0) ≥ 0.2;
- Holm-adjusted p < 0.05.

| method | converged? | ΔM range over 6 conditions | best Holm p |
|---|---|---|---|
| VDN | **no** in all 6 | +0.05 … +1.44 (critical: +0.79 … +1.44) | 0.0005 (does not count: not converged) |
| QMIX | yes in 5/6 | +0.00 … +0.49 | 0.074 (b = 4, critical) |
| COMA | yes | +0.02 … +0.16 | 1.0 |
| MAPPO | yes | −1.73 … −0.21 | 1.0 (meaningless: §2) |

- **VDN's critic stops fitting once there is any delay.** Final TD loss is:
  - 0.03–0.38 at d = 0;
  - about 0.95 at d = 10;
  - 1.0–1.4 at d ≥ 25.

  That is the expected consequence of VDN's mixer ignoring the global state: the payout countdown added in Fix 1 never reaches it, so its target is non-Markov under delay. VDN's large ΔM is real end-to-end behaviour, but the pre-registered gate correctly attributes it to failed fitting rather than to credit assignment.
- **QMIX** has the right sign in every condition. The effect is moderate in the critical setting (b = 4: M goes from 0.003 at d = 0 to 0.49 at d = 50), but it is not significant after Holm with 5 seeds.

### H1b: slack, not delay (level 2, method-independent)

Rule, for each per-step notion: seed-mean level-2 missed ≥ 0.8 in every slack cell and ≤ 0.3 in every critical cell.

| notion | slack cells (missed) | critical cells (missed) |
|---|---|---|
| noop effect | 0.991 in all 12 ✓ | 0.653–0.667 ✗ |
| behaviour-baseline effect | 0.964–0.970 in all 12 ✓ | 1.020–1.039 ✗ |

- **The slack half is clean, and it is the key mechanism result.** A perfect per-step critic gives a slack enabler essentially zero credit (missed ≈ 1). The value is identical to three decimals at every delay. The semantic gap between per-step and removal credit comes from slack (A's pull is not on the critical path), and delay plays no part in it.
- **The critical half fails, partly for a reason stated in advance.** The amendment warned that the critical setting makes A critical in only about 50% of noisy episodes. Measured on the test episodes: 54% on average (range 37–63% across seeds), against 3% in the slack setting. This dilution is only part of the story. If the ~46% non-critical episodes have missed ≈ 0.99, as slack cells do, a seed mean of 0.66 implies missed ≈ 0.38 in the critical episodes. That is still above the 0.3 threshold. §3.1 proposes testing this directly, post hoc.
- **The behaviour-baseline notion is above 1 even when A is critical**: the enabler gets *negative* episode credit. The likely mechanism, not yet verified step by step: under the behaviour policy A's pull is the likely action, so its advantage is near zero. A's movement steps, which ε-noise frequently perturbs, then dominate with small negative advantages. If so, a COMA-style baseline underrates a critical enabler even with a perfect critic.

### H2: leakage vs bystanders

Rule per (method, delay, enabler):

- Λ(b=4) − Λ(b=2) ≥ 0.05;
- Holm p < 0.05 over all 32 conditions;
- Λ(b=2) > 0.1.

A method passes on ≥ 75% of its conditions.

- **COMA:** 8/8.
  - Λ(b=2) is 0.39–0.47.
  - ΔΛ is +0.12 … +0.21.
  - All Holm p ≤ 0.018.
- **VDN 1/8, QMIX 1/8.** The direction is consistent (see §3.2), but most conditions miss the corrected threshold with 5 seeds. QMIX at d = 0 also fails the Λ(b=2) > 0.1 floor (0.06–0.07); it barely leaks at all there.
- **MAPPO 8/8 does not count as evidence** (§2).

### H2′: COMA vs VDN/QMIX at d = 0

Pre-registered expectation: COMA's counterfactual baseline should leak less. The opposite holds in all 8 comparisons.

- Λ_other − Λ_COMA is between −0.22 and −0.48.
- One-sided p ≈ 1 in every case.

At d = 0 with b = 2:

| method | slack | critical |
|---|---|---|
| COMA | 0.42 | 0.39 |
| VDN | 0.14 | 0.17 |
| QMIX | 0.07 | 0.06 |

Pilot 2 showed the same reversal. COMA's per-agent advantage is noisy: its level-1 fidelity is r = 0.04–0.32. That noise spreads |share| uniformly, and bystanders pick it up.

### H5: accuracy ≠ fidelity

- The best third by end-to-end L1 is 32 of 96 (method × condition) cells, with L1 ≤ 1.103.
- **30 of them** have level-1 step r < 0.3.

Examples:

| method | condition | end-to-end L1 | step r |
|---|---|---|---|
| QMIX | d = 0, b = 2, slack | 0.44 | 0.14 |
| QMIX | d = 0, b = 0, slack | 0.60 | 0.04 |

A method can give approximately right episode-level credit while its per-step read-out barely tracks the counterfactual it is supposed to estimate. Level-1 fidelity across the whole grid is r ≤ 0.36 for every method.

## 2. The MAPPO read-out artifact (affects H1a, H2)

MAPPO's read-out is its **shared** advantage (n-step, on standardised rewards). It is the same number for every agent at every step. After share normalisation, every agent therefore gets ±1/n of the episode's credit:

- **Leakage is b/n by construction.** Measured Λ is within 0.006 of b/n on average: 0.400 = 2/5 at b = 2 and 0.571 = 4/7 at b = 4. The maximum deviation is 0.09, at d = 50, where the sign of the episode sum varies between episodes.

  So "leakage grows with bystanders" is guaranteed by arithmetic for MAPPO, and its **H2 8/8 says nothing about credit assignment**.
- **The sign is negative.** At d ≤ 25, end-to-end L1 is exactly 2.000 in every seed. The episode-summed advantage is negative on every test episode, so every agent gets −1/n: the critic slightly overestimates the value of the successful scripted episodes. Missed ≈ 1.5–2.0 and ΔM ≈ −1.7 for MAPPO are therefore read-out artifacts, not evidence on H1a.
- **The meaningful MAPPO quantity is the policy-gradient SNR**, which is per-agent through each agent's own log-probability gradient:

  | delay | coalition SNR | bystander SNR |
  |---|---|---|
  | d ≤ 10 | 0.04–0.07 | 0.03–0.04 |
  | d = 50 | 0.08–0.11 | ≈ 0.03 |

  At d ≤ 10 the two are indistinguishable. At d = 50 the coalition is about 3× the bystanders. This was not pre-registered as a test and is reported descriptively.

**Decision for the report:** MAPPO is excluded from the H1a/H2 conclusions. It is discussed only through SNR, and the reason is stated. This exclusion is made *after* seeing the data. It follows from the read-out's definition, not from the results, but it is still post-hoc and is labelled so.

## 3. Post-hoc exploratory analyses

*These were done after looking at the main-grid results, were not pre-registered, and should be read as hypothesis-generating.*

### 3.1 H1b restricted to genuinely critical episodes (planned, not yet run)

Prediction from the mixture argument in §1: restricted to test episodes where A is actually on the critical path, level-2 missed for the noop notion should be about 0.38. If it comes out at that level, the critical half fails for two reasons: the manipulation is diluted, and per-step credit under-credits A modestly even when A is critical.

- **What it needs:** per-episode criticality flags. The cell JSONs store only the rate (`test_enabler_critical`).
- **Cost:** only the ground-truth stage needs rerunning (about 4 minutes per data configuration, no critic training).
- **Status:** not done yet. Until then, H1b's critical half stays *not supported* as pre-registered.

### 3.2 Direction of the VDN/QMIX leakage effect (sign test)

Pre-registered H2 failed for VDN and QMIX, but ΔΛ = Λ(b=4) − Λ(b=2) is positive in:

- **8/8 conditions for each method**, with ≥ 0.05 in 7/8 (VDN) and 6/8 (QMIX);
- 38/40 (VDN) and 39/40 (QMIX) individual seed pairs.

A one-sided sign test over the 16 condition means gives p = 1.5 × 10⁻⁵. The conditions share data, so they are not independent, and this p-value is indicative only.

The reasonable reading: bystander leakage grows for all value-decomposition methods. The pre-registered per-condition test, with 5 seeds and Holm over 32 conditions, was underpowered to show it condition by condition.

## 4. What the results support, in plain terms

1. **Per-step credit loses enablers through slack, not delay** (confirmatory, H1b slack half). Even a perfect per-step critic gives a non-critical enabler about zero credit, whatever the delay. Removal-based credit gives it a full share.
2. **COMA's counterfactual baseline does not protect against bystanders; it leaks most** (confirmatory: H2′ reversed, H2 supported for COMA).
3. **Good end-to-end credit does not mean faithful per-step credit** (confirmatory, H5). QMIX is the best end to end at d = 0, but with step r ≈ 0.1.
4. **Delay hurts the methods whose critic cannot represent it.**
   - VDN fails to fit with any delay (a training failure that the H1a gate excludes).
   - QMIX shows a consistent but not significant increase in missed credit.

   (H1a not supported.)
5. **Shared-advantage methods (MAPPO) cannot be evaluated by per-agent credit read-outs at all.** Only their gradient SNR is informative.

## 5. Limitations

- **Statistical power.** 5 seeds and Holm across up to 32 conditions make the per-condition tests underpowered; §3.2 suggests H2 for VDN/QMIX is a power failure rather than a null.
- **Planted vs realised coalitions.** At b = 4 the realised coalition differs from the planted one in up to 47% of episodes (H0).
- **Training mode.** Mode P trains critics only, on scripted data. VDN/QMIX learn Q*, while COMA/MAPPO learn the behaviour policy's value. The conclusions are about critics as credit estimators, not about online learning (Mode T, Step 8).
- **The critical-enabler manipulation is weak.** It produces only about 54% critical episodes.
