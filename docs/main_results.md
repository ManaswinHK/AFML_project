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
| H-fid: end-to-end accuracy does not imply fidelity | **Supported** | 30 of the 32 best end-to-end cells have step r < 0.3 |

H3 (coalition level) is in `docs/h3_results.md`; the coalition estimator H5 is in `docs/h5_results.md`. H4 (Mode T) was assessed descriptively for VDN and QMIX only (`docs/mode_t_results.md`).

**Sensitivity to the removal protocol** (`docs/sensitivity.md`). Re-scoring the grid against a closed-loop truth leaves H2, H2′ and H-fid unchanged, but **H1a becomes supported through one condition** (QMIX, b = 4, critical: Holm p 0.074 → 0.030). Treat H1a as borderline, not as a robust negative. The closed-loop truth also recovers the planted coalition in 100% of test episodes (open loop: 83%, 66% at b = 4), so the H0 failures are an artefact of open-loop replay. Shapley vs leave-one-out changes no verdict.
> **Naming.** Earlier drafts of this document called the finding "accuracy without fidelity" *H5*. The problem statement's H5 is the low-order coalition estimator (`docs/h5_results.md`), so this finding is now **H-fid**. The generated `hypotheses.md` files still print the old label.


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
- **The critical half fails, partly for a reason stated in advance.** The amendment warned that the critical setting makes A critical in only about 50% of noisy episodes. Measured on the test episodes: 54% on average (range 37–63% across seeds), against 3% in the slack setting. The post-hoc split in §3.1 shows that this dilution explains the noop failure completely:
- in critical episodes, noop missed is 0.11–0.12;
- in the late-start episodes where A was *not* the bottleneck, it is 1.31–1.35.

The baseline notion still fails even in critical episodes (0.52–0.56).
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

### H-fid: accuracy ≠ fidelity

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

### 3.1 H1b split by realised enabler criticality

**Method.** `scripts/posthoc_h1b.py` regenerates each cell's 30 test episodes and their ground truth (no critic training). It labels each episode critical or not with the same rule the main run used: the event completed in the first step its gate allowed. Level-2 missed is then scored per episode.
- Every one of the 120 cells reproduced its stored `semantic_gap` means exactly before being used.
- The pre-registered bounds are reused unchanged.
- Outputs: `results/mode_p/main_posthoc_h1b/` (per-episode JSON and `h1b_posthoc.md`).

**Results.** Seed means; the values are the same to two decimals at every delay.

| notion | setting | episodes | missed, critical episodes | missed, non-critical episodes |
|---|---|---|---|---|
| noop | critical (12) | 80–81 of 150 per (delay, b) | **0.11–0.12** (sd ≈ 0.10) ✓ ≤ 0.3 | 1.31–1.35 |
| noop | slack | 5 of 150 | 1.22 ± 2.07 (3 seeds; too few to interpret) | **1.00** ✓ ≥ 0.8 |
| baseline | critical (12) | 80–81 of 150 | **0.52–0.56** ✗ | 1.61–1.62 |
| baseline | slack | 5 of 150 | 1.30 ± 2.15 (too few) | **0.97** ✓ |

Pooled over all cells:

| notion | critical episodes | non-critical episodes |
|---|---|---|
| noop | mean 0.14, median 0.00 (n = 1028) | mean 1.10, median 1.00 (n = 2572) |
| baseline | mean 0.54, median 0.00 (n = 1028) | mean 1.18, median 1.00 (n = 2572) |

**Reading.**
1. **For the noop notion, H1b's pattern holds once criticality is conditioned on what actually happened.**
   - A perfect per-step critic gives the enabler nearly its full removal share when A was the bottleneck (missed ≈ 0.12).
   - It gives A nothing when A had slack (missed ≈ 1.0).
   - This holds at every delay.

   The pre-registered failure was the weak manipulation (54% critical), not the mechanism. This corrects the 0.38 estimate in the first version of this document. That estimate assumed non-critical episodes behave like slack ones, but in the late-start setting they score about 1.3.
2. **Non-critical episodes in the late-start setting score above 1.** The enabler gets *negative* per-step credit there. A plausible reading, not checked step by step: A starts late but still isn't the bottleneck, so its late pull is one of the actions that delays the discounted reward.
3. **For the behaviour-baseline (COMA-style) notion, the critical half fails even after conditioning.** The median critical episode has missed 0.00, but the mean is 0.54. A large minority of critical episodes give A strongly negative credit, so the notion is unreliable even when the enabler is decisive.
4. **Caveats.**
   - The slack setting has only 5 critical episodes in total, too few to say anything about criticality there.
   - Critical and non-critical episodes are not randomised: criticality is an outcome of the ε-noise, so the two groups may differ in other ways too.

**Verdict:** H1b stays *not supported* as pre-registered. As a post-hoc, hypothesis-generating result, the slack mechanism is confirmed for the noop notion. It is not confirmed for the baseline notion.

### 3.2 Direction of the VDN/QMIX leakage effect (sign test)

Pre-registered H2 failed for VDN and QMIX, but ΔΛ = Λ(b=4) − Λ(b=2) is positive in:

- **8/8 conditions for each method**, with ≥ 0.05 in 7/8 (VDN) and 6/8 (QMIX);
- 38/40 (VDN) and 39/40 (QMIX) individual seed pairs.

A one-sided sign test over the 16 condition means gives p = 1.5 × 10⁻⁵. The conditions share data, so they are not independent, and this p-value is indicative only.

The reasonable reading: bystander leakage grows for all value-decomposition methods. The pre-registered per-condition test, with 5 seeds and Holm over 32 conditions, was underpowered to show it condition by condition.

### 3.3 Uniform-credit baseline

Uniform credit gives 1/n to every agent. It needs no training. It is scored against the same leave-one-out truth, on the same valid test episodes as each method. Script: `scripts/posthoc_uniform.py`; full table in `results/mode_p/main_posthoc_uniform/uniform_posthoc.md`.

| method | conditions where its mean L1 beats uniform | mean (method − uniform) at b = 0 / 2 / 4 |
|---|---|---|
| VDN | 1/24 | +0.86 / +0.53 / +0.34 |
| QMIX | 3/24 | +0.72 / +0.17 / +0.18 |
| COMA | 0/24 | +1.27 / +0.72 / +0.56 |
| MAPPO | 1/24 (artifact, §2) | +1.73 / +1.08 / +0.85 |

- **Uniform L1 is 0.000 at b = 0** in every cell. With no bystanders, the true leave-one-out credit is split equally among A, B and C, so uniform is the exact answer there. The b = 0 rows therefore only show how far each method is from the right answer; they say nothing about uniform being a strong baseline.
- **With bystanders, uniform L1 is 0.72–0.73 (b = 2) and 1.00–1.02 (b = 4).** The planted values would be 0.80 and 1.14; the realised truth is a little less concentrated (H0).
- **The only wins:**
  - QMIX at d = 0, slack, b = 2 (0.44 vs 0.73) and b = 4 (0.66 vs 1.02), 5/5 seeds each;
  - QMIX at d = 10, critical, b = 4 (0.85 vs 1.00), 5/5 seeds;
  - VDN at d = 0, slack, b = 4 (0.79 vs 1.02), 5/5 seeds;
  - MAPPO at d = 50, critical, b = 2, by 0.01. This is the ±1/n artifact landing near uniform, not information.
- **No method beats uniform at d ≥ 25.** COMA never beats it.

Reading: across most of the main grid, the L1 values do not show that a method carries information about who earned the reward beyond "share equally". The wins are confined to QMIX and VDN at short delay, which is where the H5 result (§1) already shows QMIX is accurate for the wrong reason. This reinforces the H3-grid finding and makes a uniform row a required reference for any L1 table in this benchmark.

## 4. What the results support, in plain terms

1. **Per-step credit loses enablers through slack, not delay** (confirmatory, H1b slack half). Even a perfect per-step critic gives a non-critical enabler about zero credit, whatever the delay. Removal-based credit gives it a full share. Post hoc (§3.1), the noop notion gives a genuinely critical enabler nearly full credit (missed ≈ 0.12), so slack is what makes the difference. The COMA-style baseline notion under-credits the enabler even then.
2. **COMA's counterfactual baseline does not protect against bystanders; it leaks most** (confirmatory: H2′ reversed, H2 supported for COMA).
3. **Good end-to-end credit does not mean faithful per-step credit** (confirmatory, H-fid). QMIX is the best end to end at d = 0, but with step r ≈ 0.1.
4. **Delay hurts the methods whose critic cannot represent it.**
   - VDN fails to fit with any delay (a training failure that the H1a gate excludes).
   - QMIX shows a consistent but not significant increase in missed credit.

   (H1a not supported.)
5. **Shared-advantage methods (MAPPO) cannot be evaluated by per-agent credit read-outs at all.** Only their gradient SNR is informative.
6. **Almost no cell beats equal shares** (post-hoc, §3.3). Mean L1 is below the uniform baseline in only 5 of 96 method-conditions, all at d ≤ 10 except one MAPPO artifact.

## 5. Limitations

- **Statistical power.** 5 seeds and Holm across up to 32 conditions make the per-condition tests underpowered; §3.2 suggests H2 for VDN/QMIX is a power failure rather than a null.
- **Planted vs realised coalitions.** At b = 4 the realised coalition differs from the planted one in up to 47% of episodes (H0).
- **Training mode.** Mode P trains critics only, on scripted data. VDN/QMIX learn Q*, while COMA/MAPPO learn the behaviour policy's value. The conclusions are about critics as credit estimators, not about online learning (Mode T, Step 8).
- **The critical-enabler manipulation is weak.** It produces only about 54% critical episodes.
