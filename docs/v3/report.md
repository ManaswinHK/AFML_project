# Who earned the reward? v3 report: credit by intervention

Team Aporia · 12 Nov 2026 · branch `intervention-credit-v3` · pre-registrations frozen at `a08b267` (Mode P) and `5a73c9c` (Phase 3)

> **Status of this document:** final for the Mode P confirmatory round, Phase 3 and Phase 4. Sections still marked with double braces are filled when the remaining post-hoc/descriptive runs finish.

## 0. Summary

{{SUMMARY}}

## 1. What v3 set out to do

The round-2 assessment (`docs/assessment_and_plan.md`) recommended reframing the project as **credit by intervention**:
- a one-step counterfactual (what one-step read-outs such as COMA or Q_tot differences estimate) cannot identify who earned a reward;
- a **persistent** counterfactual (an agent stops for the rest of the episode) can, and it is learnable by an ordinary value critic once the data contains agents stopping.

v3 executes the plan:
- **Phase 0**: fix the record;
- **Phase 1**: confirmatory repeat on fresh seeds;
- **Phase 2**: generalisation, and where a model is needed;
- **Phase 3**: does exact credit change learning?
- **Phase 4** (reduced): does a robustness-trained critic give credit for free?

It also adds three things the plan did not have:
1. **The commitment split** (`theory.md` §3–4), an exact decomposition of persistent credit into *shared surprise* (luck) and *structural contribution*. It extends to per-step Shapley values and coalition dividends that sum exactly to the episode's, with credit landing when an outcome becomes *certain* rather than when its delayed reward is paid.
2. **A permutation-invariant critic** motivated by a stated symmetry (relabelling agents and the stopped set together leaves the value unchanged). Tested on shuffled roles, with role flags hidden, and for zero-shot transfer to unseen team sizes.
3. **A precise account of when credit matters for learning.** Expected-value credit is provably equivalent to the team reward with a good baseline. Only *realised* counterfactuals, which split luck between agents, can speed up learning. Phase 3 tests the prediction.

## 2. Phase 0: the record, fixed

`errata.md` lists ten corrections (E1–E10). The ones that change conclusions:
- **H5** is a statement about the *data* (removal interventions), not ECE's network: a presence-only Lasso table matches it.
- **STAS**'s equal split on the coalition layout is forced by constant-return training data (every main-grid episode returns exactly 10).
- **H4** was uninformative, not "mildly supportive".
- **Closed-loop truth** is the right default.

Code fixes:
- the partial-layout state is now Markov (pending payout amount);
- finishers proceed after an absent enabler has pulled;
- persistent stops are an environment operation, with acting flags in the global state;
- agents' role flags ("unit types") are in the state.

All are covered by tests (`tests/test_v3_phase0.py`, 23 tests).

## 3. Method

**Credit by intervention** (`src/whoearned/ablation.py`, `critics.py`):
1. **Collect.** 200 full-team episodes plus 200 *stopped continuations*: from a random step of a random training episode, a random set of agents stops for good.
   - Who stops: half single agents, half random subsets.
   - When: stratified, half before the event completes and half in the payout wait.
   - Null: no-op (the agent stays inert; teammates are not told) unless stated.
2. **Train** a state-value critic by Monte-Carlo regression on the return-to-go; the acting flags are part of the state.
   - **MLP**: 2×128.
   - **Set critic**: shared per-agent encoder, one self-attention layer, sum and mean pooling, value head. Exactly invariant to relabelling agents.
3. **Read out** on factual test states only: W_{−i}(s_t) = critic(s_t with i's acting flag off).
   - **κ (ablation-TD credit)**: κ_i(t) = r_t + W_{−i}(s_{t+1}) − W_{−i}(s_t), which sums to removal credit.
   - **Structural read-out**: c_i(t) = D_i(s_t) − D_i(s_{t+1}) with D_i = V − W_{−i}.
   - **Coalition dividends** from W_S(s_0) for all S.

**Truth.** The exact persistent-ablation values for every test episode (closed loop, common random numbers, both nulls), the closed-loop removal game, the closed-loop no-op game, and open-loop truth. Each method is graded against the truth of its own null.

**Baselines and controls:**
- equal split;
- no-intervention critic;
- presence-only table (`table3_mix`, ECE's own data);
- ECE-3_mix;
- VDN, QMIX, COMA, MAPPO;
- **control V2**: MAPPO and QMIX trained on the *same* stopped episodes, read out with their standard one-step read-out and with a masked read-out;
- an **expected-credit floor**: the best any expected-credit estimator can do on a fixed layout.

## 4. Confirmatory results (seeds 5–9)

Full per-condition tables, p-values and the one analysis-script correction (P5 argument order): `confirmatory_results.md`. Values below are condition means over seeds 5–9 (ranges are over conditions). L1: share distance to the exact credit of the method's own null (0 = exact, 2 = disjoint). The equal split scores 0.80 (b = 2) and 1.14 (b = 4).

### 4.1 Verdicts

| ID | Hypothesis | Verdict | Key numbers |
|---|---|---|---|
| P2a | MLP ablation critic learns removal credit (L1 < 0.15) | **Supported** 14/14 | L1 0.012–0.063 (main), 0.014–0.025 (overlap) |
| P2a-set | Set critic does too (L1 < 0.05) | Partial 12/14 | L1 0.002–0.032. The 2 misses have means 0.026 and 0.032, below the threshold but not significant with 5 seeds |
| P2b | Beats the equal split | **Supported** 14/14 | |
| P3 | Interventions are necessary (gap > 0.3) | Partial 12/14 | No-intervention critic 0.85–1.50 vs 0.01–0.06. The 2 misses (overlap) have gaps 0.65–0.72, not significant after Holm |
| P4a | MLP per-step fidelity r > 0.85 | Partial 10/12 | r 0.91–1.00. Misses: critical cells at d = 25/50 (r 0.92, 0.91, sd 0.1), the MLP's seed instability seen in the pilot |
| P4a-set | Set critic per-step r > 0.85 | **Supported** 12/12 | r 0.98–1.00 |
| P4b | One-step read-outs are not faithful to their own per-step targets (r < 0.5) | **Supported** for VDN, QMIX, COMA, MAPPO, 12/12 each | r −0.00 to 0.19 |
| P5 | Presence-only table not worse than ECE-3 (dividend error, removal game) | **Supported** 6/6 (after correcting the script's argument order) | equal on the single coalition (0.142 vs 0.143); better on overlap (0.47 vs 0.99–1.02) and partial (0.41 vs 1.04–1.05) |
| P6a | Ablation credit keeps exactly the coalition (top-k > 0.9) | **Supported** 12/12 | 100% in every condition |
| P6b | QMIX / COMA usually do not (< 0.5) | QMIX partial 9/12; COMA **supported** 12/12 | QMIX keeps the coalition in 77% of d = 0 slack episodes, "right for the wrong reason" again; COMA 1–15% |
| P7a | Shuffled roles: the table cannot use role information | **Supported** 2/2 | table = equal split (0.78, 1.15) |
| P7b | Shuffled roles: MLP below the equal split by ≥ 0.25 | **Not supported** 0/2 | gaps 0.27 and 0.26: above the margin as point estimates, not significant |
| P8a | Shuffled roles: set critic near exact (L1 < 0.05) | **Supported** 2/2 | **0.010 and 0.008** |
| P8b | … beats the MLP by ≥ 0.25 | **Supported** 2/2 | MLP 0.53 and 0.88 |
| P8c-a | … recovers coalition structure (dividend error < 0.4) | **Supported** 2/2 | 0.13 and 0.21 |
| P8c-b | … better than ECE-3_mix on dividends | Partial 1/2 | b = 4: 0.21 vs 0.57 (yes); b = 2: 0.13 vs 0.20 (p = 0.055) |
| P8d | Set critic's advantage holds with role flags hidden | **Supported** 2/2 | 0.019 / 0.007 vs MLP 0.60 / 0.93 |
| P9a | MAPPO trained on stops: masked read-out beats its standard read-out by ≥ 0.5 | Partial 8/12 | d ≤ 25: 0.32–0.69 vs 1.99–2.00 (all supported). d = 50: gaps 0.26–0.89, not all significant |
| P9b | QMIX trained on stops: masked vs standard (two-sided, descriptive) | – | At d > 0 the masked read-out is better in 7 of 8 conditions (by 0.02–0.49). The pilot's reversal was a d = 0 effect |
| P9c | Data alone does not fix one-step read-outs (MAPPO gain < 0.1) | Partial 9/12 | d ≤ 25: no gain (2.00 → 2.00). d = 50: the stops help MAPPO's standard read-out a little (gains 0.06–0.20) |
| P10a | Partial layout: structural read-out within 0.1 of the expected-credit floor | **Not supported** 0/2 | 0.49 / 0.51 vs floor 0.39 / 0.38 |
| P10b | … and better than κ | **Supported** 2/2 | κ 0.65 / 0.66 |
| P11a | Size transfer keeps exactly the coalition | **Supported** 2/2 | 100% in every seed, both directions |
| P11b | … better than the equal split by > 0.3 | **Supported** 2/2 | 5 → 7 agents: 0.60 vs 1.14; 7 → 5 agents: 0.29 vs 0.80 |

**Tally (28 entries, counting P4b's four methods and P6b's two separately):** 18 supported, 7 partial, 2 not supported, 1 descriptive.
- In the partial results, the failing conditions have point estimates on the hypothesis's side in P2a-set, P3, P4a and P8c-b. They fail only on significance with 5 seeds under Holm correction.
- The other partials are real exceptions: QMIX's top-k at d = 0 (P6b) and MAPPO's read-outs at d = 50 (P9a, P9c).
- The two failures (P7b, P10a) are honest limits, discussed below.

### 4.2 What the confirmatory round establishes

1. **Persistent interventions make removal credit learnable by an ordinary critic, and nothing else in the comparison does.** On the 12 main conditions:
   - the MLP ablation critic is at 0.012–0.063 and keeps exactly the coalition in 100% of episodes;
   - the same critic without stops is at 0.85–1.50;
   - every one-step method is at 0.50–2.00 (VDN 0.71–1.72, QMIX 0.50–1.42, COMA 1.45–1.62, MAPPO 0.83–2.00);
   - the one-step methods are unfaithful even to their own per-step targets (r ≤ 0.19). This is the measured counterpart of Identity 2.
2. **Better data does not rescue one-step read-outs; a different read-out of the same critic does (control V2).**
   - MAPPO trained on exactly the same stops still gives 2.00 with its own read-out at d ≤ 25.
   - Queried at the stopped state, the same critic gives 0.32–0.69.
   - The plan's open question ("better data or better read-out?") has a clear answer: **the read-out**, given data that contains stops. The data alone is not enough.
3. **ECE's coalition result belongs to the data.** A presence-only Lasso table, which never sees a trajectory, matches ECE-3_mix on the single coalition and beats it on overlap and partial (errata E1, now confirmatory).
4. **A critic that respects the relabelling symmetry is what makes credit generalise.**
   - With roles reassigned every episode, the set critic is near exact: 0.010 / 0.008, step r ≈ 1.0, top-3 100%, dividend error 0.13 / 0.21.
   - The MLP, with the same 200 stops and the same inputs, is at 0.53 / 0.88, and needs 4× the stops to reach 0.17 / 0.52.
   - The advantage does not come from role flags: with flags hidden, 0.02 / 0.007 vs 0.60 / 0.93.
   - The set critic also transfers zero-shot to team sizes it never saw: always the right coalition, half the equal split's error. It leaks 14–30% of the credit to unseen bystanders.
5. **Luck is real and measurable.** On the partial layout, realised credit flips with exploration noise.
   - Removing the luck term (the structural read-out) improves the critic from 0.65 to 0.49.
   - It does not reach the expected-credit floor (0.38–0.39).
   - About 0.1 of L1 there is ordinary estimation error on top of the luck.

## 5. Does credit change learning? (Phase 3)

Pre-registration: `preregistration_v3_phase3.md` (frozen before any gate run). Full tables: `phase3_results.md`. Figure: `figures/gate_curves.png`.

**Setting.** K walled lanes (one enabler + one finisher each, reward 10), IPPO from EPyMARL with default hyperparameters (shared parameters), seeds 5–9. Three arms:
- **team reward**;
- **team_vector**: the team reward delivered per agent, a control for the per-agent-reward code path;
- **difference**: exact difference rewards from lockstep shadow copies, r − r^(−i).

Metric: episodes to 90% greedy test success.

| K | team | difference | ratio (diff/team) | team_vector / team |
|---|---|---|---|---|
| 1 | 2,730 | 2,730 | **1.00** | 1.00 |
| 2 | 13,846 (2 runs stuck at 0.5) | 7,208 (1 stuck) | 0.52 | 1.00 |
| 4 | 1,370 | 810 | 0.59 | 1.00 |
| 8 | 1,370 | 570 | **0.42** (per seed 0.25–0.67) | 1.00 |

| Hypothesis | Verdict |
|---|---|
| H3.1: no headroom with one event (ratio in [0.8, 1.25] at K = 1) | **Supported** (1.00) |
| H3.2: headroom with many events (K = 8, log ratio < log 0.67) | **Supported** (mean ratio 0.40, p = 0.018) |
| H3.3: the per-agent-reward control is inert | **Supported** (1.00 at every K) |
| H3.4: headroom grows monotonically with K | Not monotone (ρ = −0.40; K = 2 is the exception) |

**Reading.**
- **The gate opens.** Exact *realised* credit speeds up learning about 2.5× at K = 8, and not at all with one event, exactly as `theory.md` §7 predicts. It is the classical difference-rewards effect, now measured against the same benchmark's ground truth.
- It also says **where** the project's original learning claim ("wrong credit trains wrong behaviour") can live: in tasks where the team reward mixes many unrelated events. The single-coalition layout (H4) could never show it.
- **K = 2 is an exploration pathology, not a credit effect.** Stuck runs plateau at exactly 0.5 success: one lane solved, the other never. It hits both arms (2/5 team, 1/5 difference), so it does not favour either.
- **Caveat on resolution.** With shared parameters, K = 8 learns *faster* per environment step than K = 1 (one policy gets 2K agents' experience), so episodes-to-90% falls on 2–4 test points of the pre-registered 10k-step test interval. Ratios are therefore coarse (multiples of about 1/3). Addendum A (post-hoc) re-runs at 5× resolution, and without parameter sharing, below.

{{PHASE3_ADDENDUM}}

## 6. Credit for free from robustness training? (Phase 4)

Pre-registration: `preregistration_v3_phase4.md`, including its post-hoc addendum, written after a 5-episode smoke test and before the full evaluation. Data: `results/v3/phase4/*.json`.

**Setting.**
- **Layout:** 2 walled lanes plus 2 task-less bystanders (n = 6), so an equal split is wrong.
- **Training:** EPyMARL MAPPO with team reward, 300k steps, seeds 5–7.
- **Arms:** without dropout, or with **dropout 0.25** (a random set of agents stops for good at a random step in 25% of training episodes).
- **Evaluation:** every 30k-step checkpoint is scored on 30 clean test episodes against the **exact** persistent-ablation truth of *that* learned policy, computed by branching the policy itself (RNG, RNN state and buffer; tested exact).

**Final checkpoint (L1 vs exact credit; equal split = 0.67):**

| Read-out | no dropout (s5, s6, s7) | dropout 0.25 (s5, s6, s7) |
|---|---|---|
| κ, as pre-registered | 0.68, 0.66, 0.68 | 0.66, 0.66, 0.67 |
| κ with γᵗ weights (post-hoc) | 0.68, 0.66, 0.68 | 0.66, 0.66, 0.67 |
| **structural V(s₀) − W₋ᵢ(s₀) (post-hoc)** | 1.91, 1.52, 1.99 | **0.43, 0.43, 0.74** |
| MAPPO's standard shared advantage | 0.67 (equal split by construction) | 0.67 |
| clean success | 1.00, 1.00, 0.98 | 1.00, 1.00, 1.00 |

| Prediction | Verdict |
|---|---|
| H4.1: dropout makes MAPPO's critic an ablation critic (κ read-out) | **Met only on the letter, negligible in size**: dropout < no dropout in 3/3 seeds and below the equal split by 0.005. The prediction had no margin; this is not meaningful support. |
| H4.2: without stops, the masked read-out is no better than an equal split | **Supported** (0.67 vs 0.67) |
| H4.3: robustness training costs little return | **Supported** (success 1.00 vs 0.99) |
| H4.4: fidelity over training (descriptive) | With dropout, the structural read-out improves steadily (1.85 → 0.43 in two seeds; s7 reaches 0.41 then drifts to 0.74). **Without dropout it gets worse with training** (dip to about 0.7–1.1, then 1.5–2.0): the critic's extrapolation to never-seen stopped states degrades. |

**Reading.**
- **A robustness-trained critic carries credit, but not in a form κ can extract.**
  - MAPPO's critic works in standardised, discounted units and is offset from realised returns.
  - κ compares the critic with the realised return, so the offset is common to all agents and swamps the per-agent differences.
  - The structural read-out compares the critic with itself, so the offset cancels. The commitment split is therefore useful in practice, not only as bookkeeping.
- Even then, **credit "for free" is much weaker than credit by design**: 0.43–0.74 against 0.02 for the Monte-Carlo intervention critic of §4. That gap is the cost of a TD-trained, reward-standardised critic whose stops land after commitment most of the time (lanes complete by step 3–5 of 25).
- **Without stops in its data, a critic's answer to "what if i stopped?" is arbitrary and drifts as training continues.** This is the learning-time version of P3.

## 7. Descriptive and exploratory results

{{EXPLORATORY}}

## 8. Limits

- **One environment family**, scripted teams in Mode P, n ≤ 7. The learned-policy results (Phases 3–4) use simple walled lanes.
- **Interventions are needed.** You must be able to make an agent stop during data collection (a wrapper suffices; no reset or replay). Logged full-team data cannot identify removal credit when teammates react (`theory.md` §6).
- **The critic estimates expected credit.** Realised credit, including who was lucky in one episode, needs a realised counterfactual. The commitment split makes the gap explicit, and the partial layout measures it (P10).
- **The null is a modelling choice.** No-op and removal define different games where teammates react to absence (overlap layout). Credit should be reported with its protocol.
- **Size transfer is partial:** the right coalition, but credit leaks to unseen bystanders. Mixed-size training is the obvious next step.

## 9. What to claim, and what not

{{CLAIMS}}

## 10. Plan items not run, and why

| Item | Why not |
|---|---|
| V3 (Identity 2 on trained critics) | With a state-value critic, summing the one-step bracket along stopped trajectories telescopes to the same quantity κ reads out. Its only empirical content (querying the critic at off-distribution states) is what V2/P9 tests. |
| 2.5 Randomised start cells | Shuffled roles already moves every agent's start cell between episodes; random starts would be a weaker variant of a test already in the confirmatory grid. |
| Phase 3 arm (c), learned κ as a reward | With exact critics its advantage equals the team TD error (theory §7), so it measures critic error, not credit. Replaced by the `team_vector` control. |
| 4.4 Level-Based Foraging | Out of scope for the one-week deadline. It needs new truth machinery for a new environment. |
| Active intervention design as a claim | Exploratory pilots showed no reliable gain (sometimes worse); reported as a negative result. |
