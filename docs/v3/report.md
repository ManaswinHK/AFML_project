# Who earned the reward? v3 report: credit by intervention

Team Aporia · 12 Nov 2026 · branch `intervention-credit-v3` · pre-registrations frozen at `a08b267` (Mode P) and `5a73c9c` (Phase 3)

> **Status of this document:** sections marked {{RESULTS}} are filled from the generated files `confirmatory_results.md` and `phase3_results.md` once all runs finish. Everything else is final.

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

{{RESULTS}}

## 5. Does credit change learning? (Phase 3)

{{PHASE3}}

## 6. Credit for free from robustness training? (Phase 4)

{{PHASE4}}

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
