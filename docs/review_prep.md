# Review Prep: Everything Done in "Who Earned the Reward?"

Read in order. Parts 1–3 give you the vocabulary and the project's logic. Part 4 is the full story of what was done and found. Part 5 is a cheat sheet of numbers. Part 6 is a bank of likely cross-questions with answers. Part 7 is what we must *not* claim.

Sources, all in this repo: `docs/planning/who_earned_the_reward_problem_statement_v2.md`, `docs/planning/ROADMAP.md`, `docs/environment_design.md`, `docs/mode_p_protocol.md`, `docs/pilot_findings.md`, `docs/preregistration_amendment.md`, `docs/main_results.md`, `docs/step7_ports.md`, `docs/h3_results.md`, `docs/results_overview.md`. If a reviewer asks for a number you do not remember, those files hold it.

---

## Part 1. The 60-second version

**The problem.** In cooperative multi-agent RL, a team gets one shared reward. The reward is often produced by a *subgroup* of agents, and it often arrives *late*. Methods exist that split that reward among agents (credit assignment). They are normally judged by how much reward the team ends up earning, not by whether the credit went to the agents who actually produced the reward.

**What we did.** We built a small gridworld where we *plant* exactly which agents produce each reward and when. Because we planted it, we can compute the correct credit exactly. Then we ask each method's credit signal how close it gets to that correct answer, while varying the reward delay and the number of useless "bystander" agents.

**What we found.**
1. The credit signals in VDN, QMIX and COMA measure a *per-step counterfactual* ("what if this agent did nothing right now?"). That is a different question from "who earned the reward?". Even a perfect critic gives an agent with slack zero credit.
2. A method can score well on end-to-end credit error while its own per-step signal is nearly uncorrelated with the thing it claims to estimate.
3. On our coalition layouts, no method beats simply giving every agent an equal share.
4. The coalition-level read-outs cannot see coalitions at all, because the teamwork happens across time steps and the read-out looks at one step.

**Where we are.** The benchmark and the first batch of experiments are done (Steps 1–7). Remaining: the coalition estimator (H5), a small full-training study (Mode T / H4), and sensitivity checks.

---

## Part 2. Concepts you must be able to explain from scratch

### 2.1 Credit assignment, two halves
- **Temporal:** which earlier action caused a later reward.
- **Structural:** which agent(s) among the team caused it.
Our setting stresses both: the enabler acts early (temporal), and bystanders sit beside the real contributors (structural).

### 2.2 Coalition value v(S)
v(S) is the reward the team would earn if only the agents in set S acted and the others were removed. For n agents there are 2ⁿ subsets, so n = 5 means 32 replays and n = 8 means 256.

### 2.3 Three ground-truth quantities

Use the running example: A is an enabler (pulls a lever), B and C are finishers (push a crate together), D and E are bystanders. Reward 10 only if A, B and C are all present.

| Quantity | Formula | Question it answers | Running example |
|---|---|---|---|
| **Harsanyi dividend** Δ(S) | Σ over A⊆S of (−1)^(\|S\|−\|A\|) v(A) | Which *coalition* created value that no smaller group creates by itself? | Δ({A,B,C}) = 10; all others 0 |
| **Shapley value** φ_i | Σ over S∋i of Δ(S)/\|S\| | What is each agent's *fair share*? The coalition's dividend split equally among its members | φ_A = φ_B = φ_C = 10/3; φ_D = φ_E = 0 |
| **Leave-one-out (LOO) marginal** m_i | v(N) − v(N∖{i}) | How much is lost if this one agent is removed? | m_A = m_B = m_C = 10; m_D = m_E = 0 |

After normalising to shares, all three give ⅓ each to A, B and C here. They differ when coalitions overlap or when partial teams earn partial reward.

**Whiteboard example (G4, the reason we need dividends):**

| Game | Values | Dividends | Shapley |
|---|---|---|---|
| Pure synergy | v(A) = v(B) = 0, v(AB) = 2 | Δ(AB) = 2 | 1, 1 |
| Independent | v(A) = v(B) = 1, v(AB) = 2 | Δ(A) = Δ(B) = 1 | 1, 1 |

Same Shapley values, opposite coordination structure. So agent-level credit can be right while coalition-level attribution is wrong. This is the motivation for H3.

### 2.4 Failure modes we measure
- **Leakage:** a bystander receives credit.
- **Missed credit:** an early enabler receives too little credit because its action was long before the reward.
- **Coalition confusion:** the method believes the wrong group created the value.

### 2.5 Methods under test and what "credit" means for each

| Method | Family | Credit read-out (episode sum of per-step) | Compared with |
|---|---|---|---|
| VDN | value decomposition (Q_tot = Σ Qᵢ) | Q_tot(s,a) − Q_tot(s, aᵢ→noop, a₋ᵢ) | leave-one-out |
| QMIX | value decomposition with monotone mixer using the global state | same as VDN | leave-one-out |
| COMA | actor-critic, counterfactual baseline | Q(s,a) − expected Q under the behaviour policy for agent i's action | leave-one-out |
| MAPPO | actor-critic, shared advantage | the shared advantage (same for every agent) | reference only |
| STAS | return decomposition (spatial-temporal) | Σₜ r̂ᵢ,ₜ, the reward the model assigns to agent i | Shapley |
| NA²Q | coalition-aware value decomposition (pair terms, attention) | Shapley value of its mixer game; dividends of the mixer game | Shapley; dividends |

**Rule behind this table:** a method is compared only with the ground-truth quantity its read-out actually estimates. Raw per-agent utilities are not used as credit, because in VDN adding a constant to one Qᵢ and subtracting it from another leaves Q_tot unchanged, so utilities have no well-defined shares.

### 2.5b Why MAPPO is only a reference
Its advantage is shared: every agent gets the same number. After normalising to shares, each agent gets ±1/n. So leakage equals b/n by arithmetic (measured within 0.006), and L1 hits exactly 2.0 when the episode-summed advantage is negative. These numbers say nothing about credit assignment. The meaningful MAPPO quantity is the **policy-gradient signal-to-noise ratio (SNR)**, per agent.

### 2.6 Metrics, with scales
- **L1 credit error** = Σᵢ |c̃ᵢ − g̃ᵢ| between normalised credit shares and the target. Range 0 (exact) to 2 (disjoint support).
- **Spearman:** rank correlation of credit and target.
- **Leakage Λ** = Σ over bystanders |c̃ᵢ|. Ideal 0.
- **Missed enabler credit M** = max(0, 1 − Σ_enablers c̃ / Σ_enablers g̃). Ideal 0. It can reach above 1 when the enabler receives *negative* credit.
- **Coalition recovery:** Jaccard recall/precision of coalitions whose estimated dividend is large (>10% of total absolute dividend) against the true coalitions.
- **Dividend error:** normalised distance between estimated and true dividends. Max 2.
- **Interaction mass:** share of absolute dividend on coalitions of 2+ agents. True value 1.0 on our H3 layouts. Added because dividend error saturated.
- **Return–faithfulness:** Spearman between return and (1 − L1) within each method. This is for H4 and has not been run yet.
- **Uniform baseline:** give 1/n to everyone, then score L1. Added post-hoc in H3.

### 2.6 Two evaluation modes
- **Mode P (policy evaluation):** a *scripted* team that completes the events generates fixed data. Only each method's critic/mixer/return model is trained on it. Purpose: isolate *credit estimation* from *learning*. No degenerate zero-reward episodes, exact ground truth.
- **Mode T (full training):** each method trains its own policies. Purpose: test whether the findings survive learning, and test H4 (return versus faithfulness). **Not done yet.**

---

## Part 3. The environment and ground-truth machinery

### 3.1 Planted Coalition Gridworld
- Grid with optional walls. Agents can physically block each other (collisions). This matters: it is why "no-op" and "true removal" differ.
- Actions: noop, up, down, left, right, interact.
- **Events.** Each event k has enablers (each pulls its own lever), finishers (must interact together in the same step on their own goal cells while the gate is open), reward r_k, and a partial flag.
  - All-or-nothing: completes only if all finishers act together.
  - Partial synergy: completes if at least one finisher acts; pays r_k × (participating finishers / all finishers).
- **Keyed stations:** only the designated agent can use its lever or goal. That is what makes the coalition planted *by identity*.
- **Gate timing:** a gate opened at step t is usable from t+1, so enablers always act strictly before finishers.
- **Delay d:** an event completing at step t pays at t + d.
- **Bystanders:** can move and block, never activate anything. **Overlap:** an agent can belong to several events.
- **True removal:** `reset(present=mask)` takes absent agents off the grid. Observations keep a fixed size, with a presence flag set to 0.
- **Determinism:** no randomness after reset, so replaying the same action sequences gives identical trajectories.

### 3.2 Observations vs the critic's state
- **Agent observation** is fixed size (4n + 7K + 4). It does **not** show the payout countdown, so agents stay partially observant and the temporal-credit difficulty is preserved.
- **Global state** (for centralised critics) includes the payout-pending flag and countdown. This was added after pilot 1 (see Part 4).

### 3.3 Scripted team
Privileged scripts that read the environment directly. Enablers wait until `enabler_start`, then pull. Finishers walk to goals and interact together once the gate is open and `finish_not_before` has passed. Bystander modes:
- `wander`: random, but never onto a coalition member's planned route (default; keeps realised roles equal to planted roles).
- `roam`: random anywhere (can interfere).
- `idle`: never moves.
Exploration noise: with probability ε a coalition member takes a random action (we used ε = 0.2). **Each agent has its own random stream** seeded by (seed, agent), consumed every step whether present or not. So removing an agent changes nothing in the others' randomness; the removal is the only change.

### 3.4 Replay ground truth
- **Open-loop (default):** record the full team's actions once, replay them for every subset S with the others removed.
- **Closed-loop (sensitivity check):** re-run the scripted team for each subset, so remaining agents can react.
- Also computed per event, since total return = Σ per-event payouts.
- Null policies: `remove` (default), `noop`, `random`.
- Cost: about 60 ms/episode at n = 5.

### 3.5 Planted vs realised
Physical interference (a noisy agent blocking someone) means the *realised* coalition can differ from the *planted* one. Rule: **always grade against the realised ground truth**, and report how often realised = planted.

---

## Part 4. What was done, in order, and what was found

### Steps 1–4: foundations (all done, unit-tested)
1. Ground-truth calculator reproduces the running example and the G4 games, and agrees with brute-force on random games.
2. Environment built.
3. Scripted team and replay. 200-seed check: the full team always succeeds. Realised = planted in 83–100% of seeds depending on noise and roaming. Deviations are due to physical interference.
4. Metrics module.

### Step 5: EPyMARL integration
EPyMARL (unmodified, commit `cbc38c0`) is cloned into `external/`. We wrote an adapter, a launcher, a bridge converting our episodes to EPyMARL batches (for Mode P), and read-outs for VDN/QMIX/COMA/MAPPO. Smoke test: all four train online. With 3,000 steps the return is 0, as expected, because random exploration rarely coordinates a lever pull and a joint push.

### Step 6: Mode P

**Protocol (per data cell = delay, bystanders, seed):**
- 200 scripted training episodes (ε = 0.2), 30 disjoint test episodes.
- Each critic trained for 1,500 updates with minibatches of 32 episodes. **Actors are never trained.**
- Per-step read-out summed over the episode, scored against ground truth.

**Pilot 1 found two method problems:**
1. *The critic's state was not Markov under delay.* The state showed an event had completed but not how long ago, so the payout step was unpredictable. Critic loss at d = 25 was 1–30 against ~0.1 at d = 0. **Fix:** put the payout countdown in the global state only. **Result:** loss converged for QMIX, COMA, MAPPO. **VDN is unchanged** because it only sees agent observations, never the global state.
2. *Per-step counterfactual read-outs are not removal credit.* COMA converged at d = 0 yet gave near-random credit. Diagnosis: the scripted team recovers from a one-step deviation (if A doesn't pull now, it pulls next step), so a single action's true value is near zero, even for the agent who matters most. **Fix:** two-level grading.

**Two-level grading.**
- Build a *per-step counterfactual truth*: for each step t and agent i, replace i's action with noop at that step only, let the scripted team continue (closed loop), and record the change in discounted return (γ = 0.99).
- Level 1, **fidelity:** read-out vs per-step truth. Is the critic right about what it estimates?
- Level 2, **semantic gap:** per-step truth vs removal credit (LOO/Shapley). Even with a perfect critic, is this the right notion of credit?
- **End to end:** read-out vs removal credit.

**Why discounting matters:** under γ, a one-step deviation costs the delay it causes. A slack enabler (acts before the finishers are ready) has a per-step effect of exactly 0. A late enabler on the critical path does not.

**Pilot 2 key result:** QMIX at d = 0 had end-to-end L1 ≈ 0.52, closer to removal credit than a *perfect* per-step critic (≈ 1.16), while matching its own per-step target at r ≈ 0.05. It looked faithful for the wrong reason.

**Pre-registration.** Thresholds were fixed after pilot 2 and before the main grid. **Amendment (disclose this proactively):** with 5 seeds the smallest possible one-sided Wilcoxon p is 1/32 = 0.031, and after Holm correction over ≥2 tests no result could ever be significant. So the primary test was changed to a one-sided *paired t-test* on per-seed means (Holm-corrected), with Wilcoxon still reported. The amendment was written after launching the main grid but before looking at any result, and is committed in `docs/preregistration_amendment.md`.

**Main grid:** d ∈ {0, 10, 25, 50} × b ∈ {0, 2, 4} × {slack, critical enabler} × 5 seeds = 120 cells, about 20 CPU-hours. Horizon 80 at d = 50 so payouts are not clipped.

### Main-grid verdicts, with the mechanism for each

| Hypothesis | Verdict | Mechanism / honest reading |
|---|---|---|
| **H0** ground truth protocol-stable (remove-vs-noop share L1 ≤ 0.15 and realised = planted ≥ 0.6) | ✗ narrowly: 104/120 pass | The 16 failures are 4 data configurations (all b = 4) repeated over 4 delays. Worst L1 0.18, worst realised = planted 0.53. Cause: physical interference. Grading uses realised truth, so grading is still valid. |
| **H1a** missed enabler credit grows with delay (ΔM ≥ 0.2, critic converged at both delays) | ✗ | VDN's critic doesn't converge under delay (gate excludes it). QMIX has the right sign in all conditions (best ΔM +0.49), but Holm p = 0.074 with 5 seeds. COMA +0.02…+0.16. |
| **H1b** slack, not delay, removes per-step enabler credit | ✗ as written; slack half ✓ 24/24 | Slack cells: level-2 missed 0.96–0.99, identical at every delay. Critical half failed because only 54% of "critical" episodes were really critical. |
| **H2** leakage grows with bystanders | COMA ✓ 8/8; VDN 1/8; QMIX 1/8; MAPPO 8/8 is an artifact | COMA ΔΛ +0.12 to +0.21, Holm p ≤ 0.018. VDN/QMIX: direction right in 16/16 conditions but underpowered. |
| **H2′** COMA leaks less than VDN/QMIX at d = 0 | ✗ reversed, 0/8 | At b = 2, COMA 0.42, VDN 0.14, QMIX 0.07. COMA's per-agent signal is noisy (level-1 r 0.04–0.32), and noise spreads credit evenly onto bystanders. |
| **H5** accuracy ≠ fidelity | ✓ | 30 of the 32 best end-to-end cells have level-1 step r < 0.3. Level-1 r ≤ 0.36 for every method. |

**Post-hoc analyses (labelled as such):**
- *H1b split by realised criticality.* For the noop notion: critical episodes missed 0.11–0.12; non-critical 1.31–1.35 (late-start) or 1.00 (slack). So the mechanism holds, and the pre-registered failure was the weak manipulation. For the COMA-style baseline notion: even in critical episodes the mean is 0.52–0.56 (median 0), so a minority of critical episodes give A strongly negative credit.
- *VDN/QMIX leakage direction.* ΔΛ > 0 in 16/16 conditions and 77/80 seed pairs. Sign test p = 1.5 × 10⁻⁵, indicative only because conditions share data. Reading: H2 for VDN/QMIX looks like a power failure, not a null.
- *MAPPO excluded from H1a/H2.* Follows from the read-out's definition, but decided after seeing data, so labelled post-hoc.

### Step 7: porting NA²Q and STAS
Both are **clean reimplementations** (the official repos have no licence, so no code was copied). Official code was compared line by line but **not executed**.

- **NA²Q:** agents with a shared GRU, a VAE on agent states, and a mixer Q_tot = Σ_k w_k f_k(Q_{S_k}) + v(s), with k over every agent and every pair. Each f_k is a monotone MLP. w is a softmax attention.
  - *Deviation:* in Mode P we use a **SARSA target** because the official max target diverges on fixed data (Q_tot 97–143 versus a true ≈ 8). Same network with SARSA converges (≈ 8.3). Consequence: NA²Q estimates the behaviour policy's Q, like COMA, not Q* like VDN/QMIX. The official target is kept for Mode T.
- **STAS (STAS_ML):** transformer with temporal self-attention and "Shapley attention" over random coalitions; trained to make Σ r̂ᵢ,ₜ match the episode return.
  - *Known effect:* it spreads credit almost uniformly (≈ 0.2 to everyone when truth is ≈ 0.32/0.32/0.32/0.03/0.03). Loss fell six-fold with 6× budget (9.2 → 1.2) and the split did not change. On a synthetic task where agent 0 deserves 100%, STAS gives it 25–27%. Interpretation: identifiability limit, because the loss constrains only the total and nothing pushes an uninvolved agent's output to zero.
  - Eval-mode sums overshoot the return by 20–70% (dropout train/eval mismatch inherited from the original).
- **Mixer-game coalition read-out.** For VDN, QMIX, NA²Q: g_t(S) = Q_tot(a_S, noop_{−S}) − Q_tot(noop_all) at each step; take exact Harsanyi dividends and sum over the episode.
  - VDN's game has only singleton dividends (by construction); NA²Q's has none above order 2. Both are unit-tested.

### H3 grid (30 cells, six methods, 87 min on 8 workers)
Layouts: overlap ({A,B,C}=6, {B,D}=4) and partial synergy ({A,B}={A,C}=10, {A,B,C}=−10); d ∈ {0, 25, 50}; 5 seeds.

- **H3 pre-registered:** ✗ on the literal rule (0 of 6 best-third-L1 pairs fall in the worst third of dividend error) but **uninformative**. Dividend error is ≈ 2.0 (maximum) everywhere, range 1.9993–2.0000, so any ranking is a ranking of noise. NA²Q's recovery cannot fall with d because it is zero at every d.
- **Interaction mass** 0.000–0.002 for VDN/QMIX/NA²Q against a true 1.0. **Why:** the mixer game swaps actions within one step, but the planted coalitions act across steps (A pulls, then B/C finish later). A per-step game is therefore additive, and summing additive games stays additive. The mixers are *not* too weak: QMIX can represent any monotone interaction and NA²Q has pair terms. The limit is the read-out.
- **Post-hoc uniform baseline:** L1 0.45 (overlap) and 0.89 (partial). No method's mean beats it in any of 36 conditions. STAS comes closest (0.52) because its near-uniform split matches the overlap truth (0.2 for three of five agents) by coincidence.
- **Post-hoc STAS delay invariance:** d = 0 and d = 25 agree to within 4e-6 per episode. STAS sees only observations, actions and total return; none change with delay inside the horizon. So STAS being "robust to delay" is trivial.
- **H0 for the partial layout:** realised = planted only 40–50%, test return 8.2–8.8 instead of 10. Weak data.

---

## Part 5. Numbers cheat sheet

| Fact | Value |
|---|---|
| Main grid | 120 cells = 4 delays × 3 bystander counts × 2 enabler settings × 5 seeds; ~20 CPU-hours |
| H3 grid | 30 cells = 2 layouts × 3 delays × 5 seeds; 6 methods; 87 min on 8 workers |
| Training per cell | 200 scripted episodes (ε = 0.2), 1,500 updates, batch 32; 30 test episodes |
| Discount | γ = 0.99 |
| Ground truth cost | ~60 ms/episode (n = 5); per-step truth ~7 s/episode (n = 5), 13 s (n = 7) |
| L1 scale | 0 to 2 |
| H0 | 104/120 pass; failures all b = 4; worst realised = planted 0.53 |
| H1a | QMIX best ΔM +0.49, Holm p = 0.074; VDN not converged (loss ≈ 1.0 vs 0.05) |
| H1b slack | level-2 missed 0.96–0.99 at every delay (24/24) |
| H1b critical episodes | only 54% (range 37–63%) truly critical; noop missed 0.11–0.12 once conditioned |
| H2 | COMA 8/8 (ΔΛ +0.12…+0.21); VDN 1/8; QMIX 1/8 |
| H2′ | 0/8; at b = 2: COMA 0.42, VDN 0.14, QMIX 0.07 |
| H5 | 30 of 32 best cells have step r < 0.3; level-1 r ≤ 0.36 for all |
| Pilot-2 QMIX | end-to-end L1 0.52 vs perfect-critic 1.16, own-target r ≈ 0.05 |
| MAPPO | Λ = b/n (0.400 at b = 2, 0.571 at b = 4); L1 = 2.000 at d ≤ 25 |
| MAPPO SNR | coalition 0.04–0.07 vs bystander 0.03–0.04 at d ≤ 10; ≈ 3× at d = 50 |
| NA²Q max target | Q_tot 97–143 vs true ≈ 8; SARSA fixes it (≈ 8.3) |
| STAS | ≈ 0.2 to every agent; synthetic single-agent task: 25–27% to the deserving agent |
| H3 interaction mass | 0.000–0.002 vs true 1.0; dividend error 1.9993–2.0000 |
| Uniform baseline | L1 0.45 overlap, 0.89 partial; beaten by 0 of 36 method means |
| Sign test (VDN/QMIX leakage) | 16/16 conditions, 77/80 seed pairs, p = 1.5e-5 (indicative) |
| Level-1 fidelity | r ≤ 0.36 everywhere; 0–0.2 in pilots |

---

## Part 6. Likely cross-questions, with answers

### About the idea and novelty

**Q1. What is new here? Hasn't credit assignment been studied for years?**
Methods have been studied extensively, but evaluated by return or win rate, or by qualitative plots. Matrix-game papers (QTRAN, WQMIX, QPLEX) compare learned joint values with a known payoff, but they are one-step, with no delay, and they check the joint value, not who gets credit. We could not find a benchmark that grades credit against a planted, exactly computable truth at both coalition and individual level under delay. "Could not find" means not found in our search, not proven absent.

**Q2. Why a synthetic gridworld? Why not SMAC?**
Ground truth needs removal-and-replay in a deterministic environment. SMAC cannot cheaply remove an agent and replay exactly. A minimal environment also lets a failure be traced to one knob (delay, bystanders, overlap, synergy). The cost is generalisation, which we state as a limitation.

**Q3. Why is Mode P a fair way to evaluate methods that are supposed to learn policies?**
Because it asks a narrower question: *given perfect data, does the method's credit signal give the right answer?* If a method cannot do that, learning end to end cannot help. The cost is that it evaluates critics, not whole algorithms; Mode T (not done) addresses that.

### About ground truth

**Q4. What is ground truth and how do you know it is correct?**
v(S) from deterministic replay with agents removed, for all 2ⁿ subsets. Dividends, Shapley and LOO are computed from v(S) by exact formulas. We have unit tests: the running example (Δ = 10, φ = 10/3), the two G4 games, and agreement with brute-force formulas on random games.

**Q5. Ground truth depends on a counterfactual choice. Isn't that arbitrary?**
Yes, which is why we declare it and test sensitivity. Default is true removal with open-loop replay. Sensitivity checks are no-op and closed-loop replay. H0 tests that the protocol choice (remove vs noop) matters little: mean share L1 ≤ 0.15 in 104/120 cells. We do not claim to settle the philosophical question of which counterfactual is right.

**Q6. Why open-loop replay by default?**
So we do not credit an agent merely because the *other* agents behave erratically when it disappears, which they never saw in training. Closed-loop is the sensitivity check (Step 9, not yet run).

**Q7. What is the difference between Shapley and LOO, and why use both?**
LOO asks what is lost if this agent alone is removed. Shapley averages marginal contributions over all orderings, equivalent to splitting each coalition's dividend equally among members. Counterfactual read-outs (COMA, Q_tot difference) estimate LOO-like quantities, so LOO is their matched target. STAS splits with Shapley values, so Shapley is its matched target. In our single-event layout they coincide in shares (⅓ each).

**Q8. Planted coalitions don't always match realised ones. Doesn't that break the benchmark?**
It breaks the *planted labels*, not the *grading*. We grade against the realised ground truth from exact replay and report how often realised = planted (83–100% in the main data; 40–50% for the partial layout, which we call weak data). Leakage uses realised bystanders.

### About the pilots and method changes

**Q9. You changed the method after the pilot. Isn't that p-hacking?**
The changes were fixes to *measurement validity*, made before any hypothesis thresholds were fixed. Pilot 1 showed the d > 0 numbers mixed "delay is hard" with "the critic cannot fit the reward" (non-Markov state). We stated that explicitly and refused to fix thresholds until both fixes were in. Thresholds were fixed after pilot 2 and before the main grid.

**Q10. You amended the pre-registration. Why, and when?**
Because with 5 seeds the smallest possible Wilcoxon p is 0.031 and Holm correction makes significance impossible whatever the data. We switched the primary test to a paired t-test and kept Wilcoxon alongside. The amendment was written after the main grid was launched but before any result was inspected, and it is committed with a timestamp. Effect-size thresholds were not changed.

**Q11. What is "non-Markov under delay" in plain words?**
If a reward arrives 25 steps after an event and the state does not say how long ago the event happened, then the same state can lead to a payout now or in 10 steps. A feedforward critic cannot predict which, so it has an irreducible error. Adding a countdown to the state fixes it.

**Q12. Why does VDN not benefit from the fix?**
VDN's Q_tot = Σ Qᵢ(oᵢ, aᵢ) uses only agent observations. The countdown is in the global state, which VDN never reads. This is a genuine property: a purely decentralised value decomposition cannot represent when a delayed team reward will arrive unless agents can observe it. VDN's delay results therefore reflect a training failure, and the H1a convergence gate excludes it for that reason.

**Q13. Is putting the countdown in the critic's state cheating?**
It is a deliberate choice: the critic is centralised and allowed global state. Agent observations do not show it, so the agents' problem is still partially observable and the early-action-to-late-reward difficulty is intact.

### About the central finding (semantic gap)

**Q14. Explain "semantic gap" like I'm new.**
COMA and Q_tot-difference read-outs answer "what would a one-step deviation cost right now?". "Who earned the reward?" is the removal question. They agree only if deviations cannot be recovered from. In our environment the team recovers: if A skips the lever now, it pulls next step. So the true one-step value of A's action is about 0, even though removing A kills the reward. Even a perfect critic then gives A zero credit. That is the gap, and it comes from slack, not delay.

**Q15. But this only happens because your scripted team is recoverable. Is that realistic?**
Partly it is the point: any robust team recovers from small deviations. We make no claim that this holds for every environment. Our claim is narrower: per-step counterfactual credit conflates "essential to the outcome" with "irreplaceable at this exact step". The H1b split shows the mechanism directly: when A genuinely is the bottleneck the noop notion gives it nearly full credit (missed 0.12), when A has slack it gives about zero.

**Q16. Your H1b failed. Why report it as a mechanism result?**
The pre-registered rule failed on its critical half, honestly reported as ✗. The cause was a weak manipulation: only 54% of "critical" episodes were critical, so the cell average mixes critical and non-critical. The post-hoc split by realised criticality shows the mechanism holds for the noop notion. We label that split post-hoc and hypothesis-generating. The confirmatory finding is the slack half (24/24).

**Q17. Why does the COMA-style notion under-credit even a critical enabler?**
Median 0 but mean 0.54: a minority of critical episodes give A strongly negative credit. Our *likely* explanation, not verified step by step, is that under the behaviour policy A's pull is the likely action so its advantage is near zero, and A's movement steps, which noise perturbs, dominate with small negative advantages. Say "likely" and "not yet verified".

**Q18. What does "QMIX looks good for the wrong reason" mean?**
At d = 0, QMIX's end-to-end L1 is about 0.52, better than a perfect per-step critic would score (≈ 1.16), yet its correlation with its own per-step target is ≈ 0.05. Its errors happen to push credit towards A. Without two-level grading it would look like a faithful credit assigner. This is the evidence for H5 (accuracy does not imply fidelity).

### About specific methods

**Q19. Why does COMA leak more than VDN/QMIX when COMA's baseline is meant to help?**
COMA's per-agent advantage is noisy (level-1 r 0.04–0.32). After normalising to shares, noise spreads |credit| roughly evenly, and bystanders pick up their share. The counterfactual baseline reduces variance in theory, but here it does not remove bystander credit. H2′ predicted the opposite and was reversed in 8/8 comparisons.

**Q20. Why is MAPPO excluded?**
Its shared advantage is identical for all agents, so normalised shares are ±1/n and leakage is b/n by arithmetic (measured within 0.006). Its H2 8/8 is therefore meaningless. We exclude it from H1a/H2 conclusions and report only its policy-gradient SNR. We state openly that this exclusion was decided after seeing the data, though it follows from the read-out's definition.

**Q21. Why does STAS give uniform credit? Is your port wrong?**
We believe it is an identifiability limit. The loss only constrains the *total* of the decomposed rewards, and each agent's reward comes from attention over a random coalition, so nothing forces an uninvolved agent's output to zero. Evidence: six times more training cut the loss six-fold without changing the split, and on a synthetic task where agent 0 deserves 100%, STAS gives it 25–27%. We compared our code with the official code line by line but did **not execute the official code**, so we cannot completely exclude a porting difference.

**Q22. Why does your NA²Q use a different target than the paper?**
On fixed scripted data, the official max target diverged (Q_tot 97–143 versus a true ≈ 8): the max picks actions the data never takes and NA²Q's monotone pair terms amplify the overestimates. With a SARSA target the same network converges. So in Mode P NA²Q estimates the behaviour policy's value, like COMA, not Q* like VDN/QMIX. The official max target is kept for Mode T. This is a documented deviation.

**Q23. VDN/QMIX learn Q*, COMA/MAPPO learn the behaviour value. Isn't that an unfair comparison?**
It is a caveat we state. The scripted policy is near-optimal, so on its own path the two are close but not identical. We report it as a limitation rather than hide it.

**Q24. Why use the official-code deviations at all? Why not run the official repos?**
Neither STAS nor NA²Q repo has a licence, so copying code is not safe, and porting the environment into each codebase is the main engineering cost. We reimplemented the same architectures and losses and listed every deviation in `docs/step7_ports.md`.

### About H3

**Q25. H3 failed. What did you learn?**
Literally ✗, but uninformative: dividend error is at its maximum (≈ 2.0, range 1.9993–2.0000) for every method, so ranking by it ranks noise. The real finding is that the mixer-game read-outs for VDN, QMIX and NA²Q see no coalition structure (interaction mass 0.000–0.002 against a true 1.0). The reason is the read-out, not the model: it swaps actions within one step, but our coalitions act across steps. A fair test needs an episode-level read-out, which is the H5 estimator.

**Q26. So you cannot say anything about coalition-level credit?**
Correct, we cannot yet say how well methods recover coalitions in general. We can say these specific per-step read-outs cannot express cross-time coalitions. That is a limitation of the read-out, and the next experiment (H5) is designed to test coalition credit properly.

**Q27. You said "no method beats uniform credit". Is that a fair baseline?**
It is a post-hoc sanity baseline: give 1/n to everyone and score L1 against the same truth. On the overlap layout uniform scores 0.45, on partial 0.89. No method's mean beats it in any of 36 conditions. It means moderate-looking L1 values may carry no information about who earned the reward, so benchmarks should report it next to L1. STAS's 0.52 on overlap is a coincidence of the layout (its near-uniform split happens to match a truth that is 0.2 for three of five agents). We have **not** yet computed it for the main grid.

### About statistics and rigour

**Q28. Five seeds is small. How can you trust these results?**
We do not over-claim. Five seeds with Holm correction over up to 32 conditions give limited power, and we say so. The VDN/QMIX H2 result looks like a power failure: the sign is right in 16/16 conditions, 77/80 seed pairs. The sign test (p = 1.5e-5) is indicative only because conditions share data. The confirmatory results we lean on most (H1b slack half 24/24, H5, H2′ 0/8) are consistent across conditions, not marginal.

**Q29. What is Holm correction and why use it?**
A step-down multiple-comparison correction that controls the family-wise error rate. We correct within each hypothesis over all its conditions (e.g. 32 conditions for H2), which is conservative.

**Q30. What is confirmatory vs post-hoc, and why does it matter?**
Confirmatory results apply a rule fixed before looking at the data. Post-hoc analyses were done after seeing results, so they can only generate hypotheses. We label each. Post-hoc items: H1b split by criticality, VDN/QMIX sign test, MAPPO exclusion, uniform baseline, STAS delay invariance, MAPPO SNR comparison.

### About status, time and next steps

**Q31. What is not done?**
Step 8 (Mode T, full training, H4), Step 9 (sensitivity analyses: no-op vs removal, open vs closed-loop, Shapley vs LOO), and the H5 low-order coalition estimator. Also the uniform baseline for the main grid. Note: the ROADMAP.md checkbox for the parent Step 6 is still unticked although every sub-item under it is ticked, so if asked, Step 6 is complete.

**Q32. What is H5 and why is it next?**
A cheap estimator: fit an episodic return model with coalition terms up to size k ∈ {2, 3}, read out low-order dividends, and allocate each dividend equally to its members, rescaled to sum to the episode return (as in TAR²). It works on whole episodes, so it can express cross-time coalitions, which is exactly what the per-step mixers could not. The required honesty check: also test it on coalitions of size k+1, which it cannot represent, so we are not grading it on its own assumptions.

**Q33. What is H4 and what could go wrong?**
H4: within each method, return and (1 − L1) are weakly correlated, so return is a weak proxy for credit quality. Risk: online training may not learn at all under delay (our smoke test returned 0), so we will restrict to pilot-validated configurations and treat H4 as a descriptive pilot.

**Q34. What is your biggest weakness?**
Honest ranking: (1) the coalition-level question is unanswered so far; (2) the critical-enabler manipulation was weak; (3) five seeds; (4) one synthetic environment; (5) NA²Q and STAS are our reimplementations, not executed official code.

**Q35. If I only remember three takeaways?**
1. Per-step counterfactual credit is not "who earned the reward": slack makes it zero, even for a perfect critic.
2. Low end-to-end error can coexist with near-zero fidelity, and no method beat equal credit on the coalition layouts.
3. The benchmark is exact and reproducible, and it exposes these problems because we planted the truth.

---

## Part 7. What not to claim

- Do **not** say methods "fail at coalition credit". Say the per-step mixer read-outs cannot express cross-time coalitions, and H3 was uninformative.
- Do **not** say STAS is "wrong". Say its credit is near-uniform, probably by an identifiability limit, and that we did not execute the official code.
- Do **not** say delay causes missed enabler credit. Say slack does; delay mainly breaks methods that cannot represent it (VDN), and QMIX shows a consistent but non-significant trend.
- Do **not** say MAPPO leaks. Say its leakage is b/n by construction and only its SNR is meaningful.
- Do **not** present post-hoc results as confirmed. Say "post-hoc, hypothesis-generating".
- Do **not** say the VDN/QMIX leakage effect is established. Say direction is consistent, per-condition tests were underpowered, and the sign test is indicative only.
- Do **not** promise Mode T results. It is not done and may only be a descriptive pilot.
- Do **not** claim novelty absolutely. Say "not found in our search".

---

## Part 8. Glossary (one line each)

- **Harsanyi dividend:** the value created by a coalition that no smaller subset creates alone.
- **Shapley value:** fair-share credit; each coalition's dividend split equally among its members.
- **Leave-one-out:** value lost when one agent is removed.
- **Enabler / finisher / bystander:** acts early to open a gate / acts jointly later / contributes nothing.
- **Slack:** the enabler acts earlier than it needs to, so it is not on the critical path.
- **Leakage Λ:** share of credit going to bystanders.
- **Missed credit M:** shortfall of enabler credit relative to its true share.
- **Semantic gap:** difference between a per-step counterfactual notion of credit and removal credit, even for a perfect critic.
- **Fidelity:** how well a read-out matches the per-step quantity it estimates.
- **Mode P / Mode T:** critic-only on scripted data / full training.
- **Open- / closed-loop replay:** replay recorded actions / let the remaining agents re-act.
- **Markov state:** the state contains everything needed to predict the next reward.
- **Mixer game:** a per-step game g_t(S) built from a learned mixer, used to read out coalition dividends.
- **Interaction mass:** share of absolute dividend on coalitions of two or more agents.
- **Holm correction:** multiple-testing correction controlling family-wise error.
- **Uniform baseline:** equal 1/n credit to everyone, scored with the same metric.
- **SNR (MAPPO):** per-agent policy-gradient signal-to-noise ratio, the only meaningful MAPPO credit quantity here.
