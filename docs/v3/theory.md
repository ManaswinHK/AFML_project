# Persistent-ablation credit: definitions, identities and what they imply

v3, 11 Nov 2026. Builds on `docs/assessment_and_plan.md` §3. Every identity below is checked to machine precision on real episodes in `tests/test_ablation.py`. The code is in `src/whoearned/ablation.py`.

## 1. Setting and notation

- n agents, N = {1..n}, horizon T, rewards r_t, undiscounted return G = Σ_t r_t. Episode credit is undiscounted, as in the repo's ground truth.
- **Stop.** A persistent intervention: from step t on, the agents in C take no-op for the rest of the episode.
  - Under the **no-op null** they stay on the grid inert, and teammates are not told.
  - Under the **removal null** they leave the grid, and teammates see them gone.
  - Either way, a lever an agent has already pulled stays pulled. Implemented as `env.stop(C, null)`.
- **Acting flags.** The environment's global state carries one acting flag per agent (Phase 0.2). A stopped agent is therefore just another state, and an ordinary state-value critic can represent it. Agents' own observations are unchanged, so under the no-op null teammates really are not told.
- **Ablation value.** W_S(s) is the expected return-to-go from state s when only the agents in S keep acting:
  - W_N = V, the ordinary value;
  - W_∅(s) is the value already **committed** at s: payouts that arrive even if everyone stops now.
- **Closed-loop coalition game.** v(S) = W_S(s₀), the repo's closed-loop protocol (`ground_truth_closed_loop`).
- **Oracle values.** With the simulator, snapshots and common random numbers (CRN), W_S(s_t) is computed *exactly for this episode's random draws* (`oracle_ablation_values`). These are realised, not expected, counterfactuals.

## 2. Ablation-TD credit (Identity 1)

  κ_C(t) = r_t + W_{N∖C}(s_{t+1}) − W_{N∖C}(s_t),   with W(s_T) = 0.

The sum over t telescopes:

  Σ_t κ_C(t) = G − W_{N∖C}(s₀).

- For C = {i} under the oracle (CRN), the right-hand side is exactly the realised closed-loop leave-one-out credit G − v(N∖i).
- For a learned critic, the right-hand side is G − Ŵ_{−i}(s₀): realised return minus an *expected* counterfactual.
- **Test.** `test_identity1_*`: the identity holds exactly on coalition, overlap and partial layouts, 3 seeds each.

## 3. The commitment split: luck versus contribution (new in v3)

Let δ_t = r_t + V(s_{t+1}) − V(s_t) be the team's TD error, and D_i(s) = V(s) − W_{−i}(s) the team's **dependence** on agent i at s. Then, exactly:

  κ_i(t) = δ_t + c_i(t),   c_i(t) = D_i(s_t) − D_i(s_{t+1}).

*Proof.* δ_t + D_i(s_t) − D_i(s_{t+1}) = r_t + V(s_{t+1}) − V(s_t) + V(s_t) − W_{−i}(s_t) − V(s_{t+1}) + W_{−i}(s_{t+1}) = κ_i(t). ∎

**Reading:**
- δ_t is **shared surprise**. It is identical for every agent, including bystanders, and has mean zero under the team's policy. All of the episode's luck (Σ_t δ_t = G − V(s₀)) is in this term.
- c_i(t) is **structural credit**. It is non-zero only where the team's dependence on i changes, and it sums to V(s₀) − W_{−i}(s₀): *expected* leave-one-out credit.
- So the realised-versus-expected distinction in assessment §5 is a clean additive split, not a limitation to apologise for:
  - an estimator can report c (contribution) and δ (luck) separately;
  - κ is their sum;
  - only a realised counterfactual, such as an oracle or a hindsight model, can attribute luck to individuals.
- **Test.** `test_commitment_split_is_exact`.

## 4. The pending game and per-step semivalues (new in v3)

At each step t define the **pending game**

  u_t(S) = W_S(s_t) − W_∅(s_t),

the value that still needs the agents in S to keep acting.
- u₀ is the closed-loop coalition game (v(∅) = 0).
- u_T = 0: at the end nothing is pending.
- Every linear credit rule φ (Shapley value, Harsanyi dividends, leave-one-out) commutes with differences. Its **step credit** φ(u_t) − φ(u_{t+1}) therefore telescopes:

  Σ_t [φ(u_t) − φ(u_{t+1})] = φ(u₀).

**Consequences** (all tested exactly in `test_pending_game_*` and `test_step_credit_lands_on_commitment_not_payout`):

1. **Per-step Shapley credit sums to the episode's closed-loop Shapley value; per-step dividends sum to its Harsanyi dividends.** The coalition-level truth that H3/H5 grade at the episode level now has a timeline.
2. **Shapley step credit is efficient at every step.** Σ_i [φ_i(u_t) − φ_i(u_{t+1})] = u_t(N) − u_{t+1}(N) = r_t − [W_∅(s_t) − W_∅(s_{t+1})], which is the value **committed** at step t.
   - The committed value is non-zero when an outcome becomes certain (the joint finish), not when its payout arrives.
   - At the payout step, r_t is exactly offset by the drop in W_∅.
   - **Delay invariance therefore holds by construction**: moving the payout later moves no credit. This is the formal version of H1a's negative result: delay is not what makes credit hard.
3. **Coalition timeline.** In the running example (no-op null, deterministic team):
   - at A's pull, the dividend of {A, B, C} (10) is released and a dividend of 10 on {B, C} appears (step credit −10 on {B, C});
   - at the joint finish, {B, C}'s dividend is released;
   - over the episode, {A, B, C} is credited 10 and {B, C} 0, matching the episode dividends;
   - leave-one-out step credit gives A +10 at the pull and B, C +10 at the finish, with nothing anywhere else.

   The Shapley *step* credit can be negative for an agent whose pivotality rises at that step (B and C at the pull: −5/3 each). That is correct bookkeeping, not an error: their remaining share of pending value grew.

## 5. Identity 2: why one-step read-outs fail, and where

(From assessment §3.2; restated because v3's controls V1–V3 test it.)

Under the no-op null, the performance-difference lemma gives

  v(N) − v(N∖i) = E_{τ ~ team without i} Σ_t γᵗ [V(s_t) − Q(s_t, null_i, a_{−i})].

The bracket is what a one-step read-out (Q_tot difference, COMA) estimates. Read-outs sum it along the *factual* trajectory instead of along trajectories where i has stopped. With a team that recovers from any single lapse, the bracket is about (1 − γ)·V on factual states (0.078 in the running example) and large only off-distribution.

## 6. Identifiability: interventions are necessary when teammates react

- Persistent-ablation values are functionals of the counterfactual system in which some agents stopped and **the others may react** (finishers give up, walk elsewhere, or wait).
- Full-team data contain no state in which an agent has stopped. The removal value is then identified only under extra structure. The clearest published case is arXiv 2304.12458 (factored transitions, separable rewards, removed agents marginalised out under their pre-dropout behaviour, **non-reactive** remaining agents), where importance sampling from pre-dropout data suffices.
- When the remaining agents' policy depends on who is present, which is exactly the coalition structure we study, no such reweighting exists. The data must contain stops.
- That is the theoretical reason for P3 (the no-intervention control fails). It is also why STAS's return decomposition on constant-return data cannot work (assessment §2.3 item 4).

## 7. What persistent credit does not give: a better learning signal

(Assessment §3.5, unchanged.) As a reward, κ_i = r_t + Φ_i(s_{t+1}) − Φ_i(s_t) with Φ_i = W_{−i} is potential-based shaping (Ng et al., 1999). It preserves optimal policies and gives no variance reduction over a good baseline. With a single all-or-nothing coalition, leave-one-out credit equals team return for every member on every trajectory, so there is no headroom. Any learning benefit must come from **difference rewards removing other events' noise** (Wolpert & Tumer), which needs a multi-event layout. Phase 3 tests exactly that, with predictions written in advance.

The commitment split adds one observation. The structural part c_i(t) is a pure potential difference (D_i(s_t) − D_i(s_{t+1})), so as a reward on its own it has zero effect on the optimal policy *and* zero learning signal. Everything a learner can use is in the reward-bearing part. So c is an **attribution** tool (who mattered, when, with whom), not a training signal. We state this so nobody tries to train on c.

## 8. Summary of what is new in v3 versus round 2

| Item | Round 2 | v3 |
|---|---|---|
| κ sums to leave-one-out (Identity 1) | shown | kept, tested in-repo |
| κ = shared surprise + structural credit | not stated | **new**, exact |
| Per-step Shapley / dividends with episode-exact sums, efficiency per step = committed value | sketched for dividends at s₀ only | **new**, exact, tested |
| Delay invariance | stated | **derived** from commitment (W_∅) |
| Identifiability boundary (reactive teammates) | asserted | **stated precisely**, with the factored non-reactive case (2304.12458) as the contrast |
