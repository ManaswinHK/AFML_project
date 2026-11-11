# Who Earned the Reward? Honest assessment, new tests, and the plan from here

**Date:** 11 November 2026 · **Repo:** `DBS13062006/AFML_project` at commit `5ddf18f` (7 Nov) · **For:** team Aporia

---

## 0. Bottom line

**Is the work so far useful?** Partly, and not for the reason the one-pager gives.

- **What is good:** the measuring instrument (exact ground truth, two-level grading, pre-registration, honest labels). It is better than what most credit-assignment papers have.
- **What is weak:** of the pre-registered hypotheses, most failed or were uninformative, and the two headline results are less than they look:
  1. "Per-step read-outs miss who earned the reward" is largely a **mathematical identity**, not an empirical discovery (§3).
  2. The H5 estimator (ECE-k) is **matched or beaten by a 25-coefficient Lasso table** that never looks at a trajectory (§4, Test C).
- **What is missing:** nothing yet shows that credit accuracy matters for *learning*. On your main layout it provably cannot (§3.5), so H4 is uninformative, not weakly supported.

**What I found, in one table.** All tests use your environment, seeds, test episodes and metrics.

| # | Test | Result |
|---|---|---|
| A | A per-step credit whose counterfactual **persists** (the agent stops for the rest of the episode, not for one step), computed exactly | Sums to leave-one-out credit with zero error in all 3,600 test episodes of your main grid. Missed enabler credit goes from your 0.99 (slack cells) to 0.00. The enabler is credited at the step it pulls the lever, finishers at the step they finish; bystanders get exactly 0. |
| B | The same credit from a **learned** critic trained on 200 episodes plus 200 "an agent stops here" continuations | Credit error L1 **0.02–0.04** against 0.80–1.14 for an equal split; against your own open-loop truth 0.09–0.20, where stored QMIX has 0.90–1.31. The same critic without the interventions scores 1.0–1.3, worse than an equal split. |
| C | A presence-only Lasso table on the exact data ECE-k_mix saw | Dividend error 0.41 / 0.76 / 0.75 (coalition / overlap / partial) against ECE's 0.43 / 1.10 / 1.08. Same or better on 6 of 6 conditions. |
| D | Downstream check from your stored results: keep the 3 highest-credit agents | The kept team is the planted coalition in 40% (QMIX), 21% (VDN), 9% (COMA) and 4% (STAS) of episodes with 2 bystanders. Chance is 10%. With the learned critic of Test B: 100%. |
| E | Roles shuffled every episode | The table collapses to an equal split (L1 0.79 vs 0.80). The learned critic still works but needs more data (0.41 at 200 interventions, 0.26 at 800). This is where a learned model earns its place, and where the open problem is. |

**What I recommend.**

1. Reframe the project as **credit by intervention**: a one-step counterfactual cannot identify who earned the reward; a persistent one can, and it is learnable with an ordinary value critic once the data contains agents stopping.
2. Fix the record first (Phase 0): the ECE claim, the STAS claim, the partial-layout state, the H4 reading.
3. Repeat Tests A–E inside the repo on fresh seeds under a new pre-registration (Phases 1–2). That is your positive result.
4. Before any more Mode T, run the headroom gate (Phase 3). It decides whether there is a learning story at all.

I do not expect the new credit, used as a reward, to speed up learning. §3.5 gives the reason. I would rather tell you now than after two weeks of training runs.

---

## 1. What I did and did not do

**Read:** the one-pager, `who_earned_the_reward_problem_statement_v2.md`, `ROADMAP.md`, every file in `docs/`, the source in `src/whoearned/` (environment, scripted team, replay, ground truth, metrics, experiments, coalition estimator), and the stored JSONs in `results/mode_p/`.

**Ran:** the 131 unit tests that do not need EPyMARL (all pass), and the tests below. My regenerated test episodes reproduce your stored leave-one-out truth in the cell I checked (5 of 5 episodes), and the equal-split L1 computed from them matches your tables (0.73 at b = 2).

**Did not do:**
- I did not re-run VDN, QMIX, COMA, MAPPO, NA²Q, STAS or ECE (EPyMARL is not in my sandbox). Every number for those methods is read from your stored JSONs for the same cells and test episodes.
- I did not read problem statement v1.
- Papers I only saw as abstracts or snippets are listed in §10.

**Two rounds of experiments.** There are two bundles in the attachment:
- `atd/` is this round: commit `5ddf18f`, PyTorch critic, no-op and removal nulls. All numbers in §4 are from it unless marked.
- `rcc_prototype/` is an earlier pass on the older commit `5943480` with a scikit-learn critic. Where it overlaps with this round the numbers agree (learned-critic L1 0.09–0.26 against open-loop truth in both). I cite a few of its results below, marked **[round 1]**. I did not re-run those in this round.

> **In this repo:** `atd/` is at `scripts/atd/` and `rcc_prototype/` is at `scripts/rcc_prototype_round1/` (force-added, because `.gitignore` excludes `results/` folders).

**Everything here is exploratory:** same seeds as your runs, analysis choices made while looking at data. §9 is the pre-registration for a confirmatory repeat.

---

## 2. Assessment of the work so far

### 2.1 What is solid

- **The instrument.** Deterministic environment, exact replay for all 2ⁿ coalitions, per-event truth, realised-versus-planted reporting, 190 tests.
- **Two-level grading.** Separating "is the critic right about its own target" from "is that target the right notion" is the best idea in the project.
- **The discipline.** Pre-registration, amendments before looking, post-hoc work labelled as such.
- **The uniform baseline and the closed-loop sensitivity check.** Both changed what the results mean, and you reported that.

### 2.2 Scorecard

| Hypothesis | Your verdict | My reading |
|---|---|---|
| H0 benchmark is measurable | fails open loop, passes closed loop | The open-loop truth is the weaker protocol. Make closed loop the default (§4, Test C shows it also inflates dividend errors). |
| H1a missed credit grows with delay | not supported | Correct, and the premise was wrong: delay is not the cause. |
| H1b slack removes per-step credit | slack half holds | This is the real finding. It is an identity (§3.2). |
| H2 / H2′ leakage | COMA only; reversed | Explained by an identity: a perfect COMA critic gives every agent zero expected credit, so what you measure is exploration luck. **[round 1]** numeric check: −0.02, −0.01, +0.07, 0.00, 0.00. |
| H3 individual hides coalition errors | uninformative | Agreed. |
| H4 return is a weak proxy | "mildly supportive" | **Uninformative.** With one all-or-nothing coalition, exact credit equals team return for every member on every trajectory (§3.5). There was never anything for return to hide. |
| H5 ECE-k beats NA²Q and uniform | supported (dividends, L1) | True as stated, but a Lasso table does the same or better (Test C). The claim should be about the **data**, not the estimator. |
| H-fid accuracy without fidelity | supported | True. Most of those "best" cells are still worse than an equal split. |

### 2.3 Problems to fix or state

| # | Issue | Evidence | Action |
|---|---|---|---|
| 1 | The read-outs are your construction. No method claimed to output removal credit. | v2 §2.4 | Claim "no standard one-step read-out recovers removal credit", not "the methods assign credit wrongly". |
| 2 | ECE-k_mix is a regression on interventions. The GRU over actions does no identifiable work on fixed layouts. | Test C | Add the table baseline to `docs/h5_results.md`; reword H5. |
| 3 | Much of the residual dividend error on the single-coalition layout comes from the open-loop truth, not the estimator. | The table, which matches ECE there, scores 0.41 against open-loop truth and 0.16 against closed-loop truth with the same dividends | Report both truths. |
| 4 | The STAS conclusion does not follow. Every training episode returns exactly 10, so any return-decomposition model has a zero-loss solution that pays everyone equally. | `train_return_mean` = 10.0; **[round 1]** | Say "our reimplementation, on constant-return data". |
| 5 | The partial-layout state is not Markov under delay. A partial event pays 5 or 10; the state shows "pending" and the countdown but not the amount. | `gridworld.py::get_state` at `5ddf18f`. **[round 1]**: adding the amount cut value RMSE at d = 25 from 1.71 to 0.88. | Phase 0.1; re-run the stored partial cells at d > 0. |
| 6 | The scripted finishers abandon an event when its enabler is absent, even if the lever is already pulled. | `scripted.py::_finish_feasible`; Test A (credit moves from the pull to the finish under the removal null) | Phase 0.3. |
| 7 | Pilot seeds are inside the confirmatory grid. | `scripts/run_mode_p.py`: pilot seeds (0, 1), main seeds (0–4) | Use seeds 5–9 next. |
| 8 | Mode P is one regime: a near-optimal scripted team that recovers from any one-step lapse. That is exactly where one-step advantages vanish. | §3.2 | Say so; Phase 4 tests trained policies. |
| 9 | Low critic fidelity (r ≤ 0.36) is confounded with coverage. The data never shows a missing contribution. | §3.3 | Controls V1–V2 in Phase 1. |

### 2.4 Against the one-pager

| Promise | Status |
|---|---|
| Stage 1: credit error grows with delay and turnover while return looks good | Not shown. What you found instead is better supported: slack, not delay. |
| Stage 2: a mechanism that lowers credit error at matched compute, keeps return, is far cheaper than exact Shapley | ECE meets the first part only by being a table. Test B is a stronger candidate. "Keeps return" is untested. |

---

## 3. The idea: one-step versus persistent ablation

### 3.1 Definitions

- π: the team's joint policy. Null: what an agent does when it "stops". I use two:
  - **no-op null:** it stays on the grid and does nothing; teammates are not told;
  - **removal null:** it leaves the grid (your `reset(present=mask)`), and teammates see it gone.
- **W_S(s):** the expected return-to-go from state s when only the agents in S keep acting from now on and the rest stop for good. I call this the **ablation value**. W_N = V, the ordinary value.
- v(S) = W_S(s₀): your coalition value (closed loop).
- **Ablation-TD credit** of agent i at step t, with −i meaning "everyone but i":

  κ(i, t) = γᵗ · [ r_t + γ·W₋ᵢ(s_{t+1}) − W₋ᵢ(s_t) ]

### 3.2 Two identities

**Identity 1 (factual path).** For any S and any trajectory, the terms telescope:

  Σ_t γᵗ [ r_t + γ·W_S(s_{t+1}) − W_S(s_t) ] = G − W_S(s₀)

where G is the discounted return. In expectation the right side is v(N) − v(S). With S = N∖{i}: **the ablation-TD credits of agent i sum to its leave-one-out removal credit**, and they are computed on the trajectory that actually happened.

**Identity 2 (ablated path).** By the performance difference lemma (Kakade & Langford, 2002), under the no-op null:

  v(N) − v(N∖i) = E over trajectories *of the team without i* of Σ_t γᵗ [ V(s_t) − Q(s_t, null_i, a₋ᵢ) ]

The bracket is exactly what a Q_tot-difference or COMA-style read-out estimates. So **your one-step read-outs are the right integrand evaluated on the wrong trajectories.** Summed along the trajectories where agent i has stopped, they give removal credit. Summed along the factual trajectory, as every read-out does, they give something else.

**Consequence (the semantic gap, exactly).** The gap between removal credit and one-step credit is a difference of state distributions:

  m_i − c_i^{one-step} = Σ_t γᵗ ( E_{s ~ team without i} − E_{s ~ full team} ) [ g_i(s) ],  g_i(s) = E_a[ Q(s, a) − Q(s, null_i, a₋ᵢ) ]

If the team recovers from any single lapse, g_i is about (1 − γ)·V on every state the full team visits, and large only on states it never visits: the lever still unpulled with the deadline here. That is your H1b slack result, derived instead of measured.

### 3.3 Why the persistent target is also easier to learn

- **Scale.** A one-step effect is about (1 − γ)·V, here 0.08 on a value of 8. A critic needs error well under 1% to see it. A persistent effect is as large as V itself. This is the known action-gap problem (Baird, 1994; Tallec et al., 2019) applied to credit read-outs.
- **Coverage.** A one-step read-out queries Q at actions the data rarely contains. The ablation value is queried at states that the intervention data contains by design.

Observed: your critics reach r ≤ 0.36 on their own targets; the ablation critic reaches 0.90–0.998 on its own (Test B).

### 3.4 Properties (each checked in Test A)

| Property | Statement |
|---|---|
| Adds up | Σ_t κ(i, t) = G − W₋ᵢ(s₀) on every trajectory. |
| Null agent | If the team never needs i (W₋ᵢ = V), its credit is zero in expectation at every step. In a deterministic episode it is exactly zero. |
| Delay-invariant | Once a payout is certain it is in W_S for every S. Moving the payout later changes κ only through γ. |
| Localised | κ(i, t) is non-zero only at transitions where the team's dependence on i changes. |
| Coalitions | The Harsanyi dividends of the game S ↦ W_S(s₀) are the closed-loop episode dividends. Along a trajectory, dependence moves down the lattice: {A,B,C} holds 10 before the pull, {B,C} holds 10 after it, nobody after the finish. |

### 3.5 What it does *not* give you: a better learning signal

1. **One all-or-nothing coalition has no headroom.** If the only reward needs every member of C, then removing any member gives 0, so each member's leave-one-out credit equals the team return on every trajectory. Exact credit changes only the bystanders' signal. **[round 1]** toy check: 215 versus 215 rounds to solve, with or without bystanders.
2. **As a reward, κ is potential-based shaping.** κ is the team reward plus γΦ(s′) − Φ(s) with Φ = W₋ᵢ, which preserves optimal policies (Ng, Harada & Russell, 1999). With Monte-Carlo returns it equals using W₋ᵢ(s_t) as agent i's baseline. For a bystander that is the usual baseline; for a member who has not yet contributed it is 0, a worse baseline. So no variance reduction.
3. **Where a benefit can come from.** When an agent's reward is mixed with rewards from events it has nothing to do with. **[round 1]** toy: with 8 independent events, exact difference rewards solved in 648 rounds against 1,961; with 16, team reward did not solve in 30,000. That is the classical difference-rewards effect (Wolpert & Tumer). It needs a multi-event layout, which you do not have yet.

So the honest scope is: **attribution** (who mattered, when, which coalition), with uses such as auditing, lazy-agent detection and team selection (Test D). Learning speed-up is a separate question with its own gate (Phase 3).

---

## 4. New tests

Conventions: L1 is your share-distance metric (0 = exact, 2 = disjoint). "Open-loop truth" is your default; "closed-loop truth" is your sensitivity protocol. Undiscounted (γ = 1) for episode credit, as in your truth. Scripts are in `atd/`.

### Test A. Exact ablation-TD credit (`run_oracle.py`)

**One episode, in full.** Layout: 5×10 grid; A pulls a lever, then B and C must interact on their goal cells in the same step for a reward of 10, paid d = 25 steps later; D and E are bystanders; horizon 60; scripted team with ε = 0.2. Cell (d = 25, b = 2, slack, seed 0), test episode 0. In this episode A pulls at t = 4, B and C finish at t = 14, the payout arrives at t = 39.

| Agent | Perfect one-step critic (your no-op notion, γ = 0.99), episode total | Exact ablation-TD credit (no-op null, γ = 1) |
|---|---|---|
| A (enabler) | 0.000 | **+10 at t = 4** (the pull) |
| B (finisher) | 0.078 | **+10 at t = 14** (the joint finish) |
| C (finisher) | 0.078 | **+10 at t = 14** |
| D, E (bystanders) | 0.000 | 0 at every step |

0.078 is 10·γ²⁵·(1 − γ): the cost of finishing one step late. That is all a one-step counterfactual can ever see here.

**All 120 main-grid cells** (3,600 test episodes, both nulls):

| Check | No-op null | Removal null |
|---|---|---|
| max over episodes and agents of abs(Σ_t κ − leave-one-out) | 0.0 | 0.0 |
| Enabler's credit lands on its pull step | 100% | 0.3% |
| Enabler's credit lands on the finish step | 0% | 99.3% |
| Finishers' credit lands on the finish step | 100% | 100% |
| Episodes with any bystander credit / any negative step credit | 0 / 0 | 0 / 0 |
| L1 against your open-loop truth, b = 0 / 2 / 4 | 0.00 / 0.09 / 0.20 | same |
| For comparison, stored perfect one-step critic: missed enabler credit | 0.99 slack, 0.66 critical | |

**Reading.**
- The total is the same under both nulls. The **timing** is not: under the removal null the enabler is credited at the finish, only because the scripted finishers give up when the enabler is absent (problem 6 in §2.3). Which null you use is a real modelling choice, not a detail.
- The residual L1 against open-loop truth (0.09, 0.20) is exactly the open-versus-closed-loop gap you already measured in `docs/sensitivity.md`. It is not an error of the credit.

### Test B. Learned ablation critic (`run_learned.py`)

**Method.**
- **Data.** Your 200 training episodes per cell (same seeds), plus 200 continuations in which a random set of agents stops at a random step (half single agents, half random subsets of every size). Closed loop, same random streams.
- **Critic.** MLP 2×128 from your global state plus an "active" bit per agent to the return-to-go, Monte-Carlo regression, 3,000 Adam steps. About 7–11 s on one CPU core.
- **Read-out.** Factual test states only, no interventions at test time: set agent i's active bit to 0, query, form κ.
- **Control.** The identical critic on the 200 episodes alone.
- **Cells.** 40 single-coalition cells (d ∈ {0, 25, 50}, b ∈ {2, 4}, slack and critical; 5 seeds for the two H5 conditions, 3 otherwise), plus your 20 overlap and partial H5 cells.

**Single-coalition cells, leave-one-out target** (mean over cells):

| | b = 2 (22 cells) | b = 4 (18 cells) |
|---|---|---|
| Equal split, L1 vs closed / open-loop truth | 0.80 / 0.73 | 1.14 / 1.03 |
| Stored VDN / QMIX / COMA, L1 vs open-loop truth | 1.30 / 0.90 / 1.44 | 1.39 / 1.31 / 1.56 |
| **Ablation critic, L1 vs closed-loop truth** | **0.020** | **0.036** |
| **Ablation critic, L1 vs open-loop truth** (floor from Test A: 0.09 / 0.20) | **0.094** | **0.202** |
| Same critic, no interventions, L1 vs open-loop truth | 0.97 | 1.20 |
| Ablation critic leakage / missed enabler credit | 0.009 / 0.012 | 0.018 / 0.022 |
| Top-3 credited agents are the coalition | 100% | 100% |
| Per-step correlation with exact κ | 0.80 | 0.80 |
| Stored methods, per-step correlation with their own target (max over these cells) | 0.31 | |

The critic is below the equal split and below stored QMIX in 40 of 40 cells.

**A flaw I found and fixed.** Per-step correlation was 0.95 at d = 0 but 0.69 at d = 25 and 0.76 at d = 50. Cause: I sampled the stopping step only before the event completed, so "payout pending, agent i stopped" states were never in the data. Sampling through the payout wait instead (`fix_delay.py`, 6 cells):

| d | Sampler | Per-step r (3 seeds) | Episode L1 vs closed-loop truth |
|---|---|---|---|
| 25 | before completion | 0.56–0.69 | 0.012–0.016 |
| 25 | through the payout wait | **0.997–0.998** | 0.019–0.056 |
| 50 | before completion | 0.70–0.80 | 0.011–0.025 |
| 50 | through the payout wait | **0.991–0.996** | 0.032–0.128 |

With a fixed budget of 200, timing fidelity and episode totals trade off. A stratified sampler (half before completion, half after) is the obvious next step; I have not run it.

**Overlap and partial layouts** (10 cells each):

| Layout | Equal split | Ablation critic (200) | Per-step r | No interventions |
|---|---|---|---|---|
| overlap | 0.56 | **0.016** | 0.97 | 0.76 |
| partial | 0.99 | 0.60 | 0.59 | 1.25 |

**The partial layout is not solved**, and 800 interventions do not help (0.59). Two causes: the state problem (§2.3 item 5), and a limit of the method: the critic estimates *expected* credit from a state, and in this layout which finisher completes alone differs episode by episode. See §5.

### Test C. Is ECE-k more than a table? (`run_learned.py`, `table*_mix`)

**Method.** I rebuilt your `mix` data exactly (200 episodes plus one open-loop replay each with a random subset removed, same RNG). I then fitted return ≈ Σ over coalitions S of size ≤ k of β_S·[all of S present] with a Lasso penalty of 0.025, which is ECE's 0.05 sparsity weight converted to scikit-learn's convention. Not tuned. The table sees only who was present. Scored with your metric code on your test episodes.

| Layout (d = 0 / 50) | Metric | ECE-k_mix (stored) | Table, same k | Table vs closed-loop truth |
|---|---|---|---|---|
| coalition, k = 3 | dividend error | 0.43 / 0.44 | **0.41 / 0.41** | 0.16 |
| | recall / precision | 0.96 / 0.96 | 0.96 / 0.96 | |
| | Shapley L1 | 0.12 | 0.12 | 0.06 |
| coalition, k = 2 (honesty check) | dividend error | 2.00 | 2.00 | |
| overlap, k = 3 | dividend error | 1.08 / 1.13 | **0.77 / 0.75** | 0.54 / 0.52 |
| | Shapley L1 | 0.16 | 0.15 / 0.14 | 0.08 |
| partial, k = 2 | dividend error | 1.09 / 1.07 | **0.76 / 0.73** | 0.37 / 0.36 |
| | Shapley L1 | 0.43 / 0.42 | **0.29** | 0.19 |
| partial, k = 3 | dividend error | 1.21 / 1.20 | **0.84 / 0.80** | 0.49 / 0.46 |

**Reading.**
- On fixed layouts the start state is always the same, so v(S) is a fixed table and 400 samples of (who was present, return) are enough to fit it. The network is not needed and on two layouts it hurts.
- What made H5 work is the **replay data**, which your own control already showed (ECE without it reproduces the equal split). The accurate claim is "removal interventions identify coalition credit; a sparse Möbius regression is sufficient".
- My own critics are *worse* than the table at coalition level with 200 interventions: dividend error 1.25 (MLP over masks) and 1.45 (state-conditioned low-order head). With an L1 penalty the low-order head reaches 0.35 on the single coalition (0.10 against closed-loop truth); with 800 interventions the MLP reaches 0.64–0.78. Sparsity matters more than architecture here.

### Test D. Does wrong credit cost anything? (`prune.py`, stored results only)

**Question.** If you used each method's credit to pick the 3 agents to keep, would you keep the coalition? If yes the pruned team still earns 10; if not, 0.

| Method | b = 2 (chance 10%) | b = 4 (chance 2.9%) |
|---|---|---|
| QMIX | 39.7% | 23.8% |
| VDN | 21.2% | 14.9% |
| COMA | 8.8% | 3.0% |
| MAPPO | 7.3% | 0.7% |
| NA²Q / NA²Q_mix | 40.0% / 51.0% | not run |
| STAS / STAS_mix | 4.0% / 18.0% | not run |
| ECE-3_mix, table, ablation critic | 100% | 100% (critic; ECE and table not run at b = 4) |

1,200 episodes per main-grid method and bystander count; 300 for the H5 methods. COMA is at chance. This is the first number in the project that says the credit error has a consequence.

### Test E. Shuffled roles (`run_perm.py`)

**Why.** Tests B–D have a fixed start state, where a table suffices. A learned model is only justified if it generalises across situations.

**Setup.** Single-coalition layout, d = 25. Every episode a random permutation decides which agent index is the enabler, which two are finishers and which are bystanders (start cells and stations move with the role). 200 training episodes, 30 test episodes, 5 seeds. The table gets the same number of interventions, all at t = 0 (its best case). Target: closed-loop leave-one-out, no-op null.

| | b = 2 | b = 4 |
|---|---|---|
| Equal split, L1 | 0.80 | 1.14 |
| Table, 200 / 800 interventions | 0.79 / 0.80 | 1.14 / 1.15 |
| **Ablation critic, 200 / 800** | **0.41 / 0.26** | **0.80 / 0.60** |
| Critic: top-3 are the coalition, 200 / 800 | 71% / 98% | 20% / 63% |
| Critic: per-step r, 200 / 800 | 0.73 / 0.82 | 0.45 / 0.64 |

**Reading.** The table is exactly an equal split, as it must be. The critic reads roles off the state and is clearly better, but its error is about 20 times larger than on fixed roles (0.02–0.04) at the same budget, and with 7 agents it is far from solved. A plain MLP has to learn every relabelling separately. This is the open problem worth an architecture (§7.2). I could not run ECE here; it sees agent identity and actions, so it may do well, and it must be compared.

---

## 5. What these tests do not show

- **Nothing is confirmatory.** Same seeds as your runs; several choices made after seeing data (the sampler fix is one).
- **The comparison with your methods is not like for like.** The ablation critic had intervention data and they did not. Your own NA²Q_mix and STAS_mix controls show that giving them replay data is not enough, but QMIX and COMA on intervention data have not been run (control V2).
- **It needs interventions.** You must be able to make an agent stop during data collection. Logged full-team data cannot identify removal credit, by the argument in §2.3 item 4.
- **It estimates expected credit, not realised credit.** The total is G − Ŵ₋ᵢ(s₀): the realised return minus an *expected* counterfactual. It cannot say that in episode 17 a bystander happened to matter. The partial layout exposes this.
- **The counterfactual is still a choice.** No-op and removal nulls give different timing (Test A) and, on the overlap layout, different games: with C inert and B not told, B waits for C forever and event 1 is lost too. Chen et al. (2026) make the same point for LLM teams: removing an agent changes what the others see. Report credit with its protocol, always.
- **Scripted, stationary policy only.** With learning policies the critic chases a moving target.
- **At the start state the critic is "measure v(S), then smooth".** The per-step fidelity and Test E are what show more than that.

---

## 6. Is the problem validated enough?

**"Standard one-step read-outs do not recover who earned the reward": yes, nearly.** It now has a measurement, a mechanism, an identity and a consequence (Test D). Three cheap controls close it:

| Control | Question | Cost |
|---|---|---|
| V1. Coverage: QMIX and COMA in Mode P at 200 / 800 / 3,200 episodes and ε ∈ {0.2, 0.4}. | Is r ≤ 0.36 a property of the read-out or of narrow data? Prediction from §3.3: fidelity stays low until critic RMSE is below about (1 − γ)·V ≈ 0.08. | 6 cells × 3 seeds × 6 settings |
| V2. Your methods on the intervention data, with two read-outs: the standard one-step one, and a masked value (Q_tot at the state with i stopped). | Separates "better data" from "better read-out". Prediction: one-step stays poor, masked value becomes good. | Test B grid |
| V3. Identity 2 on real critics: sum QMIX's one-step read-out along trajectories where agent i has stopped. | Direct test of §3.2. With a perfect critic this equals leave-one-out. | needs the Test A rollouts |

**"It matters for learning": not validated.** Do not write "wrong credit trains wrong behaviour" until Phase 3 says so.

**External validity: thin.** One environment family, scripted policies, n ≤ 7. Phase 4 addresses it; it is not the next step.

---

## 7. The proposal

### 7.1 Credit by intervention (tested: Tests A, B, D)

1. **Collect.** In a fraction of episodes, make a random set of agents stop at a random step for the rest of the episode. A wrapper that overrides an agent's action with no-op does this in any environment; no reset or replay is needed.
2. **Train** one centralised state-value critic on all episodes, with an "active" bit per agent in its input. In EPyMARL that is MAPPO's critic with n extra inputs.
3. **Read out.**

| Level | Formula | Cost |
|---|---|---|
| Step, agent | κ(i, t) = r_t + γ·V̂(s_{t+1}, i stopped) − V̂(s_t, i stopped) | n forward passes per step |
| Episode, agent | G − V̂(s₀, i stopped) | n forward passes |
| Coalitions at the start state | sparse Möbius (Lasso) regression of return on presence, or the same on V̂(s₀, S) | 2ⁿ passes for n ≤ 8, or sampled masks |

This is a data protocol and a read-out, not a combination of two architectures.

### 7.2 A hypothesis to test, not a result: an equivariant critic

- **Observation.** With shuffled roles the MLP critic needs far more data (Test E).
- **Reason.** W(s, active set) does not change if you relabel the agents and relabel the active set the same way. An MLP on a flat state vector does not know this and must learn up to n! copies.
- **Proposal.** One shared encoder per agent over its own features (position, role flags, active bit) plus the event features; pool with attention or a sum; a value head on the pooled vector. For coalition terms, a shared head per order on pooled member encodings with an L1 penalty, gated by the active set.
- **Prediction.** On shuffled roles with 200 interventions, L1 ≤ 0.15 at b = 2 (MLP: 0.41) and ≤ 0.35 at b = 4 (MLP: 0.80).
- **Kill criterion.** If it does not beat the MLP by at least 0.15 L1 at b = 4, drop it and report the MLP result as the limit.

Build it only after Phase 1 holds. It is the one place where I think a new model is justified by a stated symmetry instead of by "nobody has combined these".

---

## 8. Plan, step by step

Effort is in person-days for someone who knows the repo. I do not know your deadline, so the order is by dependency and each phase ends with something reportable. Roles: **A** integration and documents, **B** environment, **C** critic and Mode P runs, **D** Mode T runs, analysis and figures.

### Phase 0. Fix the record (2 days)

| # | Task | Where | Done when |
|---|---|---|---|
| 0.1 | Add each event's pending payout amount (÷ its reward) to the global state. | `envs/gridworld.py::get_state`, `state_size` (+K) | New test: two partial episodes paying 5 and 10 have different states while pending. |
| 0.2 | Mid-episode stopping: `env.remove_agents(agents)` and an action-override no-op wrapper. | `envs/gridworld.py`; logic in `atd/atd.py::remove_agent`, `continue_ablated` | Test: removal at t = 0 equals `reset(present=mask)`. |
| 0.3 | Finishers proceed when an absent enabler has already pulled. | `scripted.py::_finish_feasible` | Existing tests pass; closed-loop v(S) unchanged; under the removal null the enabler's credit lands on the pull. |
| 0.4 | Table baseline next to ECE. | `mode_p.py`: add `table2_mix`, `table3_mix` (20 lines; see `run_learned.py`) | Rows appear in `results/mode_p/h5/summary.md`; my numbers reproduced within 0.02. |
| 0.5 | Top-k selection metric. | `metrics.py`, `summarise_mode_p.py` (logic in `prune.py`) | Column in every summary. |
| 0.6 | Dividend error against both truths. | `experiments.py::_test_truth` | Two columns in the H3 and H5 tables. |
| 0.7 | Re-run the stored partial cells at d ∈ {25, 50} after 0.1. | `scripts/run_mode_p.py --grid h3`, `--grid h5` | Critic losses at d = 25 within 2× of d = 0. |
| 0.8 | Correct the documents: H5 (data, not estimator), STAS wording, H4 as uninformative, COMA zero-mean mechanism, closed loop as default truth. | `docs/`, problem statement status table | §2.3 items 1–5 addressed. |

### Phase 1. Confirmatory repeat in the repo (5 days)

| # | Task | Detail | Done when |
|---|---|---|---|
| 1.1 | Write `docs/preregistration_v3.md` **before any run**. | §9. Seeds 5–9. | Committed first. |
| 1.2 | Exact truth. | `replay.py::ablation_values(config, make_team, null)` from `atd.oracle_ablation_values`; add to `TestTruth`. About 0.2 s per 5-agent episode. | Test: Σ_t κ equals closed-loop leave-one-out to 1e-9. |
| 1.3 | Intervention data. | `DataCell.episodes("ablate", n, null, sampler)`; sampler **stratified**: half of the stops before the last event completes, half between completion and payout. | With n = 0 the episodes are byte-identical to today's `train`. |
| 1.4 | The critic. | `src/whoearned/ablation.py`: MLP on state + active bits, Monte-Carlo regression; also a TD(0) variant using MAPPO's critic and `train_critic_sequential`. Register `"atd"` in `mode_p.ALL_ALGS` and `read_credit`. | On cell (d = 0, b = 2, seed 0) reproduces L1 0.02 and step r 0.96 within 0.03. |
| 1.5 | Grid. | Main (d ∈ {0, 25, 50} × b ∈ {2, 4} × slack/critical) plus overlap and partial; seeds 5–9. Methods: atd, atd without interventions, table, ECE-3_mix, uniform, and the stored four. | Tables as in Tests B–D. |
| 1.6 | Controls V1–V3 (§6). | | Reported whatever the result. |
| 1.7 | Data scaling. | Interventions ∈ {25, 50, 100, 200, 400, 800}. | L1 and dividend error against budget, with seeds. **[round 1]**: 19 interventions already beat the equal split on individual credit; dividends needed about 200. |
| 1.8 | Both nulls, same cells. | No-op and removal. | A table of where they agree (totals) and differ (timing; overlap game). This is your Step 9 "no-op vs removal". |

### Phase 2. Generalisation: where a model is needed (5 days)

| # | Task | Detail | Done when |
|---|---|---|---|
| 2.1 | Shuffled-roles layout in the repo. | `layouts.py::coalition_layout(..., role_perm=…)` from `run_perm.py::perm_config`; `DataCell(shuffle_roles=True)`. | Test: a permuted episode has the same return and the permuted truth. |
| 2.2 | Run table, ECE-3_mix, atd (MLP) at 200 and 800. | b ∈ {2, 4}, d = 25, seeds 5–9. | Reproduces Test E for table and atd; ECE reported. |
| 2.3 | Equivariant critic (§7.2). | Per-agent tokens: own position, role flags, active bit, event flags. | Meets or misses the §7.2 prediction; kill criterion applied. |
| 2.4 | Size k+1 honesty check. | New layout: 2 enablers + 2 finishers, n = 6. | Does the sparse fit find the size-4 dividend from the same budget? Reported either way. |
| 2.5 | Randomised start cells. | Same as 2.1 with random starts. | Second generalisation axis. |

### Phase 3. Headroom gate: can exact credit help learning here? (5 days, decision point)

| # | Task | Detail |
|---|---|---|
| 3.1 | Multi-event layout. | `layouts.py::multi_event_layout(K)`: K walled lanes, each with one enabler and one finisher, reward 10; n = 2K; K ∈ {1, 2, 4, 8}. Short lanes, horizon about 25, so exploration is feasible. |
| 3.2 | Feasibility first. | IPPO on K = 1 must reach 90% success within about 300k steps. If not, shorten the lanes before anything else. |
| 3.3 | Exact difference rewards online. | Run n shadow copies of the environment, copy i without agent i, all fed the same joint action. Agent i's reward is r − r_copy_i. This is your open-loop replay in lockstep. Needs per-agent rewards in EPyMARL (`common_reward=False`; check your pinned commit). |
| 3.4 | Runs. | IPPO and MAPPO with (a) team reward, (b) shadow difference rewards, (c) ablation-TD credit from a critic trained alongside. K ∈ {1, 2, 4, 8}, 5 seeds. Count environment steps with and without shadow copies. |
| 3.5 | **Gate.** | If (b) needs under 0.67× the episodes of (a) at the largest K: there is a learning story; continue to Phase 4. If not: stop the learning track and write the measurement-and-attribution paper. Predictions written down now: K = 1 shows no difference; (c) is no better than (a) (§3.5). |

### Phase 4. Under learning, and a second environment (stretch, 1 week)

| # | Task | Detail |
|---|---|---|
| 4.1 | Dropout during training. | Wrapper that stops agents with probability p ∈ {0, 0.1, 0.25} per episode; MAPPO on the overlap layout and on K = 4. Its critic *is* the ablation critic. |
| 4.2 | Fidelity over training. | At 10 checkpoints: read-outs on 30 clean evaluation episodes against exact κ. |
| 4.3 | Side effects. | Return with and without dropout; do agents become robust to absences, and does that change what the credit means? |
| 4.4 | Level-Based Foraging. | Coalitions arise naturally (food needs a level sum). Use trained EPyMARL policies, the no-op wrapper, Monte-Carlo truth by re-running with an agent stopped. Only if Phases 1–2 hold. |

### Phase 5. Write-up (1 week)

Benchmark and two-level grading → the two identities → credit by intervention against truth, with the table and no-intervention controls → shuffled roles → the headroom result, whichever way it went → limits.

---

## 9. Pre-registration v3 (draft)

Fresh seeds 5–9. One-sided paired t-test on per-seed means, Holm within each hypothesis, exact Wilcoxon alongside, as in your amendment. Thresholds come from today's exploratory numbers with a margin.

| ID | Hypothesis | Supported if | Today |
|---|---|---|---|
| P1 | Exact persistent credit recovers the enabler. | Missed enabler credit ≤ 0.05 in every single-coalition cell. | 0.00 |
| P2 | It is learnable from 200 interventions. | L1 vs closed-loop truth ≤ 0.15 in every single-coalition and overlap condition, and below uniform (paired). | largest in any cell 0.066 |
| P3 | Interventions are necessary. | The no-intervention critic's L1 exceeds the ablation critic's by ≥ 0.3 in every condition. | smallest gap in any cell 0.39; mean gap about 1.0 |
| P4 | The persistent target is learnable per step; one-step targets are not. | Step r ≥ 0.85 at every delay with the stratified sampler; no stored-method step r above 0.5. | 0.90–0.998; ≤ 0.36 |
| P5 | ECE-k is not better than a table on fixed layouts. | Table dividend error ≤ ECE's + 0.05 on every layout. | lower on 6 of 6 |
| P6 | Wrong credit has a cost. | Top-3 selection ≥ 95% for the ablation critic; ≤ 50% for QMIX and COMA. | 100%; 40%, 9% |
| P7 | A table cannot handle shuffled roles; a state critic can. | Table L1 within 0.05 of uniform; MLP critic below uniform by ≥ 0.25 at 200 interventions. | 0.01; 0.39 and 0.34 |
| P8 | Equivariance helps. | §7.2 prediction. | not run |
| P9 | Partial layout after the state fix. | No threshold. Report L1 and the gap to realised truth; this is the method's stated limit. | 0.60 |
| P10 | No headroom with one coalition. | K = 1: episodes-to-90% within ±20% across reward types. | toy: 215 vs 215 |
| P11 | Headroom with many events. | Largest K: difference rewards need ≤ 0.67× the episodes. | toy: 0.33× at K = 8 |

---

## 10. What I need from you, and resources

**Data sets:** none. Everything is generated by your environment.

**Models:** none. No LLM is involved, so Ollama is not needed.

**Hardware.**
- Your MacBook Air (16 GB) is enough for Phases 0–2. My 2-core sandbox ran the 60 cells of Tests B–C, with exact truths, in about 50 minutes; a critic fits in about 10 s on one core. Use CPU; the networks are too small for MPS to help.
- Phases 3–4 are many EPyMARL runs. Your notes say 20 Mode T runs took 27 minutes on the 8-worker Ryzen machine. Use it if you still have it; on the Air alone, time one run first and cut to 3 seeds if needed.

**Papers.** I saw these only as abstracts, snippets or automated summaries. If you upload the PDFs I will read them properly. The first three decide the novelty wording.

1. Wang et al., *SHAQ* (arXiv 2105.15013): how coalition values are defined and obtained.
2. Li, Lee & Bareinboim, *Counterfactual Shapley Credit Assignment* (arXiv 2607.16999): do replacement actions persist across steps?
3. Chen et al., *Exact Is Easier: Credit Assignment for Cooperative LLM Agents* (arXiv 2603.06859 v2): Proposition 1 on distribution shift under agent removal.
4. Heuillet et al., *Collective eXplainable AI* (arXiv 2110.01307): their removal protocols.
5. Chen et al., *STAS* (arXiv 2304.07520): to check your port against the paper's own Alice-and-Bob numbers.

**Decisions I need.** Your deadline, and whether the Ryzen machine is available. They decide how much of Phases 3–4 is realistic.

---

## 11. Prior work, and what I would and would not claim

"Not found" below means not found in my searches, not proven absent.

| Work | What it does | Relation |
|---|---|---|
| Wolpert & Tumer, difference rewards | Agent's reward = G minus G with the agent clamped to a null action | Your leave-one-out truth is this. Needs the reward function or a resettable simulator. |
| COMA; Shapley Counterfactual Credits (Li et al., 2021); SQDDPG | Replace an agent's *action* inside a learned critic, for one step | One-step. This is what your benchmark shows failing. |
| Difference Advantage Estimation (Li, Xie & Lu, ICML 2022) | Potential-based difference rewards inside GAE; the potential marginalises the agent's own action at each step | One-step marginalisation; the agent keeps acting. |
| Devlin et al., AAMAS 2014 | Counterfactual-as-potential shaping (CaP, DRiP) | Closest older idea to "ablation value as potential". As far as I can tell from the abstract, the potential is the difference reward's counterfactual term, which needs the system utility; I did not read the full paper. |
| MACD (AAMAS 2024) | World-model rollouts with the agent's first action replaced by a default | One-step replacement. Its authors note that a default action is not the same as the agent leaving. |
| Dr.Reinforce (Castellini et al.) | Learns a reward network and differences it per step | Per-step; cannot see delayed effects. |
| Mahjoub et al., Agent Importance; Heuillet et al. | Leave-one-out or Shapley by re-running trained policies | Post-hoc, needs the simulator, not a learned estimator. |
| Triantafyllou et al., agent-specific effects (ICML 2024, 2025) | Causal decomposition of one action's effect through other agents | Explanation in an SCM; not a learned credit read-out graded against planted truth. |
| C3 / Exact Is Easier; CCPO; Agents that Matter (all 2026, LLM agent teams) | Measured removal or replay counterfactuals instead of critics | Same philosophy. CCPO uses whole-episode removal as a baseline. C3 warns that removal shifts teammates' inputs. |
| Li, Lee & Bareinboim (2026) | Shapley credit over *time steps* for one agent, with baseline-policy replacement | Single agent; nearest causal-theory relative. |
| TOCA (2026) | Separates action credit from exogenous turnover in open teams | Different question; turnover there is not an intervention for credit. |
| Covert et al.; Frye et al. (explainable ML) | Removal-based explanation with surrogates trained under masking | The same lesson, transferred from features to agents. |

**I would claim, if the confirmatory round holds:** (i) a ground-truth benchmark with two-level grading; (ii) two identities that explain why one-step read-outs cannot recover removal credit and what does; (iii) removal interventions make it identifiable with a standard critic, shown against truth, with a table baseline and a no-intervention control; (iv) an honest account of when credit matters for learning.

**I would not claim:** that persistent ablation is new as a *definition* (it is difference rewards and Heuillet's protocol); that the critic is a new architecture; that it improves return.

**I did not find** a learned persistent-ablation value with a temporal read-out graded against planted truth in classical CTDE MARL. Read papers 1–3 above before writing that sentence anywhere.

---

## Appendix A. Reproducing the numbers

Unzip the bundle; put `atd/` in `<repo>/scripts/atd/`, or anywhere with `WHOEARNED_REPO` set to the repo path. Needs numpy, scipy, scikit-learn and PyTorch (CPU). Run from inside `atd/`. Per-cell JSON outputs are in `atd/out/`.

| Test | Command | Time (2 cores) |
|---|---|---|
| A | `python run_oracle.py` then `python agg_oracle.py` | about 60 min |
| A (one episode) | `python credit_profile.py` | 10 s |
| B, C | `python run_learned.py all` then `python agg.py` | about 50 min |
| B (sampler fix) | `python fix_delay.py` | 6 min |
| D | `python prune.py` | seconds |
| E | `python run_perm.py` | 8 min |

Scripts skip cells whose JSON exists; delete `out/` to recompute. `rcc_prototype/` has its own README and the round-1 write-up.

## Appendix B. Terms

| Term | Meaning |
|---|---|
| Removal credit | Credit defined by what the team earns without the agent: leave-one-out (v(N) − v(N∖i)) or Shapley. |
| Open-loop / closed-loop truth | Replay recorded actions with agents removed / re-run the team with agents removed. |
| One-step counterfactual | Change one agent's action at one step, then the team continues as usual. |
| Persistent counterfactual | The agent stops at a step and stays stopped. |
| No-op / removal null | The stopped agent stays on the grid inert / leaves the grid. |
| Ablation value W_S(s) | Expected return-to-go from s when only S keeps acting. |
| Ablation-TD credit κ(i, t) | r_t + γ·W₋ᵢ(s_{t+1}) − W₋ᵢ(s_t), discounted by γᵗ. |
| Table | Lasso regression of episode return on which agents were present, with coalition terms up to size k. |
| Interaction mass | Share of total absolute dividend on coalitions of two or more agents. |
| Headroom | How much faster learning is with exact credit than with the team reward. |

## Sources

- [Potential-Based Difference Rewards for Multiagent Reinforcement Learning (Devlin et al., AAMAS 2014)](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2014/aamas/p165.pdf)
- [Difference Advantage Estimation for Multi-Agent Policy Gradients (Li, Xie & Lu, ICML 2022)](https://proceedings.mlr.press/v162/li22w/li22w.pdf)
- [Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination (MACD, AAMAS 2024)](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf)
- [Difference Rewards Policy Gradients (Castellini et al.)](https://arxiv.org/pdf/2012.11258v2.pdf)
- [Shapley Counterfactual Credits for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.00285)
- [Efficiently Quantifying Individual Agent Importance in Cooperative MARL](https://arxiv.org/abs/2312.08466)
- [Agent-Specific Effects (Triantafyllou et al.)](https://arxiv.org/abs/2310.11334)
- [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](https://proceedings.mlr.press/v267/triantafyllou25a.html)
- [Counterfactual Shapley Credit Assignment (Li, Lee & Bareinboim)](https://arxiv.org/abs/2607.16999)
- [Exact Is Easier: Credit Assignment for Cooperative LLM Agents](https://arxiv.org/html/2603.06859v2)
- [Counterfactual Credit Policy Optimization for Multi-agent Collaboration](https://arxiv.org/html/2603.21563v5)
- [Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based Attribution](https://arxiv.org/html/2605.27621v1)
- [Turnover-Orthogonal Credit Assignment for Open-Team MARL (TOCA)](https://arxiv.org/abs/2610.02847)
- [Collective eXplainable AI (Heuillet et al.)](https://www.researchgate.net/publication/355060153_Collective_eXplainable_AI_Explaining_Cooperative_Strategies_and_Agent_Contribution_in_Multiagent_Reinforcement_Learning_with_Shapley_Values)

Cited from memory, not re-checked today: Kakade & Langford (2002); Ng, Harada & Russell (1999); Baird (1994); Tallec, Blier & Ollivier (2019); Wolpert & Tumer (2001); Covert, Lundberg & Lee (2021); Frye et al. (2021). Verify each before citing.
