# Who Earned the Reward? (v2)
## Problem Statement, Research Gaps and Hypotheses

**Project:** Subgroup-Level Credit Assignment for Delayed Coalition Rewards in Cooperative MARL
**Team:** Aporia (AFML Course Project, Custom Track)
**Title:** *Measuring Who Earned the Reward: Ground-Truth Evaluation of Subgroup Credit Assignment under Delayed Coalition Rewards in Cooperative MARL*
**Version:** v2.1. v2 was trimmed from v1 (literature check as of 23 October 2026); v2.1 revises the research gaps (Section 3) after an external literature re-check of 27 October 2026. See Section 9 for what changed.

---

## 0. Summary

In cooperative MARL, team reward is often produced by one **subgroup** of agents and arrives **after a delay**. Coalition-aware credit methods are evaluated by return, win rate and qualitative plots. Delayed-reward methods split credit per agent and per step, and are also evaluated by return. Ground-truth credit checks exist in multi-agent settings, but only per agent or per component, and only for the proposing paper's own estimator (Section 3). As far as we found, no work checks the credit read-outs of *standard* learners against a known correct answer **at the coalition level, under delay**.

This project:

1. builds a gridworld where the coalition behind each reward, and the timing of each contribution, is **planted**;
2. defines ground truth at two levels, **which coalition created the value** (Harsanyi dividends) and **each agent's share** (leave-one-out marginals and Shapley values), under a declared counterfactual protocol;
3. compares methods' credit signals with that ground truth **only through read-outs that estimate the same quantity**;
4. separates **credit estimation** from **learning** by evaluating critics on fixed scripted data before evaluating trained policies;
5. optionally tests a cheap low-order coalition estimator.

---

## 1. Background and Running Example

The credit-assignment problem has a **temporal** half (linking actions to later rewards) and a **structural** half (attributing an outcome to the agents who produced it) [1]. MACA writes the global reward as a sum of rewards produced by (possibly overlapping) agent subsets [2]. STAS and TAR² address episodic or delayed team reward by redistributing it across time and agents [3][4].

**Running example.** Five agents A–E.

| Time | Event |
|---|---|
| t = 5 | A pulls a lever that opens a gate (**enabler**) |
| t = 25 | B and C jointly push a crate through the gate (**finishers**) |
| whole episode | D and E wander (**bystanders**) |
| t = 40 | Team receives +10 (**delay**) |

If any of A, B or C is absent, the reward is 0. So:

- Harsanyi dividends: Δ({A,B,C}) = 10, all others 0.
- Shapley values: φ_A = φ_B = φ_C = 10/3, φ_D = φ_E = 0.
- Leave-one-out marginals: m_A = m_B = m_C = 10, m_D = m_E = 0.

Normalised to shares, all three give {A, B, C} one third each. They diverge once coalitions overlap or partial coalitions earn partial value (Section 2.3).

**Failure modes**, all compatible with full team return:

- **Leakage:** D or E receive credit.
- **Missed credit:** A receives little credit because its action was 35 steps before the reward.
- **Coalition confusion:** the method concludes {B, C} or {B, C, D} created the value.

---

## 2. Problem Statement

### 2.1 Formal setting

Cooperative Dec-POMDP, agents $N = \{1,\dots,n\}$, shared team reward, episodic return $R(\tau)$.

**Planted coalition-reward model.**

$$
R(\tau) = \sum_{k=1}^{K} r_k \cdot \mathbb{1}[\text{event } k \text{ completed in } \tau]
$$

Each event $k$ has a required coalition $G_k = E_k \cup F_k$ (enablers who act by $t^e_k$, finishers who act jointly at $t^f_k$) and a payout at $t^f_k + d$. Bystanders are $B = N \setminus \bigcup_k G_k$. Partial synergy is allowed: a partial coalition may earn a fraction of $r_k$.

**Knobs**

| Knob | Values (initial) |
|---|---|
| Delay $d$ | 0, 10, 25, 50 |
| Enabler–finisher gap | short / long |
| Bystanders | 0, 2, 4 |
| Overlap | none / one agent in two coalitions |
| Synergy | all-or-nothing / partial |
| Coalition size | 2, 3, **4** (larger than the estimator's order $k$; see H4) |
| Team size | $n \le 8$ |

We start from a small factorial subset (delay × bystanders at fixed other knobs) and add the rest only if time allows.

### 2.2 Two evaluation modes

Credit can be wrong because a method **estimates** credit badly, or because it never **learns** a policy that reaches the reward. v1 mixed these. v2 separates them.

| Mode | Behaviour | What is trained | Ground truth | Purpose |
|---|---|---|---|---|
| **P (policy evaluation)** — primary | Fixed scripted policies that complete the planted events (with exploration noise), plus scripted bystanders | Each method's critic / mixer / return model only, on logged data | Exactly the planted structure: scripted policies are written to be well-defined when any agent is removed | Isolates credit estimation; no degenerate zero-reward cases |
| **T (full training)** — secondary | Each method's own learned policies | Everything | Computed by replay on trained-policy trajectories | Tests whether findings survive learning, and links to return (H1) |

In Mode T, a planted bystander may learn to help or obstruct. We therefore report the **realised** role of every agent (from replay ground truth), not just the planted one.

### 2.3 Ground-truth credit

**Counterfactual protocol (declared).** Contribution is only meaningful relative to a stated counterfactual [6][7]. Let

$$
v(S) = \mathbb{E}\big[R(\tau) \mid i \in S \text{ act},\ j \notin S \text{ follow a null policy}\big].
$$

- **Null policy (default): true removal.** The agent is taken off the grid. Its observation slot is zeroed and a presence flag is set to 0, so network input sizes stay fixed. This avoids the no-op proxy's known weakness that a no-op agent can still physically obstruct others [8].
- **Replay (default): open-loop.** Agents in $S$ repeat their recorded actions in the deterministic environment. This avoids crediting an agent merely because the *other* trained policies behave erratically when it disappears (they never saw that situation in training).
- **Sensitivity checks:** no-op null policy; closed-loop replay (agents in $S$ re-act from their policies, with per-agent fixed random-number streams so the removal is the only change).

**Coalition level: Harsanyi dividends** [34]:
$$
\Delta(S) = \sum_{A \subseteq S} (-1)^{|S|-|A|} v(A)
$$

**Agent level, two targets:**

- **Leave-one-out marginal** $m_i = v(N) - v(N \setminus \{i\})$. This is what counterfactual read-outs (COMA, difference-of-$Q_{tot}$) estimate, so it is the **matched target** for those methods.
- **Shapley value** $\phi_i = \sum_{S \ni i} \Delta(S)/|S|$ [35]. This is the fair-division target, and the one used for coalition-level comparisons.

In all-or-nothing games with disjoint coalitions the two targets give the same shares. With overlap or partial synergy they differ, and this difference is itself reported.

**Cost.** $n \le 8$ gives at most 256 coalitions, each evaluated by deterministic replay. This is affordable without sampling.

### 2.4 What counts as a method's credit (read-out rules)

**Principle:** a method is compared only with the ground-truth quantity that its read-out actually estimates. Raw per-agent utilities are **not** used as credit. In VDN, $Q_1 + c$ and $Q_2 - c$ give the same $Q_{tot}$, so utilities have no well-defined shares, and QMIX utilities are arbitrary up to the mixer.

| Method | Read-out $\hat c_i$ | Compared with |
|---|---|---|
| VDN / QMIX | $\sum_t \big[Q_{tot}(s_t, \mathbf a_t) - Q_{tot}(s_t, a^0_i, \mathbf a_{-i,t})\big]$ (offset-invariant, uses the learned mixer) | $m_i$ |
| COMA | $\sum_t A_i(s_t, \mathbf a_t)$ (its counterfactual advantage) | $m_i$ |
| STAS (or TAR² as fallback) | $\sum_t \hat r_{i,t}$, overall and per event | $\phi_i$ (STAS splits with Shapley values) |
| NA²Q | individual terms vs. $\phi_i$; pairwise/coalition terms vs. $\Delta(S)$ | $\phi_i$, $\Delta$ |
| MAPPO | Shared advantage: identical for all agents by construction [11] | **Reference only**, not a hypothesis (see below) |

**MAPPO.** A shared advantage gives bystanders the same scalar, but a bystander's *expected* policy gradient is about zero; what it actually receives is gradient noise. For MAPPO we therefore report a learning-signal quantity instead of leakage: the variance of each bystander's policy-gradient estimate relative to that of coalition members.

**Normalisation.** Shares are $\tilde c_i = \hat c_i / \sum_j |\hat c_j|$, and are only computed when $\sum_j |\hat c_j|$ is above a small threshold. Episodes below it are counted and reported separately rather than dropped silently.

### 2.5 Metrics

| Metric | Definition | Ideal |
|---|---|---|
| Credit error | $L_1 = \sum_i |\tilde c_i - \tilde g_i|$ against the matched target $g \in \{m, \phi\}$; plus Spearman rank | 0 / 1 |
| Leakage | $\Lambda = \sum_{i \in B} |\tilde c_i|$ | 0 |
| Missed enabler credit | $M = \max\big(0, 1 - \sum_{i\in E}\tilde c_i / \sum_{i\in E}\tilde g_i\big)$ | 0 |
| Coalition recovery | Jaccard similarity between $\hat G_k$ (coalitions whose estimated dividend exceeds a threshold) and $G_k$ | 1 |
| Dividend error | $\sum_{|S| \le k} |\hat\Delta(S) - \Delta(S)|$, for methods exposing coalition terms | 0 |
| Return–faithfulness | Spearman($\text{return}$, $1 - L_1$) **within each method** across seeds and configurations, and pooled | measured |
| Cost | critic evaluations per update; wall-clock | lower |

### 2.6 Research questions

- **RQ1.** In Mode P, how far do credit read-outs deviate from ground truth, and how do leakage and missed credit change with delay and bystanders?
- **RQ2.** Do individual-level metrics hide coalition-level errors?
- **RQ3.** In Mode T, does team return reveal these errors?
- **RQ4 (optional).** Can a low-order coalition estimator reduce coalition error cheaply?

### 2.7 Problem statement (final form)

> Given a cooperative multi-agent task in which team reward is produced by specific, possibly overlapping subgroups and delivered after a delay, and a declared counterfactual protocol defining ground-truth coalition credit (Harsanyi dividends) and individual credit (leave-one-out marginals and Shapley values), we (i) measure how far existing MARL methods' credit read-outs deviate from the ground-truth quantity each read-out estimates, as a function of delay and coalition structure, first with learning removed (fixed scripted data) and then with full training; (ii) test whether these deviations are visible in team return; and (iii) optionally evaluate a low-order coalition estimator at polynomial cost.

---

## 3. Research Gaps

**Revised 27 October 2026 (v2.1).** The gaps G1–G5 of v2.0 are replaced by F1–F7 below.

- **Sources.** This version draws on an external re-check of the literature through 27 October 2026, and is narrowed to match it.
- **Our results so far.** Each gap lists what this project has found, with its status. Full results: `docs/results_overview.md`.
- **Numbering.** The external analysis used its own G1–G9 numbering, from an earlier gap list. That numbering is not the G1–G5 used in v2.0.

**Citation flags.**
- References [1]–[38] were checked against their primary sources in the v2 fact-check (23 October 2026).
- **†** marks a reference added from the external analysis whose existence and content **we have not checked**. Several are 2026 preprints.
- **‡** marks a reference we know exists, but whose bibliographic details (venue, pages, numbers quoted) are taken from the external analysis and not re-checked.

Verify every †/‡ item before the final write-up.

"Not found" means not found in our search or in the external analysis, not proven absent. A final OpenReview check, including ICLR 2027 submissions, is needed before any publication claim.

**What is not new.**
- **Planted-ground-truth attribution grading** is standard in explainable ML: shapiq‡ [44] (Möbius and interaction indices against exact values), XAI-Bench‡ [45] and AttributionLab‡ [46]. Our dividend-error and Jaccard metrics adapt that methodology.
- **Ground-truth credit checks in multi-agent settings exist**, per agent or per component, but only for the proposing paper's own estimator:
  - Agent Importance [8]: against true Shapley values and individual rewards in LBF/RWARE;
  - COSAC† [39]: closed-form per-agent advantages in a sequential bandit;
  - TOCA [27]: planted value components.

  The claim "credit is never graded against ground truth in MARL" is therefore **not** made.
- **What is new** is the MARL instantiation: planted coalitions × delay × credit read-outs of *standard* learners, graded at both the individual and the coalition level.

### F1. Coalition-level attribution of standard learners is not graded against planted coalitions (merges v2.0 G1 and G4)

**Gap.**
- Coalition methods validate their groups by return, win rate or qualitative plots:
  - Nucleolus [12], CORA [11], MACA [2], NA²Q [14], QCoFr [13];
  - HySTAR [15] and HIVE† [48].
- PRD-MAPPO [16] checks planted teams only visually.
- One-step matrix games check the joint value against a known payoff matrix (QTRAN [36], WQMIX [37], QPLEX [38]), not the credit.
- Harsanyi dividends have been computed in a multi-agent setting [24], but over analytic Gaussian coalition models, not learners' read-outs.

**Why individual credit is not enough.** The worked counterexample (from v2.0 G4) is kept:

| Game | Values | Dividends | Shapley |
|---|---|---|---|
| Pure synergy | v(A) = v(B) = 0, v(AB) = 2 | Δ(AB) = 2 | 1, 1 |
| Independent | v(A) = v(B) = 1, v(AB) = 2 | Δ(A) = Δ(B) = 1 | 1, 1 |

Identical Shapley values, opposite coordination structure.

**Our evidence: H3 (negative).**
- We built the benchmark and coalition metrics, and applied them to VDN, QMIX and NA²Q through a per-step mixer-game read-out.
- **Every learned mixer game is additive.** Interaction mass is 0.000–0.002, against a true 1.0, and dividend error is about 2.0 (the maximum) in every configuration.
- So the pre-registered H3 test is uninformative.
- A positive answer requires an episode-level coalition read-out (H5).

**Status: open; benchmark built, negative first result.**

### F2. Coalition credit under delayed reward, across time steps (v2.0 G1, delay part)

**Gap.**
- Agent–temporal redistribution methods allocate credit to (agent, step) cells: STAS [3], TAR² [4], CAST [18] and LaRe‡ [47].
- None allocates to coalitions, or is graded against planted interactions *across time*: agents who contribute at different steps.

**Our evidence.**
- **The per-step coalition game cannot express synergy across time.** The planted coalitions act across steps (an enabler pulls first, finishers act later), so any game defined within one step is additive. This holds by construction, and the H3 grid confirms it.
- **STAS's credit does not change with delay.** Its results at d = 0 and d = 25 agree to within 4e-6 per episode: it sees only observations, actions and the total return, and none of these change with delay inside the horizon.
- **STAS's agent split is near uniform** at 6× the budget, and on a synthetic single-agent task.

**Status: open. We contribute a mechanism for why per-step read-outs fail here; H5 is the planned positive test.**

### F3. Missed credit for early enablers as delay grows (v2.0 G2)

**Gap.**
- The effect an upstream agent has *through* downstream agents is formalised in structural causal models: agent-specific effects‡ [40] and their Shapley decomposition‡ [41].
- It is graded against closed-form truth only in delay-free sequential bandits (COSAC's indirect effect† [39]).
- Methods that model delayed influence are scored by return or proxies:
  - STAS's proximity correlation [3];
  - MAGIC's artificial-delay study [19];
  - counterfactual Shapley credit, which is single-agent [7].
- No study measures the ground-truth enabler credit that standard learners miss as an explicit delay d grows.

**Our evidence: H1a and H1b (answered, with a correction to the gap's premise).**
- **Once the critic's state is Markov, delay barely matters.** H1a is not supported. VDN cannot see the payout countdown and fails to fit; QMIX has the right sign, but Holm p = 0.074.
- **Slack is what removes enabler credit, not delay.** A *perfect* per-step critic gives an enabler with slack about zero credit at every delay: level-2 missed is 0.96–0.99 (H1b slack half, confirmatory).
- When the enabler was actually decisive (post-hoc split), the noop notion gives it nearly full credit: missed 0.11–0.12.

**Status: addressed for Mode P.**

### F4. Bystander leakage is not measured against planted zero-contribution agents (v2.0 G3)

**Gap.**
- The lazy-agent problem is well known [20][21]: SHAQ's QMIX anecdote [23], influence diagnostics in two-agent LLM teams† [42].
- Agent Importance [8] measures trained policies after the fact.
- No study measures the share of credit a learner assigns to agents planted to contribute nothing.

**Our evidence: H2 and H2′ (answered).**
- **Leakage grows with bystanders.** For COMA this is confirmatory (8/8). For VDN/QMIX it holds in direction in 16/16 conditions (post-hoc sign test), but the per-condition tests are underpowered.
- **COMA's counterfactual baseline leaks the most**, the reverse of the expectation: 0.22–0.48 more than VDN/QMIX.
- **MAPPO's shared advantage leaks b/n by construction**, so it is excluded from leakage conclusions.

**Status: addressed for Mode P.**

### F5. Return can hide unfaithful credit, across standard learners (v2.0 G5)

**Gap.**
- TOCA [27] shows, within its own ablations, a variant with higher return and worse ground-truth credit. The external analysis reports ≈374 vs ≈341 return and turnover-event error 3.63 vs 2.14.
- MARL benchmarks are summarised by aggregate return or win rate (Coordination Matters† [49]).
- In single-agent LLM settings, implicit credit can be indistinguishable from a shuffled control ("Credit Without Ground Truth"† [50]).
- How common this is across standard learners, seeds and delays is unmeasured.

**Our evidence: related but not decisive.**
- **Accuracy without fidelity (H-fid, confirmatory).** 30 of the 32 best end-to-end cells have per-step fidelity r < 0.3. QMIX looks right for the wrong reason.
- **No method beats uniform credit (post-hoc, H3 grid).** On the overlap and partial layouts, no method's mean L1 beats giving each agent 1/n.
- The return–faithfulness test itself is H4, in Mode T. Descriptively, for VDN and QMIX with a seeded replay buffer, return is only weakly correlated with credit accuracy (ρ ≈ +0.15 to +0.20).

**Status: partly addressed. H4 was run descriptively for VDN and QMIX (seeded replay buffer); open for COMA and MAPPO.**

### F6. Credit read-outs from implicit learners have not been validated (new)

**Gap.**
- Value-decomposition and actor-critic learners have no explicit credit vector. Theirs must be *extracted*, for example as a Q_tot difference, a counterfactual advantage, a shared advantage or a mixer game.
- There is no accepted protocol for doing so.
- Audits graded against the method's own outputs can be circular [50]†.

**Our evidence: the project's main contribution so far.**
- **Two-level grading** separates fidelity (read-out vs the per-step counterfactual it estimates) from the semantic gap (a perfect per-step critic vs removal credit). Results:
  - Fidelity is r ≤ 0.36 for every method on the main grid.
  - The semantic gap is large even for a perfect critic, because of slack (F3) and because the team recovers from a one-step deviation.
- **MAPPO's shared-advantage read-out carries no per-agent information.** Every agent gets ±1/n.
- **The mixer-game coalition read-out is blind to coalitions across time** (F2).

**Status: open in the literature; addressed here in Mode P.**

### F7. Sensitivity of the ground truth to the counterfactual protocol (partly addressed; v2.0 Section 2.3)

**Gap.**
- Counterfactual protocols are formalised:
  - agent-specific effects in structural causal models‡ [40][41];
  - fixed-history replay and counterfactual distribution shift in agent removal† [42];
  - do-operator replay validated on planted structural causal models, single-agent† [43];
  - no-op vs random vs replace-with-teammate removal proxies, compared with Monte Carlo Shapley‡ [52].
- How much planted-coalition ground truth, and method rankings, change between removal, no-op, random, closed-loop and open-loop protocols in MARL is not quantified.

**Our evidence: partial (H0).**
- Removal vs no-op ground truth differs by a mean share L1 ≤ 0.18 per cell.
- The only failures are at 4 bystanders, from physical interference.
- Whether method rankings change under different protocols is Step 9.

**Status: partially addressed; treated as an empirical sub-question.**

### Demoted or not claimed

- **Polynomial cost is not claimed as a gap.**
  - At n ≤ 8, exact ground truth is feasible: 2⁸ = 256 coalition values.
  - Polynomial approximators exist: Agent Importance [8], Monte Carlo/kernel Shapley and interaction indices [44], and O(N) Aumann–Shapley† [51].
  - The low-order estimator (H5) is a **design contribution**, not a gap.
- **Agent turnover and departed-agent credit** is partly addressed (TOCA [27]; review [1]) and moved to future work.
- **Also not claimed:**
  - improving return with temporal redistribution [3][4][18];
  - improving win rate with grouping [2][11][12][15][30][31][32];
  - a fast proxy for individual importance [8];
  - new attribution metrics (they are adapted from [44]–[46]).

**Novelty risk.** Credit assignment for LLM multi-agent systems is moving quickly (COSAC† [39], C3† [42], and the call for "multi-agent tasks with designed credit structure" in survey [6]). We position this project as **classical CTDE MARL with state dynamics and explicit delay**, which those settings do not have.

---

## 4. Hypotheses

Predictions are **directional**. Numeric thresholds are fixed after a pilot (2 seeds, $d \in \{0, 25\}$) and before main runs.

### H0. The benchmark is measurable

Ground-truth variance across replays and disagreement between the default and sensitivity protocols are small compared with differences between methods. If H0 fails, the benchmark is fixed before H1–H4 are interpreted.

### H1. Missed enabler credit grows with delay (F3), Mode P

$M$ increases with $d$ and with the enabler–finisher gap for VDN, QMIX and COMA. The redistribution method (STAS) has lower $M$ but not zero at the largest $d$.
**Falsified if** $M$ does not increase with $d$ for any method.

### H2. Leakage grows with bystanders (F4), Mode P

$\Lambda$ increases with the number of bystanders for VDN and QMIX. COMA leaks less than VDN and QMIX at $d = 0$.
**Falsified if** $\Lambda$ is negligible for all methods at every bystander count.

### H3. Individual metrics hide coalition errors (F1, F2), Mode P

With overlap or partial synergy, some method–configuration pair has individual credit error in the best third but dividend error in the worst third. NA²Q's coalition recovery falls with $d$.
**Falsified if** individual and dividend errors are tightly correlated across all configurations.

### H4. Return is a weak proxy for credit faithfulness (F5), Mode T

Within each method, return and $1 - L_1$ are weakly correlated. At least two methods with statistically indistinguishable return differ significantly in $L_1$.
**Falsified if** return strongly predicts $1-L_1$ within methods. That would itself be useful: return would be a reasonable proxy.

### H5 (optional, only if H0–H3 are done). Low-order coalition estimator

**Mechanism.**
1. Fit an episodic return model with coalition terms up to size $k \in \{2, 3\}$.
2. Read out low-order dividends.
3. Allocate each dividend equally to its members, rescaled so redistributed rewards sum to the episode return (as in TAR² [4]).

**Prediction.** It has lower dividend error and leakage than NA²Q at matched compute when coalition size ≤ $k$.
**Required honesty check.** It must also be evaluated on planted coalitions of size $k+1$, which it cannot represent exactly. We report how much it degrades there, so we are not grading the method only on its own assumptions.

| | Gap | Metric | Manipulation | Mode |
|---|---|---|---|---|
| H1 | F3 | $M$ | delay, enabler gap | P |
| H2 | F4 | $\Lambda$ | bystanders | P |
| H3 | F1, F2 | $L_1$ vs. dividend error; Jaccard | overlap, synergy, delay | P |
| H4 | F5 | return vs. $L_1$ | across seeds | T |
| H5 | F1, F2 (design contribution) | all + cost | method; coalition size vs. $k$ | P |

**Status of each hypothesis (7 November 2026).** Details: `docs/results_overview.md`. "Supported" and "not supported" are the pre-registered rules applied as written.

| | Status | In one line |
|---|---|---|
| H0 | not supported (narrow) | 16 of 120 cells (4 data configurations, all b = 4) have realised ≠ planted too often under open-loop replay; under closed-loop replay realised = planted in 100% of episodes |
| H1a | not supported (sensitive) | VDN's critic does not converge under delay; QMIX has the right sign (Holm p = 0.074). Supported through one condition if scored against a closed-loop truth |
| H1b | not supported | slack half holds in 24/24 cells; critical half fails because only 54% of "critical" episodes were critical |
| H2 | supported for COMA; not for VDN, QMIX | MAPPO's 8/8 is an artefact of its shared advantage |
| H2′ | not supported (reversed) | COMA leaks more than VDN and QMIX in all 8 comparisons |
| H3 | not supported literally; uninformative | dividend error is 1.9993–2.0000 for every method, so it cannot rank methods |
| H4 | descriptive only: first clause consistent, second not shown | VDN and QMIX with a seeded replay buffer (plain online training never earns reward): Spearman ρ(return, 1 − L1) = +0.20 and +0.15, no CI reaching 0.7; QMIX earns more than VDN without assigning credit more accurately; COMA and MAPPO not assessed; `docs/mode_t_results.md` |
| H5 | supported on dividend error and on beating uniform credit; not on leakage | ECE-3_mix recovers the order-3 coalition (dividend error 0.43 vs 2.0) |
| H-fid (extra finding) | supported | 30 of the 32 best end-to-end cells have step r < 0.3 |

---

## 5. Experimental Design

### 5.1 Environment

One custom **Planted Coalition Gridworld**, deterministic, supporting true removal, open-loop replay and fixed-size observations with presence flags. The Alice & Bob-style enabler chain [3] is a configuration of this environment, not a separate one. An LBF variant is future work.

### 5.2 Methods

| Method | Source |
|---|---|
| VDN, QMIX, COMA, MAPPO | EPyMARL [33] |
| STAS | public repo [3] (TAR² [4] as fallback if porting fails) |
| NA²Q | public repo [14] |

Dropped from v1 for scope: QPLEX, TAR² (unless needed as the fallback), HYGMA, PRD-MAPPO, MACA, CORA, QCoFr, MAGIC. The main engineering cost is porting the environment into each codebase, so we limit ourselves to three: EPyMARL, STAS, NA²Q.

### 5.3 Protocol

- **Mode P:** one scripted dataset per configuration; each critic trained to convergence on it; 5 seeds (seeds change network initialisation and data order).
- **Mode T:** 5 seeds per method at the pilot-selected configurations only; matched environment-step budgets; EPyMARL defaults.
- Ground truth is computed on the same trajectories as the read-outs. Report means with 95% confidence intervals and paired tests.

### 5.4 Milestones

1. Environment + exact ground truth + unit tests reproducing the Section 1 and Section 3 (F1 counterexample) numbers.
2. Mode P with EPyMARL methods; H0, H1, H2.
3. Port STAS and NA²Q; H3.
4. Mode T; H4.
5. H5 if time remains.

---

## 6. Scope, Assumptions, Non-Goals

- **In scope:** fully cooperative, shared reward, $n \le 8$, one custom environment.
- **Assumptions:** delays are external; ground truth is relative to the declared protocol; credit is evaluated on the trajectories being scored.
- **Non-goals:** SMAC/GRF state of the art; settling which counterfactual is philosophically correct; mixed-motive settings; agent turnover (v1's H7 is future work).

---

## 7. Positioning

| Work | Delayed reward | Subgroup-level | Graded vs. ground truth |
|---|---|---|---|
| QTRAN / WQMIX / QPLEX matrix games [36][37][38] | ✗ | ✗ | partial: joint value vs. known payoff, one-step |
| STAS [3], TAR² [4], CAST [18] | ✓ | ✗ | ✗ |
| MACA [2], Nucleolus [12], CORA [11], NA²Q [14], QCoFr [13] | ✗ | ✓ | ✗ |
| PRD-MAPPO [16] | ✗ | ✓ | partial (qualitative) |
| Agent Importance [8] | ✗ | ✗ | ✓ post-hoc, trained policies |
| TOCA [27] | turnover only | ✗ | ✓ own heads only |
| COSAC† [39] | ✗ (bandit) | ✗ | ✓ per-agent advantages, own estimator |
| Agent-specific effects‡ [40][41] | ✗ | ✗ | causal explanation, not grading of learned credit |
| shapiq‡ [44], XAI-Bench‡ [45] (non-RL) | ✗ | ✓ interaction indices | ✓ (methodology we adapt) |
| **This project** | ✓ | ✓ | ✓ dividends, marginals, Shapley; standard learners' read-outs; learning separated from estimation; read-out fidelity graded separately (F6) |

---

## 8. Threats to Validity

| Threat | Mitigation |
|---|---|
| Read-out measures a different quantity than ground truth | Offset-invariant counterfactual read-outs; each compared with its matched target ($m$ or $\phi$); raw utilities not used |
| Ground truth contaminated by policy brittleness after removal | Open-loop replay by default; closed-loop as a check; scripted policies in Mode P |
| Methods fail to learn under long delay | Mode P does not depend on learning; Mode T restricted to pilot-validated configurations; near-zero-credit episodes reported |
| Planted roles differ from learned roles | Report realised roles from replay in Mode T |
| Synthetic environment may not generalise | Kept minimal so failures can be traced to a knob; LBF variant listed as future work |
| Estimator graded on its own assumptions | Coalitions of size $k+1$ included |
| Recency of sources | Several key sources are 2026 preprints; re-check each before the final write-up. References marked † or ‡ (Section 3) are not yet checked by us |

---

## 9. Changes from v1

- **Credit read-outs:** raw VDN/QMIX utilities dropped, since they have no well-defined shares. Each read-out is now compared with the ground-truth quantity it estimates; leave-one-out marginals added as a target alongside Shapley.
- **MAPPO:** no longer a leakage hypothesis (its leakage is true by construction); it is a reference, with a gradient-variance diagnostic instead.
- **New Mode P** (scripted data, critics only) separates credit estimation from learning; full training (Mode T) is secondary.
- **Replay default** changed from closed-loop to open-loop; per-agent fixed random-number streams for the closed-loop check; fixed-size observations with presence flags.
- **Scope:** 3 environments → 1; ~10 methods → 6 across three codebases; 8 hypotheses → H0–H4 plus optional H5; turnover (H7) moved to future work.
- **Thresholds:** directional predictions; numbers fixed after a pilot. The return–faithfulness correlation is computed within each method.
- **Estimator:** must also be tested on coalitions larger than its order $k$.
- **Gap claims softened:** matrix-game evaluations against known payoffs (QTRAN, WQMIX, QPLEX) are acknowledged as precedent; gaps are stated as "not at coalition level and not under delay".
- **References:** the Substack explainer replaced with Harsanyi (1963) and Shapley (1953).

**v2.1 (27 October 2026):**
- **Research gaps** G1–G5 replaced by F1–F7 after an external literature re-check:
  - G1 and G4 merged into F1; the delay part of G1 is now F2.
  - G2 → F3, G3 → F4, G5 → F5.
  - New: F6 (validity of read-outs) and F7 (protocol sensitivity).
  - Polynomial cost demoted to a design contribution; turnover stays future work.
- **Each gap now states our results so far** and its status.
- **The claim is narrowed** from "never graded against ground truth" to "not at coalition level, under delay, for standard learners". XAI benchmarks are credited as the source of the grading methodology.
- **References [39]–[52] added**, flagged † or ‡ where not yet checked by us. Notes added to [4] (two arXiv IDs) and [42] (two titles).
- **Hypotheses H1–H5 and the sections after them are unchanged** (pre-registered). Only their gap labels were updated.

---

## 10. References

Numbering follows v1 so cross-references stay stable; [34]–[38] are new in v2; v1's [9] is removed.

- [1] Saleh Abadi, A., & Soh, L.-K. *Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems.* arXiv 2510.27659, 2025.
- [2] Zhao, X., & Xie, Y. *Multi-level Advantage Credit Assignment for Cooperative Multi-Agent Reinforcement Learning (MACA).* AISTATS 2025. arXiv 2508.06836.
- [3] Chen, S., Zhang, Z., Yang, Y., & Du, Y. *STAS: Spatial-Temporal Return Decomposition for Solving Sparse Rewards Problems in Multi-agent Reinforcement Learning.* AAAI 2024. (arXiv 2304.07520 uses the shorter title *STAS: Spatial-Temporal Return Decomposition for Multi-agent Reinforcement Learning*.) Code: https://github.com/zowiezhang/STAS
- [4] Kapoor, A., Tessera, K., Baranwal, M., Khadilkar, H., Peters, J., Albrecht, S., & Sun, M. *Redistributing Rewards Across Time and Agents for Multi-Agent Reinforcement Learning (TAR²).* arXiv 2502.04864, v2 2025. (Per the external analysis, not re-checked: an earlier version exists as arXiv 2412.14779, *Agent-Temporal Credit Assignment for Optimal Policy Preservation…*.)
- [6] Zhang, C. *From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models.* arXiv 2604.09459, v3 2026.
- [7] Li, M., Lee, K.-Z., & Bareinboim, E. *Counterfactual Shapley Credit Assignment.* arXiv 2607.16999, v2 2026.
- [8] Mahjoub, O., de Kock, R., Singh, S., Khlifi, W., Vall, A., Tessera, K., & Pretorius, A. *Efficiently Quantifying Individual Agent Importance in Cooperative MARL.* AAAI XAI4DRL workshop; arXiv 2312.08466.
- [11] Ji, M., Xu, G., Jia, K., Duan, Z., Qiu, Y., Ge, J., & Li, M. *Cooperative Game-Theoretic Credit Assignment for Multi-Agent Policy Gradients via the Core (CORA).* arXiv 2506.04265, v3 2026.
- [12] Li, Y., Cao, Z., Qiao, J., & Hu, S. *Nucleolus Credit Assignment for Effective Coalitions in Multi-agent Reinforcement Learning.* AAMAS 2025. arXiv 2503.00372.
- [13] Xu, Q., Zhu, Y., Wu, X., & Chen, C. *High-order Interactions Modeling for Interpretable Multi-Agent Q-Learning (QCoFr).* NeurIPS 2025. arXiv 2510.20218.
- [14] Liu, Z., Zhu, Y., & Chen, C. *NA²Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning.* ICML 2023, PMLR 202:22539–22558. Code: https://github.com/zichuan-liu/NA2Q
- [15] Luo, X., Zhang, Y., Kuang, Y., Yuan, S., Zeng, Z., Zhu, W., Ji, Z., & Wang, Z. *HySTAR: Anchored Hypergraphs for Stable Credit Assignment in Cooperative Multi-Agent Reinforcement Learning.* arXiv 2609.31531, 2026.
- [16] Kapoor, A., Freed, B., Choset, H., & Schneider, J. *Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization.* Reinforcement Learning Journal (RLC 2024). arXiv 2408.04295.
- [18] Zhang, Y., Du, Y., Huang, B., Pechenizkiy, M., & Fang, M. *A Causality-Inspired Spatial-Temporal Return Decomposition Approach for Multi-Agent Reinforcement Learning (CAST).* Reinforcement Learning Journal vol. 7 (RLC 2026, pre-proceedings). Earlier version: NeurIPS 2024 Workshop on Causal Representation Learning (author order there: Zhang, Du, Huang, Fang, Pechenizkiy).
- [19] Yu, H., Cong, J., Wang, S., Wang, L., & Liu, C. *MAGIC: Multi-Step Advantage-Gated Causal Influence for Multi-agent Reinforcement Learning.* arXiv 2605.01805, v2 2026.
- [20] Li, T., Zhu, K., Luong, N. C., Niyato, D., Wu, Q., Zhang, Y., & Chen, B. *Applications of Multi-Agent Reinforcement Learning in Future Internet: A Comprehensive Survey.* arXiv 2110.13484 (lazy-agent discussion in its VDN section).
- [21] Liu, B., Pu, Z., Pan, Y., Yi, J., Liang, Y., & Zhang, D. *Lazy Agents: A New Perspective on Solving Sparse Reward Problem in Multi-agent Reinforcement Learning (LAIES).* ICML 2023, PMLR 202:21937–21950.
- [23] Wang, J., Zhang, Y., Gu, Y., & Kim, T.-K. *SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning.* NeurIPS 2022. arXiv 2105.15013.
- [24] Bouchaffra, D., Ykhlef, F., Lebbah, M., & Azzag, H. *Coalition Free Energy and Adaptive Precision in Multi-Agent Cooperation.* arXiv 2605.26278, 2026.
- [27] Thakur, A., & Singhal, M. *Turnover-Orthogonal Credit Assignment for Open-Team Multi-Agent Reinforcement Learning (TOCA).* arXiv 2610.02847, 2026.
- [30] Phan et al. *VAST.* NeurIPS 2021. (Not re-verified.)
- [31] Zang et al. *GoMARL.* NeurIPS 2023. (Not re-verified.)
- [32] Liu & Li. *HYGMA.* ICML 2025. (Not re-verified.)
- [33] EPyMARL. https://github.com/uoe-agents/epymarl
- [34] Harsanyi, J. C. *A Simplified Bargaining Model for the n-Person Cooperative Game.* International Economic Review 4(2), 1963. doi:10.2307/2525487
- [35] Shapley, L. S. *A Value for n-Person Games.* In H. W. Kuhn & A. W. Tucker (eds.), Contributions to the Theory of Games II (Annals of Mathematics Studies 28), Princeton University Press, 1953, pp. 307–317. doi:10.1515/9781400881970-018
- [36] Son, K., Kim, D., Kang, W. J., Hostallero, D. E., & Yi, Y. *QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning.* ICML 2019, PMLR 97. https://arxiv.org/abs/1905.05408
- [37] Rashid, T., Farquhar, G., Peng, B., & Whiteson, S. *Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.* NeurIPS 2020. https://arxiv.org/abs/2006.10800
- [38] Wang, J., Ren, Z., Liu, T., Yu, Y., & Zhang, C. *QPLEX: Duplex Dueling Multi-Agent Q-Learning.* ICLR 2021. https://arxiv.org/abs/2008.01062

**Added in v2.1** from the external literature re-check (27 October 2026):
- **†** = not yet checked by us: existence, authors, venue and quoted numbers all unverified.
- **‡** = known to exist, but the details given here are not re-checked.

- [39] † Deshmukh, Addanki, Subramanian & Vlassis. *COSAC* (per-agent advantages with closed-form ground truth in a sequential-bandit testbed). arXiv 2604.17693, 2026.
- [40] ‡ Triantafyllou, S., Sukovic, A., Mandal, D., & Radanovic, G. *Agent-Specific Effects: A Causal Effect Propagation Analysis in Multi-Agent MDPs.* ICML 2024. arXiv 2310.11334.
- [41] ‡ Triantafyllou, S., et al. *Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making.* ICML 2025. arXiv 2410.12539.
- [42] † *C3 / Exact Is Easier: Credit Assignment for Cooperative LLM Agents.* arXiv 2603.06859, 2026. v1 is reportedly titled *Contextual Counterfactual Credit Assignment…*; cite the version used.
- [43] † Shah. *CAR (Causal Agent Replay).* arXiv 2606.08275, 2026.
- [44] ‡ Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., & Hüllermeier, E. *shapiq: Shapley Interactions for Machine Learning.* NeurIPS 2024 Datasets & Benchmarks.
- [45] ‡ *Synthetic Benchmarks for Scientific Research in Explainable Machine Learning (XAI-Bench).* arXiv 2106.12543, 2021.
- [46] ‡ *AttributionLab: Faithfulness of Feature Attribution Under Controllable Environments.* arXiv 2310.06514, 2023.
- [47] ‡ *LaRe: latent-reward, LLM-assisted episodic multi-agent credit assignment.* arXiv 2412.11120.
- [48] † Zhou et al. *HIVE* (hypergraph Position-value decomposition). Neural Networks 203:109103, 2026.
- [49] † *Coordination Matters* (process diagnostics beyond aggregate return in MARL benchmarks). arXiv 2605.06557, 2026.
- [50] † *Credit Without Ground Truth* (single-agent ALFWorld audit of implicit and judge credit). arXiv 2608.19760, 2026.
- [51] † *Million-agent Aumann–Shapley attribution* with analytic per-agent ground truth. arXiv 2605.11404, 2026.
- [52] ‡ Heuillet, A., Couthouis, F., & Díaz-Rodríguez, N. *Collective eXplainable AI: Explaining Cooperative Strategies and Agent Contribution in Multiagent Reinforcement Learning with Shapley Values.* IEEE Computational Intelligence Magazine 17(1), 2022. arXiv 2110.01307.
