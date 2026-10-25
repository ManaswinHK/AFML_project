# Who Earned the Reward? (v2)
## Problem Statement, Research Gaps and Hypotheses

**Project:** Subgroup-Level Credit Assignment for Delayed Coalition Rewards in Cooperative MARL
**Team:** Aporia (AFML Course Project, Custom Track)
**Title:** *Measuring Who Earned the Reward: Ground-Truth Evaluation of Subgroup Credit Assignment under Delayed Coalition Rewards in Cooperative MARL*
**Version:** v2, trimmed from v1 (literature check as of 23 October 2026; see Section 9 for what changed)

---

## 0. Summary

In cooperative MARL, team reward is often produced by one **subgroup** of agents and arrives **after a delay**. Coalition-aware credit methods are evaluated by return, win rate and qualitative plots. Delayed-reward methods split credit per agent and per step, and are also evaluated by return. As far as we found, no work systematically checks, against a known correct answer, **whether the right subgroup received the credit when reward is delayed**.

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

Each gap is stated narrowly. "Not found" means not found in our search, not proven absent; a final OpenReview check is needed before any publication claim.

### G1. Coalition-level credit is not graded against a planted ground truth, and not under delay

**Precedent we must acknowledge.** Value-decomposition papers compare learned values with a known payoff matrix in one-step games. QTRAN shows learned $Q_1$, $Q_2$ and $Q_{jt}$ next to the payoff matrix (its Table 1) [36]; QPLEX shows learned $Q_{tot}$ for five methods against the payoff matrix (its Table 4, Appendix C) [38]; WQMIX analyses the idealised tabular QMIX projection of known payoff matrices (its Tables 1–2) [37]. These check whether the **joint** value and greedy action are represented correctly, not whether credit goes to the right agents or coalitions, and they are one-step with no delay. SHAQ's credit evaluation is qualitative, on SMAC [23].

**Gap.** Coalition-aware methods (Nucleolus [12], CORA [11], MACA [2], NA²Q [14], QCoFr [13]) validate their groups by return, win rate or qualitative plots. PRD-MAPPO [16] comes closest: in its Collision Avoidance task, relevant-set attention concentrates on agents of the same known team, but this is a qualitative heatmap under a per-step distance-based reward. CORA checks its sampled approximation against its own full solution, which is self-consistency, not truth.

### G2. Missed credit for early enablers under delay is not measured in MARL

STAS's Alice & Bob task is an enabler chain, but it is scored by reward and success rate. STAS's only quantitative credit check correlates credit with inverse distance to prey [3], a proximity proxy that by construction under-credits distant enablers. MAGIC [19] rewards delayed teammate influence and includes an artificial-delay study, but is scored by return and win rate. Counterfactual Shapley credit [7] credits enabling actions (a DoorKey key pickup five steps before the door toggle), but only for a single agent in small tabular environments, using simulator rollouts.

### G3. Bystander leakage is described but not measured against known bystanders

The lazy-agent problem is well known [20][21], and SHAQ gives an anecdote of QMIX crediting an agent that fled the battle [23]. Agent Importance [8] measures the contributions of trained policies after the fact, not the credit signal the learner uses.

### G4. Individual credit can be right while coalition attribution is wrong

**Worked counterexample**

| Game | Values | Dividends | Shapley |
|---|---|---|---|
| Pure synergy | v(A) = v(B) = 0, v(AB) = 2 | Δ(AB) = 2 | 1, 1 |
| Independent | v(A) = v(B) = 1, v(AB) = 2 | Δ(A) = Δ(B) = 1 | 1, 1 |

The Shapley values are identical, but the coordination structure is opposite. We found no MARL credit benchmark that grades dividend-level attribution; the one 2026 multi-agent use of Harsanyi dividends we found [24] uses independent Q-learning on trajectory data.

### G5. Return can hide unfaithful credit

TOCA [27] reports a variant with higher return (≈374 vs ≈341) and worse credit error on turnover events (3.63 vs 2.14), for its own heads only. Systematic evidence across standard methods is missing.

**Not claimed:** improving return with temporal redistribution [3][4][18]; improving win rate with grouping [2][11][12][15][30][31][32]; a fast proxy for individual importance [8]; turnover-robust credit [27]; departed-agent credit (moved to future work).

---

## 4. Hypotheses

Predictions are **directional**. Numeric thresholds are fixed after a pilot (2 seeds, $d \in \{0, 25\}$) and before main runs.

### H0. The benchmark is measurable

Ground-truth variance across replays and disagreement between the default and sensitivity protocols are small compared with differences between methods. If H0 fails, the benchmark is fixed before H1–H4 are interpreted.

### H1. Missed enabler credit grows with delay (G2), Mode P

$M$ increases with $d$ and with the enabler–finisher gap for VDN, QMIX and COMA. The redistribution method (STAS) has lower $M$ but not zero at the largest $d$.
**Falsified if** $M$ does not increase with $d$ for any method.

### H2. Leakage grows with bystanders (G3), Mode P

$\Lambda$ increases with the number of bystanders for VDN and QMIX. COMA leaks less than VDN and QMIX at $d = 0$.
**Falsified if** $\Lambda$ is negligible for all methods at every bystander count.

### H3. Individual metrics hide coalition errors (G1, G4), Mode P

With overlap or partial synergy, some method–configuration pair has individual credit error in the best third but dividend error in the worst third. NA²Q's coalition recovery falls with $d$.
**Falsified if** individual and dividend errors are tightly correlated across all configurations.

### H4. Return is a weak proxy for credit faithfulness (G5), Mode T

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
| H1 | G2 | $M$ | delay, enabler gap | P |
| H2 | G3 | $\Lambda$ | bystanders | P |
| H3 | G1, G4 | $L_1$ vs. dividend error; Jaccard | overlap, synergy, delay | P |
| H4 | G5 | return vs. $L_1$ | across seeds | T |
| H5 | — | all + cost | method; coalition size vs. $k$ | P |

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

1. Environment + exact ground truth + unit tests reproducing the Section 1 and G4 numbers.
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
| **This project** | ✓ | ✓ | ✓ dividends, marginals, Shapley; across methods; learning separated from estimation |

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
| Recency of sources | Several key sources are 2026 preprints; re-check each before the final write-up |

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

---

## 10. References

Numbering follows v1 so cross-references stay stable; [34]–[38] are new in v2; v1's [9] is removed.

- [1] Saleh Abadi, A., & Soh, L.-K. *Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems.* arXiv 2510.27659, 2025.
- [2] Zhao, X., & Xie, Y. *Multi-level Advantage Credit Assignment for Cooperative Multi-Agent Reinforcement Learning (MACA).* AISTATS 2025. arXiv 2508.06836.
- [3] Chen, S., Zhang, Z., Yang, Y., & Du, Y. *STAS: Spatial-Temporal Return Decomposition for Solving Sparse Rewards Problems in Multi-agent Reinforcement Learning.* AAAI 2024. (arXiv 2304.07520 uses the shorter title *STAS: Spatial-Temporal Return Decomposition for Multi-agent Reinforcement Learning*.) Code: https://github.com/zowiezhang/STAS
- [4] Kapoor, A., Tessera, K., Baranwal, M., Khadilkar, H., Peters, J., Albrecht, S., & Sun, M. *Redistributing Rewards Across Time and Agents for Multi-Agent Reinforcement Learning (TAR²).* arXiv 2502.04864, v2 2025.
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
