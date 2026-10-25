# Who Earned the Reward?
## Problem Statement, Verified Research Gaps and Hypotheses

**Project:** Subgroup-Level Credit Assignment for Delayed Coalition Rewards in Cooperative MARL
**Team:** Aporia (AFML Course Project, Custom Track)
**Sharpened title:** *Measuring Who Earned the Reward: Ground-Truth Evaluation of Subgroup Credit Assignment under Delayed Coalition Rewards in Cooperative MARL*
**Literature verified as of:** 23 October 2026

---

## Table of Contents

0. [Executive Summary](#0-executive-summary)
1. [Background](#1-background)
2. [Problem Statement](#2-problem-statement)
3. [Verified Research Gaps](#3-verified-research-gaps)
4. [Hypotheses](#4-hypotheses)
5. [Experimental Design Sketch](#5-experimental-design-sketch)
6. [Scope, Assumptions and Non-Goals](#6-scope-assumptions-and-non-goals)
7. [Positioning Against the Closest Work](#7-positioning-against-the-closest-work)
8. [Threats to Validity](#8-threats-to-validity)
9. [Changes Relative to the Submitted One-Pager](#9-changes-relative-to-the-submitted-one-pager)
10. [Verification Log](#10-verification-log)
11. [References](#11-references)

---

## 0. Executive Summary

In cooperative multi-agent reinforcement learning (MARL), a team often succeeds because one specific **subgroup** of agents did the work, and the reward for that work frequently arrives **later** than the actions that produced it. Many recent methods claim to assign credit at the subgroup/coalition level, and others claim to handle delayed rewards. After re-checking the literature up to 23 October 2026, we find that **none of them verifies, against a known correct answer, whether the right subgroup received the credit**. Coalition-aware methods are judged by win rate, return and qualitative visualisations; delayed-reward methods credit individual agents per time step and are also judged mainly by return.

This project:

1. builds environments where the coalition that created each unit of reward, and the timing of every contribution, is **planted and therefore known**;
2. defines ground-truth credit at two levels: **which coalition created the value** (Harsanyi dividends) and **what share each agent deserves** (Shapley values), under an explicitly declared counterfactual protocol;
3. measures, for standard and recent methods, **credit leakage** to bystanders, **missed credit** for early enablers, **coalition recovery**, and **credit error**, as a function of reward delay and coalition structure;
4. tests whether **team return hides these errors**; and
5. evaluates a **low-order subgroup credit estimator** whose cost grows polynomially rather than exponentially in team size.

### Verification verdict (detailed in Section 3)

| ID | Gap | Status on 23 Oct 2026 |
|---|---|---|
| G1 | Coalition credit is never graded against a known correct answer | **Open** |
| G2 | Coalition-level credit under delayed reward is unstudied | **Open** |
| G3 | Missed credit for early enablers is unmeasured in MARL | **Open, narrowed** (new related methods exist, none measures it) |
| G4 | Bystander leakage ("lazy agents") is named but never measured against planted truth | **Open** |
| G5 | "Who gets credit" is conflated with "which coalition created the value" | **Open** |
| G6 | Ground-truth credit depends on an undeclared counterfactual protocol | **Open (methodological)**, partially informed by recent work |
| G7 | Exact game-theoretic credit is exponentially expensive | **Open** |
| G8 | Team return can mask unfaithful credit | **Open**, newly documented in 2026 |
| G9 | Credit for enablers who leave before a delayed payout | **Partially addressed** (turnover methods exist; this specific case is unmeasured) |

---

## 1. Background

### 1.1 Setting

In cooperative MARL, several agents share one team reward. Each agent must learn its own policy from that shared signal, usually under centralised training with decentralised execution (CTDE). The **credit assignment problem** has two halves [1]:

- **Temporal credit assignment:** linking actions to rewards that arrive later.
- **Structural credit assignment:** attributing an outcome to the specific agents (or components) that produced it.

### 1.2 Why subgroups matter

Team reward is rarely produced by the whole team at once. MACA [2] gives a warehouse example: agent A helps agents B and C carry a fridge while agent D does nothing, and separating these contributions is what makes cooperation learnable. MACA generalises this by writing the global reward as a **sum of rewards produced by (possibly overlapping) subsets of agents** [2]. This is exactly the structure our project targets.

### 1.3 Why delay matters

Real rewards often arrive long after the actions that caused them. STAS [3] notes that standard credit methods typically fail in episodic settings where the team reward is revealed only at the end of an episode. TAR² [4] describes the double problem: identify **when** the helpful actions happened and **which agent or subset of agents** was responsible. A review of cooperation in multi-agent learning [5] notes that sparse and delayed rewards in MARL remain relatively unexplored and that the episodic-reward setting is not well understood.

### 1.4 Running example (used throughout this document)

Five agents: **A, B, C, D, E**.

| Time | Event |
|---|---|
| t = 5 | A pulls a lever that opens a gate. A is the **enabler**. |
| t = 25 | B and C jointly push a crate through the gate. They are the **finishers**. |
| whole episode | D and E wander. They are **bystanders**. |
| t = 40 | The team receives **+10** when the crate reaches the depot. This is the **delay**. |

**Correct answer.** Coalition {A, B, C} created the value, and each member was necessary. If any of A, B or C is absent, the reward is 0. Formally (Section 2.3):

- the only non-zero coalition dividend is Δ({A,B,C}) = 10;
- the individual shares are φ_A = φ_B = φ_C = 10/3 ≈ 3.33;
- φ_D = φ_E = 0.

**Three failure modes.** All three can happen while the team still earns the full +10:

- **Leakage:** D and E receive credit they did not earn.
- **Missed credit:** A receives little credit, because its action happened 35 steps before the reward and far from the crate.
- **Coalition confusion:** the method concludes that {B, C} alone, or {B, C, D}, created the value.

Team return cannot reveal any of these. That is the core motivation.

---

## 2. Problem Statement

### 2.1 Informal statement

When a cooperative team's reward is produced by a specific, possibly overlapping subgroup of agents and is delivered after a delay, current MARL credit-assignment methods may:

- give credit to agents outside that subgroup (**leakage**);
- under-credit subgroup members whose contribution came early (**missed credit**);
- misidentify which subgroup created the value (**coalition confusion**).

They may do all of this while still achieving high team return. Because credit-assignment methods are validated almost entirely on return, these failures are currently unmeasured. The problem is to:

1. define a ground truth for subgroup credit under delay;
2. measure how far existing methods deviate from it, and under which conditions;
3. test whether a lightweight subgroup-level mechanism reduces that deviation without the exponential cost of exact game-theoretic credit.

### 2.2 Formal setting

We use a cooperative Dec-POMDP with agent set $N = \{1,\dots,n\}$, a shared team reward, and episodic return $R(\tau)$ for trajectory $\tau$.

**Planted coalition-reward model.** We design environments where the return is a sum of $K$ coalition events:

$$
R(\tau) \;=\; \sum_{k=1}^{K} r_k \cdot \mathbb{1}\big[\text{event } k \text{ is completed in } \tau\big]
$$

Each event $k$ has:

- a required coalition $G_k \subseteq N$, split into **enablers** $E_k$ (must act at some time $\le t^{e}_k$) and **finishers** $F_k$ (must act jointly at time $t^{f}_k$);
- a **payout time** $t^{f}_k + d$, where $d$ is the delay;
- **bystanders** $B = N \setminus \bigcup_k G_k$.

This extends MACA's formalism, in which the global reward is a sum over rewards produced by agent subsets [2], by adding (i) an enabler/finisher time structure, (ii) an explicit delay, and (iii) planted, known coalitions.

**Controlled knobs**

| Knob | Range / values |
|---|---|
| Delay $d$ | 0 to ~50 steps |
| Enabler–finisher gap | $t^{f}_k - t^{e}_k$ |
| Number of bystanders | 0 to $n-2$ |
| Coalition overlap | whether one agent belongs to several $G_k$ |
| Synergy shape | all-or-nothing vs. partial value from partial coalitions |
| Team size | $n \le 8$ for exact ground truth; larger with Monte-Carlo estimates |

### 2.3 Ground-truth credit

**Counterfactual protocol (declared explicitly).** "Contribution" is only meaningful relative to a stated counterfactual. A 2026 survey on credit-assignment benchmarks argues the target should be a reproducible, protocol-specific contrast rather than an assumption-free "true credit" label, and calls for multi-agent tasks with designed coalition interventions [6]. Counterfactual Shapley credit assignment [7] likewise shows that attributions change with the baseline policy used for "absent" actions. We therefore define

$$
v_\pi(S) = \mathbb{E}\big[R(\tau)\;\big|\; i \in S \text{ follow } \pi_i,\; j \notin S \text{ follow a null policy } \pi^0_j\big].
$$

- **Default null policy:** the agent is removed from the episode. Because our environment is custom, true removal is implementable. This avoids a known weakness of the no-op proxy: a no-op agent can still physically obstruct others [8].
- **Sensitivity protocols:** (a) no-op, which prior work found to be the most accurate removal proxy among those it tested [8]; (b) uniformly random actions.
- **Replay mode:** closed-loop by default (agents in $S$ keep acting from their policies and may react to the absence), with open-loop replay (agents in $S$ repeat their recorded actions) as a sensitivity check.

**Which coalition created the value: Harsanyi dividends** (Möbius inversion):

$$
\Delta(S) = \sum_{A \subseteq S} (-1)^{|S|-|A|}\, v_\pi(A)
$$

**Who gets the credit: Shapley values.** A Shapley value equals the sum of an agent's coalition dividends, each split equally among that coalition's members [9]:

$$
\phi_i = \sum_{S \ni i} \frac{\Delta(S)}{|S|}
$$

**Per-event extension.** Credit can also be computed per (agent, event), so that missed enabler credit is defined precisely even when several events overlap in one episode.

**Cost.** With $n \le 8$ there are at most $2^8 = 256$ coalitions. Episodes are replayed from saved seeds in a deterministic environment, so exact values are affordable. For larger $n$, use Monte-Carlo permutation estimates with reported error bars.

### 2.4 What counts as a method's "credit" (extraction protocol)

Methods expose credit in incompatible forms, so a fixed read-out rule is part of the problem definition. Every result is reported under at least two rules.

| Method family | Raw signal | Credit read-out |
|---|---|---|
| VDN / QMIX / QPLEX | per-agent utilities inside a mixer | **Rule A:** gradient attribution (mixer gradient × utility), as used for QMIX in prior work [10]. **Rule B:** learned-model counterfactual, $Q_{tot}(s,a) - Q_{tot}(s, a^0_i, a_{-i})$. |
| COMA | per-agent counterfactual advantage | the advantage, summed over the episode or per event |
| MAPPO / MAA2C | one shared advantage | identical for all agents by construction. Many policy-gradient methods share a single global advantage across agents [11]. This is a built-in "maximum leakage" reference point. |
| STAS / TAR² | redistributed reward $\hat r_{i,t}$ | $\sum_t \hat r_{i,t}$, and per-event sums |
| NA²Q / QCoFr | explicit individual and coalition (interaction) terms | coalition terms compared with $\Delta$; individual credit compared with $\phi$ |
| MACA-style correlated sets, PRD relevant sets | groups inferred from attention | inferred group compared with planted $G_k$ |

All credits are normalised to shares, $\tilde c_i = \hat c_i / \sum_j |\hat c_j|$, before comparison with $\tilde\phi_i = \phi_i / \sum_j |\phi_j|$.

### 2.5 Metrics

| Metric | Definition | Correct value |
|---|---|---|
| Credit error | $L_1 = \sum_i \lvert \tilde c_i - \tilde \phi_i \rvert$, plus Spearman rank correlation | 0 / 1 |
| Leakage | $\Lambda = \sum_{i \in B} \lvert \tilde c_i \rvert$ (credit share given to bystanders) | 0 |
| Missed credit | $M = \max\!\big(0,\; 1 - \sum_{i\in E}\tilde c_i \,/\, \sum_{i\in E}\tilde\phi_i\big)$ | 0 |
| Coalition recovery | Jaccard similarity between the method's identified group $\hat G_k$ and the true $G_k$ | 1 |
| Dividend error | $\sum_{\lvert S \rvert \le k} \lvert \hat\Delta(S)-\Delta(S) \rvert$, for methods that expose coalition values | 0 |
| Return–faithfulness link | Spearman correlation, across methods, seeds and configurations, between normalised return and $1 - L_1$ | measured |
| Cost | critic/value evaluations per update; wall-clock time | lower is better |

### 2.6 Research questions

- **RQ1 (existence).** Do standard methods misassign credit in planted, delayed coalition tasks, even when they reach near-maximal return?
- **RQ2 (drivers).** How do leakage, missed credit and coalition recovery change with delay, enabler–finisher gap, number of bystanders and coalition overlap?
- **RQ3 (who vs. which).** Are individual-level metrics sufficient, or are there coalition-level errors they hide?
- **RQ4 (cheap fix).** Can a low-order subgroup estimator reduce coalition credit error at matched compute, without hurting return?
- **RQ5 (optional stress test).** Do these failures worsen when enablers leave the team before the delayed reward arrives?

### 2.7 Problem statement (final form)

> Given a cooperative multi-agent task in which team reward is produced by specific, possibly overlapping subgroups and delivered after a delay, and given a declared counterfactual protocol that defines ground-truth coalition credit (Harsanyi dividends) and individual credit (Shapley values), we aim to (i) quantify how far the credit signals of existing MARL methods deviate from this ground truth, via leakage, missed enabler credit, coalition recovery and credit error, as a function of delay and coalition structure; (ii) determine whether this deviation is visible in team return; and (iii) evaluate whether a low-order subgroup credit estimator reduces the deviation at polynomial rather than exponential cost.

---

## 3. Verified Research Gaps

### 3.0 How the gaps were verified

For each gap we searched specifically for 2023–2026 work that could close it (arXiv, NeurIPS, ICML, ICLR, AAMAS, AAAI, AISTATS, IJCAI, RLC), then read the closest papers' evaluation sections, not just their abstracts.

**Status labels**

| Label | Meaning |
|---|---|
| **Open** | No work found that closes the gap. |
| **Open, narrowed** | Related work now exists; the specific gap as stated here remains. |
| **Partially addressed** | Some components are solved; an identifiable piece remains. |
| **Closed** | Solved; would not be claimed by this project. |

Absence of evidence is not proof of absence. A final OpenReview check (e.g. ICLR 2027 submissions) is recommended before any publication claim.

---

### G1. Coalition credit is never graded against a known correct answer. **Status: Open**

**Claim.** Methods that assign credit to subgroups or coalitions are validated by return, win rate or qualitative visualisation. None compares its identified coalitions or coalition credit with a planted, known coalition structure.

**Evidence: what each closest method actually checks**

| Method | Venue / date | What it does with groups | How its groups/credit are validated |
|---|---|---|---|
| Nucleolus credit [12] | AAMAS 2025 | Splits agents into multiple small coalitions instead of one grand coalition | Win rate and reward; coalition formation shown as visualisations on three SMAC maps |
| CORA [11] | arXiv, v3 Mar 2026 | Allocates per-agent advantages via a regularised ε-core over coalition advantages; samples coalitions | Returns and win rates; approximation measured against its *own* full-coalition solution (violation ratio, objective gap), i.e. self-consistency, not truth |
| MACA [2] | AISTATS 2025 | Multi-level counterfactual advantages; correlated set chosen by thresholding attention weights | Win rate; one attention heatmap shown to match the map layout |
| QCoFr [13] | NeurIPS 2025 | Continued-fraction mixer exposing individual and coalition interaction terms | LBF and SMAC performance; interpretability shown qualitatively on SMAC frames (e.g. pairwise coalition weights for focus-firing agents) |
| NA²Q [14] | ICML 2023 | Generalised additive model with shape functions over coalitions | Performance on LBF and SMAC; qualitative interpretability |
| HySTAR [15] | arXiv, Sep 2026 | Anchored hypergraph for credit to agents and high-order coalitions | Win rate, success rate, episode reward |
| PRD-MAPPO [16] | RLC 2024 | Estimates each agent's "relevant set" of teammates | Closest precedent: in a task with three known teams, attention concentrated on same-team agents. Qualitative heatmap, dense rewards, relevance rather than credit |
| REFIL [17] | ICML 2021 | Group matching game with ground-truth groups | Known groups used as an *oracle* to build the method's masks, not to grade credit |

**Why it matters.** Without a correct answer, "this method assigns credit better" cannot be distinguished from "this method trains faster for some other reason".

**How we address it.** Planted coalitions with exact dividends; coalition-recovery and dividend-error metrics.

---

### G2. Coalition-level credit under delayed reward is unstudied. **Status: Open**

**Claim.** Coalition-aware methods compute credit from per-step critics under dense rewards. Delayed-reward methods redistribute return per agent and per step, not per coalition.

**Evidence**

- The Nucleolus authors note in their related work that nucleolus-based allocations fall short in dynamic environments where rewards are delayed [12].
- MACA uses SMAC's default reward, which pays for damage dealt, kills and winning, i.e. a dense signal [2].
- STAS first decomposes the return per time step, then uses Shapley values to split each step's reward among *agents* [3].
- TAR² decomposes sparse global rewards into agent-specific, time-step-specific components [4].
- CAST (NeurIPS 2024 workshop) proves identifiability of team and individual rewards in episodic MARL. It is evaluated by MPE return and causal-structure visualisation [18].
- Cooperation review: sparse/delayed MARL is relatively unexplored [5].

**Why it matters.** Under delay, a "coalition" stretches across time: enablers act early, finishers act late. Neither family represents that structure explicitly.

**How we address it.** Delay and enabler–finisher-gap knobs; per-event ground truth.

---

### G3. Missed credit for early enablers is unmeasured in MARL. **Status: Open, narrowed**

**Claim.** No work measures how much credit existing MARL credit methods give to agents whose early action enabled a later team reward, compared with a ground truth, as delay grows.

**Evidence that the problem is real**

- STAS's own toy environment, Alice & Bob, is an enabler chain. Alice must reach a key that unlocks Bob's room, then Bob reaches a key that unlocks Alice's room, and only then can both reach the treasure. It is evaluated only by average reward and treasure-reaching rate [3].
- STAS's only quantitative credit check is elsewhere: a Pearson correlation between predator credit and inverse distance to prey (STAS 0.36, SQDDPG 0.27, QMIX −0.16, COMA ≈ 0) [3]. This is a proximity proxy, which is exactly the kind of signal that under-credits enablers.

**What changed since our first pass (why "narrowed")**

| Work | What it does | Why it does not close G3 |
|---|---|---|
| MAGIC [19] (arXiv, May 2026) | Rewards an agent's *multi-step* counterfactual effect on teammates' futures (e.g. a give-way action that helps a teammate later), gated by team advantage; includes an artificial-delay study | It is an intrinsic-reward method evaluated by return and win rate. Its fidelity check (separability of predicted vs. true branch effects) validates its *own* influence estimator, not credit faithfulness. "Influence on teammates" is not "share of the delayed reward". |
| Counterfactual Shapley Credit Assignment [7] (arXiv, 2026) | Single-agent temporal credit via counterfactual Shapley values; credits a key pickup that enables a door toggle five steps later | Single-agent, tabular, simulator-based. It shows enabler credit is achievable and measurable in principle, but says nothing about MARL credit methods. |

**How we address it.** The missed-credit metric $M$, measured for every baseline across delay levels. Counterfactual Shapley's simulator-based approach is a methodological precedent for our ground truth.

---

### G4. Bystander leakage is named but never measured against planted truth. **Status: Open**

**Claim.** Credit given to agents that contributed nothing (the "lazy agent" pathology) is widely acknowledged but is not measured as a quantity against known bystanders.

**Evidence**

- The lazy-agent problem: under shared reward, an agent can receive decent team reward because teammates did the work [20].
- LAIES (ICML 2023) defines lazy agents mathematically and proposes a fix, evaluated by performance on sparse-reward SMAC and Google Research Football [21].
- The same pathology is reported in multi-agent LLM reasoning, where one agent contributes only trivially [22].
- The best quantitative-style evidence in classic MARL is anecdotal. SHAQ shows QMIX assigning the lowest credit to fighting agents and the highest to an agent escaping the battle [23].
- Agent Importance [8] measures post-hoc contributions of *trained policies* by environment reward with a no-op replacement. It does not measure the credit signal the learner itself uses.

**How we address it.** Planted bystanders; the leakage metric $\Lambda$.

---

### G5. "Who gets credit" is conflated with "which coalition created the value". **Status: Open**

**Claim.** MARL credit methods report individual credit. Shapley values are a many-to-one summary of coalition dividends, so individual credit can look correct while the coalition structure is misread. No MARL credit benchmark grades coalition-level attribution.

**Evidence**

- Harsanyi dividends answer "which coalition created the value". Shapley values split each dividend equally among its coalition's members [9].
- The only 2026 work found that uses Harsanyi dividends for multi-agent credit [24]:
  - evaluates on roundabout trajectory data with independent Q-learning;
  - lists integration with MAPPO/QMIX as future work;
  - reports gains that are very small in absolute terms.
- MACA formalises reward as a sum over subset rewards [2], and QCoFr/NA²Q expose coalition terms [13][14]. None compares these against true dividends.

**Worked counterexample (Shapley cannot tell synergy from independence)**

| Game | Coalition values | Dividends | Shapley values |
|---|---|---|---|
| 1 (pure synergy) | v(A) = v(B) = 0, v({A,B}) = 2 | Δ({A,B}) = 2 | φ_A = φ_B = 1 |
| 2 (independent) | v(A) = v(B) = 1, v({A,B}) = 2 | Δ(A) = Δ(B) = 1, Δ({A,B}) = 0 | φ_A = φ_B = 1 |

In Game 1, A and B must work together; in Game 2, they contribute independently. The Shapley values are identical. A method can therefore be "Shapley-correct" while completely misreading the team's coordination structure.

**How we address it.** Dividend-level ground truth; RQ3 / H5.

---

### G6. Ground-truth credit depends on an undeclared counterfactual protocol. **Status: Open (methodological), partially informed**

**Claim.** "Contribution" depends on how an absent agent is modelled. MARL credit papers rarely declare this, so their claims are not comparable.

**Evidence**

- The 2026 credit-assignment survey rejects assumption-free ground-truth labels in favour of protocol-specific contrasts with uncertainty intervals [6].
- Counterfactual Shapley credit assignment shows attributions differ under random vs. task-specific baselines and measure credit relative to the baseline policy [7].
- Agent Importance reports that a previous study [25] compared three agent-removal proxies (no-op, random action, another agent's action) and found no-op closest to the true Shapley value. It also notes that a no-op agent can still obstruct others, and that the computation needs copyable or resettable simulators [8].

**How we address it.** Declared protocol (true removal) plus two sensitivity protocols; closed- vs. open-loop replay check.

---

### G7. Exact game-theoretic credit is exponentially expensive. **Status: Open**

**Evidence**

- Agent Importance's timing on LBF: exact Shapley takes about 3.5 s per environment step with 10 agents and about 8,065 s (≈ 2.2 h) per step with 20 agents. The linear-time proxy takes about 0.17 s at 20 agents [8].
- Nucleolus credit enumerates excesses over all $2^n$ coalitions and lists computational efficiency as future work [12].
- CORA lists more scalable coalition evaluation as future work [11].
- QCoFr motivates itself by the combinatorial explosion of explicit interaction expansions such as NA²Q's [13].
- For multi-agent LLM training, Shapley sampling is described as intractable at industrial team sizes [26].

**How we address it.** RQ4 / H6: a low-order estimator with $O(n^k)$ terms ($k \le 3$).

---

### G8. Team return can mask unfaithful credit. **Status: Open, newly documented**

**Evidence**

- **TOCA** (Oct 2026), controlled turnover benchmark with known credit components [27]:

  | Variant | Return | ShockMAE (turnover-event credit error, lower is better) |
  |---|---|---|
  | TOCA without centering | ≈ 374 | 3.63 |
  | Full TOCA | ≈ 341 | 2.14 |

  The variant with the higher return assigns credit less faithfully. Component errors are reported only for TOCA's own decomposition heads, not for its MAPPO baselines.
- An August 2026 audit of LLM agents (single-agent) found that step-level credit signals used in training rank steps no better than their own shuffled controls against executed-replay ground truth [28].
- The 2026 survey explicitly calls for credit-assignment benchmarks with designed multi-agent credit structure [6].

**How we address it.** The return–faithfulness metric; H1.

---

### G9. Credit for enablers who leave before a delayed payout. **Status: Partially addressed**

**Evidence**

- Openness (agents joining or leaving) has been shown to cause credit misattribution, visible as unstable losses and degraded performance [1].
- TOCA separates action, pure-turnover and interaction effects in open teams, but assumes turnover is exogenous and grades faithfulness only for its own heads [27].
- MARS-RA builds a benchmark where agents leave and re-enter. Its primary metric is task success rate; dense per-agent rewards are used only to score the LLM's pairwise comparisons [29].

**Remaining piece.** Missed credit for enablers who depart between their action and the delayed payout, measured against ground truth.

**How we address it.** RQ5 / H7, as an optional stress test.

---

### 3.10 What is *not* a gap (do not claim these)

| Already done | By |
|---|---|
| Improving return with agent-temporal reward redistribution | STAS [3], TAR² [4], CAST [18] |
| Improving win rate with grouping or coalition structures | VAST [30], GoMARL [31], HYGMA [32], Nucleolus [12], CORA [11], MACA [2], HySTAR [15] |
| A linear-time proxy for individual contribution, validated against Shapley values on LBF | Agent Importance [8] |
| Shaping rewards for delayed teammate influence | MAGIC [19] |
| Turnover-orthogonal credit decomposition | TOCA [27] |

Our novelty is the **ground-truth measurement** of subgroup credit under delay across these families, plus a cheap estimator judged on that measurement.

---

## 4. Hypotheses

Numeric thresholds are proposed pre-registration values; fix them before running experiments.

### H0 (meta)

Every quantity in Section 2.5 can be measured reliably: ground-truth variance across replays and extraction-rule disagreement are small relative to between-method differences. If H0 fails, the benchmark itself must be fixed before H1–H7 are interpreted.

### H1. Return–credit decoupling (gaps G1, G8)

- **Statement.** Among runs that reach at least 90% of maximum return, credit error varies widely across methods. Return is a weak predictor of credit faithfulness.
- **Rationale.** TOCA's ablation shows high return can coexist with unfaithful credit [27].
- **Prediction.** Spearman correlation between return and $1-L_1$ across methods and seeds has |ρ| < 0.3, and at least two methods with statistically indistinguishable returns differ significantly in $L_1$.
- **Falsified if** |ρ| ≥ 0.6 consistently. Return would then be a reasonable proxy, which is itself a useful finding.

### H2. Shared-signal methods leak credit to bystanders (G4)

- **Statement.** MAPPO, VDN and QMIX assign non-trivial credit to planted bystanders. Leakage grows with the number of bystanders. Counterfactual methods (COMA) leak less at zero delay.
- **Rationale.** MAPPO shares a single advantage [11]. Additive and monotonic mixers spread value across utilities: the lazy-agent problem [20].
- **Prediction.**
  - MAPPO: Λ approaches the bystanders' share of the team.
  - VDN and QMIX: Λ > 0.1, increasing with the number of bystanders.
  - COMA: Λ significantly lower than MAPPO at d = 0.
- **Falsified if** Λ < 0.05 for all methods.

### H3. Delay produces missed enabler credit (G2, G3)

- **Statement.** Missed enabler credit $M$ increases with delay $d$ and with the enabler–finisher gap. Temporal redistribution (STAS, TAR²) reduces $M$ relative to per-step methods but does not eliminate it.
- **Rationale.** Per-step critics and proximity-correlated credit [3] tie credit to when and where the reward appears.
- **Prediction.** $M$ rises monotonically with $d$ for value-decomposition methods and MAPPO. For STAS and TAR², $M$ is below the others but above 0.2 at the largest $d$.
- **Falsified if** $M$ is flat in $d$, or close to 0, for all methods.

### H4. Coalition recovery degrades under delay (G1, G2)

- **Statement.** Coalition-aware read-outs recover planted coalitions above chance at $d = 0$, and recovery falls toward chance as $d$ grows. The read-outs are NA²Q/QCoFr coalition terms, MACA-style correlated sets, and PRD relevant sets.
- **Rationale.** These read-outs are inferred from per-step critics or attention trained under dense signals.
- **Prediction.** Jaccard similarity is clearly above chance at $d = 0$ and declines significantly by the largest $d$.
- **Falsified if** recovery is unaffected by $d$. That would be a positive finding for those methods.

### H5. Individual credit can look right while coalition attribution is wrong (G5)

- **Statement.** With overlapping coalitions or partial synergy, a method can have low individual credit error but high dividend error.
- **Rationale.** Shapley values are a many-to-one summary of dividends (Section 3, G5 counterexample).
- **Prediction.** At least one method–configuration pair has $L_1$ in the best tercile but dividend error in the worst tercile.
- **Falsified if** individual and coalition errors are tightly correlated (ρ > 0.8) across all configurations. Individual metrics would then suffice.

### H6. A low-order subgroup estimator reduces error cheaply (G7)

- **Statement.** A candidate mechanism, compared with baselines at matched compute:
  - reduces dividend error, leakage and missed credit;
  - keeps return within 5% of the best baseline;
  - has cost that grows as $O(n^k)$ rather than $2^n$.
- **Candidate mechanism.**
  1. Learn an episodic return model whose terms are restricted to coalitions of size $k \le 2$ or $3$.
  2. Read out low-order dividends.
  3. Allocate them to members, normalised so the redistributed rewards sum exactly to the episode return (as TAR² does to preserve the optimal policy [4]).
- **Falsified if** there is no significant improvement in coalition metrics at matched compute. The project still stands on H1–H5.

### H7 (optional). Departed enablers lose credit (G9)

- **Statement.** An enabler that leaves between its action and the payout receives more missed credit than one that stays, and the gap grows with delay.
- **Falsified if** there is no difference in $M$ between departed and present enablers.

### Hypothesis summary

| | Gaps | Main metric | Key manipulation |
|---|---|---|---|
| H1 | G1, G8 | return vs. $L_1$ | across methods and seeds |
| H2 | G4 | Λ | number of bystanders |
| H3 | G2, G3 | $M$ | delay $d$, enabler gap |
| H4 | G1, G2 | Jaccard | delay $d$ |
| H5 | G5 | $L_1$ vs. dividend error | overlap, synergy |
| H6 | G7 | all metrics + cost | method |
| H7 | G9 | $M$ | departure before payout |

**What counts as success.** Each hypothesis, confirmed or rejected, is a finding. The project fails only if H0 fails, i.e. the quantities cannot be measured reliably.

---

## 5. Experimental Design Sketch

### 5.1 Environments

| ID | Environment | Purpose |
|---|---|---|
| E1 | **Planted Coalition Gridworld** (custom) | Full control of $G_k$, enablers/finishers, delay $d$, bystanders, overlap, synergy; supports true agent removal and deterministic replay |
| E2 | **Level-Based Foraging (LBF) variant** | Familiar benchmark. Add delayed payout, planted bystanders, and fixed agent levels (as in Agent Importance's deterministic LBF variant with levels 1, 2, 3 [8]). LBF also provides per-agent rewards as a secondary reference. |
| E3 | **Alice & Bob-style enabler chain** | Reuses STAS's enabler-chain structure [3] with ground truth added |

### 5.2 Baselines and code availability

| Method | Family | Code source (verified) |
|---|---|---|
| IQL, VDN, QMIX, COMA, MAA2C, MAPPO | value decomposition / policy gradient | **EPyMARL** [33]. COMA, VDN, QMIX and QTRAN support common-reward settings only, which suits our shared-reward design. |
| QPLEX | value decomposition | **Not in EPyMARL's algorithm list.** Use PyMARL2 or NA²Q's PyMARL fork, which lists QPLEX [14]. |
| STAS | agent-temporal redistribution | public repo [3] |
| TAR² | agent-temporal redistribution | public repo [4] |
| NA²Q | coalition interaction terms | public repo [14] |
| HYGMA | dynamic grouping | public repo [32] |
| MACA, CORA, QCoFr, MAGIC | coalition / influence methods | public code not confirmed; implement only if time permits |

### 5.3 Protocol

- At least 5 seeds per configuration; matched training budgets; EPyMARL defaults where available.
- Credit evaluated on trajectories from the trained policy; ground truth computed on the same trajectories by replay.
- Report means with 95% confidence intervals and paired tests.

---

## 6. Scope, Assumptions and Non-Goals

**In scope**

- Fully cooperative tasks with a shared team reward.
- $n \le 8$ for exact ground truth; Monte-Carlo estimates beyond that.
- Planted environments and an LBF variant.

**Assumptions**

- Delays and turnover are external, not caused by the agents' own actions.
- Ground truth is defined relative to the declared counterfactual protocol.
- Credit is evaluated on trajectories generated by the trained policy.

**Non-goals**

- Not claiming new state-of-the-art results on SMAC or Google Research Football.
- Not resolving which counterfactual protocol is philosophically correct.
- Not addressing mixed-motive or competitive settings.

---

## 7. Positioning Against the Closest Work

| Work | Delayed reward | Subgroup-level | Graded against ground-truth credit |
|---|---|---|---|
| STAS (AAAI 2024) [3] | ✓ | ✗ individual Shapley per step | ✗ distance proxy |
| TAR² (arXiv 2025) [4] | ✓ | ✗ | ✗ |
| CAST (NeurIPS 2024 workshop) [18] | ✓ | ✗ individual rewards | ✗ visualisation |
| MACA (AISTATS 2025) [2] | ✗ dense SMAC | ✓ | ✗ heatmap |
| CORA (arXiv 2026) [11] | partial | ✓ | ✗ self-consistency only |
| Nucleolus (AAMAS 2025) [12] | ✗ | ✓ | ✗ visualisation |
| NA²Q / QCoFr (ICML 2023 / NeurIPS 2025) [13][14] | ✗ | ✓ interaction terms | ✗ qualitative |
| PRD-MAPPO (RLC 2024) [16] | ✗ | ✓ relevant sets | partial: qualitative, known teams |
| MAGIC (arXiv 2026) [19] | ✓ influence | ✗ | partial: own estimator vs. true dynamics |
| Agent Importance (AAAI 2024 workshop) [8] | ✗ | ✗ | ✓ post-hoc, trained policies |
| TOCA (arXiv 2026) [27] | turnover only | ✗ | ✓ own heads only |
| **This project** | ✓ | ✓ | ✓ dividends and Shapley values, all methods |

---

## 8. Threats to Validity

| Threat | Mitigation |
|---|---|
| Credit-extraction ambiguity | At least two read-out rules per method; conclusions must hold under both |
| Counterfactual choice | Three protocols; closed- vs. open-loop replay |
| Synthetic environments may not generalise | Add the LBF variant; keep E1 minimal so failures can be traced to causes |
| Training noise | ≥ 5 seeds, confidence intervals, paired tests |
| Ground-truth noise | Deterministic replay for exact $v(S)$; Monte-Carlo error bars otherwise |
| Unfair baselines | Matched budgets; EPyMARL defaults; published code where available |
| Recency of evidence | Several key sources are 2026 arXiv preprints (TOCA, MAGIC, HySTAR, Counterfactual Shapley, CORA v3); re-check before final write-up |

---

## 9. Changes Relative to the Submitted One-Pager

- **Stage 1** becomes the ground-truth benchmark (H0–H5). **Stage 2** becomes the low-order estimator (H6). Agent turnover becomes optional (H7).
- **Baselines** expand beyond VDN, QMIX, QPLEX and COMA to include MAPPO, STAS, TAR², NA²Q and PRD-MAPPO (if code allows).
- **Correction:** QPLEX is not part of EPyMARL; source it from PyMARL2 or NA²Q's repository.
- **Primary metrics** gain dividend error and coalition recovery alongside leakage and missed credit.
- **Related work** must now cite MAGIC (delayed teammate influence) and Counterfactual Shapley (single-agent enabler credit) and explain why they do not close G3.

---

## 10. Verification Log

| Item checked | Finding | Source |
|---|---|---|
| QCoFr evaluation | LBF and SMAC performance; coalition interpretability qualitative only; no ground-truth comparison | [13] |
| Counterfactual Shapley Credit Assignment | Single-agent, tabular, simulator-based; baseline-dependent attributions; credits enabling actions in DoorKey | [7] |
| MAGIC | Delayed teammate influence as intrinsic reward; evaluated by return/win rate; estimator-separability diagnostic | [19] |
| CAST | Episodic MARL identifiability; MPE return and visualisation | [18] |
| LAIES | Lazy agents defined mathematically; performance-evaluated | [21] |
| Agent Importance details | No-op removal; Shapley timing table; deterministic LBF levels 1/2/3; correlation ≈ 0.96–0.97 with Shapley | [8] |
| PRD-MAPPO | Relevant-set heatmap matches known teams (qualitative) | [16] |
| STAS details | Alice & Bob enabler chain; proximity-correlation credit check | [3] |
| EPyMARL algorithm list | IQL/COMA/VDN/QMIX/QTRAN plus IA2C/IPPO/MADDPG/MAA2C/MAPPO/PAC; **no QPLEX** | [33] |
| MACA, CORA public code | Not confirmed | — |
| HIVE (Neural Networks 2026), a hypergraph game-theoretic value decomposition | Seen only as a citation in [24]; contents **not verified** | — |
| Worked examples (running example, G5 counterexample, G7 timing) | Recomputed in code by exact enumeration: Δ({A,B,C}) = 10 is the only non-zero dividend; φ = 10/3 for A, B, C and 0 for D, E; Shapley-from-dividends identity holds; both G5 games give φ_A = φ_B = 1; 8,065 s = 2.24 h; 2⁸ = 256 coalitions | this document |

---

## 11. References

Status in brackets: peer-reviewed venue or preprint.

1. Abadi, A. S., & Soh, L.-K. *Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems.* [arXiv 2510.27659, 2025] https://arxiv.org/abs/2510.27659
2. Zhao, X., & Xie, Y. *Multi-level Advantage Credit Assignment for Cooperative MARL (MACA).* [AISTATS 2025, PMLR 258] https://arxiv.org/abs/2508.06836
3. Chen, S., Zhang, Z., Yang, Y., & Du, Y. *STAS: Spatial-Temporal Return Decomposition for MARL.* [AAAI 2024] https://arxiv.org/abs/2304.07520 · code: https://github.com/zowiezhang/STAS
4. Kapoor, A., et al. *TAR²: Redistributing Rewards Across Time and Agents for MARL.* [arXiv 2502.04864] https://arxiv.org/abs/2502.04864 · code: https://github.com/AdityaKapoor74/MARL_Agent_Temporal_Credit_Assignment
5. Du, Y., Leibo, J. Z., Islam, U., Willis, R., & Sunehag, P. *A Review of Cooperation in Multi-agent Learning.* [arXiv 2312.05162, 2023] https://arxiv.org/abs/2312.05162
6. *From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models.* [arXiv 2604.09459, 2026] https://arxiv.org/abs/2604.09459
7. Li, M., Lee, K.-Z., & Bareinboim, E. *Counterfactual Shapley Credit Assignment.* [arXiv 2607.16999, 2026] https://arxiv.org/abs/2607.16999
8. Mahjoub, O., et al. *Efficiently Quantifying Individual Agent Importance in Cooperative MARL.* [AAAI 2024 workshop; arXiv 2312.08466] https://arxiv.org/abs/2312.08466
9. Harsanyi dividends and Shapley values (explainer): https://domezsolt.substack.com/p/when-to-use-harsanyi-dividends-in
10. *Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition (CIA).* [AAAI 2023] https://arxiv.org/abs/2211.12712
11. Ji, M., Xu, G., et al. *Cooperative Game-Theoretic Credit Assignment for Multi-Agent Policy Gradients via the Core (CORA).* [arXiv 2506.04265, v3 Mar 2026] https://arxiv.org/abs/2506.04265
12. Li, Y., Cao, Z., Qiao, J., & Hu, S. *Nucleolus Credit Assignment for Effective Coalitions in MARL.* [AAMAS 2025] https://arxiv.org/abs/2503.00372
13. Xu, Q., Zhu, Y., Wu, X., & Chen, C. *High-order Interactions Modeling for Interpretable Multi-Agent Q-Learning (QCoFr).* [NeurIPS 2025] https://arxiv.org/abs/2510.20218
14. Liu, Z., Zhu, Y., & Chen, C. *NA²Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning.* [ICML 2023] https://proceedings.mlr.press/v202/liu23be.html · code: https://github.com/zichuan-liu/NA2Q
15. Luo, X., et al. *HySTAR: Anchored Hypergraphs for Stable Credit Assignment in Cooperative MARL.* [arXiv 2609.31531, 2026] https://arxiv.org/abs/2609.31531
16. Kapoor, A., Freed, B., Choset, H., & Schneider, J. *Assigning Credit with Partial Reward Decoupling in MAPPO.* [RLC 2024] https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_45.pdf
17. Iqbal, S., et al. *Randomized Entity-wise Factorization for MARL (REFIL).* [ICML 2021] https://arxiv.org/abs/2006.04222
18. Zhang, Y., Du, Y., Huang, B., Fang, M., & Pechenizkiy, M. *A Causality-Inspired Spatial-Temporal Return Decomposition Approach for MARL (CAST).* [NeurIPS 2024 Workshop on Causal Representation Learning] https://openreview.net/forum?id=tlNkThyt18
19. Yu, H., Wang, L., Cong, J., Wang, S., & Liu, C. *MAGIC: Multi-Step Advantage-Gated Causal Influence for MARL.* [arXiv 2605.01805, 2026] https://arxiv.org/abs/2605.01805
20. *Applications of MARL in Future Internet: A Comprehensive Survey* (lazy-agent discussion of VDN). [arXiv 2110.13484] https://arxiv.org/abs/2110.13484
21. Liu, B., et al. *Lazy Agents: A New Perspective on Solving Sparse Reward Problem in MARL (LAIES).* [ICML 2023] https://proceedings.mlr.press/v202/liu23ac.html
22. Zhang, Z., et al. *Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation.* [arXiv 2511.02303] https://arxiv.org/abs/2511.02303
23. Wang, J., Zhang, Y., Gu, Y., & Kim, T.-K. *SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning.* [NeurIPS 2022] https://arxiv.org/abs/2105.15013
24. Bouchaffra, D., Ykhlef, F., Lebbah, M., & Azzag, H. *Coalition Free Energy and Adaptive Precision in Multi-Agent Cooperation.* [arXiv 2605.26278, 2026] https://arxiv.org/abs/2605.26278
25. Heuillet, A., Couthouis, F., & Díaz-Rodríguez, N. *Collective eXplainable AI: Explaining Cooperative Strategies and Agent Contribution in MARL with Shapley Values.* [IEEE Computational Intelligence Magazine, 2022] (cited via [8])
26. *Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces.* [arXiv 2605.02801, 2026] https://arxiv.org/abs/2605.02801
27. Thakur, A., & Singhal, M. *Turnover-Orthogonal Credit Assignment for Open-Team MARL (TOCA).* [arXiv 2610.02847, Oct 2026] https://arxiv.org/abs/2610.02847
28. *Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay.* [arXiv 2608.19760, 2026] https://arxiv.org/abs/2608.19760
29. Wang, D., et al. *MARS-RA: Rank Aggregation for Credit Assignment via Multimodal Comparisons in Embodied Multi-Agent Cooperation.* [arXiv 2607.27967, 2026] https://arxiv.org/abs/2607.27967
30. Phan, T., et al. *VAST: Value Function Factorization with Variable Agent Sub-Teams.* [NeurIPS 2021] https://proceedings.neurips.cc/paper/2021/hash/c97e7a5153badb6576d8939469f58336-Abstract.html
31. Zang, Y., et al. *Automatic Grouping for Efficient Cooperative MARL (GoMARL).* [NeurIPS 2023] https://proceedings.neurips.cc/paper_files/paper/2023/hash/906c860f1b7515a8ffec02dcdac74048-Abstract-Conference.html
32. Liu, C., & Li, D. *HYGMA: Hypergraph Coordination Networks with Dynamic Grouping for MARL.* [ICML 2025] https://arxiv.org/abs/2505.07207 · code: https://github.com/mysteryelder/HYGMA
33. EPyMARL (Extended PyMARL), University of Edinburgh. https://github.com/uoe-agents/epymarl
