# Related work for v3, and what we do and do not claim

11 Nov 2026. "Not found" means not found in our searches, not proven absent. Papers marked † were read for v3, in full or in the sections cited. Everything else is cited from abstracts, prior reading, or memory and must be verified before external use.

## 1. Closest work, by question

| Work | What it does | How v3 differs |
|---|---|---|
| Wolpert & Tumer, difference rewards (2001 onward) | Agent i's reward is G minus G with i clamped to a null action | Our leave-one-out truth is this. It needs the reward function or a resettable simulator. v3 Phase 3 uses it online (shadow copies) as the *learning* arm. |
| COMA (Foerster et al., 2018); Shapley Counterfactual Credits (Li et al., KDD 2021); SQDDPG | Replace one agent's **action** inside a learned critic, for **one step** | One-step. The benchmark shows these read-outs failing, and Identity 2 (`theory.md` §5) explains why. |
| †SHAQ (Wang et al., arXiv 2105.15013) | Markov Shapley value over persistent, state-level coalition values. The coalition value is defined by the coalition's own policy (Eq. 5); learned via a Shapley-Bellman operator with sampled coalitions | Closest *definition*: coalition values are persistent. But the target is optimal coalition values in a convex game, learned action-level (Q^ϕ_i). There is no intervention data, no read-out graded against planted truth, and the outside agents' behaviour is not specified. |
| †Counterfactual Shapley Credit Assignment (Li, Lee & Bareinboim, arXiv 2607.16999) | Shapley credit over **time steps** of a **single agent**. An intervened step takes a baseline-policy action at that step only (Eq. 7); computed by simulation | Single-agent, one-step replacement: the opposite corner from v3 (multi-agent, persistent stop). Its causal (SCM) treatment of realised vs expected counterfactuals parallels our commitment split (`theory.md` §3). |
| †C3 / "Exact Is Easier" (Chen et al., arXiv 2603.06859) | Measures per-decision credit for LLM agent teams by restoring a checkpoint and re-running. Proposition 1: **removing** an agent shifts teammates' observation distribution, a bias that more samples cannot remove | Same philosophy: measure counterfactuals instead of trusting critics. Their Proposition 1 is our point that the null is a modelling choice (no-op vs removal, `errata.md` E8, Test A). C3 replaces only the evaluated action; v3 learns a critic over persistent stops. |
| CCPO (arXiv 2603.21563), Agents that Matter (arXiv 2605.27621) | Whole-episode removal baselines for LLM teams | Removal-based, but not learned from intervention data and not graded against planted truth. |
| †Probabilistic agent dropout (arXiv 2304.12458) | Post-dropout value estimated from **pre-dropout** samples by importance sampling | Possible only under factored transitions, separable rewards, and post-dropout dynamics that **marginalise** the removed agents (Assumptions 1–4), i.e. teammates that do not react. Our coalition teams react (finishers give up or wait), which is exactly why interventions are necessary (`theory.md` §6). |
| †Fault tolerance in MARL (arXiv 2412.00534) | Critic receives a fault flag and zeroed action for a failed agent; attention down-weights it; random faults injected during training | Closest *mechanism* (critic input flags a stopped agent). It is used for robustness, not credit. No per-agent contribution measure is defined. |
| Mahjoub et al., Agent Importance (arXiv 2312.08466); Heuillet et al., Collective XAI (2110.01307) | Leave-one-out or Shapley values by re-running trained policies | Post-hoc, needs the simulator, not a learned estimator. |
| Triantafyllou et al., agent-specific effects (2023, ICML 2025) | Causal decomposition of one action's effect through other agents | Explanation in an SCM; not a learned credit read-out graded against planted truth. |
| STAS (Chen et al., AAAI 2024) | Temporal return decomposition, then Shapley over agents per step | Learns from return-only data, so it cannot identify removal credit when returns are constant (`errata.md` E2). |
| RUDDER (Arjona-Medina et al., 2019) | Redistributes return to the steps where the predicted return changes | Single-agent ancestor of "credit at the step where the prediction changes". Our commitment point (`theory.md` §4) is its multi-agent, coalition-lattice analogue: credit moves when the team's *dependence on a coalition* changes. |
| †Mesnard et al., Counterfactual Credit Assignment (ICML 2021, arXiv 2011.09464) | Future-conditional baselines V(x, Φ). Unbiased if the hindsight statistic Φ is independent of the action given the state (Thm 2) | The theory behind our claim that only **realised** counterfactuals can split luck. Their condition is the one a hindsight credit model must satisfy (v3 `theory.md` §3, §7; Phase 3). |
| Ng, Harada & Russell (1999) | Potential-based shaping preserves optimal policies | κ_i = team reward + potential difference (Φ = W_{−i}), so expected-value credit cannot change the optimum (`theory.md` §7). |
| Devlin et al., AAMAS 2014 (CaP, DRiP) | Counterfactual-as-potential shaping | Closest older idea to "ablation value as potential". Their potential needs the system utility; ours is learned from intervention data. |

## 2. Uploaded papers, and whether they are used

Read for v3: the abstracts and method sections of each.

| Paper | Idea | Used in v3? |
|---|---|---|
| †Stepwise-Flow-GRPO (Savani et al., arXiv 2603.28718) | Step credit g_t = r̂_{t−1} − r̂_t from intermediate reward estimates (Tweedie predictions), telescoping to the final reward | **As a parallel, not as a component.** Its gain is a potential difference of a progress estimate. Our κ is the same construction with a *counterfactual, agent-specific* potential (W_{−i}). The team-level case (S = N) is exactly the team TD error δ_t. We cite it to show the structure is general; we add the agent dimension. |
| †GraphGPO (Cheng et al., arXiv 2605.26684) | Pools all rollouts into one state-transition graph; step credit is the reduction of a graph distance-to-goal | **As motivation for pooling across trajectories.** Our critic is queried at states pooled from factual and stopped trajectories, a parametric version of the same idea. Not combined: our states are not discrete enough for a graph to share nodes across stopped trajectories. |
| †DACA-GRPO (Monsefi et al., arXiv 2605.16342) | Stratified masking: positions are split into strata so each is evaluated with most of the context | **One design choice.** The *stratified stop-time sampler* (half before completion, half in the payout wait) is the same principle: guarantee coverage of a stratum the default sampling misses. Round 2 showed the missing stratum (payout wait) cost per-step fidelity (r 0.69 → 0.998). |
| †Natural emergent misalignment from reward hacking (MacDiarmid et al., Anthropic 2025) | Learning to exploit a mis-specified reward generalises to broadly misaligned behaviour | **Motivation for auditing, not a method.** It shows why *who earned the reward* matters beyond return: if credit rewards the wrong behaviour (a free-riding bystander, a hacked shortcut), the learned behaviour generalises. Persistent-ablation credit is an audit tool that can flag agents credited without being needed (Test D). We make no claim about LLMs. |

## 3. What we claim, and what we do not

**We claim** (subject to the v3 confirmatory results):
1. A planted-truth benchmark with two-level grading, and closed-loop truth as the default.
2. Identities, all exact and tested:
   - why one-step read-outs cannot recover removal credit (Identity 2);
   - persistent ablation-TD credit sums to removal credit (Identity 1);
   - it splits into shared surprise plus structural credit;
   - its per-step Shapley values and dividends sum to the episode's, with credit at commitment rather than payout.
3. Intervention data (agents stopping mid-episode) makes removal credit learnable by an ordinary value critic, shown against exact truth with a no-intervention control and a table baseline.
4. With a critic that respects the relabelling symmetry, the same data generalises across role assignments and recovers coalition structure.
5. An honest answer to whether credit accuracy changes learning here (Phase 3).

**We do not claim:**
- that persistent ablation is new as a *definition* (it is difference rewards / leave-one-out with a persistent null);
- that a set/attention critic is a new architecture (DeepSets, attention critics such as MAAC exist);
- that ablation credit improves return.

**Not found in our searches:** a value critic trained on mid-episode persistent-stop data, read out per step as ablation-TD credit, graded against exact planted truth in classical cooperative MARL. Closest: SHAQ (definition), 2412.00534 (mechanism), C3 (measurement).

## Sources

- [SHAQ, arXiv 2105.15013](https://arxiv.org/abs/2105.15013)
- [Counterfactual Shapley Credit Assignment, arXiv 2607.16999](https://arxiv.org/abs/2607.16999)
- [Exact Is Easier / C3, arXiv 2603.06859](https://arxiv.org/abs/2603.06859)
- [Counterfactual Credit Policy Optimization, arXiv 2603.21563](https://arxiv.org/abs/2603.21563)
- [Agents that Matter, arXiv 2605.27621](https://arxiv.org/html/2605.27621v1)
- [Probabilistic agent dropout, arXiv 2304.12458](https://arxiv.org/abs/2304.12458)
- [Fault tolerance in MARL, arXiv 2412.00534](https://arxiv.org/abs/2412.00534)
- [Counterfactual Credit Assignment in Model-Free RL, arXiv 2011.09464](https://arxiv.org/abs/2011.09464)
- [Shapley Counterfactual Credits, arXiv 2106.00285](https://arxiv.org/abs/2106.00285)
- [STAS, arXiv 2304.07520](https://arxiv.org/abs/2304.07520)
- [Agent Importance, arXiv 2312.08466](https://arxiv.org/abs/2312.08466)
- [MACD, AAMAS 2024](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf)
- Uploaded: arXiv 2603.28718, 2605.26684, 2605.16342; MacDiarmid et al. (Anthropic, 2025).
