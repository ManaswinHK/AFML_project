# Progress Review: Slide Contents

**Project:** Who Earned the Reward? Ground-truth evaluation of subgroup credit assignment under delayed coalition rewards in cooperative MARL

Four slides: progress, results, issues, plan. Results are labelled **confirmatory** (pre-registered rule applied as written) or **post-hoc** (found after seeing the data). Sources: `ROADMAP.md`, `docs/results_overview.md`, `docs/main_results.md`, `docs/h3_results.md`.

---

## Slide 1: Progress so far

**Title:** Progress: the benchmark is built, and 7 of 9 roadmap steps are done

**Project in one line.** We test whether MARL credit assignment (VDN, QMIX, COMA, MAPPO, STAS, NA²Q) credits the agents that actually earned a delayed team reward. We grade it against exact ground truth in a planted environment.

**Gap we fill** (problem statement v2.1, Section 3). Ground-truth credit checks in MARL exist only per agent or per component, and only for each paper's own method. Nobody grades *standard* learners' credit at the **coalition level under delay**, or checks whether their credit read-outs are valid at all (F1–F6).

**Pipeline (diagram, left to right, each box ticked):**
1. Planted Coalition Gridworld: enabler, then finisher, then delayed payout, plus bystanders
2. Scripted agents
3. Replay with agents removed, giving v(S) for all 2ⁿ subsets
4. Harsanyi dividends, Shapley values, leave-one-out marginals
5. Metrics: L1, leakage, missed credit, coalition recovery
6. Method read-outs

**Roadmap status**
- Done, Tier 1:
  - Step 1: ground-truth calculator
  - Step 2: environment
  - Step 3: scripted policies and replay
  - Step 4: metrics
  - Step 5: EPyMARL integration
  - Step 6: Mode P experiments
- Done, Tier 2: Step 7 (ported NA²Q and STAS, plus a coalition-level read-out).
- Not started: Step 8 (Mode T, full training), Step 9 (sensitivity analyses), optional H5 estimator.

**Experiments run**
- Main grid: 120 cells × 5 seeds, about 20 CPU-hours, 4 methods.
- H3 grid: 30 cells × 6 methods, 87 minutes.
- Two pilots that exposed and fixed two method problems.

**Speaker notes**
- The ground truth is exact and unit-tested. It reproduces the running example (Δ({A,B,C}) = 10, φ = 10/3).
- Thresholds were fixed after the pilot and before the main runs (pre-registration), which is why the negative results are credible.
- NA²Q and STAS are clean reimplementations, because the official repositories have no licence.

---

## Slide 2: Results so far

**Title:** Results: low credit error is not the same as correct credit

**Pre-registered verdicts (compact table, left)**

| Hypothesis | Verdict |
|---|---|
| H0 ground truth is stable | ✗ narrowly (104/120 cells pass) |
| H1a missed enabler credit grows with delay | ✗ (VDN does not converge; QMIX has the right sign, Holm p = 0.074) |
| H1b slack removes enabler credit | slack half ✓ 24/24; critical half ✗ (weak manipulation) |
| H2 leakage grows with bystanders | COMA ✓ 8/8; VDN and QMIX 1/8 each |
| H2′ COMA leaks less than VDN/QMIX | ✗ reversed (COMA leaks most) |
| H3 individual metrics hide coalition errors | ✗ literally, and uninformative |
| H5 accuracy does not imply fidelity | ✓ 30 of the 32 best cells |
| H4 return is a weak proxy for credit | not yet run (Mode T) |

**Four headline findings (right)**

1. **"Credit" from these methods is a per-step counterfactual, not "who earned the reward."** Even a perfect per-step critic gives a slack enabler about zero credit, because the scripted team recovers from a one-step deviation.
2. **Good scores can be accidental.** QMIX at d = 0, b = 2 is the best end to end (L1 0.44), yet it barely matches its own per-step target (step r 0.14). Level-1 fidelity is r ≤ 0.36 for every method.
3. **No method beats a uniform split on the H3 layouts** (post-hoc). Uniform L1 is 0.45 on overlap and 0.89 on partial synergy, and no method's mean beats it in any of 36 conditions. Any credit benchmark should report a uniform baseline.
4. **Per-method behaviour**
   - COMA's counterfactual baseline leaks the most.
   - MAPPO's shared advantage makes credit read-outs meaningless (Λ = b/n by construction).
   - STAS spreads credit almost uniformly, an identifiability limit and not undertraining.
   - The mixer games for VDN, QMIX and NA²Q cannot see coalitions: interaction mass is 0.000–0.002 against a true 1.0.

**Speaker notes**
- Each ✗ comes with a mechanism: weak manipulation (H1b), a structurally additive read-out (H3), a method that cannot see the delay (VDN). Present them as findings, not failures.
- Tag as post-hoc: the uniform baseline, the MAPPO exclusion, and the H1b split by realised criticality.
- Optional chart: L1 per method against the uniform-baseline line.

---

## Slide 3: Issues and blockers

**Title:** Issues: what we fixed, what is still open

**Resolved**
- The critic state was not Markov under delay. We added a payout countdown, which fixed QMIX, COMA and MAPPO. VDN cannot use it.
- NA²Q's official max target diverges on fixed data (Q_tot ≈ 115–140 against a true ≈ 8). We switched to a SARSA target for Mode P.
- Per-step read-outs are not removal credit. We added two-level grading (fidelity and semantic gap).

**Open issues**

| Issue | Why it matters | Response |
|---|---|---|
| The per-step mixer game cannot express cross-time coalitions, so H3 is uninformative | No usable coalition-level result yet | Build the H5 episode-level estimator, the fair test |
| The critical-enabler manipulation is weak (only 54% critical) | H1a and H1b's critical half are underpowered | Report with the realised-criticality split |
| 5 seeds and Holm correction over up to 32 conditions | Low power (VDN/QMIX H2 looks like a power failure) | State as a limitation |
| Partial-synergy data realises its planted structure in only 40–50% of episodes | Weakens H3 | State as a limitation |
| Mode T may not learn: 3k-step online smoke runs returned 0, because random exploration rarely coordinates the lever and the joint push | H4 depends on it | Time one longer run first; restrict to pilot-validated configurations; report as a descriptive pilot |
| Uniform baseline not yet run on the main grid | Unknown whether main-grid L1 carries information | A cheap re-analysis |
| NA²Q and STAS are our reimplementations, compared line by line but not run against the official code | Credibility | State as a limitation |

**Speaker notes**
- Lead with the first row. It is what turns "H3 failed" into "here is the experiment that tests it properly."

---

## Slide 4: Timeline and plan to completion

**Title:** Plan: five phases to finish within two weeks

| Phase | Work | Output |
|---|---|---|
| **1. Close out Mode P** | Run the uniform baseline on the main grid (a re-analysis, no new training). | Revised main-grid L1 table |
| **2. Coalition estimator (H5)** | Build the low-order episodic coalition estimator. Test it on the overlap and partial layouts, including coalitions of size k+1. | The coalition-level result that replaces the uninformative H3 |
| **3. Full training (Mode T)** | Launch early and run in the background while H5 is built. Use 2 pilot-selected configurations and 3 seeds. Then analyse H4. | A descriptive return-vs-credit result |
| **4. Sensitivity checks** | Compare Shapley with leave-one-out, and run an open vs closed-loop replay check. | A robustness section |
| **5. Write-up** | Report, figures, final presentation, plus buffer. | Final deliverables |

**Cut to fit two weeks**
- More seeds on key cells: dropped. VDN/QMIX H2 stays "indicative, underpowered".
- Stronger critical-enabler manipulation: dropped. We report H1b with the realised-criticality split.
- Fixing the partial-synergy layout: dropped. It is a stated limitation.
- Sensitivity analyses: cut from three checks to two. The no-op vs removal check is skipped unless time remains.
- Mode T: reduced in scope, so H4 is a descriptive pilot, not a confirmatory test.

**Risks**
- The critical path is H5 plus launching Mode T early. If Mode T cannot start early, or does not learn, H4 is reported as not completed. The 3k-step smoke runs returned 0.
- Mode T compute is the main unknown. Time one run first and shrink the scope right away if it is too slow.
- The core contribution already stands without H5 and Mode T. The benchmark plus the Mode P findings are complete, so a slip only costs the extras.
