# Who Earned the Reward? Results and Inferences So Far

A summary of every result to date (Steps 1–7), as of 27 October 2026. Details and full tables are in the linked documents. Each result is labelled:
- **confirmatory**: a pre-registered rule, applied as written;
- **post-hoc**: done after seeing the data; hypothesis-generating only.

| Source | Covers |
|---|---|
| `who_earned_the_reward_problem_statement_v2.md` | question, hypotheses H0–H5, metrics |
| `docs/environment_design.md` | Planted Coalition Gridworld |
| `docs/pilot_findings.md` | pilots 1–2, method fixes, thresholds |
| `docs/preregistration_amendment.md`, `docs/mode_p_protocol.md` | statistics and protocol |
| `docs/main_results.md` | main grid (120 cells): H0, H1a, H1b, H2, H2′, H5 |
| `docs/step7_ports.md` | NA²Q and STAS ports |
| `docs/h3_results.md` | H3 grid (30 cells, six methods) |

---

## 1. Setup in brief

**Question.** When a cooperative multi-agent RL method assigns credit, does it credit the agents that actually earned the team reward?

**Environment: the Planted Coalition Gridworld.** Rewards come from planted coalitions:
- an enabler A pulls a lever, and finishers B and C complete the event later;
- the payout can be delayed by d steps;
- bystanders do nothing useful;
- variants: overlapping coalitions and partial synergy.

**Ground truth.**
- Exact replay with any subset of agents removed gives coalition values v(S).
- From v(S): Harsanyi dividends, Shapley values and leave-one-out (LOO) marginals.
- Mode P adds a **per-step counterfactual truth**: one agent's action at one step is set to noop, and the team continues (closed loop, γ = 0.99).

**Mode P.** Critics and mixers are trained on fixed scripted data (200 noisy episodes, 1,500 updates), then read out on 30 test episodes. Methods:
- main grid: VDN, QMIX, COMA, MAPPO (EPyMARL);
- H3 grid: these four plus our ports of NA²Q and STAS.

**Two-level grading.** Added after pilot 1.

| Level | Compares | Question it answers |
|---|---|---|
| 1, fidelity | read-out vs the per-step counterfactual it estimates | Is the critic right about its own target? |
| 2, semantic gap | a *perfect* per-step critic vs removal credit | Is that target the right notion of credit? |
| End to end | read-out vs its matched removal target (LOO or Shapley) | |

**Metrics:**
- L1 between credit shares (0 = exact, 2 = disjoint);
- Spearman rank correlation;
- leakage Λ (share of credit going to realised bystanders);
- missed enabler credit M;
- at the coalition level: dividend error, Jaccard recall/precision and interaction mass.

---

## 2. Infrastructure checks (Steps 1–5)

- **The ground-truth calculator reproduces the running example:** Δ({A, B, C}) = 10 and φ = 10/3. It also matches brute-force formulas on random games.
- **Scripted data.** The full team always succeeds. Realised = planted dividends in 83–100% of seeds, falling under ε-noise and roaming. Physical interference (a noisy agent blocks or unblocks someone) is the cause.
  - **Inference:** always grade against the *realised* ground truth, and report the match rate.
- **Cost.** Exact ground truth takes about 60 ms per episode at n = 5. The per-step counterfactual truth (about 300 rollouts per episode) dominates Mode P's runtime.
- **EPyMARL.** All four methods train online. A 3k-step smoke run gives return 0, as expected: random exploration rarely coordinates.

## 3. Pilots: two method problems found and fixed

From `docs/pilot_findings.md`:

1. **The critic's state was not Markov under delay.** The state showed that an event had completed, but not how long ago, so the payout time was unpredictable. Critic losses at d = 25 were 1–30.
   - **Fix:** a payout countdown in the global (critic) state only.
   - **Effect:** losses converged for QMIX, COMA and MAPPO.
   - **VDN cannot use the fix.** It never sees the global state, and is bit-for-bit unchanged.
2. **Per-step counterfactual read-outs are not removal credit.** The scripted team recovers from a one-step deviation: if A doesn't pull now, it pulls next step. A single action's value is therefore genuinely about 0, even for the agent who matters most.
   - **Fix:** two-level grading with the per-step truth.

**Pilot 2 already showed the project's central pattern.** QMIX at d = 0 was close to removal credit (L1 ≈ 0.52), closer than a *perfect* per-step critic would be (L1 ≈ 1.16). Yet it barely matched its own per-step target (r ≈ 0.05). It looked faithful for the wrong reason.

**Changes before the main grid:**
- an enabler-slack knob (A slack vs A critical);
- horizon 80 for d = 50;
- thresholds fixed and pre-registered.

## 4. Main grid: 120 cells, VDN, QMIX, COMA, MAPPO

The grid is d ∈ {0, 10, 25, 50} × b ∈ {0, 2, 4} × {slack, critical} × 5 seeds, about 20 CPU-hours.

**Statistics:**
- one-sided paired t-test on per-seed means;
- Holm correction within each hypothesis;
- exact Wilcoxon p reported alongside.

### 4.1 Confirmatory verdicts

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| **H0** ground truth is protocol-stable | ✗, narrowly | 104/120 cells pass; the 16 failures are 4 b = 4 data configurations × 4 delays (worst remove-vs-noop L1 0.18; worst realised = planted 0.53) |
| **H1a** missed enabler credit grows with delay | ✗ | VDN's critic does not converge under delay (excluded by the gate); QMIX has the right sign everywhere (best ΔM +0.49), Holm p = 0.074 |
| **H1b** slack, not delay, removes per-step enabler credit | ✗ as written | slack half ✓ in 24/24 cells (level-2 missed 0.96–0.99, the same at every delay); critical half ✗, because only 54% of "critical" episodes were critical |
| **H2** leakage grows with bystanders | COMA ✓ 8/8; VDN 1/8; QMIX 1/8 | COMA ΔΛ +0.12 to +0.21, Holm p ≤ 0.018 |
| **H2′** COMA leaks less than VDN/QMIX at d = 0 | ✗, **reversed** 0/8 | COMA leaks 0.22–0.48 *more* (b = 2: COMA 0.42, VDN 0.14, QMIX 0.07) |
| **H5** end-to-end accuracy does not imply fidelity | ✓ | 30 of the 32 best end-to-end cells have step r < 0.3; level-1 r ≤ 0.36 for every method |

### 4.2 The MAPPO artifact

MAPPO's read-out is its *shared* advantage, the same number for every agent.
- After normalisation every agent gets ±1/n, so **Λ = b/n by construction** (measured within 0.006).
- L1 = 2.000 at d ≤ 25, because the summed advantage is negative.
- Its H2 8/8 and its M values are therefore **not evidence**.
- Only its policy-gradient SNR is informative:
  - coalition ≈ bystanders (about 0.04) at d ≤ 10;
  - about 3× the bystanders at d = 50 (descriptive, not pre-registered).

Excluding MAPPO from the H1a/H2 conclusions is post-hoc, but it follows from the read-out's definition rather than from the results.

### 4.3 Post-hoc analyses

- **H1b split by realised criticality** (`scripts/posthoc_h1b.py`; all 120 cells reproduced exactly first).

  | notion | critical episodes (missed) | non-critical episodes (missed) |
  |---|---|---|
  | noop | 0.11–0.12 ✓ | 1.31–1.35 in the late-start setting; 1.00 with slack |
  | behaviour baseline (COMA-style) | 0.52–0.56 ✗ | |

  - **For the noop notion the mechanism holds.** The pre-registered failure was the weak manipulation, not the mechanism.
  - **The COMA-style notion under-credits the enabler even when A is decisive.** Its median is 0 but its mean is 0.54, so a minority of critical episodes give A strongly negative credit.
- **VDN/QMIX leakage direction.** ΔΛ > 0 in 16/16 conditions and in 77/80 seed pairs. The sign test gives p = 1.5 × 10⁻⁵, indicative only because the conditions share data. So H2 for VDN/QMIX looks like a power failure, not a null.

## 5. Step 7: porting NA²Q and STAS

From `docs/step7_ports.md`. Both are clean reimplementations: the official repositories have no licence, so no code is copied.

**NA²Q.**
- **The official max target diverges on fixed data:** Q_tot reaches 97–143 against a true return of about 8.
- With a SARSA target the same network converges (Q_tot ≈ 8.3).
- Mode P therefore uses SARSA, so NA²Q estimates the behaviour policy's Q, as COMA does.
- The official target is kept for Mode T.

**STAS.**
- **It spreads credit almost uniformly across agents.**
  - On a main-grid cell, the true shares are about 0.32 each for A, B and C and about 0.03 for bystanders. STAS gives about 0.2 to everyone.
  - A 6× larger budget cuts the loss six-fold (9.2 → 1.2) without changing the split.
  - On a synthetic task where agent 0 deserves 100%, STAS gives it 25–27%.
- **Inference:** this is an identifiability limit of the method, not undertraining. The loss constrains only the total, and nothing pushes an uninvolved agent's output to zero.
- Eval-mode sums overshoot the return by 20–70%, a dropout train/eval mismatch inherited from the original.

**Mixer-game coalition read-out.**
- At each step, g_t(S) = Q_tot(a_S, noop_{−S}) − Q_tot(noop_all).
- Its exact Harsanyi dividends are summed over the episode.
- By construction, VDN's game has only singleton dividends and NA²Q's has none above order 2 (both unit-tested).

## 6. H3 grid: 30 cells, six methods

Overlap and partial-synergy layouts × d ∈ {0, 25, 50} × 5 seeds. It ran in 87 minutes on 8 workers, with STAS on the RTX 4060.

| Result | Type | Numbers |
|---|---|---|
| **H3** "best-third individual error but worst-third dividend error"; NA²Q recovery falls with d | confirmatory: ✗ literally, **uninformative** | 0/6 pairs; dividend error 1.9993–2.0000 everywhere (spread 0.0007); NA²Q recovery is zero at every d |
| Mixer games are additive | descriptive | interaction mass 0.000–0.002 (VDN, QMIX, NA²Q) vs a true 1.0 |
| No method beats uniform credit | post-hoc | uniform L1 0.45 (overlap) / 0.89 (partial); no method's mean beats it in any of 36 conditions; STAS comes closest (0.52; 2/5 seeds) |
| Main grid: methods rarely beat uniform | post-hoc | 5 of 96 method-conditions beat it (QMIX 3, VDN 1, MAPPO 1 artifact), all at d ≤ 10 except the artifact; none at d ≥ 25; COMA 0/24 (`main_results.md` §3.3) |
| STAS ignores delay | post-hoc | d = 0 and d = 25 agree to within 4e-6 per episode |
| Partial-synergy data is weak (H0) | descriptive | realised = planted 40–50%; return 8.2–8.8 instead of 10 |

**Why the coalition read-out is blind.** The planted coalitions act across steps: A pulls first, and the finishers act later. A per-step game swaps actions within one step, so it can only be additive. This is the coalition-level form of the semantic gap from H1b.

---

## 7. Overall inferences

1. **"Credit" from value-decomposition and actor-critic read-outs is a per-step counterfactual, and that is a different quantity from "who earned the reward".**
   - Even a perfect per-step critic gives a slack enabler zero credit (H1b slack half, level 2).
   - It cannot see coalitions whose members act at different times (H3).
   - The gap comes from slack and from team recovery after a one-step deviation, not from delay.
2. **Critics are poor estimators even of their own per-step target.** Level-1 fidelity is r ≤ 0.36 for every method, and r ≈ 0–0.2 in the pilots.
3. **Low end-to-end error is not evidence of correct credit assignment** (H5).
   - QMIX's good d = 0 scores come with near-zero fidelity.
   - On the H3 layouts, **no method beats a uniform split**.
   - Any credit benchmark should therefore report a uniform-credit baseline next to L1.
4. **COMA's counterfactual baseline does not protect against bystanders; it leaks most** (H2′ reversed). The behaviour-baseline notion also under-credits a decisive enabler (post-hoc).
5. **Delay mainly breaks methods that cannot represent it.**
   - VDN cannot see the payout countdown and fails to fit.
   - QMIX degrades consistently but not significantly.
   - STAS is trivially delay-invariant, because it sees only the total return.
6. **Shared-advantage methods (MAPPO) cannot be evaluated by per-agent credit read-outs at all.** Only their gradient SNR is meaningful.
7. **Return decomposition (STAS) as implemented cannot separate agents.** Its credit is near uniform by an identifiability argument.
8. **Leakage does grow with bystanders for all value-decomposition methods.** The per-condition tests were underpowered (post-hoc sign test).

## 8. Caveats that apply throughout

- 5 seeds, and Holm across up to 32 conditions, give limited power.
- Mode P trains critics on scripted data, not policies online.
  - VDN and QMIX learn Q*.
  - COMA, MAPPO and NA²Q (SARSA) learn the behaviour policy's value.
- Planted ≠ realised coalitions in a sizeable share of episodes (b = 4; the partial layout). Grading always uses the realised truth.
- The critical-enabler manipulation is weak (54% critical).
- STAS and NA²Q are our reimplementations; the official code was compared line by line but not executed.
- The uniform baseline is post-hoc on both grids. On the main grid it is exact at b = 0 (true credit is equal shares there), so only b > 0 cells test it.

## 9. Still to do

| Item | What it answers |
|---|---|
| **H5** low-order episodic coalition estimator | whether coalition credit is recoverable at all with an episode-level read-out; includes the size k+1 honesty check |
| **Step 8** Mode T (online training) | H4: is return a weak proxy for credit faithfulness? |
| **Step 9** sensitivity analyses | noop vs removal, open- vs closed-loop replay, Shapley vs LOO |
