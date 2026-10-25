# Mode P Pilots: Findings

- **Pilot 2** (both fixes applied): the section below. Read this first.
- **Pilot 1** (before the fixes): further down. Kept for the record of why the method changed.

---

# Pilot 2 (Markov critic state + two-level grading; 2 seeds, d ∈ {0, 25}, b ∈ {0, 2, 4})

Raw: `results/mode_p/pilot2/summary.md`. 12 cells, 7 workers, about 45 minutes (10–25 minutes per cell; the per-step truth dominates at b=4). All episodes valid.

## Fix 1 worked, except for VDN, by construction

Final critic loss at d=25 dropped from 13–30 (COMA), 1–2.7 (QMIX) and ≈3 (MAPPO) to 0.05–0.23, the same as at d=0. **VDN is unchanged.** Its Q_tot is Σ_i Q_i(o_i, a_i), computed from agent observations only, and the countdown lives in the global state, which VDN never sees. VDN's results are bit-for-bit identical to pilot 1, which is also a determinism check. *This is a property of VDN worth reporting:* a purely decentralised value decomposition cannot represent when a delayed team reward arrives unless agents can observe it.

## Level 2: the semantic gap is large, and independent of the critic

A **perfect** per-step critic, summed over the episode and scored against leave-one-out:

| Per-step notion | L1 | Missed enabler credit | Leakage (b=2 / b=4) | Mean shares A, B, C |
|---|---|---|---|---|
| Noop effect (what VDN/QMIX estimate) | 1.12–1.19 | **1.02** | 0.04 / 0.05 | 0.00, 0.23, 0.37 |
| Behaviour-baseline effect (what COMA estimates) | 1.07–1.13 | **1.00** | 0.07 / 0.10 | 0.00, 0.12–0.19, 0.20–0.22 |

The enabler gets **zero** credit, at every delay and bystander count. A pulls the lever before the finishers are ready, so no single action of A's changes when the reward arrives. Per-step counterfactual credit only sees agents on the critical path at each step. The gap is identical at d=0 and d=25: delay only rescales the values (γ^d) once the state is Markov.

**Implication for H1.** In this layout, *slack*, not delay, is what removes enabler credit. The main runs need the enabler-gap knob: make A the bottleneck (`enabler_start` late) versus leave it with slack.

## Level 1: critics are poor at per-step effects

Pearson between the read-out and its per-step truth, over all (episode, step, agent): **0.0–0.2 for every method**. Per-step effects are tiny (≈ 10·(1−γ)·γ^k ≈ 0.1) next to the value scale (≈ 7–9), and almost all of them are 0. No method is "faithful" at level 1.

## End to end: QMIX looks good for the wrong reason

| d = 0 | L1 | Spearman | Leakage (b=2) | Missed (b=2) |
|---|---|---|---|---|
| QMIX | 0.52–0.56 | 0.76 | 0.05 | 0.17 |
| VDN | 0.54–0.89 | 0.41–0.52 | 0.13 | 1.17 |
| COMA | 1.24–1.53 | ≈ 0 | 0.42 | 0.90 |
| MAPPO (shared) | 2.00 | – | 0.40 | 1.63 |

QMIX's credit is **closer to removal credit than a perfect per-step critic's would be** (L1 0.52 vs 1.16), while matching its own per-step target poorly (r ≈ 0.05). Its errors happen to push credit towards A. Without two-level grading, QMIX would look like a faithful credit assigner. This is the clearest evidence so far for the project's thesis: **low end-to-end error does not mean the method is assigning credit for the right reasons.**

With delay (critics now converged), QMIX degrades end to end:

| QMIX | d = 0 | d = 25 |
|---|---|---|
| L1 | 0.52–0.56 | 0.88–1.73 |
| Leakage | 0.05–0.13 | 0.44–0.63 |
| Missed | 0.17–0.59 | 0.95–1.31 |

COMA is flat in delay.

## H0 (measurability): holds on average

Same as pilot 1 (the ground truth does not depend on the critic state). Remove-vs-noop mean share L1 is ≤ 0.12 per cell, against between-method differences of 0.5–1.5. Realised = planted in 67–100% of episodes, lowest at b=4.

## Other

- **MAPPO SNR:** coalition ≈ bystanders (≈ 0.03) in every cell. The shared advantage carries no differential signal.
- **H2:**
  - Leakage grows with bystanders for VDN, QMIX and COMA (both pilots).
  - "COMA leaks less than VDN/QMIX at d=0" is the opposite in both pilots (0.42 vs 0.05–0.13). Level 1 explains why: COMA's per-step signal is near zero, so noise spreads its credit evenly.
- **H3** (coalition-level) cannot be tested with these four methods: none exposes coalition terms. It needs NA²Q (Step 7).

## Proposed pre-registered thresholds (for the main runs)

Two seeds give seed-sds of 0.01–0.6 on L1, so effects are judged with paired tests across 5 seeds (Wilcoxon signed-rank on per-seed means, α = 0.05, Holm-corrected within each hypothesis).

| ID | Hypothesis (main-run form) | Supported if |
|---|---|---|
| H0 | Ground truth is protocol-stable | Mean remove-vs-noop share L1 ≤ 0.15 in every cell, and realised = planted ≥ 0.6. Per-episode maxima reported. |
| H1a | Missed enabler credit grows with delay | M(d=50) − M(d=0) ≥ 0.2, paired, for a method whose critic converged at both delays (final loss within 2× of d=0) |
| H1b *(new)* | Slack, not delay, removes per-step enabler credit | Level-2 missed ≥ 0.8 when A has slack, and ≤ 0.3 when A is on the critical path |
| H2 | Leakage grows with bystanders | Λ(b=4) − Λ(b=2) ≥ 0.05, paired, and Λ(b=2) > 0.1, per method |
| H2′ | COMA leaks less than VDN/QMIX at d=0 | Λ_COMA < Λ_VDN and Λ_COMA < Λ_QMIX, paired. *Pilot predicts rejection; keep it pre-registered as stated in v2.* |
| H5 *(two-level)* | End-to-end accuracy does not imply fidelity | At least one method with end-to-end L1 in the best third has level-1 step r < 0.3 |

H3 waits for NA²Q; H4 is Mode T.

## Changes recommended before the main runs

1. **Enabler-slack knob.** Add `enabler_start` ∈ {0 (slack), late (A on the critical path)} to the data cell, for H1b.
2. **Horizon 80 for d = 50**, so noisy completions are not clipped. The pilot had 0% clipped at d ≤ 25.
3. **Grid.** d ∈ {0, 10, 25, 50} × b ∈ {0, 2, 4} × slack ∈ {slack, critical} × 5 seeds = 120 cells ≈ 4–5 hours on 7 workers.

---

# Pilot 1 (before the fixes; 2 seeds, d ∈ {0, 25}, b ∈ {0, 2, 4})

Raw results: `results/mode_p/pilot/summary.md` (git-ignored; regenerate with `scripts/run_mode_p.py --grid pilot`). Runtime: 12 data cells, 6 workers, about 20 minutes. All 1,440 test episodes valid; no clipped payouts; every scripted test episode earned the full 10.

## What worked

- **Pipeline.** End-to-end and reproducible. Seeds give identical ground truth across delays (delay does not change actions), as it should.
- **H0 (measurability), mostly.** Ground truth under removal vs no-op disagrees by a mean share-L1 of 0.00 (b=0), 0.03–0.08 (b=2) and 0.10–0.12 (b=4). Between-method differences are 0.5–1.5, so on average the protocol choice is small. But single episodes reach 0.8, and realised = planted falls to 67–70% at b=4 with ε=0.2. Both must be reported.
- **Directional signal at d=0.** QMIX recovers the coalition reasonably (L1 ≈ 0.5, Spearman ≈ 0.74 at b=2, low leakage). Leakage grows with bystanders for every method, as H2 predicts.

## Problem 1: the critic's state is not Markov under delay (blocks interpreting d > 0)

| Final critic loss | d = 0 | d = 25 |
|---|---|---|
| VDN | 0.02–0.46 | ≈ 1.0, flat from the first update |
| QMIX | 0.07–0.11 | 1–2.7, unstable |
| COMA | 0.09–0.14 | 13–30, still falling |
| MAPPO | 0.008 | ≈ 3.0 |

The global state shows that the event completed, but not how long ago. Completion time varies across episodes because of noise, so under delay the payout step cannot be predicted from the state. That is an irreducible error floor for feedforward critics (VDN/QMIX/COMA/MAPPO here are all feedforward). The problem statement assumes a Dec-POMDP, where the reward is a function of the state; my environment violates that.

**Consequence.** The pilot's d=25 numbers (leakage ≈ 0.6 for VDN/QMIX, Spearman collapsing) mix "delay makes credit hard" with "the critic cannot fit the reward". They are not interpretable as H1/H2 evidence yet.

**Recommended fix.** Add each event's pending-payout countdown to the **global state only**, which centralised critics see. Agent observations stay as they are, so the agents' problem stays partially observable and the temporal-credit difficulty (linking A's early pull to a late reward) remains. Then re-run the pilot and check that d=25 losses converge.

## Problem 2: per-step counterfactual read-outs are not removal credit (changes how we grade)

COMA at d=0 converged (loss 0.08) yet gives near-random credit (Spearman ≈ 0, missed ≈ 1). Diagnostic on `d0_b2_s0`:

- At A's lever-pull step, COMA's Q-values over A's six actions are 0.07, −0.02, 0.06, 0.01, 0.05, 0.06. "Interact" is not the best, and the spread is tiny.
- Mean advantage at A's pull step over the test episodes is 0.005.

This is **correct** for the data. The scripted team recovers from a one-step deviation: if A does not pull at t=4, it pulls at t=5. So the true value of a single action, against any alternative, really is near zero. Per-step counterfactual read-outs (COMA advantage, Q_tot difference) answer *"what would a one-step deviation cost?"*. Leave-one-out and Shapley answer *"how much does this agent matter?"*. The v2 assumption that the sum of per-step counterfactuals ≈ leave-one-out does not hold when the team can recover.

**Recommended fix: two-level grading.** Add a third ground truth, a **per-step counterfactual truth**. For each test episode, agent i and step t, replace i's action with noop at that one step, let the scripted team continue (closed loop), and record the change in return. That costs T × n rollouts per episode (≈ 300, under 1 s). Then grade every read-out twice:

1. **Estimation fidelity:** read-out vs per-step counterfactual truth. Is the critic accurate about what it is supposed to estimate?
2. **Semantic gap:** per-step counterfactual truth vs removal credit (leave-one-out / Shapley). How far is "what these methods estimate" from "who earned the reward", even with a perfect critic?

This separates "the method's critic is wrong" from "the method's notion of credit is the wrong notion". It also directly answers the project's core question.

## Thresholds

Not fixed yet. Both problems change what the d > 0 numbers mean, so fixing pre-registration thresholds now would anchor them on artefacts. Fix them after the pilot is re-run with both changes.

## Directional notes (pilot, d=0 only; indicative)

- **H2 leakage grows with bystanders:** yes for VDN (0.13 → 0.28), QMIX (0.09 → 0.13) and COMA (0.42 → 0.55).
- **H2 "COMA leaks less than VDN/QMIX at d=0":** no in the pilot (0.42 vs 0.13 / 0.09). Problem 2 likely explains this: COMA's per-step signal is near zero, so noise spreads its credit evenly.
- **MAPPO SNR:** coalition and bystanders have similar signal-to-noise (≈ 0.03 each). The shared advantage gives coalition members no more usable gradient signal than bystanders in Mode P.
