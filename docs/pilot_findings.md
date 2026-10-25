# Mode P Pilot: Findings (2 seeds, d ∈ {0, 25}, b ∈ {0, 2, 4})

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
