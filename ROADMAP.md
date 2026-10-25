# Roadmap

Plan for implementing `who_earned_the_reward_problem_statement_v2.md`. Each step ends with something tested before the next one starts.

## Tier 1: must-have

- [x] **Step 1. Ground-truth calculator** (`src/whoearned/ground_truth.py`)
  Coalition values → Harsanyi dividends, Shapley values, leave-one-out marginals, normalised shares.
  Done when the unit tests reproduce the running example (Δ({A,B,C}) = 10, φ = 10/3), the two G4 games, and agree with brute-force formulas on random games.
- [x] **Step 2. Planted Coalition Gridworld** (design: `docs/environment_design.md`) (`src/whoearned/envs/`)
  Events with enablers/finishers, payout delay d, bystanders, overlap, partial synergy; true agent removal; fixed-size observations with presence flags; deterministic replay.
  Done when scripted episodes give the planted rewards, and removing any coalition member zeroes the event.
- [x] **Step 3. Scripted policies + replay-based ground truth** (`scripted.py`, `replay.py`)
  Scripted agents that complete the events; open-loop replay with any subset removed; v(S) for all 2^n subsets.
  Done when the replay ground truth equals the planted structure on scripted data.
- [ ] **Step 4. Metrics** (`src/whoearned/metrics.py`)
  L1 credit error, Spearman, leakage Λ, missed credit M, coalition Jaccard, dividend error.
- [ ] **Step 5. EPyMARL integration**
  Register the environment with EPyMARL; implement the read-outs (difference of Q_tot, COMA advantages, MAPPO gradient-noise diagnostic).
- [ ] **Step 6. Mode P experiments**
  Train critics/mixers on scripted data across the configuration sweep. Pilot (2 seeds) → fix thresholds → main runs (5 seeds). H0, H1, H2, H3.

## Tier 2: should-have

- [ ] **Step 7.** Port STAS and NA²Q.
- [ ] **Step 8.** Mode T (full training) on pilot-selected configurations; H4.
- [ ] **Step 9.** Sensitivity analyses: no-op vs. removal, open- vs. closed-loop, Shapley vs. leave-one-out.

## Tier 3: stretch (pick one)

- [ ] H5 low-order coalition estimator, tested on coalitions larger than its order k.
- [ ] Interactive credit visualisation.

## Findings log

- **Step 3 (200 seeds per cell).** The full team always succeeds. How often the replay ground truth equals the planted dividends:

  | Layout | wander, ε=0 | wander, ε=0.1 | roam, ε=0 | roam, ε=0.1 |
  |---|---|---|---|---|
  | running_example | 200 | 188 | 179 | 166 |
  | overlap_example | 200 | 162 | 200 | 164 |

  Deviations come from physical interference: a noisy or roaming agent ends up next to someone, so removing a bystander changes the outcome. Grading always uses the **realised** ground truth. Mode P must report the deviation rate (`GroundTruth.matches(planted_dividends(cfg))`).
- **Cost.** About 60 ms per episode for the exact ground truth with n = 5 (32 replays). n = 8 means 256 replays, roughly 0.5 s per episode. Fine for Mode P, and it parallelises across episodes if needed.

## Conventions

- Agents are indexed 0..n-1. A coalition is a bitmask `int` (bit i set ⇔ agent i present). So with agents A..E = 0..4, {A,B,C} = 0b00111 = 7.
- Coalition values are a NumPy array `v` of length 2^n with `v[mask]` = value of that coalition; `v[0]` is the empty coalition.
- Run tests: `.venv/Scripts/python -m pytest`
