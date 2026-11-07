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
- [x] **Step 4. Metrics** (`src/whoearned/metrics.py`)
  L1 credit error, Spearman, leakage Λ, missed credit M, coalition recall/precision (Jaccard) with exact chance level, normalised dividend error, return–faithfulness (pooled and per method), realised bystanders. `evaluate_credit(...)` scores one episode, in total or per event.
- [x] **Step 5. EPyMARL integration**
  EPyMARL is cloned (unmodified) into `external/epymarl` at commit `cbc38c0`; it is git-ignored, so re-clone it there on a new machine.
  - `whoearned/epymarl_env.py`: adapter (`env_args.map_name` selects the layout).
  - `scripts/run_epymarl.py`: online training launcher, e.g. `.venv/Scripts/python scripts/run_epymarl.py --config=qmix --env-config=planted with t_max=20000 use_cuda=False`.
  - `whoearned/epymarl_bridge.py`: builds mac + learner as `run.py` does and converts our episodes into `EpisodeBatch`es (for Mode P).
  - `whoearned/readouts.py`: difference of Q_tot (VDN/QMIX), COMA advantage, MAPPO shared advantage + policy-gradient SNR. All are per step (B, T, n).
  - Dependencies added to the venv: `sacred`, `gymnasium`, `pyyaml` (EPyMARL's pinned `PyYAML==5.3.1` does not build on Python 3.12). `smaclite` is stubbed because EPyMARL imports it but we never use it.
- [ ] **Step 6. Mode P experiments**
  Train critics/mixers on scripted data across the configuration sweep. Pilot (2 seeds) → fix thresholds → main runs (5 seeds). H0, H1, H2, H3.
  - [x] Runner (`experiments.py`, `mode_p.py`, `scripts/run_mode_p.py`, `scripts/summarise_mode_p.py`); protocol in `docs/mode_p_protocol.md`.
  - [x] Pilot run 1, which found two method problems: `docs/pilot_findings.md`.
  - [x] Fix 1: payout countdown in the global (critic) state, so the state is Markov under delay.
  - [x] Fix 2: per-step counterfactual ground truth (discounted, γ = 0.99) and two-level grading (fidelity + semantic gap).
  - [x] Pilot run 2: fixes confirmed (except VDN, by construction); thresholds proposed in `docs/pilot_findings.md`.
  - [x] Add the enabler-slack knob and horizon 80 for d=50 → main grid (120 cells, 5 seeds). Results: `docs/main_results.md`.
  - [x] Post-hoc: H1b split by realised criticality (`scripts/posthoc_h1b.py`; `docs/main_results.md` §3.1).
  - [x] Post-hoc: uniform-credit baseline on the main grid (`scripts/posthoc_uniform.py`; `docs/main_results.md` §3.3).

## Tier 2: should-have

- [x] **Step 7.** Port STAS and NA²Q (design and deviations: `docs/step7_ports.md`).
  - [x] NA²Q (`na2q.py`): agents + VAE + order-≤2 monotone mixer. Mode P uses a SARSA target, because the official max target diverges on fixed data.
  - [x] STAS (`stas.py`): STAS_ML return decomposition.
  - [x] Mixer-game Harsanyi dividends for VDN, QMIX and NA²Q (coalition level, H3); `run_mode_p.py --algs`.
  - [x] H3 grid: overlap and partial-synergy layouts × delay; run all six methods. Results: `docs/h3_results.md` (`scripts/analyse_h3.py`).
- [x] **Step 8.** Mode T; H4 (descriptive). QMIX and MAPPO (400k steps) and QMIX (1M steps) earned no reward from plain online training, because a random policy completes the task in 0 of 20,000 episodes. With a replay buffer seeded with scripted episodes, VDN and QMIX learn, and H4 was assessed descriptively for those two. 20 runs (VDN, QMIX × d ∈ {0, 25} × 5 seeds) scored at every checkpoint: within-method ρ(return, 1 − L1) = +0.20 (QMIX) and +0.15 (VDN). COMA and MAPPO are not covered. `docs/mode_t_results.md`.
- [x] **Step 9.** Sensitivity analyses (`docs/sensitivity.md`).
  - [x] Shapley vs leave-one-out (`scripts/sensitivity_shapley_loo.py`): no ranking or verdict changes.
  - [x] Open- vs closed-loop replay (`scripts/sensitivity_loop.py`): H2, H2′, H-fid unchanged; H1a becomes supported through one condition.
  - [ ] No-op vs removal: dropped as a separate check (partly covered by H0's remove-vs-noop comparison).

## Tier 3: stretch (pick one)

- [x] H5 low-order coalition estimator, tested on coalitions larger than its order k (`docs/h5_results.md`).
- [ ] Interactive credit visualisation.

## Findings log

- **Step 3 (200 seeds per cell).** The full team always succeeds. How often the replay ground truth equals the planted dividends:

  | Layout | wander, ε=0 | wander, ε=0.1 | roam, ε=0 | roam, ε=0.1 |
  |---|---|---|---|---|
  | running_example | 200 | 188 | 179 | 166 |
  | overlap_example | 200 | 162 | 200 | 164 |

  Deviations come from physical interference: a noisy or roaming agent ends up next to someone, so removing a bystander changes the outcome. Grading always uses the **realised** ground truth. Mode P must report the deviation rate (`GroundTruth.matches(planted_dividends(cfg))`).
- **Cost.** About 60 ms per episode for the exact ground truth with n = 5 (32 replays). n = 8 means 256 replays, roughly 0.5 s per episode. Fine for Mode P, and it parallelises across episodes if needed.

- **Step 5 smoke tests.** VDN, QMIX, COMA and MAPPO all train online through EPyMARL. COMA and MAPPO use the parallel runner, which works on Windows. With 3k steps the return is 0, as expected: random exploration rarely coordinates the lever and the joint push.
- **First Mode P preview** (VDN; 200 scripted episodes, ε = 0.2; 1,500 updates ≈ 75 s on CPU; 20 test episodes; leave-one-out target; **one seed, untuned, not a result**):

  | delay | L1 | leakage | missed |
  |---|---|---|---|
  | 0 | 1.05 | 0.15 | 1.53 (enabler A gets negative credit) |
  | 25 | 0.91 | 0.46 | 0.65 |

  The pipeline gives non-trivial, delay-dependent credit. Before Step 6, decide: VDN/QMIX learn Q* (greedy max) from the scripted data rather than the value of the scripted policy. Since the scripted policy is near-optimal these are close on its own path, but this should be stated in the method.

- **Step 6 main grid** (120 cells, 5 seeds, about 20 CPU-hours on the Ryzen 5 7600; full write-up in `docs/main_results.md`).

  Pre-registered verdicts:
  - H0 ✗, narrowly: only 4 b = 4 data configurations fail.
  - H1a ✗: VDN's critic does not converge under delay; QMIX has the right sign, Holm p = 0.074.
  - H1b ✗: the slack half holds in 24/24 cells (missed ≈ 0.97–0.99, independent of delay); the critical half fails, since only 54% of episodes are critical.
  - H2: COMA ✓ 8/8; VDN and QMIX 1/8 each.
  - H2′: reversed (COMA leaks most).
  - H-fid ✓ (the finding earlier labelled H5): 30 of 32 best cells have step r < 0.3.

  **MAPPO artifact:** its shared advantage gives every agent ±1/n, so Λ = b/n by construction. Its H2 8/8 and its missed values are not evidence; only its SNR is informative.

  **Post-hoc:**
  - ΔΛ > 0 in all 16 VDN/QMIX conditions (sign test p = 1.5e-5, conditions not independent).
  - H1b split by realised criticality: noop missed is 0.11–0.12 in critical episodes vs 1.0 with slack, so the mechanism holds and the weak manipulation explains the failure. The baseline notion is still 0.52–0.56 when critical.
- **Step 7 ports** (`docs/step7_ports.md`):
  - NA²Q's official max target diverges on fixed data (Q_tot ≈ 115–140 vs a true ≈ 8). A SARSA target fixes it, so it is used in Mode P.
  - STAS spreads credit almost uniformly across agents. This persists at 6× the budget and on a synthetic single-agent task, so it is the method's identifiability limit, not undertraining.
  - Smoke cell: NA²Q L1 0.51, leakage 0.04; STAS L1 0.74, leakage 0.42.
- **Step 7 H3 grid** (30 cells, 6 methods, 87 min on 8 workers; write-up in `docs/h3_results.md`).
  - Pre-registered H3: ✗ on the literal rule (0 of 6 best-third-L1 pairs fall in the worst third of dividend error), but **uninformative**. Dividend error is about 2.0 (the maximum) in every pair, with a spread of 0.0007. NA²Q's recovery cannot fall with d, because it is zero at every d.
  - Interaction mass is 0.000–0.002 for VDN, QMIX and NA²Q, against a true 1.0. The per-step mixer game cannot express coalitions whose members act at different steps. This is the coalition-level form of the semantic gap; a fair test needs an episode-level read-out (H5).
  - **Post-hoc uniform baseline** (1/n to everyone): L1 0.45 on overlap and 0.89 on partial. No method's mean beats it in any of 36 conditions; STAS comes closest (0.52 on overlap, 2 of 5 seeds), because its near-uniform split matches the overlap truth.
  - **Post-hoc STAS delay invariance:** d = 0 and d = 25 agree to within 4e-6 per episode. STAS sees only observations, actions and the total return, and none of these change with delay inside the horizon.
  - H0: the partial layout realises its planted structure in only 40–50% of test episodes (return 8.2–8.8).
- **H5 grid** (30 cells + 10 `h5_ref`; `docs/h5_results.md`). The episodic coalition estimator trained on episodes plus open-loop replays with agents removed (ECE-k_mix) beats NA²Q on dividend error in 8/8 conditions (ECE-3_mix on the order-3 layout: 0.43 vs 2.0; recall and precision 0.96) and beats uniform credit on individual L1 in 8/8. Leakage is not significantly lower (2/8 vs NA²Q, 1/8 vs data-matched NA²Q_mix). Without the replay data, ECE only reproduces uniform credit. STAS ran on a mix of devices (see the write-up).
- **Mode T** (`docs/mode_t_results.md`): plain online training never earns reward (random play completes the task in 0 of 20,000 episodes). Seeding the replay buffer with 200 scripted episodes lets VDN and QMIX learn (`scripts/run_epymarl_seeded.py`). H4, descriptive: return is only weakly correlated with credit accuracy (ρ = +0.20 QMIX, +0.15 VDN, CIs below 0.7); QMIX earns more than VDN without assigning credit more accurately. Needed a Python 3.12 environment (`sacred` does not import under 3.14).
- **Sensitivity** (`docs/sensitivity.md`): Shapley vs LOO changes nothing; the closed-loop truth always recovers the planted coalition and flips H1a to supported through one QMIX condition.

## Conventions

- Agents are indexed 0..n-1. A coalition is a bitmask `int` (bit i set ⇔ agent i present). So with agents A..E = 0..4, {A,B,C} = 0b00111 = 7.
- Coalition values are a NumPy array `v` of length 2^n with `v[mask]` = value of that coalition; `v[0]` is the empty coalition.
- Run tests: `.venv/Scripts/python -m pytest`
