# v3 exploratory pilots (seeds 0–4), and what they decided

11–12 Nov 2026. **Exploratory**: same seeds as every earlier round, and settings chosen while looking at them. Nothing here is a confirmatory result; `docs/v3/preregistration_v3.md` (frozen after this) is tested on seeds 5–9.

All L1 values are share distances (0 = exact, 2 = disjoint) against the **exact persistent-ablation truth of the method's own null** (closed loop, common random numbers), unless marked. "div" is dividend error against the closed-loop removal game. Values are means over seeds, with [min, max] where useful. Critics use 200 training episodes + 200 stops unless marked.

Raw tables: `results/v3/pilot_*` (cell JSONs); `.venv/bin/python scripts/v3/aggregate.py <grid>`.

## 1. Reproduction of round 2 (plan 1.4)

Cell d = 0, b = 2, seed 0, round-2 sampler (`atd_bd`): **L1 0.006, step r 0.98**. Round 2 reported 0.02 and 0.96; both are within the 0.03 tolerance. Passed.

## 2. Fixed single-coalition layout (`pilot_fixed`, seeds 0–1, 24 cells)

| Method | b = 2: L1 | step r | top-3 | div | b = 4: L1 | step r | top-3 | div |
|---|---|---|---|---|---|---|---|---|
| no interventions (`atd_noint`) | 0.88 [0.46, 1.18] | 0.10 | 0% | 1.98 | 1.14 | 0.15 | 0% | 1.99 |
| MLP, round-2 sampler (`atd_bd`) | 0.012 | 0.88 [0.66, 0.98] | 100% | 0.82 | 0.024 | 0.94 | 100% | 1.37 |
| MLP, **stratified** (`atd`) | 0.036 [0.010, 0.151] | 0.96 [0.92, 1.00] | 100% | 0.93 | 0.037 [0.011, 0.087] | 0.93 [0.69, 1.00] | 100% | 1.45 |
| MLP, through-payout (`atd_tp`) | 0.017 | 0.97 | 100% | 0.93 | 0.037 | 0.95 | 100% | 1.45 |
| MLP, active design | 0.020 | 0.96 | 100% | 0.78 | 0.071 [0.03, 0.16] | 0.99 | 100% | 1.26 |
| **Set critic** (`atd_set`) | **0.004** [0.001, 0.014] | **0.998** | 100% | **0.094** | **0.005** | **0.979** [0.86, 1.00] | 100% | 0.317 |
| Table (`table3_mix`, removal game) | – | – | 100% | 0.215 | – | – | 100% | 0.221 |

**Readings:**
- **Interventions are necessary** (no-intervention L1 0.46–1.42 vs 0.01–0.15).
- **Stratified sampling trades a little episode L1 for timing.** At d = 25 its step r is 0.96–1.00 against 0.81–0.86 for the round-2 sampler. The set critic gets both.
- **On fixed layouts the set critic also recovers coalition dividends at b = 2** (0.09 vs table 0.22), **not at b = 4** (0.32 vs 0.22). No superiority claim on fixed layouts is pre-registered.
- **The MLP has occasional outlier seeds** (L1 0.15; step r 0.69). Pre-registered thresholds for the MLP stay at the plan's values (L1 ≤ 0.15, r ≥ 0.85), so the test can catch that instability.

## 3. Shuffled roles (`pilot_perm`, seeds 0–4, 10 cells)

| Method | b = 2: L1 | step r | top-3 | div | b = 4: L1 | step r | top-3 | div |
|---|---|---|---|---|---|---|---|---|
| no interventions | 0.98 | 0.05 | 11% | 1.98 | 1.19 | 0.06 | 3% | 1.99 |
| table3_mix | (= equal split) | – | 8% | 1.80 | (= equal split) | – | 3% | 1.94 |
| ECE-3_mix | – | – | 100% | 0.21 [0.09, 0.50] | – | – | 100% | 0.68 [0.40, 1.47] |
| MLP, 200 stops | 0.46 [0.36, 0.59] | 0.76 | 63% | 1.87 | 0.87 [0.78, 1.00] | 0.52 | 27% | 1.98 |
| MLP, 800 stops | 0.15 | 0.97 | 100% | 1.44 | 0.50 | 0.80 | 87% | 1.91 |
| MLP, active, 200 | 0.39 | 0.82 | 83% | 1.77 | 0.83 | 0.58 | 19% | 1.98 |
| **Set critic, 200** | **0.004** [0.001, 0.014] | **0.999** | **100%** | **0.059** | **0.009** [0.004, 0.021] | **0.995** | **100%** | **0.325** |
| Set critic, 800 | 0.010 | 0.999 | 100% | 0.042 | 0.009 | 0.999 | 100% | 0.214 |
| Set critic, active, 200 | 0.003 | 0.999 | 100% | 0.045 | 0.040 [0.003, 0.17] | 0.998 | 100% | 0.195 |

**Readings:**
- **The table collapses to the equal split**, as it must.
- **ECE-3_mix handles shuffled roles well at the individual level** (top-3 100%): it reads role flags from observations.
- **The set critic with 200 stops beats the MLP with 800 stops by a factor of 15–55 in L1**, and beats ECE on dividends (0.06 vs 0.21; 0.33 vs 0.68).
- This is far beyond the assessment's §7.2 prediction (L1 ≤ 0.15 at b = 2 and ≤ 0.35 at b = 4).
- The MLP sees the same role flags. A control with role flags hidden from both critics is pre-registered (P8d; pilot in §7).

## 4. Overlap and partial layouts (`pilot_h5`, seeds 0–1)

| Layout | MLP no-op | MLP removal | Set | Table3 div | MLP div |
|---|---|---|---|---|---|
| overlap | 0.016 (r 0.97) | 0.013 | 0.029 | 0.56 | 1.20 |
| partial | 0.66 (r 0.81) | 0.63 | 0.65 | 0.46 | 1.44 |

**Overlap layout: the null is a definition, not a detail.** The no-op critic is 0.016 from no-op truth but 0.26 from removal truth. Under the no-op null, B keeps waiting for an inert C and loses event 1 too (assessment §5). Each critic is graded against its own null.

**Partial layout: the error is luck, not estimation.**
- The state fix removed the delay penalty: d = 50 is now the same as d = 0.
- What remains: in a third of test episodes, exploration noise makes one finisher complete the event alone (exact check, seed 0: 20 of 30 episodes pay 10, 10 pay 5; realised no-op LOO patterns (10, 5, 5), (5, 0, 0), (10, 10, 5), …).
- A state critic can only know *expected* credit. Two additions follow:
  - an expected-credit floor (`floor_expected`: every episode predicted as the cell's mean realised truth);
  - the structural read-out of the commitment split (`atd_struct`).
- Tiny-cell check: κ 0.45, structural 0.25, floor 0.19 (equal split 0.80).

## 5. Control V2: better data or better read-out? (`smoke_v2`, `pilot_v2`, partial)

The same EPyMARL learners are trained on the same 200 stopped episodes as `atd`.

| Cell | MAPPO standard | MAPPO masked | QMIX standard | QMIX masked | ATD (MLP) |
|---|---|---|---|---|---|
| d = 25, b = 2, seed 0 (L1 vs exact / vs removal LOO) | 1.85 / 2.00 | **0.32** / 0.32 | 1.10 / **0.70** | 0.78 / 0.78 | 0.015 |
| d = 0, b = 2 slack, seeds 1–2 (vs removal LOO) | 1.93 | **0.41** | 0.42 | 0.50 | 0.020 |

- **Data alone does not fix one-step read-outs.** MAPPO trained on stops still gives 1.93–2.00.
- **The read-out fixes MAPPO.** Its masked read-out falls to 0.32–0.41.
- **QMIX is the exception.** Its standard read-out is already as close to removal credit as its masked one (0.42–0.70 vs 0.50–0.78). This matches the earlier pilot-2 finding that QMIX looks right at d = 0 for the wrong reason.
- So the plan's directional prediction for QMIX is **not** pre-registered as a claim. P9b is registered as a two-sided, descriptive comparison.
- **TD-trained critics are far less accurate than the Monte-Carlo critic** (0.32 vs 0.015). Plausible causes: discounting over the delay, bootstrapping, and 1,500 updates. The point is the read-out comparison, not the critic.

## 6. Zero-shot size transfer (set critic; `results/v3/size_transfer_pilot.json`, seeds 0–2)

| Train → test | Transfer L1 | In-size L1 | Equal split | Top-3 | Step r | Leakage |
|---|---|---|---|---|---|---|
| b = 2 → b = 4 (5 → 7 agents) | 0.60 [0.57, 0.63] | 0.006 | 1.14 | **100%** | 0.94 | 0.30 |
| b = 4 → b = 2 (7 → 5 agents) | 0.29 [0.28, 0.30] | 0.002 | 0.80 | **100%** | 0.93 | 0.14 |

The transferred critic always picks exactly the right coalition and gets the timing right, but it leaks 0.15–0.3 of the credit share to bystanders.
- Likely cause: **sum pooling** trained at one team size never forces a bystander's token to contribute zero, so an unseen extra (or missing) bystander shifts the value.
- Pre-registered: only the robust part (top-3 ≥ 90%, and clearly below the equal split).
- Mixed-size training is the natural fix and is left as follow-up.

## 7. Role flags hidden (`pilot_roles`)

Pending at the time of writing; filled in before the pre-registration is frozen.

## 8. Decisions taken from these pilots

| Decision | Reason |
|---|---|
| Primary sampler: stratified | Timing fidelity at d > 0 (r 0.96–1.0 vs 0.81–0.88) at little cost in episode L1. |
| Pre-register the set critic alongside the MLP | It is the strongest result, motivated by a stated symmetry. |
| Do not pre-register active design | No reliable gain (sometimes worse). Reported as an exploratory negative. |
| Grade against the method's own null | Overlap layout: the nulls define different games. |
| Partial layout: grade the structural read-out against the expected-credit floor | The realised-versus-expected gap is luck; a state critic cannot see it. |
| P9b (QMIX masked vs standard) two-sided | The pilot contradicts the plan's direction for QMIX. |
| Size transfer: register top-3 and gap to the equal split only | Leakage at unseen sizes is real; mixed-size training is follow-up. |
