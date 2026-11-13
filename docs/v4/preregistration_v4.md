# Pre-registration v4 (next steps after v3)

Seeds **10–14** (unused by any earlier run). Each part is frozen by its own commit before any of its confirmatory runs; the commit hashes are recorded in `docs/v4/PROGRESS.md`. Exploratory pilots use seeds 0–4 and are labelled as such.

## Part C: is it the *luck*? Expected-value credit vs realised difference rewards (frozen first)

**Question.** v3 found that exact (realised) difference rewards speed up non-shared IPPO 38× at K = 8 lanes. `docs/v3/theory.md` §7 predicts the gain comes from removing *other agents' luck*, which only a realised counterfactual can do. An expected-value credit (what any learned critic gives) cannot. This part tests that directly.

**Design.**
- Environment: `multi_event_layout(K)` (walled lanes, d = 0, horizon 25), K ∈ {4, 8}.
- Learner: EPyMARL `ippo_ns` (no parameter sharing), default hyperparameters, test every 2,000 steps (32 episodes); budgets K = 4: 1.2M, K = 8: 2M steps; early stop after 5 consecutive test points ≥ 0.9 (does not change the metric).
- **Arms** (same seeds):
  - `team`: shared reward.
  - `difference`: realised difference reward, r − r^(−i), from lockstep shadow copies = agent i's own lane reward exactly.
  - `expected` (new): κ_i(t) = r_t + W₋ᵢ(s_{t+1}) − W₋ᵢ(s_t), with W₋ᵢ(s) = Σ_{j ≠ lane(i)} V_j(s_j) and per-lane value tables V_j learned online by TD(0) (α = 0.1) in each environment worker. It equals *own lane reward + the other lanes' TD errors* exactly. With perfectly learned tables it is the exact expected-value ablation credit. It differs from `difference` by exactly the other lanes' luck.
- Metric: episodes to 90% greedy test success (first crossing; censored at budget).

| ID | Prediction | Test |
|---|---|---|
| C1 | Expected credit does not deliver the speed-up | K = 8: mean ratio episodes(expected)/episodes(team) > 0.5, i.e. not more than 2× faster (one-sided paired t-test on log ratios > log 0.5, p < 0.05, pairs by seed) |
| C2 | Realised beats expected by a wide margin | K = 8: episodes(difference)/episodes(expected) < 0.2 (paired one-sided t-test on log ratios < log 0.2) |
| C3 | Replication of v3 Addendum A2 on new seeds | K = 8: episodes(difference)/episodes(team) < 0.2 |
| C4 | Descriptive | The same three ratios at K = 4; learning curves |

**Interpretation fixed in advance.**
- C1 + C2 supported: the learning benefit of credit assignment here is the removal of other agents' luck; expected-value credit (any critic) does not provide it.
- C1 not supported: expected credit captures a meaningful part of the gain, i.e. better baselines matter; theory §7's practical claim would be wrong for non-shared IPPO.

## Part A: mixed-size training of the set critic (frozen second)

**Question.** v3 found that a set critic trained at one team size picks the right coalition at unseen sizes but leaks 14–30% of the credit to bystanders. Hypothesis: the leak is caused by training at a single size (nothing forces a bystander's token to contribute zero to the pooled value). Training on several sizes should remove it.

**Design.** `scripts/v4/size_mixed.py`. Shuffled roles, d = 25.
- Per training size: 200 full-team episodes + 200 stops (stratified, no-op null).
- **mixed**: one set critic on all training sizes.
- **single**: one set critic on a single size with the *same total* number of stops (data-matched control).
- Read out on the 30 test episodes of each held-out size against exact closed-loop no-op truth. The in-size critic and the equal split are references.
- Designs:
  - **interp**: train b ∈ {1, 3, 5} (4, 6, 8 agents), test b ∈ {2, 4} (5, 7 agents);
  - **extrap**: train b ∈ {1, 2, 3}, test b ∈ {4, 5} (7, 8 agents).
- Seeds 10–14. Pilot (seeds 0–2; `results/v4/size_mixed_pilot_*.json`): interp mixed 0.002–0.062 vs single 0.16–0.32; extrap mixed 0.03–0.17 vs single 0.55–0.94.

| ID | Prediction | Test (per held-out size; Holm across sizes within the design) |
|---|---|---|
| A1 | Interpolation: mixed-size credit is near exact | L1 < 0.10 |
| A2 | Interpolation: mixed beats the data-matched single-size critic | paired, single − mixed > 0.10 |
| A3 | Extrapolation: mixed-size credit is good | L1 < 0.25 |
| A4 | Extrapolation: mixed beats single by a wide margin | paired, single − mixed > 0.30 |
| A5 | Mixed keeps exactly the coalition at every held-out size | top-k > 0.90 |

## Part D: randomised start cells (frozen third)

**Question.** Deferred v3 item 2.5. Shuffled roles (v3 P8) move start cells only among three fixed role slots, so travel times and the moment the lever is pulled are identical in every episode. With **random start cells for every role** they vary per episode, and the per-step credit (when a stop stops mattering) depends on them. Does the set critic's near-exact credit survive a continuous geometry axis?

**Design.** `scripts/v3/run_cells.py --grid confirm_rs`. Coalition layout, d = 25, horizon 60, b ∈ {2, 4} bystanders; roles shuffled and start cells drawn per episode (`envs.layouts.random_role_starts`: coalition members anywhere except stations; bystanders anywhere off the coalition's planned routes, so their true credit stays zero, as with the fixed slots). 200 training / 30 test episodes, 200 stratified no-op stops; methods as v3 P8 (`atd` = MLP, `atd_set`, `atd_set_struct`, `atd_n800`, `atd_n800_set`, `atd_noint`, `table3_mix`, `uniform`). Truth: exact closed-loop no-op LOO (`loo_cn`) and exact per-step κ (`step_r`). Seeds 10–14.

Pilot (seeds 0–2, `results/v4/pilot_rs`): `atd_set` L1 0.014 (b = 2), 0.022 (b = 4), step r 1.00 / 0.99; `atd` 0.48 / 0.92; equal split 0.80 / 1.14.

| ID | Prediction | Test (per b; Holm across b) |
|---|---|---|
| D1 | Set-critic credit stays near exact | `atd_set` L1 (loo_cn) < 0.05 |
| D2 | Its per-step credit stays faithful | `atd_set` step r > 0.95 |
| D3 | The MLP critic does not generalise over geometry as the set critic does | paired, `atd` − `atd_set` > 0.30 |

## Part B: Level-Based Foraging (frozen fourth)

**Question.** Does credit by persistent intervention, with exact truth, carry over to a standard MARL benchmark where every episode has a new random layout? How the design was chosen (and which designs failed, and why) is in `docs/v4/lbf_design.md`, all on pilot seeds 0–4.

**Design.** `scripts/v4/lbf_cells.py`, deterministic LBF (`envs/lbf.py`, tested step for step against `lbforaging` 2.0.0). Scripted greedy coalition team, ε = 0.2. **Removal null**, **heavy food** (`food_cap="all"`: food levels up to the sum of all agent levels), so that which agents are pivotal for which food is the credit question. Truth: exact closed-loop removal LOO and the exact t = 0 game v(S) for every subset, on 30 test layouts per cell; luck floor = the layout's expected LOO over 16 extra noise draws, used as a prediction.
- Variants: **heavy** (4 agents, 3 food, 8×8, H = 50; primary) and **heavy_large** (6 agents, 4 food, 10×10, H = 60; descriptive).
- Budgets: **std** = 200 training layouts, 200 stops, 3,000 critic steps (v3's budget); **large** = 800 layouts, 800 stops, 6,000 steps.
- Methods: `set` (two-set critic, primary, as in v3), `set_noint`, `mlp`, `mlp_hi` (4× stops), `food` / `food_noint` (per-food reward-decomposed critic, secondary), structural read-outs, `uniform`, `floor_expected`.
- Seeds 10–14. Output: `results/v4/lbf_confirm/{std,large}/`.

Pilot (seeds 0–4), heavy: set L1 0.30 (std) / 0.14 (large); equal split 0.45; luck floor 0.05; MLP 0.67 / 0.58; no stops 0.96 / 1.18; Shapley L1 set 0.13 / 0.07 vs equal split 0.27.

| ID | Prediction (heavy) | Test (per budget; Holm across the two budgets) |
|---|---|---|
| B1 | The set critic beats the equal split | paired, uniform − set > 0 (LOO L1) |
| B2 | Stops are necessary | paired, set_noint − set > 0.30 |
| B3 | The symmetric critic beats the MLP | paired, mlp − set > 0.15 |
| B4 | It also recovers the t = 0 Shapley values better than the equal split | paired, uniform − set > 0.05 (Shapley L1) |
| B5 | With the large budget it is close to exact | large only: set L1 < 0.25 |
| B6 | More data helps | paired by seed, set(std) − set(large) > 0 |

**heavy_large (descriptive, expectation written now).** From the pilot: the set critic beats the equal split on LOO L1 (0.96–0.99 vs 1.66) but not on Shapley L1 (0.80–1.05 vs 0.89), picks the right agents in only 35–53% of layouts, and does not improve with 4× data. Expectation: the same pattern. Interpretation fixed in advance: with 6 agents and 4 foods the critic is far from exact at these budgets; the LBF result is claimed for 4 agents only.

**Interpretation fixed in advance.** B1–B5 supported: credit by intervention carries over to random-layout LBF with 4 agents, and its error approaches the luck floor with data. B1 not supported at std: at v3's budget a random-layout benchmark needs more data than a fixed layout, which B5/B6 then quantify.
