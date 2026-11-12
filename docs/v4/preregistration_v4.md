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
