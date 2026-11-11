# Pre-registration v3, Phase 3: the headroom gate (frozen)

**Frozen at the commit that adds this file. No `gate` run (seeds 5–9) existed before it.** The only Phase 3 runs before it were the single-lane feasibility run (seed 0, team reward) and a K = 8 sizing run on seed 0 that was stopped at 20k steps and deleted, before it produced any learning signal.

## Question

Does exact credit change how fast a team learns, and when? `docs/v3/theory.md` §7 predicts:
- **expected**-value credit (any state-based counterfactual, including ablation-TD credit) gives the same per-agent advantage as the team reward with a good baseline, so it cannot help;
- **realised** counterfactuals (difference rewards from shadow copies under common random numbers) remove other agents' luck from each agent's signal, so they can help, but only when the team reward mixes in many unrelated events.

## Design

- **Environment:** `multi_event_layout(K)`: K walled lanes, each with one enabler and one finisher, reward 10 per completed lane, delay 0, horizon 25, lane length 4. n = 2K.
- **Learner:** EPyMARL IPPO, default `ippo.yaml` hyperparameters, unmodified.
- **Arms:**
  - **team**: shared reward, `common_reward=True`;
  - **team_vector**: each agent receives the team reward as its own (`common_reward=False`), the control for the per-agent-reward code path;
  - **difference**: r − r^(−i), where r^(−i) comes from a shadow copy of the environment fed the same joint actions with agent i at no-op from t = 0.
- **K ∈ {1, 2, 4, 8}. Seeds 5–9.** Budgets: K = 1: 400k, K = 2: 800k, K = 4: 1.2M, K = 8: 2M environment steps (shadow copies are not counted in the budget; their cost is reported separately as (n + 1)× environment steps).
- **Metric:** greedy test success rate (fraction of lanes completed; 32 test episodes every 10k steps). **Episodes to 90%** = first test point with success ≥ 0.9, divided by 25 steps per episode. A run that never reaches 0.9 is censored at its budget. Censoring the team arm makes difference/team ratios *larger*, so it is conservative for H3.2.

## Hypotheses

| ID | Hypothesis | Supported if |
|---|---|---|
| H3.1 | No headroom with one event | K = 1: the ratio of mean episodes-to-90% (difference / team) lies in [0.8, 1.25] |
| H3.2 | Headroom grows with events | K = 8: one-sided paired t-test (pairs by seed) on log(episodes_difference / episodes_team) < log(0.67), p < 0.05 |
| H3.3 | The control is inert | team_vector / team ratio in [0.8, 1.25] at every K (descriptive; a failure means plumbing differences, and H3.2 is then re-tested against team_vector) |
| H3.4 | Monotone | The mean log ratio (difference / team) decreases with K (Spearman over the four K values, descriptive) |

**Gate decision (plan §8, 3.5):** if H3.2 is supported, there is a learning story, and Phase 4 is run as planned. If not, the learning track stops, and the write-up is a measurement-and-attribution paper. Either way, every number is reported.

## Not run, and why

The plan's arm (c), ablation-TD credit from a critic trained alongside, is not run. With exact critics its per-agent advantage *equals* the team TD error (`theory.md` §7), so it can only differ from the team arm through critic error. `team_vector` controls the per-agent-reward code path instead.

## Implementation notes (added before any gate result was looked at)

- The first launch failed in 2 s per run: sacred rejects env keys not declared in `configs/envs/planted_lanes.yaml`, and `dropout_p` was not declared. No run produced data. The keys were declared and the gate relaunched.
- The second launch ran for about one minute (at most about 20k steps per run, 3 runs) and was then stopped to add the early stop below. Its partial logs were deleted without any success value being read. All gate runs restart from scratch.
- **Early stop:** a run is stopped once its last 5 consecutive test points (50k steps) are ≥ 0.9. This changes no pre-registered quantity: episodes-to-90% is the *first* test point ≥ 0.9, which is fixed by then. Runs that never stabilise run to their budget. Each run records `stopped_early` in `run_info.json`.
