# Mode P Protocol (as implemented)

Mode P measures how faithfully each method's **credit-bearing component** assigns credit when learning is taken out of the picture. Code: `whoearned/experiments.py`, `whoearned/mode_p.py`, `scripts/run_mode_p.py`.

## Data cell

A data cell is `(delay d, number of bystanders b, seed s)` on `coalition_layout(b, d, horizon=60)`:

- Coalition {A, B, C}: A pulls the lever, then B and C push together. Reward 10, paid d steps after completion.
- b bystanders on row 4 (`wander` mode: they never step onto coalition routes).

For each cell:

1. **Training data.** 200 scripted episodes (`ScriptedTeam`, ε = 0.2), seeds `s·100000 + 0..199`.
2. **Test data.** 30 scripted episodes, seeds `s·100000 + 50000 + 0..29`, disjoint from training.
3. **Ground truth for each test episode.**
   - Default: open-loop replay with true removal.
   - H0 sensitivity: also computed with the no-op null policy.
   - Recorded per episode: Shapley values, leave-one-out marginals, whether realised = planted, and whether the payout was clipped.
4. **Algorithms.** Every algorithm is trained and scored on exactly this data.

## Training (critic only)

| Algorithm | What is trained | Update |
|---|---|---|
| VDN, QMIX | Agent utilities + mixer (the Q-learner) | `learner.train` (double Q, target network, standardised rewards: EPyMARL defaults) |
| COMA | Centralised Q critic only | `_train_critic` (n-step targets, q_nstep = 10) + EPyMARL's target updates |
| MAPPO | Centralised V critic only | `train_critic_sequential` (q_nstep = 5) + soft target updates (τ = 0.01) |

Defaults for all: 1,500 updates, minibatches of 32 episodes drawn from the 200, network sizes and learning rates from EPyMARL's algorithm configs. Actors are **never** trained.

**Caveat (state in the report).** VDN/QMIX learn Q* (greedy targets), not Q of the scripted policy. COMA and MAPPO critics learn the value of the behaviour policy. On the near-optimal scripted path these are close, but not identical.

## Read-outs and targets

| Algorithm | Per-step read-out | Episode credit | Compared with |
|---|---|---|---|
| VDN, QMIX | Q_tot(s, a) − Q_tot(s, a_i := noop, a_−i) | Sum over t | Leave-one-out marginals |
| COMA | Q(s, a) − Σ_a′ π_b(a′) Q(s, (a′, a_−i)), with π_b = the exact scripted behaviour probabilities | Sum over t | Leave-one-out marginals |
| MAPPO | n-step return − V(s) (shared; reference only) | Sum over t | Leave-one-out marginals |

MAPPO also reports the policy-gradient signal-to-noise ratio per agent: g_i = Σ_t A_t (onehot(a_i,t) − π_b,i,t) across test episodes.

Shapley-target scores are stored too (`summary_shapley`). In the single-event coalition layouts, both targets give the same shares (⅓ each for A, B, C).

## Two-level grading (added after pilot 1)

Each test episode also gets a **per-step counterfactual ground truth** (`replay.step_counterfactuals`). For every step t, agent i and alternative action a, the scripted team is re-run with only that one action changed (closed loop, same random streams), and the **discounted return from t** is recorded (γ = 0.99, the critics' discount). From it:

- **noop effect** = G_t − G_t(i does noop at t). VDN and QMIX (Q_tot difference) estimate this; MAPPO is scored against it as a reference.
- **baseline effect** = G_t − Σ_a π_b(a) G_t(i does a at t). COMA's counterfactual advantage estimates this.

Three comparisons per method and cell:

| Level | Compares | Answers |
|---|---|---|
| **1. Fidelity** | Read-out vs its per-step truth: Pearson over all (episode, step, agent), and L1 of episode-summed shares | Is the critic accurate about what it is meant to estimate? |
| **2. Semantic gap** (method-independent) | Per-step truth, summed over the episode, vs leave-one-out removal credit | Even with a perfect critic, how far is this notion of credit from "who earned the reward"? |
| **End to end** | Read-out vs leave-one-out | The combined effect, as in v2 |

**Why discounting matters.** The scripted team recovers from one-step deviations, so a single action rarely changes the *undiscounted* return. Under γ, it costs the delay it causes. An enabler with slack (it acts before the finishers are ready) therefore has a per-step effect of exactly 0, even for a perfect critic. A late enabler on the critical path does not (`tests/test_step_truth.py`).

Cost: about 7 s per test episode with 5 agents, 13 s with 7. Counterfactual runs stop early when they re-merge with the factual trajectory; this is checked against brute-force rollouts.

## Critic state (changed after pilot 1)

The global state now includes each event's payout-pending flag and countdown, so it is Markov under delay. Agent observations are unchanged.

## Metrics per test episode

L1 credit error, Spearman, leakage Λ (planted bystanders), missed enabler credit M. Episodes where the credit sums to ~0 are counted as invalid, not dropped. Per data cell we store per-episode rows and their means. The summary reports mean ± sd over seeds.

## Outputs

`results/mode_p/<grid>/`:

- `d{d}_b{b}_s{s}.json`: one per data cell;
- `per_seed.csv`;
- `summary.md`.
