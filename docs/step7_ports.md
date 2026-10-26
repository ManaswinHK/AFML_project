# Step 7: Porting NA²Q and STAS

Both methods are reimplemented in this repository from their official code, rather than run inside the official codebases. The official code is the reference; where we deviate, it is listed below with the reason.

| Method | Paper | Official code (commit read) | Our module |
|---|---|---|---|
| NA²Q | Liu, Zhu & Chen, ICML 2023 | github.com/zichuan-liu/NA2Q @ `adc10b4` | `src/whoearned/na2q.py` |
| STAS | Chen, Zhang, Yang & Du, AAAI 2024 | github.com/zowiezhang/STAS @ `6312725` | `src/whoearned/stas.py` |

**Licence.** Neither repository has a licence file. We therefore wrote clean implementations of the same architecture and losses rather than copying code. Both are cited in the problem statement ([3], [14]). Clone the official repositories into `external/` (git-ignored) only to compare.

## NA²Q

**Model** (from `src/modules/mixers/qnam.py`, `src/learners/qnam_learner.py`, `src/modules/agents/rnn_sd_agent.py` and `src/modules/intrinsic/qnam_context.py`):

- **Agents** (`rnn_sd`): input is observation + previous action (no agent id). A shared GRU (hidden 64) feeds a shared Q head plus a per-agent Q head.
- **VAE** on each agent's GRU state. Its decoder outputs a sigmoid mask over the agent's observation. Loss: MSE(mask·obs, obs) + KL + L1(mask).
- **Mixer.** Q_tot = Σ_k w_k f_k(Q_{S_k}) + v(s), where k runs over every single agent and every pair (`mix_nary: [1, 2]`).
  - Each f_k is a 1→8→4→1 MLP with non-negative weights and ELU, so it is monotone.
  - w is a softmax over terms: an attention whose query comes from the state and whose key comes from the VAE latents of all agents.
  - v(s) is a 3-layer MLP.
- **Training.**
  - Loss: TD loss + 0.1 × VAE loss.
  - Optimisers: RMSprop (lr 5e-4) for the agents; Adam (lr 5e-4) for mixer + VAE.
  - Gradient clipping 10 on each.
  - Double-Q, γ = 0.99, target update every 200 episodes.
  - No reward standardisation (the official NA²Q config has none).

**Deviation 1, Mode P only: SARSA target.** Bootstrapping uses the data's next joint action, not the double-Q max.

- **Why.** On fixed scripted data the official max target diverges.
  - With 200 noisy episodes, batch 32 and 1,500 updates, Q_tot at t = 0 reached 97–143, against a true discounted return of about 8.
  - The max selects per-agent actions the data never takes. NA²Q's per-agent heads and monotone pair terms amplify their overestimates.
- **The divergence is in the target, not the network.** The same network and data with a SARSA target converge (Q_tot(t=0) ≈ 8.3, loss 0.04).
- **Comparison.** On a 6-episode test batch without reward standardisation, EPyMARL's QMIX overshoots far less: max Q_tot is about 1.7× the true return, against about 12× for NA²Q with the max target.
- **Consequence.** In Mode P, NA²Q estimates the scripted behaviour policy's Q, as COMA's critic does, rather than Q* as VDN/QMIX do. `NA2Q(target="max")` restores the official update for Mode T.

**Deviation 2: deterministic read-out.** The read-out uses the VAE mean as the latent instead of a sample.

## STAS

**Model** (from `models/reward_model/mard/{mard.py, modules.py}`, `utils/util.py::make_train_step` and `configs/STAS_AandB.py`). We use the default model, STAS_ML (`reward_model_version: v2`).

- **Input embedding:** observation embedding + action embedding + positional embedding.
- **Three layers**, each:
  1. causal temporal self-attention per agent;
  2. "Shapley attention" over agents. Each agent attends to a random coalition (Bernoulli(0.5) mask, always including itself), averaged over 3 sampled masks.
- **Head:** a linear layer on the concatenated layer outputs gives r̂_{i,t}.
- **Loss:** MSE between Σ_{i,t} r̂_{i,t} and the episode return.
- **Hyperparameters:** hidden 128, 4 heads, dropout 0.5, Adam lr 1e-4, weight decay 1e-5.

**Kept from the original despite looking odd:**
- the residual wiring of its encoder layer;
- agent identity embeddings that are randomly permuted on every forward;
- dropout = 0.5. The runner passes `dropout=0.3`, but STAS_ML only uses `emb_dropout`, whose default is 0.5.

**Deviations:**
1. **Time mask.** The time mask is repeated per agent with `repeat_interleave`. The original's `repeat` pairs masks with the wrong episodes when lengths differ; all our episodes have the same length, so the result is identical.
2. **Read-out averaging.** The read-out averages 16 stochastic forwards in eval mode, where the original uses one. Each forward samples coalition masks, so a single one is a noisier estimate of the same expectation.
3. **Training budget and batch.** Mode P gives every method the same budget: 1,500 updates on batches of 32 episodes. The original trains with batches of 256 trajectories. Known effect 2 below shows that a 6× larger budget does not change STAS's agent split.

4. **Device.** STAS runs on the GPU when one is available. Every other network stays on CPU.

**Known effect 1: eval-mode sums overshoot.** The original decomposes in eval mode (dropout off) but trains with dropout 0.5. So the eval-mode r̂ sum overshoots the return: by about 20% on one run and about 70% on another with the same data. The train-mode sum, which is what the loss constrains, matches. We keep the original's eval-mode read-out. Normalised shares remove a common scale, but not any non-uniform distortion.

**Known effect 2: STAS barely separates agents. This is a property of the method, not of the budget.** It is measured, and it matters for every STAS result.

- **Main-grid cell (d = 0, b = 2, seed 0).** True Shapley shares are A, B, C ≈ 0.32 and bystanders ≈ 0.03. STAS gives ≈ 0.2 to every agent.

  | updates (on GPU) | train loss | bystander share |
  |---|---|---|
  | 1,500 | 9.2 | 0.54 |
  | 3,000 | 7.2 | 0.39 |
  | 6,000 | 3.4 | 0.38 |
  | 9,000 | 1.2 | 0.41 |

  The loss falls six-fold, but the agent split stays uniform.
- **Synthetic check.** n = 5 and T = 20, with random observations except a flag in agent 0's own observation. The return is the number of steps on which agent 0 chose action 1, so agent 0 deserves 100%. After 3,000 updates:
  - agent 0 receives 25–27% of the credit;
  - in time, credit does concentrate on agent 0's rewarding steps (0.50 vs ≈ 0 elsewhere).
- **Why.** Each agent's reward is read from attention over a random coalition that usually includes the agents who matter, and the loss constrains only the total. Nothing pushes an uninvolved agent's output to zero.
- **Is it a porting bug?** We compared our code with the official model line by line. We did not execute the official code.
- **Mode P setting.** Mode P keeps STAS at the shared budget (1,500 updates, batch 32). This finding should be reported alongside any STAS number.

## Read-outs

| Method | Individual credit | Matched target | Coalition credit |
|---|---|---|---|
| STAS | Σ_t r̂_{i,t} | Shapley φ | none (individual only) |
| NA²Q | Shapley value of its mixer game | Shapley φ | mixer-game dividends |
| VDN, QMIX | unchanged: Q_tot difference with agent i switched to noop | leave-one-out | mixer-game dividends (new) |

**Mixer game.** At step t, g_t(S) = Q_tot(s_t, a_S, noop_{−S}) − Q_tot(s_t, noop_all), using the learned mixer and the agents' own utilities. Its exact Harsanyi dividends (fast Möbius transform over all 2^n coalitions) are summed over the episode. They are then compared with the true dividends of the episode's coalition values (`experiments.coalition_scores`) using:
- normalised dividend error, over all orders and over orders ≤ 2;
- Jaccard recall/precision of the coalitions holding > 10% of |dividends|.

Two properties:
- **NA²Q's mixer game has no dividends above order 2**, because every mixer term depends on at most two agents' utilities (unit test `test_na2q_mixer_game_has_no_dividends_above_order_two`). Its Shapley value is therefore its unary term plus half of each pairwise term.
- **VDN's mixer game has only singleton dividends** (unit test with an additive mixer). Any coalition structure VDN reports is therefore impossible by construction, which is a useful control.

The planted coalition {A, B, C} is order 3. NA²Q (pairs at most) and VDN (singletons) cannot represent it exactly, while QMIX can in principle. Under H3 this is the point: dividend error measures whether a method's coalition terms are right. The problem statement's H5 honesty check makes the same point for the low-order estimator.

## Smoke cell (main-grid data, d = 0, b = 2, seed 0)

| method | target | L1 | leakage | missed | mean shares (A, B, C, D, E) | dividend error | minutes |
|---|---|---|---|---|---|---|---|
| NA²Q | Shapley | 0.51 | 0.04 | 0.65 | 0.11, 0.42, 0.43, −0.02, 0.01 | 2.00 | 6 (CPU) |
| STAS | Shapley | 0.74 | 0.42 | 0.39 | 0.19, 0.21, 0.19, 0.19, 0.23 | n/a | 31 (CPU) |
| QMIX (main grid, for comparison) | leave-one-out | 0.44 | 0.10 | 0.33 | | | |

- NA²Q's dividend error is the maximum (2.0) by construction. This layout's only true dividend is {A, B, C} (order 3), and NA²Q has none above order 2. That is why the H3 grid uses the overlap and partial-synergy layouts, which have lower-order dividends.
- STAS takes about 3 minutes for 1,500 updates on a GTX 1650, against 31 on CPU.

## H3 smoke cells (d = 0, seed 0; all six methods; about 24 minutes per cell with 2 workers)

- **Pipeline.** Both layouts run end to end (`results/mode_p/h3_smoke/`).
- **Dividend error saturates at 2.0 for VDN, QMIX and NA²Q, over all orders and at order ≤ 2. This is not a scoring bug.**
  - Inspecting the raw dividends shows the learned mixer games are almost purely **additive**: nearly all |dividend| mass sits on single agents.
  - The true dividends are all coalitions:
    - overlap: {A, B, C} = 6 and {B, D} = 4;
    - partial: {A, B} = {A, C} = 10 and {A, B, C} = −10.
  - Disjoint support gives the maximum error.
- **Why.** The mixer game swaps actions *within one step*, but these coalitions act *across* steps: A pulls the lever, then B and C finish later. A per-step game cannot express synergy between actions at different times. This is the coalition-level form of the semantic gap found for H1b.
- **Added metric.** Saturated dividend error cannot rank methods, so `coalition_scores` now also reports *interaction mass*: the share of |dividends| on coalitions of 2+ agents. The true value is 1.0 on both layouts. Raw per-episode dividends are stored too.
- **Data caveats (H0).**
  - The partial-synergy layout matches its planted structure in only 43% of test episodes (overlap: 67%).
  - The partial cell's mean test return is 8.3, not 10.
  - Grading uses the realised dividends, as everywhere else.
- **STAS** again leaks heavily (0.48–0.54), and its Spearman rank correlation is negative (−0.72).

## Tests

`tests/test_ported.py` (11 tests):
- mixer-game dividends for additive and pairwise-product mixers;
- per-step game Shapley = exact Shapley;
- NA²Q attention weights form a distribution and its terms are monotone;
- NA²Q has no dividends above order 2;
- NA²Q with the SARSA target fits without diverging;
- STAS is causal in time;
- STAS learns to decompose the return;
- VDN/QMIX/NA²Q return dividends.
