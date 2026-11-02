# H5 Pre-registration: Low-Order Episodic Coalition Estimator (2026-11-02)

**Written after three pilot cells (seed 0, d = 0, one per layout) and before running the H5 grid.** The pilot cells are also grid cells. They are reported with the rest, and this document says where they shaped the design.

Problem statement v2, H5:
- **Prediction:** lower dividend error and leakage than NA²Q at matched compute when the coalition size is ≤ k.
- **Honesty check:** evaluate on coalitions of size k + 1 too.

This document makes that rule concrete.

## 1. The estimator (ECE-k)

Code: `src/whoearned/coalition_estimator.py`.

**Return model.** R̂(τ) = Σ g_S(h_S) over the present coalitions S with 1 ≤ |S| ≤ k.
- h_i is a GRU encoding of agent i's identity and its **own action sequence only**.
- Terms of one order share weights and are symmetric in their members.

**Coalition read-out.**
- The model's game is v̂(M) = Σ over S ⊆ M of g_S.
- Its Harsanyi dividends are the terms themselves: d̂(S) = g_S for |S| ≤ k, and 0 above.
- Individual credit is the Shapley value: each term is split equally among its members.
- No replay is needed at test time.

**Loss.** MSE to the episode return, plus 0.05 · Σ|g_S| (L1) over the present terms.

**Training data, two conditions:**
- **Full-team** (`ece2`, `ece3`): the same 200 scripted episodes as every other method.
- **Replay-varied** (`ece2_mix`, `ece3_mix`): those 200 episodes, plus one open-loop replay of each with a random subset removed. Each agent is kept with probability 0.75, and removed agents are taken off the grid, as in the ground truth.
  - Cost: 200 extra replays per cell.
  - For comparison, the exact ground truth needs 2ⁿ replays *per test episode* (32 at n = 5).

**Compute.** Matched with the other methods: 1,500 updates, batch 32. Wall-clock time and parameter counts are reported.

## 2. What the pilot changed (disclosed)

1. **Full-team data cannot identify credit.** Every coalition and overlap training episode returns 10. The full-team estimator spreads credit over all agents: leakage 0.40, L1 equal to uniform. This was predicted, and it is kept as a control.
2. **Version 1 encoded the agent's own positions and flags, and trained on closed-loop rollouts with agents absent.** It fitted the coalition layout with three *pairs*, not the planted triple:
   - A finisher's own path reveals whether the enabler opened the gate.
   - So {B, C} alone predicts success, and removing A changes nothing in the model.
   - An L½ penalty did not fix this, and it was dropped.
3. **Version 2 (frozen) uses actions only and trains on open-loop replays.** This matches the ground-truth protocol, which holds the actions fixed and removes agents. Pilot, coalition layout:

   | | ECE-3_mix | ECE-2_mix |
   |---|---|---|
   | d̂({A,B,C}) (true 10) | 9.5 | – |
   | dividend error | 0.52 | 2.0 |
   | recall / precision | 0.95 / 0.95 | – |
   | L1 (uniform 0.71) | 0.13 | 0.15 |

   Overlap and partial were only partly recovered (dividend error 1.24 and 1.18; training loss had not reached 0).

No other hyperparameter was tuned. Hidden size 64, learning rate 1e-3 and L1 weight 0.05 are the first values tried.

## 3. Grid

- **Layouts:** coalition (b = 2; planted {A,B,C}, order 3), overlap (planted {A,B,C} and {B,D}, max order 3), partial (planted {A,B} or {A,C} per episode, order 2).
- **Delay:** d ∈ {0, 50}, with horizon 80 at d = 50.
- **Seeds:** 0–4. That gives 30 cells (`--grid h5`).
- **Methods on every cell:** `ece2`, `ece3`, `ece2_mix`, `ece3_mix`, `stas_mix`, `na2q_mix`.
- **Reference NA²Q and STAS on the same data cells:**
  - overlap and partial: from the H3 grid;
  - coalition: a new `h5_ref` run.
- The analysis checks that every result directory has identical ground truth for each shared cell.

A layout is **eligible** for ECE-k when its planted max order is ≤ k:
- ECE-3: all three layouts;
- ECE-2: partial only.

That gives 8 (k, layout, d) conditions.

## 4. Confirmatory tests

All tests use a one-sided paired t-test on per-seed means (5 seeds), with Holm correction within each test family. Exact Wilcoxon p is reported alongside. A test family is **supported** only if every eligible condition is significant after Holm.

- **H5a, dividend error:** ECE-k_mix < NA²Q (full-team, as in v2).
- **H5a, leakage:** ECE-k_mix < NA²Q.
- **H5a′ (data-matched), dividend error and leakage:** ECE-k_mix < NA²Q_mix. This separates the model from the extra data.
- **H5b, individual L1 against Shapley:** ECE-k_mix < uniform credit (1/n each). This bar is added because no method beat it on the H3 grid (`docs/h3_results.md`).

## 5. Descriptive (no verdict)

- **Honesty check:** ECE-2_mix against ECE-3_mix on the order-3 layouts (coalition, overlap): dividend error and L1.
- **Full-team controls:** `ece2` and `ece3` against uniform.
- **STAS_mix against STAS:** whether the replay data alone helps an existing return-decomposition method.
- **Delay:** d = 0 against d = 50. ECE sees only actions and the total return, so it should be delay-invariant except for clipped payouts. STAS showed the same property.
- **Cost:** seconds and parameters per method.

## 6. Analysis code

`scripts/analyse_h5.py` writes `results/mode_p/h5/h5_analysis.md`.
