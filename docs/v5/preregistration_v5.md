# Pre-registration v5 (response to the external review)

Frozen in one commit before any confirmatory run of Parts E and F (hash recorded in `docs/v5/PROGRESS.md`). Seeds **10–14**. Part E's arms are run on the same seeds as v4 Part C, which it is compared with; Part F's seeds were never used for Phase 4. The only pilot data are the v3 Phase 4 checkpoints (seeds 5–7), re-scored with the new robustness metric (`results/v5/robust_pilot`).

## Part E: a MAPPO arm for the learning question (review: "Phase 3: no MAPPO arm")

**Question.** v4 Part C found that expected-value credit speeds up *non-shared IPPO* 4.2× at K = 8. Part C's reading: §7's equivalence assumes a critic that sees the full state, and IPPO's critics see only their own observation. The direct test: change only the critic. EPyMARL's `mappo_ns` is `ippo_ns` with a centralised state critic (`cv_critic_ns` instead of `ac_critic_ns`). Everything else (non-shared actors, hyperparameters, budgets, test interval, early stop, seeds) is as in Part C.

**Design.** Grid `credit_mappo_ns`: K ∈ {4, 8} walled lanes; arms team, difference, expected; seeds 10–14. Descriptive grid `credit_mappo`: shared-parameter MAPPO, K = 8, the same three arms (this is the v3 plan's MAPPO arm). Metric: episodes to 90% greedy test success; simulator cost (difference = 2K + 1 simulators per step) reported for every arm.

| ID | Prediction (K = 8, `mappo_ns`) | Test |
|---|---|---|
| E1 | With a centralised critic, expected credit does **not** speed learning up more than 2× | expected/team > 0.5 (one-sided paired t-test on log ratios, p < 0.05) |
| E2 | Realised difference rewards still do | difference/team < 0.5 (one-sided paired t-test on log ratios) |
| E3 | The expected-credit gain shrinks when the critic sees the state | log(expected/team) under `mappo_ns` > the same under `ippo_ns` (v4 Part C), one-sided Welch t-test |
| E4 | Descriptive | K = 4; shared MAPPO at K = 8; drawdowns; simulator cost; team(`mappo_ns`)/team(`ippo_ns`) |

**Interpretation fixed in advance.**
- E1–E3 supported: §7 holds when its assumption holds. Part C's expected-credit gain is the information a local critic lacks, which a centralised critic supplies.
- E1 fails: expected credit helps even with a state critic, e.g. through the variance of 5-step targets. §7's practical claim is then wrong for PPO-style learners, whatever critic they use.

## Part F: Phase 4 extended (review: "one dropout level, one layout and 3 seeds; robustness not measured")

**Question.** Does training with random persistent stops (dropout) make MAPPO's critic carry credit, across dropout levels and layouts? And does it make the *team* robust to absences?

**Design.** Grid `phase4_v5`: EPyMARL MAPPO, team reward, 300k steps.
- K ∈ {2, 4} walled lanes, each layout with 2 task-less bystanders (n = 6 and n = 10).
- Dropout p ∈ {0, 0.1, 0.25, 0.5}.
- Seeds 10–14.

The **final** checkpoint is scored on 30 clean test episodes (`scripts/v5/eval_final.py`) against the exact persistent-ablation truth of that policy (as in v3 4.2): κ, structural V(s₀) − W₋ᵢ(s₀), and the equal split.

**Robustness (new metric).** Collateral damage of an absence: agent j stopped from t = 0 (no-op null, exact closed-loop continuation of the learned policy) should cost at most the payouts of j's own events. Collateral = (return of the other events in the clean episode) − (return with j stopped), as a fraction of the other events' return.

Pilot (v3 checkpoints, seeds 5–7, K = 2, p ∈ {0, 0.25}):
- structural L1 1.52–1.99 (p = 0) vs 0.43–0.74 (p = 0.25);
- equal split 0.67;
- collateral 0–2% in both arms. Walled lanes give an absence almost no way to hurt other events.

| ID | Prediction | Test (Holm across the conditions listed) |
|---|---|---|
| F1 | Dropout makes the critic carry credit | paired by seed, structural L1(p = 0) − structural L1(p) > 0, for each K and each p > 0 (6 conditions) |
| F2 | With dropout 0.25 the structural read-out beats the equal split | paired, uniform − structural > 0 at p = 0.25, each K (2 conditions) |
| F3 | Robustness training costs little return | clean success ≥ 0.95 at p = 0.5, each K (one-sided test of mean > 0.95; 2 conditions) |
| F4 | Descriptive, expectation written now | collateral < 5% of the other events' return in **every** arm, including p = 0. If so, these layouts cannot show a robustness benefit, because there is no fragility to remove. That is reported as a limit of the test, not as evidence for or against robustness. |
| F5 | Descriptive | structural L1 as a function of p, per K |
