# Pre-registration v3, Phase 4 (reduced): credit for free from robustness training? (frozen)

Frozen at the commit that adds this file. No `phase4` run existed before it.

## Design

- **Environment:** `multi_event_layout(K = 2, n_bystanders = 2)`: two walled lanes (enabler + finisher, reward 10 each) and two task-less bystanders in their own row. n = 6, horizon 25. Bystanders make the credit non-uniform: an equal split is wrong.
- **Learner:** EPyMARL MAPPO, default hyperparameters, team reward; 300k steps; a checkpoint every 30k steps.
- **Arms:** `dropout_p` ∈ {0, 0.25}. With probability p per training episode, a random set of agents stops for good at a random step (no-op null; the global state's acting flags show it). Seeds 5, 6, 7.
- **Evaluation of each checkpoint:** 30 clean test episodes with the trained policy (its own sampling noise, no dropout). The **exact** persistent-ablation truth for that policy comes from branching it (`BranchableMacTeam`: env clone + policy RNG, RNN hidden state and batch buffer; tested exact).
- **Read-outs:**
  - **masked**: MAPPO's V at the state with agent i stopped → ablation-TD credit;
  - **standard**: MAPPO's shared advantage;
  - **uniform**: equal split.
- **Metric:** L1 against the exact leave-one-out truth.

## Predictions

| ID | Prediction | Test |
|---|---|---|
| H4.1 | Dropout training makes MAPPO's own critic an ablation critic | At the final checkpoint: masked L1 (p = 0.25) < masked L1 (p = 0) in each of the 3 seeds, and the mean masked L1 (p = 0.25) is below the mean uniform L1 |
| H4.2 | Without stops in its data, the masked read-out is no better than the equal split | Final checkpoint, p = 0: mean masked L1 ≥ mean uniform L1 − 0.1 |
| H4.3 | Robustness training costs little return | Final clean success (p = 0.25) ≥ 0.9 × final clean success (p = 0), mean over seeds |
| H4.4 | Fidelity over training (descriptive) | Masked L1 per checkpoint, both arms |

Three seeds: these are directional predictions with descriptive statistics, not significance tests.
