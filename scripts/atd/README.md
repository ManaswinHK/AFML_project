# atd: ablation-TD credit prototype (round 2, 11 Nov 2026)

Runs against `DBS13062006/AFML_project` at commit `5ddf18f`. Nothing in the repository is modified.

Setup: put this folder at `<repo>/scripts/atd/`, or set `WHOEARNED_REPO=<path to repo>`.
Needs numpy, scipy, scikit-learn, PyTorch (CPU is fine). EPyMARL is not needed.
Run every command from inside this folder.

| File | What it does |
|---|---|
| `atd.py` | Library: mid-episode stopping (no-op and removal nulls), exact ablation values, ablation-TD credit, intervention data, the MLP critic, the low-order dividend critic. |
| `run_oracle.py`, `agg_oracle.py` | Test A: exact credit on the 120 main-grid cells. |
| `credit_profile.py` | Test A: the one-episode example. |
| `run_learned.py all`, `agg.py` | Tests B and C: learned critic, no-intervention control, Lasso table on the repo's `mix` data, compared with stored results. |
| `fix_delay.py` | Test B: sampler that covers the payout wait. |
| `prune.py` | Test D: top-3 selection from the repo's stored shares. |
| `run_perm.py` | Test E: shuffled roles. |
| `out/` | Per-cell JSON outputs behind every number in the write-up. Scripts skip cells whose JSON exists. |

Everything is exploratory (seeds 0-4, the same seeds as the repo's runs).
