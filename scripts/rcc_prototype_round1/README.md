# rcc_prototype

Scripts and raw outputs for the tests in `WHO_EARNED_review_and_plan.md` (10 Nov 2026).
Nothing here modifies the repository. Requires numpy, scipy, scikit-learn; no PyTorch.

Place this folder at `<repo>/experiments/rcc_prototype/`, or set the environment variable
`WHOEARNED_REPO` to the AFML_project checkout.

| File | Purpose |
|---|---|
| `rcc_lib.py` | Mid-episode removal, intervention sampler, exact persistent values, sparse dividends, torch-free `Cell` |
| `run_rcc.py` | Per-cell pipeline for Tests 3-5 (`h3`, `main_b2`, `main_b4`, `smoke` grids) |
| `aggregate.py`, `readout_variants.py` | Tables from the per-cell JSONs, next to the repo's stored results |
| `scaling.py` | Data-scaling test (needs the per-cell JSONs; set `RCC_RESULTS=results`) |
| `uniform_main.py` | Uniform baseline on the stored main grid (Test 1) |
| `coma_zero_mean.py` | Numerical check of the zero-mean identity (Test 2) |
| `aug_test.py` | Pending-payout-amount state test (Test 6a) |
| `stepgame_truth.py` | Exact per-step coalition game (Test 6b) |
| `toy_headroom.py`, `toy16.py` | Toy learning test (Test 7) |
| `results/` | My outputs: `out_main/`, `out_h3/` (one JSON per cell) and the summary files |

Quick check that everything is wired up (seconds):

    python aggregate.py results        # should reproduce results/agg_final.txt
