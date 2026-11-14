# v5: response to the external review

Start with **`review_response.md`**: each review point, what was checked, what was corrected, and the new runs.

| File | What it is |
|---|---|
| `review_response.md` | Point-by-point response, with the results of Parts E and F |
| `preregistration_v5.md` | Parts E (MAPPO arm) and F (Phase 4 extended), frozen at `0ab52d7` before any run |
| `results.md` | Generated verdicts (`scripts/v5/analyse_v5.py`) |
| `cost_accounting.md` | Simulator cost of difference rewards for every Phase 3 grid (`scripts/v5/cost_accounting.py`) |
| `figures/` | `scripts/v5/figures.py` |
| `../v3/errata.md` E11–E15 | Corrections to the v3 record |

Raw results: `results/v3/**.gz` and `results/v5/**.gz` are the published, lossless copies (`results/` is otherwise git-ignored). Restore with:

```bash
.venv/bin/python scripts/v5/restore_results.py results/v3 results/v5
```

Reproduce the new runs:

```bash
.venv/bin/python scripts/v3/run_phase3.py --grid credit_mappo_ns --workers 4
.venv/bin/python scripts/v3/run_phase3.py --grid credit_mappo --workers 3
.venv/bin/python scripts/v3/run_phase3.py --grid phase4_v5 --workers 4
.venv/bin/python scripts/v5/eval_final.py --runs 'results/v3/phase3/phase4_v5/*' --out results/v5/phase4_v5 --workers 3
.venv/bin/python scripts/v5/analyse_v5.py && .venv/bin/python scripts/v5/figures.py
```
