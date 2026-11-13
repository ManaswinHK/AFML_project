# v4: the next steps after v3

Start with **`report.md`**. Then:

| File | What it is |
|---|---|
| `report.md` | Results of all four parts, with readings and limits |
| `preregistration_v4.md` | Predictions and tests, frozen per part before its confirmatory runs (C `fe679a6`, A `21285a0`, D `fae1df9`, B `e0198de`) |
| `results.md` | Generated verdict tables (`scripts/v4/analyse_v4.py`) |
| `decisions.md` | Every design decision with alternatives and reasons (E1–E12) |
| `lbf_design.md` | How the LBF test was chosen, including the designs that failed (pilot seeds 0–4) |
| `PROGRESS.md` | Checklist and how to resume |
| `figures/` | `scripts/v4/figures.py` |

Reproduce (Python 3.12 venv, see the repo README):

```bash
.venv/bin/python scripts/v4/size_mixed.py --design interp --seeds 10,11,12,13,14 --out results/v4/size_mixed_confirm_interp.json   # Part A (and extrap)
.venv/bin/python scripts/v4/lbf_cells.py --variants heavy,heavy_large --seeds 10,11,12,13,14 --null remove --eps 0.2 --out results/v4/lbf_confirm/std
.venv/bin/python scripts/v4/lbf_cells.py --variants heavy,heavy_large --seeds 10,11,12,13,14 --null remove --eps 0.2 --n-train 800 --n-int 0,800,3200 --steps 6000 --out results/v4/lbf_confirm/large
.venv/bin/python scripts/v3/run_phase3.py --grid online_credit --workers 4                 # Part C
.venv/bin/python scripts/v3/run_cells.py --grid confirm_rs --workers 3 --out results/v4/confirm_rs   # Part D
.venv/bin/python scripts/v4/analyse_v4.py && .venv/bin/python scripts/v4/figures.py
```
