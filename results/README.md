# Raw results

Every v3, v4 and v5 result file is in this directory. v3 and v5 files are **gzipped** (`*.json.gz`, lossless) to keep the repository small (418 MB → 27 MB); v4 files are plain JSON. Model checkpoints and Sacred's copies of the source code are not included; every reported number is computed from the files here.

Restore the plain files (needed before running any analysis script):

```bash
python scripts/v5/restore_results.py results/v3 results/v5
```

Then regenerate the reports, which reproduce the committed versions byte for byte (checked from a fresh clone on 15 Nov 2026):

```bash
python scripts/v3/analyse_prereg.py      # docs/v3/confirmatory_results.md
python scripts/v3/analyse_phase3.py      # docs/v3/phase3_results.md
python scripts/v4/analyse_v4.py          # docs/v4/results.md
python scripts/v5/analyse_v5.py          # docs/v5/results.md
python scripts/v5/cost_accounting.py     # docs/v5/cost_accounting.md (prints to stdout)
```

| Directory | Contents |
|---|---|
| `v3/confirm_*` | v3 confirmatory cells (seeds 5–9): one JSON per cell, with per-episode credit and exact truth for every method |
| `v3/pilot_*`, `v3/smoke*` | exploratory pilots (seeds 0–4) |
| `v3/phase3/<grid>/<run>/` | EPyMARL runs (Sacred `config.json`, `metrics.json`, `run.json`, `info.json`): Phase 3 gate, v4 Part C (`online_credit`), v5 Part E (`credit_mappo*`), v5 Part F training (`phase4_v5`) |
| `v3/phase4/` | v3 Phase 4 checkpoint scores |
| `v4/` | v4 Parts A, B, D (plain JSON) |
| `v5/phase4_v5/` | v5 Part F final-checkpoint scores (credit and collateral damage) |
| `mode_p/`, `mode_t/` | rounds 1–2: main grid, H3, H5, sensitivity (`mode_p`); learning from scratch (`mode_t`) |
| `viz/` | per-setting data behind `docs/credit_viz.html` (rebuild with `scripts/export_credit_viz.py --build-only`) |
