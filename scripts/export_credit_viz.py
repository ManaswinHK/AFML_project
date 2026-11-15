"""Build the interactive credit visualisation, `docs/credit_viz.html`.

Two stages:

1. **Export** (slow, about 8 minutes of CPU per cell): re-run selected main-grid cells with
   :func:`whoearned.viz_export.export_cell`, which validates the regenerated credit against
   `results/mode_p/main/<key>.json`, and cache each cell as `results/viz/<key>.json`. Cached cells are
   skipped.
2. **Build** (instant): inject every cached cell into `scripts/credit_viz_template.html` and write a
   single self-contained page, `docs/credit_viz.html`, that opens from disk with no server or network.

    .venv/Scripts/python scripts/export_credit_viz.py --workers 8      # default cells, then build
    .venv/Scripts/python scripts/export_credit_viz.py --only d0_b2_s0 --workers 1
    .venv/Scripts/python scripts/export_credit_viz.py --build-only
    .venv/Scripts/python scripts/export_credit_viz.py --workers 8 --l1-tol 0.15  # on another machine
    .venv/Scripts/python scripts/export_credit_viz.py --add-estimators --workers 7  # add our methods, then build

``--add-estimators`` trains our two methods (the coalition estimator ECE-3 and the v3 critic trained with agents
sitting out) on every exported cell and merges them in. It leaves the four EPyMARL read-outs untouched, so it runs
on any machine (about 10-15 minutes of CPU per cell).

Exact mode needs the machine that ran the main grid: PyTorch's CPU arithmetic is not bit-identical
across machines, so elsewhere training drifts slowly from the stored run (on the laptop, VDN's losses
match the stored trace to update 100 and then drift; QMIX drifts sooner). ``--l1-tol`` then checks that the update-0 loss matches
and each method's mean L1 is within the tolerance, and the page reports the measured deviations.

Default cells: coalition layout, 2 bystanders, seed 0, every delay (0, 10, 25, 50), slack and
critical enabler: 8 cells, 30 test episodes each.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from whoearned.experiments import DataCell  # noqa: E402

MAIN = os.path.join(ROOT, "results", "mode_p", "main")
CACHE = os.path.join(ROOT, "results", "viz")
TEMPLATE = os.path.join(ROOT, "scripts", "credit_viz_template.html")
OUT = os.path.join(ROOT, "docs", "credit_viz.html")
PLACEHOLDER = "/*__VIZ_DATA__*/null"
DEFAULT_CELLS = [f"d{d}_b2{es}_s0" for es in ("", "_es12") for d in (0, 10, 25, 50)]


def _export(key: str, l1_tol: float | None = None) -> str:
    from whoearned.viz_export import export_cell

    with open(os.path.join(MAIN, f"{key}.json")) as f:
        stored = json.load(f)
    cell = DataCell(**stored["cell"])
    if cell.key != key:
        raise RuntimeError(f"{key}: stored cell has key {cell.key}")
    t0 = time.time()
    data = export_cell(cell, stored, l1_tol=l1_tol)
    data["export_seconds"] = round(time.time() - t0, 1)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    return path


def _add(key: str) -> str:
    from whoearned.viz_export import add_estimators

    path = os.path.join(CACHE, f"{key}.json")
    with open(path) as f:
        data = json.load(f)
    with open(os.path.join(MAIN, f"{key}.json")) as f:
        cell = DataCell(**json.load(f)["cell"])
    data = add_estimators(cell, data)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    return path


def _has_estimators(key: str) -> bool:
    with open(os.path.join(CACHE, f"{key}.json")) as f:
        return "estimators" in json.load(f)


def build(keys: list[str]) -> str:
    cells = []
    for key in keys:
        path = os.path.join(CACHE, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                cells.append(json.load(f))
        else:
            print(f"  skipped {key}: not exported", flush=True)
    if not cells:
        raise SystemExit("no exported cells; run without --build-only first")
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    if PLACEHOLDER not in page:
        raise SystemExit(f"template has no {PLACEHOLDER!r} placeholder")
    payload = json.dumps({"cells": cells}, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, payload)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {OUT} ({len(cells)} cells, {os.path.getsize(OUT) / 1e6:.1f} MB)", flush=True)
    return OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="comma-separated cell keys (default: the 8 showcase cells)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--add-estimators", action="store_true",
                   help="train our estimators on every exported cell that lacks them, merge them in, then build")
    p.add_argument("--l1-tol", type=float, default=None,
                   help="tolerant check for a machine other than the one that ran the main grid: each method's "
                        "mean L1 must be within this of the stored value (default: exact match)")
    a = p.parse_args()
    keys = a.only.split(",") if a.only else DEFAULT_CELLS

    if a.add_estimators:
        todo = [k for k in keys if os.path.exists(os.path.join(CACHE, f"{k}.json")) and not _has_estimators(k)]
        print(f"{len(todo)} cells to add estimators to with {a.workers} workers", flush=True)
        failed = []
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_add, k): k for k in todo}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    print(f"  done {k} -> {fut.result()}", flush=True)
                except Exception as e:  # noqa: BLE001
                    failed.append(k)
                    print(f"  FAILED {k}: {e}", flush=True)
        if failed:
            raise SystemExit(f"{len(failed)} cells failed; nothing built")
        build(keys)
        return

    if not a.build_only:
        todo = [k for k in keys if not os.path.exists(os.path.join(CACHE, f"{k}.json"))]
        print(f"{len(todo)} cells to export with {a.workers} workers -> {CACHE}", flush=True)
        failed = []
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_export, k, a.l1_tol): k for k in todo}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    print(f"  done {k} -> {fut.result()}", flush=True)
                except Exception as e:  # noqa: BLE001 - report every failing cell, then stop
                    failed.append(k)
                    print(f"  FAILED {k}: {e}", flush=True)
        if failed:
            raise SystemExit(f"{len(failed)} cells failed validation; nothing built")
    build(keys)


if __name__ == "__main__":
    main()
