"""Run a grid of Mode P data cells in parallel, then summarise.

    .venv/Scripts/python scripts/run_mode_p.py --grid pilot --workers 6
    .venv/Scripts/python scripts/run_mode_p.py --grid pilot --summarise-only

Cells whose JSON already exists are skipped, so an interrupted run can be resumed.
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from whoearned.experiments import CRITICAL_ENABLER_START, DataCell, TrainSpec, run_data_cell  # noqa: E402

GRIDS = {
    # Problem statement v2, Section 4: pilot with 2 seeds and d in {0, 25}.
    "pilot": dict(delays=(0, 25), bystanders=(0, 2, 4), enabler_starts=(0,), seeds=(0, 1)),
    "smoke": dict(delays=(0,), bystanders=(2,), enabler_starts=(0,), seeds=(0,)),
    # Main grid (docs/pilot_findings.md): 4 delays x 3 bystander counts x {slack, critical} x 5 seeds.
    "main": dict(delays=(0, 10, 25, 50), bystanders=(0, 2, 4),
                 enabler_starts=(0, CRITICAL_ENABLER_START), seeds=(0, 1, 2, 3, 4)),
}


def horizon_for(delay: int, enabler_start: int) -> int:
    """Horizon 80 for d = 50, 60 otherwise (as adopted). Same horizon for slack and critical
    cells at a given delay, so the slack comparison is not confounded by episode length;
    clipped payouts are counted per cell."""
    return 80 if delay >= 50 else 60


def cells(grid: str):
    g = GRIDS[grid]
    for d, b, es, s in itertools.product(g["delays"], g["bystanders"], g["enabler_starts"], g["seeds"]):
        yield DataCell(delay=d, n_bystanders=b, seed=s, enabler_start=es, horizon=horizon_for(d, es))


def _run(cell, train, out_dir):
    return run_data_cell(cell, train, out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="pilot", choices=sorted(GRIDS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--updates", type=int, default=TrainSpec.updates)
    p.add_argument("--out", default=None)
    p.add_argument("--summarise-only", action="store_true")
    a = p.parse_args()

    out_dir = a.out or os.path.join(ROOT, "results", "mode_p", a.grid)
    train = TrainSpec(updates=a.updates)
    if not a.summarise_only:
        todo = [c for c in cells(a.grid) if not os.path.exists(os.path.join(out_dir, f"{c.key}.json"))]
        print(f"{len(todo)} data cells to run with {a.workers} workers -> {out_dir}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_run, c, train, out_dir): c for c in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                cell = futures[fut]
                try:
                    fut.result()
                    print(f"[{i}/{len(todo)}] {cell.key} done ({time.time() - t0:.0f}s elapsed)", flush=True)
                except Exception as e:  # report and keep going; the cell can be re-run
                    print(f"[{i}/{len(todo)}] {cell.key} FAILED: {e!r}", flush=True)

    from summarise_mode_p import summarise_dir  # noqa: E402

    print(summarise_dir(out_dir))


if __name__ == "__main__":
    main()
