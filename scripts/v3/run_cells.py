"""Run v3 Mode P grids (persistent-ablation critics, tables, EPyMARL methods) in parallel.

    .venv/bin/python scripts/v3/run_cells.py --grid smoke --workers 1
    .venv/bin/python scripts/v3/run_cells.py --grid confirm_main --workers 8

Every cell writes results/v3/<grid>/<key>.json via whoearned.experiments.run_data_cell; cells whose JSON
exists are skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from whoearned.experiments import CRITICAL_ENABLER_START, DataCell, TrainSpec, run_data_cell  # noqa: E402

ES = CRITICAL_ENABLER_START
EPYMARL = ("vdn", "qmix", "coma", "mappo")
CORE = ("atd", "atd_noint", "atd_remove")
# grid name -> (cells, algs). Seeds 0-4 are exploratory (the repo's earlier seeds); 5-9 are confirmatory.
def _main(seeds):
    return [DataCell(delay=d, n_bystanders=b, seed=s, enabler_start=es, horizon=80 if d >= 50 else 60)
            for d, b, es, s in itertools.product((0, 25, 50), (2, 4), (0, ES), seeds)]


def _h5(seeds, layouts=("overlap", "partial", "coalition")):
    return [DataCell(delay=d, n_bystanders=1 if lay == "overlap" else 2, seed=s, horizon=80 if d >= 50 else 60,
                     layout=lay) for lay, d, s in itertools.product(layouts, (0, 50), seeds)]


GRIDS = {
    "smoke": ([DataCell(delay=0, n_bystanders=2, seed=0)], ("atd_bd", "atd", "atd_noint", "table3_mix")),
    "smoke_epymarl": ([DataCell(delay=0, n_bystanders=2, seed=0)], ("qmix", "ece3_mix", "atd")),
}


def grid(name):
    if name in GRIDS:
        return GRIDS[name]
    raise KeyError(name)


def _run(args):
    cell, train, out_dir = args
    t = time.time()
    path = run_data_cell(cell, train, out_dir)
    return f"{cell.key} {time.time() - t:.0f}s -> {path}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--algs", default=None, help="comma list overriding the grid's algorithms")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cells, algs = grid(a.grid)
    algs = tuple(a.algs.split(",")) if a.algs else algs
    out_dir = a.out or os.path.join(ROOT, "results", "v3", a.grid)
    train = TrainSpec(algs=algs)
    todo = [c for c in cells if not os.path.exists(os.path.join(out_dir, f"{c.key}.json"))]
    print(f"{a.grid}: {len(todo)} of {len(cells)} cells to run, algs {algs}", flush=True)
    jobs = [(c, train, out_dir) for c in todo]
    if a.workers == 1:
        for j in jobs:
            print(_run(j), flush=True)
    else:
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_run, jobs):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
