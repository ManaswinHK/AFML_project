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


def _perm(seeds, bs=(2, 4), delay=25):
    return [DataCell(delay=delay, n_bystanders=b, seed=s, horizon=60, shuffle_roles=True)
            for b, s in itertools.product(bs, seeds)]


PILOT_FIXED_ALGS = ("atd", "atd_bd", "atd_tp", "atd_noint", "atd_remove", "atd_set", "atd_active", "table3_mix")
PILOT_PERM_ALGS = ("atd", "atd_n800", "atd_set", "atd_n800_set", "atd_active", "atd_set_active", "atd_noint",
                   "table3_mix", "ece3_mix")
PILOT_H5_ALGS = ("atd", "atd_remove", "atd_set", "atd_n800", "atd_active", "table2_mix", "table3_mix")

CONFIRM_SEEDS = (5, 6, 7, 8, 9)
CONFIRM_MAIN_ALGS = ("uniform", "floor_expected", "atd", "atd_struct", "atd_noint", "atd_remove", "atd_set",
                     "atd_set_struct", "atd_n25", "atd_n50", "atd_n100", "atd_n400", "atd_n800", "table3_mix",
                     "ece3_mix", "vdn", "qmix", "coma", "mappo", "mappo_int", "mappo_int_masked", "qmix_int",
                     "qmix_int_masked")
CONFIRM_H5_ALGS = ("uniform", "floor_expected", "atd", "atd_struct", "atd_noint", "atd_remove", "atd_set",
                   "atd_set_struct", "atd_set_remove", "table2_mix", "table3_mix", "ece2_mix", "ece3_mix")
CONFIRM_PERM_ALGS = ("uniform", "atd", "atd_n800", "atd_noint", "atd_set", "atd_set_struct", "atd_n800_set",
                     "atd_noroles", "atd_set_noroles",
                     "atd_set_n25", "atd_set_n50", "atd_set_n100", "table3_mix", "ece3_mix")

GRIDS = {
    # ---- confirmatory (seeds 5-9): run only after docs/v3/preregistration_v3.md is frozen
    "confirm_main": (_main(CONFIRM_SEEDS), CONFIRM_MAIN_ALGS),
    "confirm_h5": (_h5(CONFIRM_SEEDS, layouts=("overlap", "partial")), CONFIRM_H5_ALGS),
    "confirm_perm": (_perm(CONFIRM_SEEDS), CONFIRM_PERM_ALGS),
    # ---- exploratory pilots (seeds 0-4, the seeds of every earlier round): choose settings, set thresholds
    "pilot_fixed": (_main((0, 1)), PILOT_FIXED_ALGS),
    "pilot_perm": (_perm((0, 1, 2, 3, 4)), PILOT_PERM_ALGS),
    "pilot_h5": (_h5((0, 1), layouts=("overlap", "partial")), PILOT_H5_ALGS),
    # ---- Phase 2.4 (descriptive, NOT pre-registered): order-4 coalition, seeds 5-9
    "double_k4": ([DataCell(delay=d, n_bystanders=2, seed=s, horizon=80 if d >= 50 else 60, layout="double")
                   for d in (0, 25) for s in CONFIRM_SEEDS],
                  ("uniform", "atd", "atd_set", "atd_set_struct", "table3_mix", "ece3_mix", "atd_noint")),
    # ---- Phase 0.7: stored partial-layout cells at d > 0, re-run with the Markov state (same methods and seeds)
    "rerun_partial_h3": ([DataCell(delay=d, n_bystanders=2, seed=s, horizon=80 if d >= 50 else 60, layout="partial")
                          for d in (25, 50) for s in range(5)],
                         ("vdn", "qmix", "coma", "mappo", "na2q", "stas", "uniform")),
    "rerun_partial_h5": ([DataCell(delay=50, n_bystanders=2, seed=s, horizon=80, layout="partial") for s in range(5)],
                         ("ece2", "ece3", "ece2_mix", "ece3_mix", "na2q_mix", "stas_mix", "table2_mix", "table3_mix")),
    "pilot_v2": ([DataCell(delay=d, n_bystanders=b, seed=s, horizon=80 if d >= 50 else 60)
                  for d in (0, 25, 50) for b in (2, 4) for s in (1, 2)],
                 ("uniform", "atd", "mappo", "mappo_int", "mappo_int_masked", "qmix", "qmix_int", "qmix_int_masked")),
    "pilot_roles": (_perm((0, 1, 2)), ("atd", "atd_noroles", "atd_set", "atd_set_noroles")),
    "pilot_budget": (_perm((0, 1, 2)), ("uniform", "atd_set_n25", "atd_set_active_n25", "atd_set_n50",
                                        "atd_set_active_n50", "atd_set_n100", "atd_set_active_n100",
                                        "atd_n100", "atd_active_n100", "atd_n400", "atd_active_n400")),
    "smoke": ([DataCell(delay=0, n_bystanders=2, seed=0)], ("atd_bd", "atd", "atd_noint", "table3_mix")),
    "smoke_epymarl": ([DataCell(delay=0, n_bystanders=2, seed=0)], ("qmix", "ece3_mix", "atd")),
    "smoke_v2": ([DataCell(delay=25, n_bystanders=2, seed=0)], ("uniform", "atd", "mappo", "mappo_int", "mappo_int_masked",
                                                                 "qmix", "qmix_int", "qmix_int_masked")),
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
