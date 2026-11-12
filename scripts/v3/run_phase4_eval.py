"""Phase 4.2: score every saved checkpoint of the phase4 MAPPO runs against exact persistent-ablation truth.

    .venv/bin/python scripts/v3/run_phase4_eval.py --workers 3
Writes results/v3/phase4/<run>.json: per checkpoint, return, success, masked / standard / uniform L1, step r.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "v3", "phase4")


def eval_run(run_dir):
    import torch as th
    th.set_num_threads(1)
    from whoearned.phase4 import evaluate_checkpoint
    name = os.path.basename(run_dir)
    out = os.path.join(OUT, f"{name}.json")
    if os.path.exists(out):
        return f"skip {name}"
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models", "*", "*")), key=lambda p: int(os.path.basename(p)))
    seed = int(name.split("_s")[1].split("_")[0])
    res = []
    for c in ckpts:
        r = evaluate_checkpoint(c, K=2, n_bystanders=2, n_test=30, seed=seed)
        r["t_env"] = int(os.path.basename(c))
        res.append(r)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"run": name, "checkpoints": res}, open(out, "w"), indent=1)
    return f"{name}: {len(res)} checkpoints"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "v3", "phase3", "phase4", "*")))
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(eval_run, runs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
