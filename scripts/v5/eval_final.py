"""Score the FINAL checkpoint of MAPPO dropout runs: credit read-outs against exact truth, plus collateral damage of
absences (robustness). Run directories are named ``mappo_K{K}_team_s{seed}[_drop{p}]``.

    .venv/bin/python scripts/v5/eval_final.py --runs 'results/v3/phase3/phase4/*' --out results/v5/robust_pilot --workers 6
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


def eval_run(args):
    run_dir, out_dir, n_bystanders, n_test = args
    import torch as th
    th.set_num_threads(1)
    from whoearned.phase4 import evaluate_checkpoint
    name = os.path.basename(run_dir)
    out = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(out):
        return f"skip {name}"
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models", "*", "*")), key=lambda p: int(os.path.basename(p)))
    if not ckpts:
        return f"no checkpoints in {name}"
    K = int(name.split("_K")[1].split("_")[0])
    seed = int(name.split("_s")[1].split("_")[0])
    drop = float(name.split("_drop")[1]) if "_drop" in name else 0.0
    r = evaluate_checkpoint(ckpts[-1], K=K, n_bystanders=n_bystanders, n_test=n_test, seed=seed)
    r.update(run=name, K=K, seed=seed, dropout=drop, t_env=int(os.path.basename(ckpts[-1])))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)
    return (f"{name}: success {r['success']:.2f} struct {r['posthoc_struct_l1']:.3f} uniform {r['uniform_l1']:.3f} "
            f"collateral {r['collateral_t0']:.3f} (frac {r['collateral_frac_t0']:.3f}, bystander "
            f"{r['collateral_bystander_t0']:.3f}, member {r['collateral_member_t0']:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="glob of run directories")
    p.add_argument("--out", required=True)
    p.add_argument("--n-bystanders", type=int, default=2)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    runs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d))
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(eval_run, [(d, a.out, a.n_bystanders, a.n_test) for d in runs]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
