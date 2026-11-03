"""Mode T (Step 8, H4): train EPyMARL methods online, then score every checkpoint's credit.

    .venv/Scripts/python scripts/run_mode_t.py train --t-max 400000 --workers 8
    .venv/Scripts/python scripts/run_mode_t.py eval --workers 8

`train` runs scripts/run_epymarl.py once per (alg, delay, seed) on coalition_layout (b = 2),
saving a checkpoint every t_max / 10 steps under results/mode_t/models. Runs whose model
directory already exists are skipped. `eval` writes one JSON per run to results/mode_t with
return and end-to-end credit error per checkpoint (``whoearned.mode_t``).
"""

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "results", "mode_t")
ALGS = ("vdn", "qmix", "coma", "mappo")
DELAYS = (0, 25)
SEEDS = (0, 1, 2)
N_BYSTANDERS = 2


def env_args(delay: int) -> dict:
    return {"map_name": "coalition_layout",
            "layout_kwargs": {"n_bystanders": N_BYSTANDERS, "delay": delay, "horizon": 60}}


def run_name(alg, delay, seed) -> str:
    return f"{alg}_d{delay}_s{seed}"


def model_dir(alg, delay, seed) -> str | None:
    hits = glob.glob(os.path.join(OUT, "runs", run_name(alg, delay, seed), "models", f"{alg}_seed{seed}_*"))
    return sorted(hits)[-1] if hits else None


def train_one(alg, delay, seed, t_max) -> str:
    run_dir = os.path.join(OUT, "runs", run_name(alg, delay, seed))
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_epymarl.py"), f"--config={alg}",
           "--env-config=planted_coalition", "with", f"seed={seed}", f"t_max={t_max}", "use_cuda=False",
           f"env_args.layout_kwargs.delay={delay}", "save_model=True", f"save_model_interval={t_max // 10}",
           f"test_interval={t_max // 20}", f"log_interval={t_max // 20}", f"local_results_path={run_dir}"]
    with open(os.path.join(run_dir, "train.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return run_name(alg, delay, seed)


def eval_one(alg, delay, seed, n_test) -> str:
    import torch as th

    from whoearned.mode_t import checkpoints, evaluate_checkpoint

    th.set_num_threads(1)
    rows = []
    for t_env, path in checkpoints(model_dir(alg, delay, seed)):
        r = evaluate_checkpoint(alg, env_args(delay), path, n_test=n_test, seed=seed)
        rows.append({"t_env": t_env, **r})
    out = os.path.join(OUT, f"{run_name(alg, delay, seed)}.json")
    with open(out, "w") as f:
        json.dump({"alg": alg, "delay": delay, "seed": seed, "checkpoints": rows}, f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("train", "eval"))
    p.add_argument("--t-max", type=int, default=400_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--algs", default=",".join(ALGS))
    p.add_argument("--delays", default=",".join(map(str, DELAYS)))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--n-test", type=int, default=30)
    a = p.parse_args()
    runs = list(itertools.product(a.algs.split(","), map(int, a.delays.split(",")), map(int, a.seeds.split(","))))
    if a.stage == "train":
        todo = [r for r in runs if model_dir(*r) is None]
        jobs = [(train_one, (*r, a.t_max)) for r in todo]
    else:
        todo = [r for r in runs if model_dir(*r) and not os.path.exists(os.path.join(OUT, f"{run_name(*r)}.json"))]
        jobs = [(eval_one, (*r, a.n_test)) for r in todo]
    print(f"{len(jobs)} {a.stage} jobs with {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(fn, *args): args for fn, args in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                print(f"[{i}/{len(jobs)}] {fut.result()} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # keep going; rerun picks it up
                print(f"[{i}/{len(jobs)}] {futures[fut]} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
