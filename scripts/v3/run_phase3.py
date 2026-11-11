"""v3 Phase 3 (headroom gate): IPPO / MAPPO on K walled lanes with team vs exact difference rewards.

    .venv/bin/python scripts/v3/run_phase3.py --grid feasibility --workers 1
    .venv/bin/python scripts/v3/run_phase3.py --grid gate --workers 8
    .venv/bin/python scripts/v3/run_phase3.py --summarise gate

Each run calls scripts/run_epymarl.py (EPyMARL, unmodified) in a subprocess with its own sacred directory
results/v3/phase3/<grid>/<run>/, and is skipped if its metrics.json exists. Learning is measured by the
greedy test success rate (fraction of lanes completed) logged every test_interval steps.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "v3", "phase3")
HORIZON = 25

# reward arm -> (EPyMARL common_reward, env reward_mode)
ARMS = {"team": (True, "team"), "team_vector": (False, "team_vector"), "difference": (False, "difference")}

GRIDS = {
    # 3.2 feasibility: can IPPO learn one lane at all, and how fast?
    "feasibility": dict(algs=("ippo",), Ks=(1,), arms=("team",), seeds=(0,), t_max=400_000),
    "feasibility_k8": dict(algs=("ippo",), Ks=(8,), arms=("team", "difference"), seeds=(0,), t_max=2_000_000),
    # 3.4 / 3.5 the gate (pre-registered in docs/v3/preregistration_v3.md): seeds 5-9.
    "gate": dict(algs=("ippo",), Ks=(1, 2, 4, 8), arms=("team", "team_vector", "difference"), seeds=(5, 6, 7, 8, 9),
                 t_max={1: 400_000, 2: 800_000, 4: 1_200_000, 8: 2_000_000}),
    # Phase 4.1: MAPPO with random persistent stops during training (dropout p), checkpoints for 4.2
    "phase4": dict(algs=("mappo",), Ks=(4,), arms=("team",), seeds=(5, 6, 7), t_max=1_600_000,
                   dropout=(0.0, 0.25), save_models=True),
    "gate_mappo": dict(algs=("mappo",), Ks=(1, 8), arms=("team_vector", "difference"), seeds=(5, 6, 7, 8, 9),
                       t_max={1: 400_000, 8: 2_000_000}),
}


def run_name(alg, K, arm, seed, p=0.0):
    return f"{alg}_K{K}_{arm}_s{seed}" + (f"_drop{p}" if p else "")


def train_one(grid, alg, K, arm, seed, t_max, p=0.0, save_models=False):
    run_dir = os.path.join(OUT, grid, run_name(alg, K, arm, seed, p))
    if os.path.exists(os.path.join(run_dir, "run_info.json")):
        return f"skip {run_dir}"
    os.makedirs(run_dir, exist_ok=True)
    common, mode = ARMS[arm]
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_epymarl.py"), f"--config={alg}",
           "--env-config=planted_lanes", "with", f"seed={seed}", f"t_max={t_max}", "use_cuda=False",
           f"env_args.layout_kwargs.n_events={K}", f"common_reward={common}", f"env_args.reward_mode={mode}",
           f"env_args.dropout_p={p}", f"env_args.dropout_seed={seed}"]
    cmd += ([f"save_model=True", f"save_model_interval={t_max // 10}", f"local_results_path={run_dir}"]
            if save_models else ["save_model=False"])
    env = dict(os.environ, WHOEARNED_SACRED_DIR=run_dir, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    stopped_early = False
    with open(os.path.join(run_dir, "stdout.log"), "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(20)
            # Early stop (does not change any pre-registered quantity): episodes-to-90% is fixed at the first
            # test point >= 0.9; once 5 consecutive test points are >= 0.9 the crossing is also shown stable.
            if not save_models and stable_success(run_dir, EARLY_STOP_LEVEL, EARLY_STOP_RUN):
                proc.terminate()
                proc.wait(timeout=60)
                stopped_early = True
        rc = proc.returncode
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump({"rc": rc, "seconds": time.time() - t0, "stopped_early": stopped_early, "t_max": t_max}, f)
    return f"{run_name(alg, K, arm, seed, p)} rc={rc} early={stopped_early} {time.time() - t0:.0f}s"


EARLY_STOP_LEVEL, EARLY_STOP_RUN = 0.9, 5


def stable_success(run_dir, level, run_length):
    c = curve(run_dir)
    if c is None or len(c[1]) < run_length:
        return False
    return bool(np.all(c[1][-run_length:] >= level))


def curve(run_dir):
    """(t_env, test success rate) from the run's sacred metrics."""
    hits = glob.glob(os.path.join(run_dir, "*", "metrics.json"))
    if not hits:
        return None
    m = json.load(open(sorted(hits)[-1]))
    key = "test_success_rate_mean"
    if key not in m:
        return None
    return np.array(m[key]["steps"]), np.array(m[key]["values"])


def episodes_to(run_dir, level=0.9):
    c = curve(run_dir)
    if c is None:
        return None
    steps, vals = c
    hit = np.flatnonzero(vals >= level)
    return float(steps[hit[0]] / HORIZON) if len(hit) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--summarise")
    a = p.parse_args()
    if a.summarise:
        for d in sorted(glob.glob(os.path.join(OUT, a.summarise, "*"))):
            c = curve(d)
            last = None if c is None else float(c[1][-1])
            print(os.path.basename(d), "episodes_to_90:", episodes_to(d), "final:", last)
        return
    g = GRIDS[a.grid]
    jobs = list(itertools.product(g["algs"], g["Ks"], g["arms"], g["seeds"], g.get("dropout", (0.0,))))
    with ProcessPoolExecutor(a.workers) as ex:
        tmax = lambda K: g["t_max"][K] if isinstance(g["t_max"], dict) else g["t_max"]
        # longest runs first, so the pool is not left waiting on one big run at the end
        jobs.sort(key=lambda j: -tmax(j[1]))
        futs = [ex.submit(train_one, a.grid, alg, K, arm, s, tmax(K), p, g.get("save_models", False))
                for alg, K, arm, s, p in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
