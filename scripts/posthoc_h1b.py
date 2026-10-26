"""POST-HOC (not pre-registered): H1b level-2 missed credit, split by whether the enabler
was actually on the critical path in each test episode.

The main-grid JSONs store only the rate of critical episodes, so this regenerates each
cell's test episodes and their ground truth (no critic training) and scores level 2 per
episode. Each cell is checked against the stored semantic_gap means first.

    .venv/Scripts/python scripts/posthoc_h1b.py --workers 7
    .venv/Scripts/python scripts/posthoc_h1b.py --report-only
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_mode_p import cells  # noqa: E402
from whoearned.experiments import _enabler_critical, _roles, _test_truth  # noqa: E402
from whoearned.metrics import evaluate_against  # noqa: E402

MAIN = os.path.join(ROOT, "results", "mode_p", "main")
OUT = os.path.join(ROOT, "results", "mode_p", "main_posthoc_h1b")
NOTIONS = ("noop", "baseline")
THRESH_SLACK, THRESH_CRIT = 0.8, 0.3  # the pre-registered H1b bounds, reused unchanged


def run_cell(cell) -> str:
    cfg = cell.config()
    test_eps = cell.episodes("test")
    truth = _test_truth(cell, test_eps)
    enablers, bystanders = _roles(cfg)
    rows = []
    for b, (ep, gt) in enumerate(zip(test_eps, truth.truths)):
        row = {"episode": b, "critical": bool(_enabler_critical(ep)), "matches_plan": bool(truth.matches_plan[b])}
        for kind in NOTIONS:
            rep = evaluate_against(truth.step[kind][b].sum(0), gt.leave_one_out, enablers, bystanders)
            row[kind] = rep.missed if rep.valid else None
        rows.append(row)
    stored = json.load(open(os.path.join(MAIN, f"{cell.key}.json")))["semantic_gap"]
    for kind in NOTIONS:
        mine = np.mean([r[kind] for r in rows if r[kind] is not None])
        if not np.isclose(mine, stored[kind]["missed"], atol=1e-9):
            raise RuntimeError(f"{cell.key} {kind}: regenerated {mine} != stored {stored[kind]['missed']}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{cell.key}.json")
    with open(path, "w") as f:
        json.dump({"cell": cell.key, "delay": cell.delay, "bystanders": cell.n_bystanders,
                   "enabler_start": cell.enabler_start, "seed": cell.seed, "episodes": rows}, f)
    return path


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def report() -> str:
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "d*.json")))]
    groups = {}
    for r in recs:
        groups.setdefault((r["enabler_start"], r["delay"], r["bystanders"]), []).append(r)
    lines = [
        "# POST-HOC: H1b split by realised enabler criticality",
        "",
        "**Not pre-registered.** Exploratory analysis done after seeing the main-grid results "
        "(docs/main_results.md §3.1). It does not change the pre-registered H1b verdict.",
        "",
        f"{len(recs)} cells. An episode is *critical* if the event completed in the first step its gate "
        "allowed (the enabler was the bottleneck). For each cell: mean level-2 missed over its critical "
        "and over its non-critical test episodes; then mean ± sd over seeds that have such episodes. "
        f"Bounds reused from H1b: ≥ {THRESH_SLACK} expected without criticality, ≤ {THRESH_CRIT} with it.",
        "",
        "| notion | setting | delay | bystanders | critical eps (total) | missed, critical eps | missed, non-critical eps |",
        "|---|---|---|---|---|---|---|",
    ]
    pooled = {k: {True: [], False: []} for k in NOTIONS}
    for kind in NOTIONS:
        for (es, d, b), rs in sorted(groups.items()):
            n_crit = sum(e["critical"] for r in rs for e in r["episodes"])
            n_all = sum(len(r["episodes"]) for r in rs)
            cols = []
            for crit in (True, False):
                per_seed = [_mean(e[kind] for e in r["episodes"] if e["critical"] == crit) for r in rs]
                per_seed = [x for x in per_seed if x is not None]
                pooled[kind][crit] += [e[kind] for r in rs for e in r["episodes"]
                                       if e["critical"] == crit and e[kind] is not None]
                cols.append(f"{np.mean(per_seed):.3f} ± {np.std(per_seed, ddof=1):.3f} ({len(per_seed)} seeds)"
                            if len(per_seed) > 1 else (f"{per_seed[0]:.3f} (1 seed)" if per_seed else "n/a"))
            setting = "critical (12)" if es else "slack"
            lines.append(f"| {kind} | {setting} | {d} | {b} | {n_crit} ({n_all}) | {cols[0]} | {cols[1]} |")
    lines += ["", "## Pooled over all cells and episodes", "",
              "| notion | missed, critical eps | missed, non-critical eps |", "|---|---|---|"]
    for kind in NOTIONS:
        c, n = pooled[kind][True], pooled[kind][False]
        lines.append(f"| {kind} | {np.mean(c):.3f} (n = {len(c)}; median {np.median(c):.3f}) "
                     f"| {np.mean(n):.3f} (n = {len(n)}; median {np.median(n):.3f}) |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "h1b_posthoc.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--only", default=None, help="run just this cell key (for testing)")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        todo = [c for c in cells("main")
                if (a.only is None or c.key == a.only) and not os.path.exists(os.path.join(OUT, f"{c.key}.json"))]
        print(f"{len(todo)} cells to run", flush=True)
        with ProcessPoolExecutor(a.workers) as pool:
            futs = {pool.submit(run_cell, c): c.key for c in todo}
            for i, f in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(todo)}] {futs[f]} -> {f.result()}", flush=True)
        if a.only:
            return
    sys.stdout.reconfigure(encoding="utf-8")
    print(report())


if __name__ == "__main__":
    main()
