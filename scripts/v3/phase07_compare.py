"""Phase 0.7: stored partial-layout cells (non-Markov state) vs re-runs with the pending-amount fix (seeds 0-4).

    .venv/bin/python scripts/v3/phase07_compare.py   -> prints a markdown table
Metric: end-to-end L1 vs open-loop leave-one-out/Shapley (the stored cells' own metric, `summary`), the final
critic loss, and the dividend error where the method has dividends.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("| grid | d | method | L1 stored -> re-run | final loss stored -> re-run | dividend error stored -> re-run |")
    print("|---|---|---|---|---|---|")
    for grid, delays in (("h3", (25, 50)), ("h5", (50,))):
        for d in delays:
            acc = {}
            for s in range(5):
                new_p = os.path.join(ROOT, "results", "v3", f"rerun_partial_{grid}", f"partial_d{d}_b2_s{s}.json")
                old_p = os.path.join(ROOT, "results", "mode_p", grid, f"partial_d{d}_b2_s{s}.json")
                if not (os.path.exists(new_p) and os.path.exists(old_p)):
                    continue
                old, new = json.load(open(old_p)), json.load(open(new_p))
                for a in old["algs"]:
                    if a not in new["algs"]:
                        continue
                    r = acc.setdefault(a, {k: [] for k in ("l1o", "l1n", "lo", "ln", "do", "dn")})
                    r["l1o"].append(old["algs"][a]["summary"]["l1"]); r["l1n"].append(new["algs"][a]["summary"]["l1"])
                    r["lo"].append(old["algs"][a]["loss_trace"][-1][1]); r["ln"].append(new["algs"][a]["loss_trace"][-1][1])
                    r["do"].append((old["algs"][a].get("summary_coalition") or {}).get("dividend_error"))
                    r["dn"].append((new["algs"][a].get("summary_coalition") or {}).get("dividend_error"))
            for a, r in acc.items():
                m = lambda v: np.mean([x for x in v if x is not None]) if any(x is not None for x in v) else float("nan")
                print(f"| {grid} | {d} | {a} (n={len(r['l1o'])}) | {m(r['l1o']):.3f} -> {m(r['l1n']):.3f} | "
                      f"{m(r['lo']):.3f} -> {m(r['ln']):.3f} | {m(r['do']):.3f} -> {m(r['dn']):.3f} |")


if __name__ == "__main__":
    main()
