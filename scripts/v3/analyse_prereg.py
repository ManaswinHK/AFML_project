"""Confirmatory analysis of docs/v3/preregistration_v3.md. Runs every hypothesis exactly as written there.

    .venv/bin/python scripts/v3/analyse_prereg.py --out docs/v3/confirmatory_results.md

Inputs: results/v3/confirm_main, confirm_h5, confirm_perm (cell JSONs), results/v3/size_transfer_confirm.json,
results/v3/phase3/gate. Thresholds live in HYPOTHESES below and must equal the frozen pre-registration.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import stats as S  # noqa: E402
from aggregate import cell_condition  # noqa: E402


def metric(rec: dict, name: str):
    """Per-cell value of a metric for one method record (mean over test episodes)."""
    get = lambda key, field: (rec.get(key) or {}).get(field)
    return {
        "loo_cn": get("summary_loo_closed_noop", "l1"),         # vs closed-loop no-op LOO (= exact no-op ablation truth)
        "loo_c": get("summary_loo_closed", "l1"),               # vs closed-loop removal LOO
        "fid_l1": get("summary_fidelity", "l1"),
        "step_r": get("summary_fidelity", "step_pearson"),
        "top_k": rec.get("top_k"),
        "div_cn": get("summary_coalition_closed_noop", "dividend_error"),
        "div_c": get("summary_coalition_closed", "dividend_error"),
        "miss_cn": get("summary_loo_closed_noop", "missed"),
        "leak_cn": get("summary_loo_closed_noop", "leakage"),
    }[name]


def collect(grid: str, alg: str, name: str, cond_filter=None) -> dict:
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "v3", grid, "*.json"))):
        r = json.load(open(p))
        cond = cell_condition(r["cell"])
        if cond_filter and not cond_filter(cond):
            continue
        rec = r["algs"].get(alg)
        if rec is None:
            continue
        v = metric(rec, name)
        if v is not None:
            out[cond].append(v)
    return dict(out)


def fmt_family(res: dict) -> str:
    lines = [f"**{res['verdict']}** ({res['k']}/{res['m']} conditions)", "",
             "| condition | n | estimate | test | p (Holm) | Wilcoxon p | supported |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if "mean" in r:
            est, test = f"{r['mean']:.3f} (sd {r['sd']:.3f})", f"mean vs {r['tau']}"
        else:
            est, test = f"{r['mean_a']:.3f} vs {r['mean_b']:.3f} (diff {r['mean_diff']:.3f})", f"diff > {r['delta']}"
        lines.append(f"| {' '.join(map(str, r['condition']))} | {r['n']} | {est} | {test} | {r['p_holm']:.4f} | "
                     f"{r['wilcoxon_p']:.3f} | {'yes' if r['supported'] else 'no'} |")
    return "\n".join(lines)


# Thresholds: must equal docs/v3/preregistration_v3.md, Section 3 (frozen).
TAU = {
    "P2a_l1": 0.10, "P3_gap": 0.30, "P4a_r": 0.85, "P4b_r": 0.50, "P5_margin": 0.05, "P6a_topk": 0.90,
    "P6b_topk": 0.50, "P7a_margin": 0.05, "P7b_gap": 0.25, "P8a_l1": 0.05, "P8b_gap": 0.25, "P8c_div": 0.40,
    "P8d_gap": 0.25, "P9a_gap": 0.50, "P9c_gain": 0.10, "P10a_margin": 0.10,
    "P11_l1": None, "P11_gap": None,
}


def hypotheses():
    main, h5, perm = "confirm_main", "confirm_h5", "confirm_perm"
    overlap = lambda c: c[0] == "overlap"
    partial = lambda c: c[0] == "partial"
    learn = lambda alg, m="loo_cn": {**collect(main, alg, m), **collect(h5, alg, m, overlap)}
    H = []
    # A. learnability and necessity of interventions
    H.append(("P2a", "atd: L1 vs exact no-op truth below P2a_l1 (main + overlap)",
              lambda: S.threshold_family(learn("atd"), TAU["P2a_l1"])))
    H.append(("P2b", "atd beats the equal split (main + overlap)", lambda: S.paired_family(learn("atd"), learn("uniform"))))
    H.append(("P3", "interventions are necessary: atd_noint worse than atd by >= P3_gap (main + overlap)",
              lambda: S.paired_family(learn("atd"), learn("atd_noint"), TAU["P3_gap"])))
    H.append(("P4a", "atd per-step fidelity (vs exact kappa) above P4a_r (main)",
              lambda: S.threshold_family(collect(main, "atd", "step_r"), TAU["P4a_r"], "above")))
    for alg in ("vdn", "qmix", "coma", "mappo"):
        H.append((f"P4b-{alg}", f"{alg}: per-step fidelity vs its own one-step target below P4b_r (main)",
                  lambda alg=alg: S.threshold_family(collect(main, alg, "step_r"), TAU["P4b_r"])))
    # B. coalition level on fixed layouts: the table is not worse than ECE
    h5conds = lambda alg: {**collect(main, alg, "div_c", lambda c: c[2] == 2 and c[3] == "slack" and c[1] in (0, 50)),
                           **collect(h5, alg, "div_c")}
    H.append(("P5", "table3_mix dividend error not worse than ece3_mix by more than P5_margin (closed-loop removal game)",
              lambda: S.paired_family(h5conds("ece3_mix"), h5conds("table3_mix"), -TAU["P5_margin"])))
    # C. consequence: top-k selection
    H.append(("P6a", "atd keeps exactly the coalition (top-k) with probability above P6a_topk (main)",
              lambda: S.threshold_family(collect(main, "atd", "top_k"), TAU["P6a_topk"], "above")))
    for alg in ("qmix", "coma"):
        H.append((f"P6b-{alg}", f"{alg} keeps the coalition with probability below P6b_topk (main)",
                  lambda alg=alg: S.threshold_family(collect(main, alg, "top_k"), TAU["P6b_topk"])))
    # D. generalisation across role assignments
    H.append(("P7a", "shuffled roles: table3_mix not better than uniform by more than P7a_margin",
              lambda: S.paired_family(collect(perm, "uniform", "loo_cn"), collect(perm, "table3_mix", "loo_cn"),
                                      -TAU["P7a_margin"])))
    H.append(("P7b", "shuffled roles: MLP atd below uniform by >= P7b_gap",
              lambda: S.paired_family(collect(perm, "atd", "loo_cn"), collect(perm, "uniform", "loo_cn"), TAU["P7b_gap"])))
    H.append(("P8a", "shuffled roles: set critic L1 below P8a_l1",
              lambda: S.threshold_family(collect(perm, "atd_set", "loo_cn"), TAU["P8a_l1"])))
    H.append(("P8b", "shuffled roles: set critic beats MLP by >= P8b_gap",
              lambda: S.paired_family(collect(perm, "atd_set", "loo_cn"), collect(perm, "atd", "loo_cn"), TAU["P8b_gap"])))
    H.append(("P8c-a", "shuffled roles: set critic dividend error (own null) below P8c_div",
              lambda: S.threshold_family(collect(perm, "atd_set", "div_cn"), TAU["P8c_div"])))
    H.append(("P8c-b", "shuffled roles: set critic dividend error below ece3_mix's",
              lambda: S.paired_family(collect(perm, "atd_set", "div_cn"), collect(perm, "ece3_mix", "div_c"))))
    H.append(("P8d", "shuffled roles, role flags hidden: set critic beats MLP by >= P8d_gap",
              lambda: S.paired_family(collect(perm, "atd_set_noroles", "loo_cn"), collect(perm, "atd_noroles", "loo_cn"),
                                      TAU["P8d_gap"])))
    # E. better data or better read-out (control V2)
    H.append(("P9a", "MAPPO on intervention data: masked read-out beats its standard read-out by >= P9a_gap (main)",
              lambda: S.paired_family(collect(main, "mappo_int_masked", "loo_cn"), collect(main, "mappo_int", "loo_cn"),
                                      TAU["P9a_gap"])))
    H.append(("P9b", "QMIX on intervention data: masked read-out beats its standard read-out (main, d > 0)",
              lambda: S.paired_family(collect(main, "qmix_int_masked", "loo_cn", lambda c: c[1] > 0),
                                      collect(main, "qmix_int", "loo_cn", lambda c: c[1] > 0))))
    H.append(("P9c", "data alone does not fix one-step read-outs: mappo_int improves on mappo by less than P9c_gain",
              lambda: S.paired_family(collect(main, "mappo", "loo_cn"), collect(main, "mappo_int", "loo_cn"),
                                      -TAU["P9c_gain"])))
    # F. luck: partial synergy
    H.append(("P10a", "partial: structural read-out within P10a_margin of the expected-credit floor",
              lambda: S.paired_family(collect(h5, "atd_struct", "loo_cn", partial), collect(h5, "floor_expected", "loo_cn", partial),
                                      -TAU["P10a_margin"])))
    H.append(("P10b", "partial: structural read-out beats kappa",
              lambda: S.paired_family(collect(h5, "atd_struct", "loo_cn", partial), collect(h5, "atd", "loo_cn", partial))))
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "v3", "confirmatory_results.md"))
    a = p.parse_args()
    parts = ["# v3 confirmatory results (generated by scripts/v3/analyse_prereg.py)", ""]
    for hid, text, fn in hypotheses():
        try:
            res = fn()
            parts += [f"## {hid}: {text}", "", fmt_family(res), ""]
        except Exception as e:  # report, never hide
            parts += [f"## {hid}: {text}", "", f"**ERROR**: {e!r}", ""]
    open(a.out, "w").write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
