"""Confirmatory analysis of docs/v5/preregistration_v5.md (Parts E and F). Writes docs/v5/results.md.

    .venv/bin/python scripts/v5/analyse_v5.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "v3"))
import stats as S  # noqa: E402
from analyse_phase3 import ratio_stats, table  # noqa: E402
from analyse_prereg import fmt_family  # noqa: E402
from run_phase3 import GRIDS, OUT, curve, run_name  # noqa: E402

ARMS = ("team", "expected", "difference")


def _one_sided(x, tau, alternative):
    if np.std(x, ddof=1) > 0:
        return float(stats.ttest_1samp(x, np.log(tau), alternative=alternative).pvalue)
    return 0.0 if (x.mean() > np.log(tau)) == (alternative == "greater") else 1.0


def drawdown(grid, alg, K, arm, seeds):
    out = []
    for s in seeds:
        c = curve(os.path.join(OUT, grid, run_name(alg, K, arm, s)))
        if c is None or not np.any(c[1] >= 0.5):
            continue
        y = c[1][np.argmax(c[1] >= 0.5):]
        out.append(float(np.max(np.maximum.accumulate(y) - y)))
    return out


def part_e(L):
    L += ["## Part E: the learning question with a centralised critic (MAPPO)", ""]
    for grid, alg, label in (("credit_mappo_ns", "mappo_ns", "`mappo_ns` (non-shared actors, state critic; confirmatory)"),
                             ("credit_mappo", "mappo", "`mappo` (shared parameters; descriptive)"),
                             ("online_credit", "ippo_ns", "`ippo_ns` (v4 Part C, for comparison)")):
        t = table(grid)
        if not t:
            L += [f"_{grid}: no data_", ""]
            continue
        g = GRIDS[grid]
        L += [f"### {label}", "", "| K | arm | n | episodes to 90% (mean) | per seed | censored | max drawdown after 50% (per seed) |",
              "|---|---|---|---|---|---|---|"]
        for K in g["Ks"]:
            for arm in ARMS:
                rows = t.get((K, arm), [])
                if rows:
                    dd = drawdown(grid, alg, K, arm, g["seeds"])
                    L.append(f"| {K} | {arm} | {len(rows)} | {np.mean([r[1] for r in rows]):,.0f} | {[int(r[1]) for r in rows]} | "
                             f"{sum(r[2] for r in rows)} | {np.round(dd, 2).tolist()} |")
        L += ["", "| K | ratio | geometric mean of per-seed ratios | x simulators per step = cost ratio | per seed |", "|---|---|---|---|---|"]
        for K in g["Ks"]:
            for num, den in (("expected", "team"), ("difference", "team"), ("difference", "expected")):
                r = ratio_stats(t, K, num, den)
                if r:
                    c = (2 * K + 1 if num == "difference" else 1) / (2 * K + 1 if den == "difference" else 1)
                    gm = np.exp(r["mean_log_ratio"])
                    L.append(f"| {K} | {num} / {den} | {gm:.3f} | {gm * c:.3f} | {r['per_seed']} |")
        L.append("")
    t, t_ip = table("credit_mappo_ns"), table("online_credit")
    tests = []
    r = ratio_stats(t, 8, "expected", "team")
    if r is not None and len(r["logr"]) > 1:
        p = _one_sided(r["logr"], 0.5, "greater")
        tests.append(f"- **E1** (K = 8, expected/team > 0.5): geometric mean {np.exp(r['mean_log_ratio']):.3f}, p = {p:.4f} -> "
                     f"**{'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}**")
    r2 = ratio_stats(t, 8, "difference", "team")
    if r2 is not None and len(r2["logr"]) > 1:
        p = _one_sided(r2["logr"], 0.5, "less")
        tests.append(f"- **E2** (K = 8, difference/team < 0.5): geometric mean {np.exp(r2['mean_log_ratio']):.3f}, p = {p:.4f} -> "
                     f"**{'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}**")
    r3 = ratio_stats(t_ip, 8, "expected", "team")
    if r is not None and r3 is not None:
        p = float(stats.ttest_ind(r["logr"], r3["logr"], equal_var=False, alternative="greater").pvalue)
        tests.append(f"- **E3** (K = 8, log(expected/team): mappo_ns > ippo_ns): {np.exp(r['mean_log_ratio']):.3f} vs "
                     f"{np.exp(r3['mean_log_ratio']):.3f}, Welch p = {p:.4f} -> **{'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}**")
    for K in (4, 8):
        a = {s: e for s, e, *_ in t.get((K, "team"), [])}
        b = {s: e for s, e, *_ in t_ip.get((K, "team"), [])}
        sd = sorted(set(a) & set(b))
        if sd:
            tests.append(f"- E4 (descriptive), K = {K}: team reward, episodes mappo_ns / ippo_ns (same seeds): "
                         f"{np.exp(np.mean(np.log([a[s] / b[s] for s in sd]))):.3f}")
    return L + tests + [""]


def part_f(L):
    L += ["## Part F: Phase 4 extended (dropout levels, two layouts, robustness)", ""]
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v5", "phase4_v5", "*.json")))
    if not files:
        return L + ["_no data_", ""]
    R = defaultdict(dict)
    for f in files:
        r = json.load(open(f))
        R[(r["K"], r["dropout"])][r["seed"]] = r
    seeds = sorted({s for d in R.values() for s in d})
    get = lambda K, p, key: [R[(K, p)][s][key] for s in seeds if s in R[(K, p)]]
    Ks = sorted({k for k, _ in R})
    ps = sorted({p for _, p in R})
    L += ["| K | dropout | n | clean success | structural L1 | kappa L1 | equal split L1 | collateral (frac of others' return) | "
          "collateral, bystanders stopped | collateral, members stopped |", "|---|---|---|---|---|---|---|---|---|---|"]
    for K in Ks:
        for p in ps:
            m = lambda key: np.mean([v for v in get(K, p, key) if v is not None])
            L.append(f"| {K} | {p} | {len(R[(K, p)])} | {m('success'):.3f} | {m('posthoc_struct_l1'):.3f} | {m('masked_l1'):.3f} | "
                     f"{m('uniform_l1'):.3f} | {m('collateral_frac_t0'):.4f} | {m('collateral_bystander_t0'):.3f} | "
                     f"{m('collateral_member_t0'):.3f} |")
    L.append("")
    a, b = {}, {}
    for K in Ks:
        for p in ps:
            if p > 0:
                a[(f"K{K}", f"p{p}")] = get(K, p, "posthoc_struct_l1")
                b[(f"K{K}", f"p{p}")] = get(K, 0.0, "posthoc_struct_l1")
    L += ["### F1: structural L1(p = 0) - structural L1(p) > 0", "", fmt_family(S.paired_family(a, b, 0.0)), ""]
    a = {(f"K{K}",): get(K, 0.25, "posthoc_struct_l1") for K in Ks}
    b = {(f"K{K}",): get(K, 0.25, "uniform_l1") for K in Ks}
    L += ["### F2: uniform - structural > 0 at p = 0.25", "", fmt_family(S.paired_family(a, b, 0.0)), ""]
    succ = {(f"K{K}",): get(K, 0.5, "success") for K in Ks}
    L += ["### F3: clean success > 0.95 at p = 0.5", "", fmt_family(S.threshold_family(succ, 0.95, "above")), ""]
    worst = max(np.max(get(K, p, "collateral_frac_t0")) for K in Ks for p in ps)
    L += [f"- **F4** (descriptive): largest per-run collateral fraction over all arms = {worst:.4f} "
          f"({'below' if worst < 0.05 else 'NOT below'} the 5% expectation)", ""]
    return L


def main():
    L = ["# v5 confirmatory results (generated by scripts/v5/analyse_v5.py)", "",
         "Pre-registration: `docs/v5/preregistration_v5.md`. Seeds 10-14.", ""]
    L = part_e(L)
    L = part_f(L)
    out = os.path.join(ROOT, "docs", "v5", "results.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
