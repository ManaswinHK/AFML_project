"""Confirmatory analysis of v4 Parts B and C (docs/v4/preregistration_v4.md). Imported by analyse_v4.py."""
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


def _ratio_test(t, K, num, den, tau, alternative):
    r = ratio_stats(t, K, num, den)
    if r is None or len(r["logr"]) < 2:
        return None, "no data"
    x = r["logr"]
    if np.std(x, ddof=1) > 0:
        p = float(stats.ttest_1samp(x, np.log(tau), alternative=alternative).pvalue)
    else:
        p = 0.0 if (x.mean() > np.log(tau)) == (alternative == "greater") else 1.0
    return r, p


def part_c(lines):
    t = table("online_credit")
    lines += ["## Part C: expected-value credit vs realised difference rewards (non-shared IPPO)", "",
              "Episodes to 90% greedy test success (first crossing; censored at budget).", "",
              "| K | arm | n | episodes to 90% (mean) | per seed | censored |", "|---|---|---|---|---|---|"]
    for K in (4, 8):
        for arm in ("team", "expected", "difference"):
            rows = t.get((K, arm), [])
            if rows:
                lines.append(f"| {K} | {arm} | {len(rows)} | {np.mean([r[1] for r in rows]):.0f} | "
                             f"{[int(r[1]) for r in rows]} | {sum(r[2] for r in rows)} |")
    lines += ["", "| K | ratio | ratio of means | geometric mean of per-seed ratios | per seed |", "|---|---|---|---|---|"]
    for K in (4, 8):
        for num, den in (("expected", "team"), ("difference", "expected"), ("difference", "team")):
            r = ratio_stats(t, K, num, den)
            if r:
                lines.append(f"| {K} | {num} / {den} | {r['ratio_of_means']:.3f} | {np.exp(r['mean_log_ratio']):.3f} | "
                             f"{r['per_seed']} |")
    lines.append("")
    for cid, num, den, tau, alt, text in (
            ("C1", "expected", "team", 0.5, "greater", "expected/team > 0.5 (expected credit is not >2x faster)"),
            ("C2", "difference", "expected", 0.2, "less", "difference/expected < 0.2"),
            ("C3", "difference", "team", 0.2, "less", "difference/team < 0.2")):
        r, p = _ratio_test(t, 8, num, den, tau, alt)
        if r is None:
            lines.append(f"- **{cid}** ({text}): no data")
            continue
        lines.append(f"- **{cid}** (K = 8, {text}; paired one-sided t-test on log ratios): geometric mean ratio "
                     f"{np.exp(r['mean_log_ratio']):.3f}, p = {p:.4f} -> **{'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}**")
    lines.append("- **C4** (K = 4): descriptive, table above.")
    return lines + [""]


def part_b(lines):
    """Filled in by the Part B pre-registration (see docs/v4/preregistration_v4.md, Part B)."""
    return lines


def _cells(grid_dir, alg, name):
    from analyse_prereg import metric
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "v4", grid_dir, "*.json"))):
        r = json.load(open(p))
        if alg in r["algs"]:
            v = metric(r["algs"][alg], name)
            if v is not None:
                out[f"b{r['cell']['n_bystanders']}"].append(v)
    return dict(out)


def part_d(lines):
    lines += ["## Part D: random start cells (set critic vs MLP)", ""]
    if not glob.glob(os.path.join(ROOT, "results", "v4", "confirm_rs", "*.json")):
        return lines + ["_no data_", ""]
    lines += ["### D1: atd_set L1 (loo_cn) < 0.05", "", fmt_family(S.threshold_family(_cells("confirm_rs", "atd_set", "loo_cn"), 0.05)), ""]
    lines += ["### D2: atd_set step r > 0.95", "", fmt_family(S.threshold_family(_cells("confirm_rs", "atd_set", "step_r"), 0.95, "above")), ""]
    lines += ["### D3: atd - atd_set > 0.30", "", fmt_family(S.paired_family(_cells("confirm_rs", "atd_set", "loo_cn"),
                                                                            _cells("confirm_rs", "atd", "loo_cn"), 0.30)), ""]
    lines += ["Descriptive (mean over seeds): | method | b2 L1 | b4 L1 | b2 step r | b4 step r |", "|---|---|---|---|---|"]
    for alg in ("uniform", "atd_noint", "atd", "atd_n800", "atd_set", "atd_set_struct", "atd_n800_set", "table3_mix"):
        l1, r = _cells("confirm_rs", alg, "loo_cn"), _cells("confirm_rs", alg, "step_r")
        f = lambda d, k: f"{np.mean(d[k]):.3f}" if d.get(k) else "-"
        lines.append(f"| {alg} | {f(l1, 'b2')} | {f(l1, 'b4')} | {f(r, 'b2')} | {f(r, 'b4')} |")
    return lines + [""]


def _lbf(variant, method, key):
    out = defaultdict(list)
    for budget in ("std", "large"):
        for p in sorted(glob.glob(os.path.join(ROOT, "results", "v4", "lbf_confirm", budget, f"{variant}_s*.json"))):
            v = json.load(open(p))["methods"].get(method, {}).get(key)
            if v is not None:
                out[f"{variant} {budget}"].append(v)
    return dict(out)


def part_b(lines):
    lines += ["## Part B: Level-Based Foraging (heavy food, removal null)", ""]
    if not glob.glob(os.path.join(ROOT, "results", "v4", "lbf_confirm", "*", "*.json")):
        return lines + ["_no data_", ""]
    h = "heavy"
    lines += ["### B1: uniform - set > 0 (LOO L1)", "", fmt_family(S.paired_family(_lbf(h, "set", "l1"), _lbf(h, "uniform", "l1"), 0.0)), ""]
    lines += ["### B2: set_noint - set > 0.30", "", fmt_family(S.paired_family(_lbf(h, "set", "l1"), _lbf(h, "set_noint", "l1"), 0.30)), ""]
    lines += ["### B3: mlp - set > 0.15", "", fmt_family(S.paired_family(_lbf(h, "set", "l1"), _lbf(h, "mlp", "l1"), 0.15)), ""]
    lines += ["### B4: uniform - set > 0.05 (Shapley L1)", "", fmt_family(S.paired_family(_lbf(h, "set", "shap"), _lbf(h, "uniform", "shap"), 0.05)), ""]
    large = {k: v for k, v in _lbf(h, "set", "l1").items() if k.endswith("large")}
    lines += ["### B5: set L1 < 0.25 (large budget)", "", fmt_family(S.threshold_family(large, 0.25)), ""]
    s = _lbf(h, "set", "l1")
    if f"{h} std" in s and f"{h} large" in s:
        lines += ["### B6: set(std) - set(large) > 0", "", fmt_family(S.paired_family({"heavy": s[f"{h} large"]}, {"heavy": s[f"{h} std"]}, 0.0)), ""]
    lines += ["### Descriptive (mean over seeds 10-14)", "",
              "| variant budget | method | LOO L1 | Shapley L1 | top-k | spurious | step r |", "|---|---|---|---|---|---|---|"]
    for v in ("heavy", "heavy_large"):
        for m in ("floor_expected", "uniform", "set_noint", "mlp", "mlp_hi", "set", "set_struct", "food_noint", "food", "food_struct"):
            cols = [_lbf(v, m, k) for k in ("l1", "shap", "topk", "spur", "step_r")]
            for cond in sorted(cols[0]):
                f = lambda d: f"{np.mean(d[cond]):.3f}" if d.get(cond) else "-"
                lines.append(f"| {cond} | {m} | " + " | ".join(f(c) for c in cols) + " |")
    return lines + [""]
