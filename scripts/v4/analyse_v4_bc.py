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
