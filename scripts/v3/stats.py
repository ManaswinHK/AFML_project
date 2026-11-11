"""Statistics for the v3 pre-registration (docs/v3/preregistration_v3.md, Section "Analysis").

Unit of analysis: one seed (a data cell). Its value is the mean over the cell's 30 test episodes.
Per condition there are 5 seeds (5-9). Within each hypothesis, conditions are a Holm family.

* threshold(values, tau, direction):  one-sided one-sample t-test of mean(values) < tau ("below") or > tau
  ("above"), df = n - 1. Supported in a condition if the Holm-adjusted p < 0.05.
* paired(a, b, delta):  one-sided paired t-test that mean(b - a) > delta (a is better than b by more than
  delta, for lower-is-better metrics). Supported if Holm-adjusted p < 0.05.
* An exact one-sided Wilcoxon signed-rank p is reported alongside (descriptive: with 5 seeds its minimum is
  1/32, so it cannot survive Holm over many conditions).
* A hypothesis is SUPPORTED if every condition in its family is supported, PARTIAL if some are, else NOT.
  Degenerate case: zero variance across seeds. Then the decision is the sign of (mean - tau) (resp. of
  mean(b - a) - delta), with p reported as 0 or 1.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def _t_one_sided(x: np.ndarray, mu: float, alternative: str) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x, ddof=1) == 0:
        m = x.mean()
        ok = m < mu if alternative == "less" else m > mu
        return 0.0 if ok else 1.0
    return float(stats.ttest_1samp(x, mu, alternative=alternative).pvalue)


def wilcoxon_exact(d: np.ndarray, alternative: str = "greater") -> float:
    """Exact one-sided signed-rank p for small n (all 2^n sign patterns); zeros dropped."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(d))
    w = ranks[d > 0].sum()
    null = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=n)]
    null = np.array(null)
    return float(np.mean(null >= w) if alternative == "greater" else np.mean(null <= w))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def threshold_family(by_condition: dict, tau: float, direction: str = "below", alpha: float = 0.05) -> dict:
    """by_condition: {condition: per-seed values}. Tests mean < tau (below) or mean > tau (above)."""
    alt = "less" if direction == "below" else "greater"
    conds = sorted(by_condition)
    raw = [_t_one_sided(by_condition[c], tau, alt) for c in conds]
    adj = holm(raw)
    rows = []
    for c, p, pa in zip(conds, raw, adj):
        x = np.asarray(by_condition[c], float)
        d = (tau - x) if direction == "below" else (x - tau)
        rows.append({"condition": c, "n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                     "tau": tau, "p": p, "p_holm": pa, "wilcoxon_p": wilcoxon_exact(d), "supported": pa < alpha})
    return _verdict(rows)


def paired_family(a: dict, b: dict, delta: float = 0.0, alpha: float = 0.05) -> dict:
    """Lower-is-better metric. Tests mean(b - a) > delta per condition (a beats b by more than delta)."""
    conds = sorted(set(a) & set(b))
    raw, rows = [], []
    for c in conds:
        d = np.asarray(b[c], float) - np.asarray(a[c], float)
        raw.append(_t_one_sided(d, delta, "greater"))
    adj = holm(raw)
    for c, p, pa in zip(conds, raw, adj):
        d = np.asarray(b[c], float) - np.asarray(a[c], float)
        rows.append({"condition": c, "n": len(d), "mean_a": float(np.mean(a[c])), "mean_b": float(np.mean(b[c])),
                     "mean_diff": float(d.mean()), "delta": delta, "p": p, "p_holm": pa,
                     "wilcoxon_p": wilcoxon_exact(d - delta), "supported": pa < alpha})
    return _verdict(rows)


def _verdict(rows: list[dict]) -> dict:
    k = sum(r["supported"] for r in rows)
    if not rows:
        return {"verdict": "NO DATA", "k": 0, "m": 0, "rows": rows}
    verdict = "SUPPORTED" if k == len(rows) else ("PARTIAL" if k else "NOT SUPPORTED")
    return {"verdict": verdict, "k": k, "m": len(rows), "rows": rows}


def two_sided_family(a: dict, b: dict) -> dict:
    """Descriptive two-sided paired comparison per condition (no verdict, no Holm decision)."""
    rows = []
    for c in sorted(set(a) & set(b)):
        d = np.asarray(b[c], float) - np.asarray(a[c], float)
        p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and np.std(d, ddof=1) > 0 else float("nan")
        rows.append({"condition": c, "n": len(d), "mean_a": float(np.mean(a[c])), "mean_b": float(np.mean(b[c])),
                     "mean_diff": float(d.mean()), "delta": 0.0, "p": p, "p_holm": p, "wilcoxon_p": float("nan"),
                     "supported": False})
    return {"verdict": "DESCRIPTIVE", "k": 0, "m": len(rows), "rows": rows}
