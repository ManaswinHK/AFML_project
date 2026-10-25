"""Test the pre-registered Mode P hypotheses (docs/pilot_findings.md, as amended in
docs/preregistration_amendment.md) on a results directory. Writes hypotheses.md there.

    .venv/Scripts/python scripts/analyse_hypotheses.py results/mode_p/main
"""

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarise_mode_p import load, per_seed_rows  # noqa: E402

ALPHA = 0.05
ALGS = ("vdn", "qmix", "coma", "mappo")


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired(x: dict, y: dict) -> dict | None:
    """One-sided paired tests that x > y over the seeds present in both."""
    seeds = sorted(set(x) & set(y))
    xs = np.array([x[s] for s in seeds], dtype=float)
    ys = np.array([y[s] for s in seeds], dtype=float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 3:
        return None
    d = xs - ys
    t_p = ttest_rel(xs, ys, alternative="greater").pvalue if d.std() > 0 else (0.0 if d.mean() > 0 else 1.0)
    w_p = wilcoxon(xs, ys, alternative="greater").pvalue if np.any(d != 0) else 1.0
    return {"n": int(len(xs)), "diff": float(d.mean()), "t_p": float(t_p), "w_p": float(w_p)}


def _num(v):
    return np.nan if v is None or v == "" else float(v)


def index(rows):
    """(alg, delay, bystanders, enabler_start) -> {metric -> {seed -> value}}"""
    idx = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["alg"], r["delay"], r["bystanders"], r["enabler_start"])
        for m in ("l1", "leakage", "missed", "final_loss", "fid_step_pearson"):
            if m in r:
                idx[key][m][r["seed"]] = _num(r[m])
    return idx


def fmt_p(p):
    return "n/a" if p is None or np.isnan(p) else (f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}")


def test_h0(records):
    lines = ["## H0: ground truth is protocol-stable", "",
             "Rule: every cell has mean remove-vs-noop share L1 ≤ 0.15 and realised = planted ≥ 0.6."]
    l1s = [r["h0"]["remove_vs_noop_l1_mean"] or 0.0 for r in records]
    plan = [r["data"]["test_matches_plan"] for r in records]
    ok = all(v <= 0.15 for v in l1s) and all(v >= 0.6 for v in plan)
    lines += ["",
              f"- Cells: {len(records)}. Max mean L1: {max(l1s):.3f}. Max per-episode L1: "
              f"{max(r['h0']['remove_vs_noop_l1_max'] or 0 for r in records):.3f}.",
              f"- Min realised = planted: {min(plan):.2f}.",
              f"- **Verdict: {'SUPPORTED' if ok else 'NOT SUPPORTED'}**", ""]
    return lines


def test_h1a(idx):
    lines = ["## H1a: missed enabler credit grows with delay", "",
             "Rule: for some method and (bystanders, enabler) condition whose critic converged at both delays "
             "(mean final loss at d=50 ≤ 2× that at d=0), M(d=50) − M(d=0) ≥ 0.2 with Holm-adjusted paired t p < 0.05 "
             "(Holm over all conditions tested here).", "",
             "| alg | bystanders | enabler | converged | ΔM | t p | Wilcoxon p | Holm t p | meets |",
             "|---|---|---|---|---|---|---|---|---|"]
    tests = []
    for alg in ALGS:
        for b in (0, 2, 4):
            for es in sorted({k[3] for k in idx}):
                lo, hi = idx.get((alg, 0, b, es)), idx.get((alg, 50, b, es))
                if not lo or not hi:
                    continue
                res = paired(hi["missed"], lo["missed"])
                if res is None:
                    continue
                conv = np.nanmean(list(hi["final_loss"].values())) <= 2 * np.nanmean(list(lo["final_loss"].values()))
                tests.append((alg, b, es, conv, res))
    if not tests:
        return lines[:2] + ["Not available: needs d=0 and d=50.", ""]
    adj = holm([t[4]["t_p"] for t in tests])
    any_ok = False
    for (alg, b, es, conv, res), a in zip(tests, adj):
        meets = conv and res["diff"] >= 0.2 and a < ALPHA
        any_ok |= meets
        lines.append(f"| {alg} | {b} | {'slack' if not es else 'critical'} | {'yes' if conv else 'no'} | "
                     f"{res['diff']:+.3f} | {fmt_p(res['t_p'])} | {fmt_p(res['w_p'])} | {fmt_p(a)} | {'✓' if meets else ''} |")
    lines += ["", f"**Verdict: {'SUPPORTED' if any_ok else 'NOT SUPPORTED'}**", ""]
    return lines


def test_h1b(records):
    lines = ["## H1b: slack, not delay, removes per-step enabler credit (level 2)", "",
             "Rule: for each per-step notion, the seed-mean level-2 missed credit is ≥ 0.8 in every slack cell "
             "and ≤ 0.3 in every critical cell.", "",
             "| notion | enabler | delay | bystanders | missed (mean over seeds) | meets |", "|---|---|---|---|---|---|"]
    g = defaultdict(list)
    for r in records:
        c = r["cell"]
        for kind, v in r.get("semantic_gap", {}).items():
            if v["missed"] is not None:
                g[(kind, c.get("enabler_start", 0), c["delay"], c["n_bystanders"])].append(v["missed"])
    verdict = {}
    for (kind, es, d, b), vs in sorted(g.items()):
        m = float(np.mean(vs))
        meets = m >= 0.8 if not es else m <= 0.3
        verdict.setdefault(kind, []).append(meets)
        lines.append(f"| {kind} | {'slack' if not es else 'critical'} | {d} | {b} | {m:.3f} | {'✓' if meets else '✗'} |")
    lines.append("")
    for kind, oks in verdict.items():
        lines.append(f"- {kind}: {sum(oks)}/{len(oks)} cells meet → **{'SUPPORTED' if all(oks) else 'NOT SUPPORTED'}**")
    if not any(k[1] for k in g):
        lines.append("- No critical-enabler cells in this directory.")
    return lines + [""]


def test_h2(idx):
    lines = ["## H2: leakage grows with bystanders", "",
             "Rule per (method, delay, enabler) condition: Λ(b=4) − Λ(b=2) ≥ 0.05 with Holm-adjusted paired t p < 0.05 "
             "(Holm over all conditions) and mean Λ(b=2) > 0.1. A method is SUPPORTED if at least 75% of its "
             "conditions meet the rule.", "",
             "| alg | delay | enabler | Λ(b=2) | ΔΛ | t p | Wilcoxon p | Holm t p | meets |",
             "|---|---|---|---|---|---|---|---|---|"]
    tests = []
    for alg in ALGS:
        for d in sorted({k[1] for k in idx}):
            for es in sorted({k[3] for k in idx}):
                b2, b4 = idx.get((alg, d, 2, es)), idx.get((alg, d, 4, es))
                if not b2 or not b4:
                    continue
                res = paired(b4["leakage"], b2["leakage"])
                if res is not None:
                    tests.append((alg, d, es, float(np.nanmean(list(b2["leakage"].values()))), res))
    adj = holm([t[4]["t_p"] for t in tests]) if tests else []
    per_alg = defaultdict(list)
    for (alg, d, es, lam2, res), a in zip(tests, adj):
        meets = res["diff"] >= 0.05 and a < ALPHA and lam2 > 0.1
        per_alg[alg].append(meets)
        lines.append(f"| {alg} | {d} | {'slack' if not es else 'critical'} | {lam2:.3f} | {res['diff']:+.3f} | "
                     f"{fmt_p(res['t_p'])} | {fmt_p(res['w_p'])} | {fmt_p(a)} | {'✓' if meets else ''} |")
    lines.append("")
    for alg in ALGS:
        if per_alg[alg]:
            frac = np.mean(per_alg[alg])
            lines.append(f"- {alg}: {sum(per_alg[alg])}/{len(per_alg[alg])} conditions → "
                         f"**{'SUPPORTED' if frac >= 0.75 else 'NOT SUPPORTED'}**")
    return lines + [""]


def test_h2_prime(idx):
    lines = ["## H2′: COMA leaks less than VDN and QMIX at d = 0", "",
             "Rule: in every (bystanders ∈ {2, 4}, enabler) condition at d=0, Λ_VDN > Λ_COMA and Λ_QMIX > Λ_COMA, "
             "each with Holm-adjusted paired t p < 0.05 (Holm over all comparisons).", "",
             "| bystanders | enabler | other | Λ_other − Λ_COMA | t p | Wilcoxon p | Holm t p | meets |",
             "|---|---|---|---|---|---|---|---|"]
    tests = []
    for b in (2, 4):
        for es in sorted({k[3] for k in idx}):
            coma = idx.get(("coma", 0, b, es))
            for other in ("vdn", "qmix"):
                o = idx.get((other, 0, b, es))
                if coma and o:
                    res = paired(o["leakage"], coma["leakage"])
                    if res is not None:
                        tests.append((b, es, other, res))
    if not tests:
        return lines[:2] + ["Not available.", ""]
    adj = holm([t[3]["t_p"] for t in tests])
    oks = []
    for (b, es, other, res), a in zip(tests, adj):
        meets = res["diff"] > 0 and a < ALPHA
        oks.append(meets)
        lines.append(f"| {b} | {'slack' if not es else 'critical'} | {other} | {res['diff']:+.3f} | "
                     f"{fmt_p(res['t_p'])} | {fmt_p(res['w_p'])} | {fmt_p(a)} | {'✓' if meets else ''} |")
    return lines + ["", f"**Verdict: {'SUPPORTED' if all(oks) else 'NOT SUPPORTED'}** ({sum(oks)}/{len(oks)})", ""]


def test_h5(idx):
    lines = ["## H5: end-to-end accuracy does not imply fidelity", "",
             "Rule: among all (method, delay, bystanders, enabler) cells, at least one in the best third by end-to-end "
             "L1 has level-1 step r < 0.3.", ""]
    cells = []
    for key, m in idx.items():
        l1 = np.nanmean(list(m["l1"].values())) if m["l1"] else np.nan
        r = np.nanmean(list(m["fid_step_pearson"].values())) if m.get("fid_step_pearson") else np.nan
        if not np.isnan(l1):
            cells.append((key, l1, r))
    if not cells:
        return lines + ["Not available.", ""]
    cut = np.quantile([c[1] for c in cells], 1 / 3)
    best = sorted([c for c in cells if c[1] <= cut], key=lambda c: c[1])
    hits = [c for c in best if not np.isnan(c[2]) and c[2] < 0.3]
    lines += [f"- Best-third cut-off: L1 ≤ {cut:.3f} ({len(best)} of {len(cells)} cells).",
              f"- Of those, {len(hits)} have step r < 0.3. Lowest-L1 examples:", "",
              "| alg | delay | bystanders | enabler | end-to-end L1 | step r |", "|---|---|---|---|---|---|"]
    for (alg, d, b, es), l1, r in best[:10]:
        lines.append(f"| {alg} | {d} | {b} | {'slack' if not es else 'critical'} | {l1:.3f} | {r:.3f} |")
    return lines + ["", f"**Verdict: {'SUPPORTED' if hits else 'NOT SUPPORTED'}**", ""]


def main(out_dir):
    records = load(out_dir)
    if not records:
        raise SystemExit(f"no results in {out_dir}")
    idx = index(per_seed_rows(records))
    lines = [f"# Hypothesis tests: `{os.path.basename(out_dir)}`", "",
             f"{len(records)} data cells. Primary test: one-sided paired t-test on per-seed means, Holm-corrected within "
             "each hypothesis (see docs/preregistration_amendment.md). Exact Wilcoxon p-values are shown for reference; "
             "with 5 seeds their floor is 1/32 = 0.031.", ""]
    lines += test_h0(records) + test_h1a(idx) + test_h1b(records) + test_h2(idx) + test_h2_prime(idx) + test_h5(idx)
    lines += ["H3 (coalition level) needs NA²Q (Step 7). H4 is Mode T (Step 8).", ""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "hypotheses.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(main(sys.argv[1] if len(sys.argv) > 1 else "results/mode_p/main"))
