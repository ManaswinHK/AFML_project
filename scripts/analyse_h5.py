"""H5 analysis: the low-order episodic coalition estimator (ECE-k) on the H5 grid.

Pre-registered rules: docs/h5_preregistration.md. Comparison methods on the same data cells:
NA²Q and STAS from the H3 grid (overlap, partial) and from the h5_ref grid (coalition layout).

    .venv/Scripts/python scripts/analyse_h5.py
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyse_h3 import uniform_l1  # noqa: E402
from analyse_hypotheses import holm  # noqa: E402

MODE_P = os.path.join(ROOT, "results", "mode_p")
# Highest order of the planted coalitions (overlap: {A,B,C} and {B,D}; partial: {A,B} / {A,C}).
MAX_ORDER = {"coalition": 3, "overlap": 3, "partial": 2}
LAYOUTS = ("coalition", "overlap", "partial")
METRICS = ("l1", "leakage", "dividend_error", "recall", "precision", "interaction_mass")


def _layout(cell):
    return cell.get("layout", "coalition")


def load(dirs):
    """(layout, delay, seed) -> {"truth": shapley list, "algs": {alg: record}} merged over directories."""
    cells = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                rec = json.load(fh)
            c = rec["cell"]
            key = (_layout(c), c["delay"], c["seed"])
            entry = cells.setdefault(key, {"truth": rec["truth"]["shapley"], "algs": {}})
            if not np.allclose(entry["truth"], rec["truth"]["shapley"]):
                raise RuntimeError(f"{key}: ground truth differs between result directories")
            entry["algs"].update(rec["algs"])
    return cells


def per_seed(cells):
    """(alg, layout, delay) -> {metric -> {seed -> value}}; alg "uniform" holds the uniform L1."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (lay, dl, s), entry in cells.items():
        out[("uniform", lay, dl)]["l1"][s] = float(np.mean([uniform_l1(v) for v in entry["truth"]]))
        for alg, a in entry["algs"].items():
            if alg not in ("na2q", "stas", "na2q_mix", "stas_mix") and not alg.startswith("ece"):
                continue
            m = out[(alg, lay, dl)]
            m["l1"][s], m["leakage"][s] = a["summary"]["l1"], a["summary"]["leakage"]
            m["seconds"][s] = a["seconds"]
            for k in METRICS[2:]:
                if "summary_coalition" in a:
                    m[k][s] = a["summary_coalition"][k]
    return out


def paired_less(x: dict, y: dict) -> dict | None:
    """One-sided paired tests that x < y over common seeds."""
    seeds = sorted(set(x) & set(y))
    if len(seeds) < 3:
        return None
    xs, ys = np.array([x[s] for s in seeds]), np.array([y[s] for s in seeds])
    d = xs - ys
    t_p = ttest_rel(xs, ys, alternative="less").pvalue if d.std() > 0 else (0.0 if d.mean() < 0 else 1.0)
    w_p = wilcoxon(xs, ys, alternative="less").pvalue if np.any(d != 0) else 1.0
    return {"n": len(seeds), "diff": float(d.mean()), "t_p": float(t_p), "w_p": float(w_p)}


def fmt(m: dict) -> str:
    v = np.array(list(m.values()), float)
    return "–" if len(v) == 0 else (f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")


def test_block(title, rows, out):
    """rows: list of (label, x, y). Holm over the block; verdict needs every row significant."""
    res = [(lab, paired_less(x, y)) for lab, x, y in rows]
    ok = [r for _, r in res if r is not None]
    adj = holm([r["t_p"] for r in ok]) if ok else []
    out += [f"### {title}", "", "| condition | mean diff | t p | Holm p | Wilcoxon p | |", "|---|---|---|---|---|---|"]
    passed, i = 0, 0
    for lab, r in res:
        if r is None:
            out.append(f"| {lab} | – | – | – | – | missing |")
            continue
        sig = r["diff"] < 0 and adj[i] < 0.05
        passed += sig
        out.append(f"| {lab} | {r['diff']:+.3f} | {r['t_p']:.4f} | {adj[i]:.4f} | {r['w_p']:.3f} | {'✓' if sig else '✗'} |")
        i += 1
    verdict = "supported" if res and passed == len(res) else "not supported"
    out += ["", f"**{passed}/{len(res)} conditions significant → {verdict}.**", ""]
    return passed, len(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=os.path.join(MODE_P, "h5"))
    p.add_argument("--refs", nargs="*", default=[os.path.join(MODE_P, "h3"), os.path.join(MODE_P, "h5_ref")])
    a = p.parse_args()
    cells = load([a.h5] + [d for d in a.refs if os.path.isdir(d)])
    cells = {k: v for k, v in cells.items() if any(x.startswith("ece") for x in v["algs"])}
    ps = per_seed(cells)
    delays = sorted({k[1] for k in cells})
    algs = [x for x in ("uniform", "na2q", "na2q_mix", "stas", "stas_mix", "ece2", "ece3", "ece2_mix", "ece3_mix")
            if any((x, lay, dl) in ps for lay in LAYOUTS for dl in delays)]

    out = [f"# H5 analysis (`{os.path.basename(os.path.normpath(a.h5))}`)", "",
           "Generated by `scripts/analyse_h5.py`. Mean ± sd over seeds of per-seed means over 30 test episodes.",
           "Individual credit is graded against Shapley shares. `_mix` = trained on the episodes plus open-loop replays with random agents removed.", ""]
    for metric in METRICS:
        out += [f"## {metric}", "", "| alg | " + " | ".join(f"{lay} d={dl}" for lay in LAYOUTS for dl in delays) + " |",
                "|---|" + "---|" * (len(LAYOUTS) * len(delays))]
        for alg in algs:
            vals = [fmt(ps[(alg, lay, dl)][metric]) if (alg, lay, dl) in ps else "–" for lay in LAYOUTS for dl in delays]
            if any(v != "–" for v in vals):
                out.append(f"| {alg} | " + " | ".join(vals) + " |")
        out.append("")

    out += ["## Cost", "", "| alg | mean seconds per cell | parameters |", "|---|---|---|"]
    params = {}
    for entry in cells.values():
        for alg, rec in entry["algs"].items():
            if "n_parameters" in rec:
                params[alg] = rec["n_parameters"]
    for alg in algs[1:]:
        secs = [v for lay in LAYOUTS for dl in delays for v in ps[(alg, lay, dl)]["seconds"].values()]
        out.append(f"| {alg} | {np.mean(secs):.0f} | {params.get(alg, '–')} |" if secs else f"| {alg} | – | – |")
    out.append("")

    out += ["## Pre-registered tests (docs/h5_preregistration.md)", ""]
    for ref, name in (("na2q", "H5a"), ("na2q_mix", "H5a′ (data-matched)")):
        for metric in ("dividend_error", "leakage"):
            rows = [(f"ece{k}_mix vs {ref}, {lay} d={dl}", ps[(f"ece{k}_mix", lay, dl)][metric],
                     ps[(ref, lay, dl)][metric])
                    for k in (2, 3) for lay in LAYOUTS if MAX_ORDER[lay] <= k for dl in delays]
            test_block(f"{name} ({metric}): ECE-k_mix below {ref} where planted order ≤ k", rows, out)
    rows = [(f"ece{k}_mix vs uniform, {lay} d={dl}", ps[(f"ece{k}_mix", lay, dl)]["l1"], ps[("uniform", lay, dl)]["l1"])
            for k in (2, 3) for lay in LAYOUTS if MAX_ORDER[lay] <= k for dl in delays]
    test_block("H5b (individual L1): ECE-k_mix below uniform credit where planted order ≤ k", rows, out)

    out += ["## Honesty check: ECE-2 on order-3 layouts (descriptive)", "",
            "| layout | d | ECE-2_mix dividend error | ECE-3_mix dividend error | ECE-2_mix L1 | ECE-3_mix L1 |",
            "|---|---|---|---|---|---|"]
    for lay in LAYOUTS:
        if MAX_ORDER[lay] == 3:
            for dl in delays:
                g = lambda alg, m: fmt(ps[(alg, lay, dl)][m])  # noqa: E731
                out.append(f"| {lay} | {dl} | {g('ece2_mix', 'dividend_error')} | {g('ece3_mix', 'dividend_error')} "
                           f"| {g('ece2_mix', 'l1')} | {g('ece3_mix', 'l1')} |")

    text = "\n".join(out) + "\n"
    with open(os.path.join(a.h5, "h5_analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
