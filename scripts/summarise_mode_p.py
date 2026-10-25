"""Summarise a directory of Mode P data-cell JSONs into summary.md and per_seed.csv."""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

METRICS = ("l1", "spearman", "leakage", "missed")


def _fmt(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def load(out_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    return records


def per_seed_rows(records):
    rows = []
    for rec in records:
        c = rec["cell"]
        for alg, a in rec["algs"].items():
            s = a["summary"]
            row = {"alg": alg, "delay": c["delay"], "bystanders": c["n_bystanders"], "seed": c["seed"],
                   "valid": f'{s["n_valid"]}/{s["n_total"]}', "seconds": round(a["seconds"], 1),
                   "final_loss": a["loss_trace"][-1][1]}
            row.update({m: s[m] for m in METRICS})
            if "snr" in a:
                n = 3 + c["n_bystanders"]
                snr = np.array(a["snr"]["snr"], dtype=float)
                row["snr_coalition"] = float(np.nanmean(snr[:3]))
                row["snr_bystanders"] = float(np.nanmean(snr[3:n])) if n > 3 else None
            rows.append(row)
    return rows


def summarise_dir(out_dir):
    records = load(out_dir)
    if not records:
        return f"no results in {out_dir}"
    rows = per_seed_rows(records)

    with open(os.path.join(out_dir, "per_seed.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["alg"], r["delay"], r["bystanders"])].append(r)

    lines = [f"# Mode P summary: `{os.path.basename(out_dir)}`", "",
             f"{len(records)} data cells. Metrics use each read-out's matched target (leave-one-out); "
             "mean ± sd over seeds of the per-seed means over test episodes.", "",
             "| alg | delay | bystanders | seeds | L1 | Spearman | leakage | missed | valid eps |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (alg, d, b), rs in sorted(groups.items(), key=lambda kv: (["vdn", "qmix", "coma", "mappo"].index(kv[0][0]), kv[0][1], kv[0][2])):
        cells = [_fmt([r[m] for r in rs]) for m in METRICS]
        if b == 0:
            cells[2] = "n/a (none)"
        valid = ", ".join(r["valid"] for r in rs)
        lines.append(f"| {alg} | {d} | {b} | {len(rs)} | " + " | ".join(cells) + f" | {valid} |")

    snr_rows = [r for r in rows if "snr_coalition" in r]
    if snr_rows:
        lines += ["", "## MAPPO policy-gradient signal-to-noise (mean over agents in each role)", "",
                  "| delay | bystanders | coalition SNR | bystander SNR |", "|---|---|---|---|"]
        g = defaultdict(list)
        for r in snr_rows:
            g[(r["delay"], r["bystanders"])].append(r)
        for (d, b), rs in sorted(g.items()):
            lines.append(f"| {d} | {b} | {_fmt([r['snr_coalition'] for r in rs])} | "
                         f"{_fmt([r['snr_bystanders'] for r in rs]) if b else 'n/a'} |")

    lines += ["", "## Data and ground truth (H0)", "",
              "| cell | test return | realised = planted | clipped | GT remove-vs-noop L1 (mean / max) | minutes |",
              "|---|---|---|---|---|---|"]
    for rec in records:
        c, d, h0 = rec["cell"], rec["data"], rec["h0"]
        lines.append(f"| d{c['delay']} b{c['n_bystanders']} s{c['seed']} | {d['test_return_mean']:.2f} | "
                     f"{d['test_matches_plan']:.2f} | {d['test_clipped']:.2f} | "
                     f"{h0['remove_vs_noop_l1_mean']:.3f} / {h0['remove_vs_noop_l1_max']:.3f} | "
                     f"{rec['seconds'] / 60:.1f} |")

    lines += ["", "## Final critic loss (mean over seeds)", "", "| alg | delay | bystanders | final loss |", "|---|---|---|---|"]
    for (alg, d, b), rs in sorted(groups.items()):
        lines.append(f"| {alg} | {d} | {b} | {_fmt([r['final_loss'] for r in rs])} |")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(summarise_dir(sys.argv[1]))
