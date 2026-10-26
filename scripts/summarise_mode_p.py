"""Summarise a directory of Mode P data-cell JSONs into summary.md and per_seed.csv."""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from whoearned.experiments import MATCHED_TARGET  # noqa: E402

METRICS = ("l1", "spearman", "leakage", "missed")
COALITION = ("dividend_error", "dividend_error_order2", "recall", "precision", "interaction_mass", "true_interaction_mass")
ORDER = ["vdn", "qmix", "coma", "mappo", "na2q", "stas"]


def _fmt(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def _slack(es):
    return "slack" if not es else f"critical ({es})"


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
            row = {"alg": alg, "delay": c["delay"], "bystanders": c["n_bystanders"],
                   "enabler_start": c.get("enabler_start", 0), "seed": c["seed"],
                   "valid": f'{s["n_valid"]}/{s["n_total"]}', "seconds": round(a["seconds"], 1),
                   "final_loss": a["loss_trace"][-1][1]}
            row.update({m: s[m] for m in METRICS})
            if "summary_fidelity" in a:
                fid = a["summary_fidelity"]
                row.update(fid_l1=fid["l1"], fid_step_pearson=fid["step_pearson"],
                           fid_valid=f'{fid["n_valid"]}/{fid["n_total"]}')
            if "snr" in a:
                n = 3 + c["n_bystanders"]
                snr = np.array(a["snr"]["snr"], dtype=float)
                row["snr_coalition"] = float(np.nanmean(snr[:3]))
                row["snr_bystanders"] = float(np.nanmean(snr[3:n])) if n > 3 else None
            if "summary_coalition" in a:
                row.update({f"coal_{k}": a["summary_coalition"].get(k) for k in COALITION})
            rows.append(row)
    return rows


def _alg_key(kv):
    (alg, d, b, es), _ = kv
    return ORDER.index(alg), es, d, b


def summarise_dir(out_dir):
    records = load(out_dir)
    if not records:
        return f"no results in {out_dir}"
    rows = per_seed_rows(records)

    with open(os.path.join(out_dir, "per_seed.csv"), "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["alg"], r["delay"], r["bystanders"], r["enabler_start"])].append(r)

    lines = [f"# Mode P summary: `{os.path.basename(out_dir)}`", "",
             f"{len(records)} data cells; mean ± sd over seeds of per-seed means over test episodes. "
             "`enabler` = slack (A pulls before finishers are ready) or critical (A starts late).", ""]

    if any("fid_l1" in r for r in rows):
        lines += ["## Level 1: fidelity (read-out vs the per-step counterfactual it estimates)", "",
                  "VDN/QMIX/MAPPO vs the noop effect; COMA vs the behaviour-baseline effect. "
                  "`step r` = Pearson over all (episode, step, agent); `L1` = episode-summed shares.", "",
                  "| alg | enabler | delay | bystanders | step r | L1 |", "|---|---|---|---|---|---|"]
        for (alg, d, b, es), rs in sorted(groups.items(), key=_alg_key):
            lines.append(f"| {alg} | {_slack(es)} | {d} | {b} | {_fmt([r.get('fid_step_pearson') for r in rs])} | "
                         f"{_fmt([r.get('fid_l1') for r in rs])} |")
        lines += [""]

    gap_recs = [rec for rec in records if "semantic_gap" in rec]
    if gap_recs:
        lines += ["## Level 2: semantic gap (a *perfect* per-step critic vs leave-one-out removal credit)", "",
                  "Method-independent: the per-step truth itself, summed over the episode, scored against leave-one-out.", "",
                  "| per-step notion | enabler | delay | bystanders | L1 | leakage | missed | mean shares (A, B, C, bystanders...) |",
                  "|---|---|---|---|---|---|---|---|"]
        g = defaultdict(list)
        for rec in gap_recs:
            c = rec["cell"]
            for kind, v in rec["semantic_gap"].items():
                g[(kind, c.get("enabler_start", 0), c["delay"], c["n_bystanders"])].append(v)
        for (kind, es, d, b), vs in sorted(g.items()):
            shares = [v["mean_shares"] for v in vs if v["mean_shares"] is not None]
            sh = ", ".join(f"{x:.2f}" for x in np.mean(shares, axis=0)) if shares else "n/a"
            lines.append(f"| {kind} | {_slack(es)} | {d} | {b} | {_fmt([v['l1'] for v in vs])} | "
                         f"{_fmt([v['leakage'] for v in vs]) if b else 'n/a'} | {_fmt([v['missed'] for v in vs])} | {sh} |")
        lines += [""]

    lines += ["## End to end: read-out vs its matched removal target", "",
              "`target`: leave-one-out (loo) or Shapley, as matched to each read-out (problem statement v2, Section 2.4).", "",
              "| alg | target | enabler | delay | bystanders | seeds | L1 | Spearman | leakage | missed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for (alg, d, b, es), rs in sorted(groups.items(), key=_alg_key):
        cells = [_fmt([r[m] for r in rs]) for m in METRICS]
        if b == 0:
            cells[2] = "n/a (none)"
        lines.append(f"| {alg} | {MATCHED_TARGET[alg]} | {_slack(es)} | {d} | {b} | {len(rs)} | " + " | ".join(cells) + " |")

    coal_rows = [r for r in rows if "coal_dividend_error" in r]
    if coal_rows:
        lines += ["", "## Coalition level: mixer-game dividends vs true Harsanyi dividends (H3)", "",
                  "Dividend error is normalised (0 = exact, 2 = disjoint), over all orders and over orders ≤ 2. "
                  "Recall and precision are Jaccard matches between coalitions holding > 10% of |dividends| "
                  "and the true non-zero coalitions. Interaction mass = share of |dividends| on coalitions of 2+ agents "
                  "(estimated vs true).", "",
                  "| alg | enabler | delay | bystanders | dividend error | order ≤ 2 | recall | precision | interaction mass | true |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        cg = defaultdict(list)
        for r in coal_rows:
            cg[(r["alg"], r["delay"], r["bystanders"], r["enabler_start"])].append(r)
        for (alg, d, b, es), rs in sorted(cg.items(), key=_alg_key):
            lines.append(f"| {alg} | {_slack(es)} | {d} | {b} | "
                         + " | ".join(_fmt([r[f'coal_{k}'] for r in rs]) for k in COALITION) + " |")

    snr_rows = [r for r in rows if "snr_coalition" in r]
    if snr_rows:
        lines += ["", "## MAPPO policy-gradient signal-to-noise (mean over agents in each role)", "",
                  "| enabler | delay | bystanders | coalition SNR | bystander SNR |", "|---|---|---|---|---|"]
        g = defaultdict(list)
        for r in snr_rows:
            g[(r["enabler_start"], r["delay"], r["bystanders"])].append(r)
        for (es, d, b), rs in sorted(g.items()):
            lines.append(f"| {_slack(es)} | {d} | {b} | {_fmt([r['snr_coalition'] for r in rs])} | "
                         f"{_fmt([r['snr_bystanders'] for r in rs]) if b else 'n/a'} |")

    lines += ["", "## Data and ground truth (H0)", "",
              "| cell | test return | realised = planted | A critical | clipped | GT remove-vs-noop L1 (mean / max) | minutes |",
              "|---|---|---|---|---|---|---|"]
    for rec in sorted(records, key=lambda r: (r["cell"].get("enabler_start", 0), r["cell"]["delay"],
                                              r["cell"]["n_bystanders"], r["cell"]["seed"])):
        c, d, h0 = rec["cell"], rec["data"], rec["h0"]
        crit = d.get("test_enabler_critical")
        lines.append(f"| d{c['delay']} b{c['n_bystanders']} {_slack(c.get('enabler_start', 0))} s{c['seed']} | "
                     f"{d['test_return_mean']:.2f} | {d['test_matches_plan']:.2f} | "
                     f"{'n/a' if crit is None else f'{crit:.2f}'} | {d['test_clipped']:.2f} | "
                     f"{h0['remove_vs_noop_l1_mean']:.3f} / {h0['remove_vs_noop_l1_max']:.3f} | "
                     f"{rec['seconds'] / 60:.1f} |")

    lines += ["", "## Final critic loss (mean over seeds)", "",
              "| alg | enabler | delay | bystanders | final loss |", "|---|---|---|---|---|"]
    for (alg, d, b, es), rs in sorted(groups.items(), key=_alg_key):
        lines.append(f"| {alg} | {_slack(es)} | {d} | {b} | {_fmt([r['final_loss'] for r in rs])} |")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(summarise_dir(sys.argv[1]))
