"""Aggregate the new tests and put them next to the team's stored results for the same cells."""
import json, glob, os, sys
import numpy as np
from collections import defaultdict
from rcc_lib import REPO as _REPO
REPO = os.path.join(_REPO, "results", "mode_p")

def ms(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs)), len(xs)) if xs else (None, None, 0)

def fmt(t, d=2):
    return "–" if t[0] is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"

def load(new_dir, old_dir):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(new_dir, "*.json"))):
        r = json.load(open(f)); c = r["cell"]
        old_path = os.path.join(REPO, old_dir, os.path.basename(f))
        old = json.load(open(old_path)) if os.path.exists(old_path) else None
        groups[(c["layout"], c["delay"], c["n_bystanders"], c["enabler_start"])].append((r, old))
    return groups

def table(new_dir, old_dir, title):
    groups = load(new_dir, old_dir)
    print(f"\n## {title}  ({sum(len(v) for v in groups.values())} cells)\n")
    hdr = ["layout", "d", "b", "enabler", "seeds", "uniform L1", "planted-oracle L1", "best stored method (L1)",
           "exact persistent credit L1", "RCC L1", "RCC, no interventions L1", "RCC value r", "RCC step r", "no-int step r",
           "stored best step r"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    allrows = []
    for key in sorted(groups):
        lay, d, b, es = key; cells = groups[key]
        tgt = "loo"
        uni = ms([r["uniform"][tgt]["l1"] for r, _ in cells])
        pl = ms([r["planted_oracle"][tgt]["l1"] for r, _ in cells])
        pt = ms([r["persistent_truth"]["telescoped"]["l1"] for r, _ in cells])
        rcc = ms([r["rcc"]["rcc"]["loo_vs_open"]["l1"] for r, _ in cells])
        noi = ms([r["rcc"]["no_intervention"]["loo_vs_open"]["l1"] for r, _ in cells])
        vr = ms([r["rcc"]["rcc"]["fidelity_value_pearson"] for r, _ in cells])
        sr = ms([r["rcc"]["rcc"]["fidelity_step_pearson"] for r, _ in cells])
        nsr = ms([r["rcc"]["no_intervention"]["fidelity_step_pearson"] for r, _ in cells])
        best, beststep = ("–", None), None
        olds = [o for _, o in cells if o is not None]
        if olds:
            algs = olds[0]["algs"].keys()
            per = {a: ms([o["algs"][a]["summary_loo"]["l1"] for o in olds]) for a in algs if a != "mappo"}
            a = min(per, key=lambda k: per[k][0]); best = (f"{a} {per[a][0]:.2f}", per[a][0])
            steps = [o["algs"][a2]["summary_fidelity"]["step_pearson"] for o in olds for a2 in algs if "summary_fidelity" in o["algs"][a2]
                     and o["algs"][a2]["summary_fidelity"]["step_pearson"] is not None]
            beststep = max(steps) if steps else None
        print(f"| {lay} | {d} | {b} | {'critical' if es else 'slack'} | {len(cells)} | {fmt(uni)} | {fmt(pl)} | {best[0]} | {fmt(pt)} | {fmt(rcc)} | {fmt(noi)} | {fmt(vr,3)} | {fmt(sr,3)} | {fmt(nsr,3)} | {'–' if beststep is None else f'{beststep:.2f} (max)'} |")
        allrows.append(dict(key=key, uni=uni[0], rcc=rcc[0], best=best[1], noi=noi[0], pt=pt[0], sr=sr[0], nsr=nsr[0], vr=vr[0],
                            seeds_rcc_beats_uni=sum(r["rcc"]["rcc"]["loo_vs_open"]["l1"] < r["uniform"]["loo"]["l1"] for r, _ in cells), n=len(cells)))
    return groups, allrows

def extra(groups, title):
    print(f"\n### {title}: other metrics (mean over seeds)\n")
    hdr = ["layout", "d", "b", "enabler", "RCC leakage", "RCC missed", "exact-persistent missed", "stored semantic-gap missed (noop)",
           "closed-vs-open-loop truth L1 (LOO)", "train return with / without interventions"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for key in sorted(groups):
        lay, d, b, es = key; cells = groups[key]
        g = lambda f: fmt(ms([f(r) for r, _ in cells]))
        olds = [o for _, o in cells if o is not None]
        sg = fmt(ms([o["semantic_gap"]["noop"]["missed"] for o in olds])) if olds else "–"
        print(f"| {lay} | {d} | {b} | {'critical' if es else 'slack'} | {g(lambda r: r['rcc']['rcc']['loo_vs_open']['leakage'])} | {g(lambda r: r['rcc']['rcc']['loo_vs_open']['missed'])} | "
              f"{g(lambda r: r['persistent_truth']['telescoped']['missed'])} | {sg} | {g(lambda r: r['persistent_truth']['closed_vs_open_loo']['l1'])} | "
              f"{ms([r['rcc']['rcc']['train']['mean_return'] for r,_ in cells])[0]:.2f} / {ms([r['rcc']['no_intervention']['train']['mean_return'] for r,_ in cells])[0]:.2f} |")

def coalition(groups, title):
    print(f"\n### {title}: coalition level (true interaction mass = 1.0)\n")
    hdr = ["layout", "d", "method", "interaction mass", "dividend error (0–2)", "recall", "precision", "Shapley L1"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for key in sorted(groups):
        lay, d, b, es = key; cells = groups[key]
        olds = [o for _, o in cells if o is not None]
        for a in ("vdn", "qmix", "na2q"):
            if olds and a in olds[0]["algs"] and "summary_coalition" in olds[0]["algs"][a]:
                sc = lambda k: fmt(ms([o["algs"][a]["summary_coalition"][k] for o in olds]), 3)
                sh = fmt(ms([o["algs"][a]["summary_shapley"]["l1"] for o in olds]))
                print(f"| {lay} | {d} | {a} (stored) | {sc('interaction_mass')} | {sc('dividend_error')} | {sc('recall')} | {sc('precision')} | {sh} |")
        if olds and "stas" in olds[0]["algs"]:
            print(f"| {lay} | {d} | stas (stored) | – | – | – | – | {fmt(ms([o['algs']['stas']['summary_shapley']['l1'] for o in olds]))} |")
        print(f"| {lay} | {d} | uniform | – | – | – | – | {fmt(ms([r['uniform']['shapley']['l1'] for r,_ in cells]))} |")
        for name, lab in (("rcc", "RCC"), ("no_intervention", "RCC, no interventions")):
            for var, vlab, shk in (("coalition", "raw Möbius", "shapley_vs_open"), ("coalition_sparse", "sparse", "shapley_sparse_vs_open")):
                sc = lambda k: fmt(ms([r["rcc"][name][var][k] for r, _ in cells]), 3)
                sh = fmt(ms([r["rcc"][name][shk]["l1"] for r, _ in cells]))
                print(f"| {lay} | {d} | {lab}, {vlab} | {sc('interaction_mass')} | {sc('dividend_error')} | {sc('recall')} | {sc('precision')} | {sh} |")

RES = sys.argv[1] if len(sys.argv) > 1 else "."   # folder that contains out_main/ and out_h3/

if __name__ == "__main__":
    g, rows = table(os.path.join(RES, "out_main"), "main", "Main-grid cells (single coalition {A,B,C} + bystanders), target = leave-one-out")
    if g: extra(g, "Main grid")
    if rows:
        print("\nRCC mean L1 below uniform in", sum(r["rcc"] < r["uni"] for r in rows), "of", len(rows), "conditions; seeds:", sum(r["seeds_rcc_beats_uni"] for r in rows), "of", sum(r["n"] for r in rows))
        print("RCC mean L1 below best stored method in", sum(r["best"] is not None and r["rcc"] < r["best"] for r in rows), "of", len(rows))
        print("ranges: RCC L1 %.2f–%.2f | no-int L1 %.2f–%.2f | exact persistent L1 %.2f–%.2f | RCC step r %.3f–%.3f | no-int step r %.3f–%.3f | value r %.3f–%.3f" % (
            min(r["rcc"] for r in rows), max(r["rcc"] for r in rows), min(r["noi"] for r in rows), max(r["noi"] for r in rows),
            min(r["pt"] for r in rows), max(r["pt"] for r in rows), min(r["sr"] for r in rows), max(r["sr"] for r in rows),
            min(r["nsr"] for r in rows), max(r["nsr"] for r in rows), min(r["vr"] for r in rows), max(r["vr"] for r in rows)))
    g, rows = table(os.path.join(RES, "out_h3"), "h3", "H3 cells (overlap, partial synergy), target = leave-one-out")
    if g:
        extra(g, "H3 grid"); coalition(g, "H3 grid")
        print("\nRCC mean L1 below uniform in", sum(r["rcc"] < r["uni"] for r in rows), "of", len(rows), "conditions; seeds:", sum(r["seeds_rcc_beats_uni"] for r in rows), "of", sum(r["n"] for r in rows))
        print("ranges: RCC L1 %.2f–%.2f | no-int L1 %.2f–%.2f | exact persistent L1 %.2f–%.2f | RCC step r %.3f–%.3f | no-int step r %.3f–%.3f" % (
            min(r["rcc"] for r in rows), max(r["rcc"] for r in rows), min(r["noi"] for r in rows), max(r["noi"] for r in rows),
            min(r["pt"] for r in rows), max(r["pt"] for r in rows), min(r["sr"] for r in rows), max(r["sr"] for r in rows),
            min(r["nsr"] for r in rows), max(r["nsr"] for r in rows)))
