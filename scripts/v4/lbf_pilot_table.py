"""Summarise every LBF design pilot (seeds 0-4) into docs/v4/lbf_design.md's table (exploratory, not confirmatory)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# directories written before the state-encoding fix (decision E6): relative food levels
ENC1 = {"lbf_pilot", "lbf_pilot_big", "lbf_pilot_eps0", "lbf_pilot_eps005", "lbf_pilot_h20rm", "lbf_pilot_h20rm_eps0"}


def rows():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "v4", "lbf_*"))):
        name = os.path.basename(d)
        if name.startswith("lbf_confirm"):
            continue
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
        for variant in sorted({r["variant"] for r in recs}):
            rs = [r for r in recs if r["variant"] == variant]
            cfg = {"null": "noop", "eps": 0.2, "n_train": 200, "n_int": [0, 200, 800], "steps": 3000, **rs[0].get("cfg", {})}
            m = lambda alg, key: (np.mean([r["methods"][alg][key] for r in rs if r["methods"].get(alg, {}).get(key) is not None])
                                  if any(r["methods"].get(alg, {}).get(key) is not None for r in rs) else float("nan"))
            loo = np.concatenate([np.array(r["truth"]["loo"]) for r in rs])
            out.append(dict(dir=name, enc="relative" if name in ENC1 else "absolute", variant=variant, null=cfg["null"],
                            eps=cfg["eps"], n_train=cfg["n_train"], stops=cfg["n_int"][1], steps=cfg["steps"],
                            seeds=",".join(str(r["seed"]) for r in rs), zero=float((np.abs(loo) < 1e-9).mean()),
                            set=m("set", "l1"), mlp=m("mlp", "l1"), uni=m("uniform", "l1"),
                            floor=m("floor_expected", "l1"), set_r=m("set", "step_r"), set_shap=m("set", "shap"),
                            uni_shap=m("uniform", "shap")))
    return out


def table():
    L = ["| dir | levels in state | variant | null | eps | train eps / stops / steps | seeds | true LOO = 0 | set L1 | MLP L1 | equal split L1 | luck floor | set step r | set Shapley L1 | equal-split Shapley L1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x != x else f"{x:.3f}"
    for r in rows():
        L.append(f"| {r['dir']} | {r['enc']} | {r['variant']} | {r['null']} | {r['eps']} | {r['n_train']} / {r['stops']} / "
                 f"{r['steps']} | {r['seeds']} | {r['zero']:.2f} | {f(r['set'])} | {f(r['mlp'])} | {f(r['uni'])} | "
                 f"{f(r['floor'])} | {f(r['set_r'])} | {f(r['set_shap'])} | {f(r['uni_shap'])} |")
    return "\n".join(L)


if __name__ == "__main__":
    print(table())
