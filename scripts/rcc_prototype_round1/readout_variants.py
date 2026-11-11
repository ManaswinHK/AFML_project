"""Post-hoc: two episode-level read-outs of the same learned critic, from the saved coalition values.
  pathwise : G(tau) - v_hat(N minus i)        (what run_rcc.py scored; carries the episode's return surprise)
  expected : v_hat(N) - v_hat(N minus i)      (no surprise term)
Scored against the repo's open-loop truth and against the closed-loop truth, with the repo's metrics."""
import json, glob, os, sys
import numpy as np
from collections import defaultdict
from rcc_lib import Cell, gt
from whoearned.metrics import evaluate_against

def rows_for(path):
    r = json.load(open(path)); c = {k: v for k, v in r["cell"].items() if k != "key"}
    cell = Cell(**c); cfg = cell.config(); n = cfg.n_agents; full = (1 << n) - 1
    en = sorted({e for ev in cfg.events for e in ev.enablers}); by = list(cfg.bystanders)
    out = defaultdict(list)
    for name in ("rcc", "no_intervention"):
        v = np.array(r["rcc"][name]["v_hat"]); wo = v[[full ^ (1 << i) for i in range(n)]]
        for vo, vc, G in zip(r["truth"]["open_v"], r["truth"]["closed_v"], r["truth"]["returns"]):
            vo, vc = np.array(vo), np.array(vc)
            for ro, cred in (("pathwise", G - wo), ("expected", v[full] - wo)):
                for tn, tv in (("open", gt.leave_one_out(vo)), ("closed", gt.leave_one_out(vc))):
                    e = evaluate_against(cred, tv, en, by)
                    if e.valid:
                        out[(name, ro, tn, "l1")].append(e.l1); out[(name, ro, tn, "leak")].append(e.leakage)
                        if e.missed is not None: out[(name, ro, tn, "missed")].append(e.missed)
    return (cell.layout, cell.delay, cell.n_bystanders, cell.enabler_start), {k: float(np.mean(v)) for k, v in out.items()}

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    for d in (os.path.join(base, "out_main"), os.path.join(base, "out_h3")):
        groups = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            k, v = rows_for(f); groups[k].append(v)
        print(f"\n{d}: learned critic with interventions; mean over seeds")
        print("layout d b es seeds | pathwise vs open: L1 leak missed | expected vs open: L1 leak missed | expected vs closed: L1 leak | no-int expected vs open L1")
        for k in sorted(groups):
            g = groups[k]; a = lambda key: np.mean([x[key] for x in g if key in x])
            print(f"{k[0]:9s} {k[1]:2d} {k[2]} {k[3]:2d} {len(g)} | {a(('rcc','pathwise','open','l1')):.3f} {a(('rcc','pathwise','open','leak')):.3f} {a(('rcc','pathwise','open','missed')):.3f} | "
                  f"{a(('rcc','expected','open','l1')):.3f} {a(('rcc','expected','open','leak')):.3f} {a(('rcc','expected','open','missed')):.3f} | "
                  f"{a(('rcc','expected','closed','l1')):.3f} {a(('rcc','expected','closed','leak')):.3f} | {a(('no_intervention','expected','open','l1')):.3f}")
