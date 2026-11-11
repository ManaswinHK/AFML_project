"""Uniform-credit baseline on the stored main grid (no retraining).
Truth per test episode is stored in each cell JSON; per-episode method L1 is stored too."""
import json, glob, os, sys
import numpy as np
from collections import defaultdict
root = sys.argv[1]
def shares(x):
    x = np.asarray(x, float); s = np.abs(x).sum()
    return None if s < 1e-8 else x / s
rows = defaultdict(list)   # (d,b,es,alg) -> list over seeds of (method_l1, uniform_l1, frac_eps_beating)
for f in sorted(glob.glob(os.path.join(root, 'd*_s*.json'))):
    r = json.load(open(f)); c = r['cell']
    key = (c['delay'], c['n_bystanders'], c['enabler_start'])
    n = len(r['truth']['loo'][0])
    uni = np.full(n, 1.0 / n)
    u_l1 = []
    for g in r['truth']['loo']:
        gs = shares(g)
        u_l1.append(None if gs is None else float(np.abs(uni - gs).sum()))
    for alg, a in r['algs'].items():
        m, u, beat = [], [], []
        for e in a['episodes']:
            ul = u_l1[e['episode']]
            if not e['loo']['valid'] or ul is None: continue
            m.append(e['loo']['l1']); u.append(ul); beat.append(e['loo']['l1'] < ul)
        rows[key + (alg,)].append((np.mean(m), np.mean(u), np.mean(beat)))
print("cond(d,b,es) alg  method_L1  uniform_L1  seeds_beating  eps_beating")
summary = defaultdict(lambda: [0, 0])
for k in sorted(rows):
    v = np.array(rows[k]); d, b, es, alg = k
    sb = int((v[:, 0] < v[:, 1]).sum())
    summary[(alg, b)][0] += int(v[:, 0].mean() < v[:, 1].mean()); summary[(alg, b)][1] += 1
    print(f"d={d:2d} b={b} es={es:2d} {alg:5s} {v[:,0].mean():.3f}  {v[:,1].mean():.3f}  {sb}/{len(v)}  {v[:,2].mean():.2f}")
print("\nconditions (delay x enabler) where method mean L1 < uniform mean L1, by bystanders:")
for (alg, b), (w, t) in sorted(summary.items()):
    print(f"  {alg:5s} b={b}: {w}/{t}")
