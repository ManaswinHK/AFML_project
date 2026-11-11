import numpy as np, json
from toy_headroom import simulate, summarise
rounds, seeds = 30000, 10
for K in (16,):
    for sig in ("team", "loo"):
        res = [summarise(simulate(K, 2, sig, rounds, 0.02, s), K) for s in range(seeds)]
        fin = np.array([r[0] for r in res]); hits = [r[1] for r in res if r[1] is not None]
        print(K, sig, "final %.3f +- %.3f"%(fin.mean(), fin.std()), "solved %d/%d"%(len(hits), seeds), "median", int(np.median(hits)) if len(hits) > seeds//2 else None)
