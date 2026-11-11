"""Downstream check from the repo's stored results: keep the 3 highest-credit agents. Is the kept team the planted coalition?
If yes the pruned team still earns 10; if not it earns 0 (coalition layout, all-or-nothing)."""
import json, glob, collections, numpy as np
import atd
R=f'{atd.REPO}/results/mode_p'
acc=collections.defaultdict(list)
for folder in ('main','h5','h5_ref'):
    for p in sorted(glob.glob(f'{R}/{folder}/*.json')):
        r=json.load(open(p)); c=r['cell']
        if c.get('layout','coalition')!='coalition' or c['n_bystanders']==0: continue
        for a,v in r['algs'].items():
            for e in v['episodes']:
                sh=e['shares']
                if sh is None: continue
                acc[(a,c['n_bystanders'])].append(float(set(np.argsort(-np.array(sh))[:3].tolist())=={0,1,2}))
print('method  b  kept-team-is-coalition  n_episodes')
for (a,b),v in sorted(acc.items()): print(f'{a:9s} {b}  {np.mean(v):.3f}  {len(v)}')
