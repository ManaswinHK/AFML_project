"""EXP-A: oracle ablation-TD credit on the main-grid test episodes (both nulls)."""
import json, sys, os, time
import numpy as np
from multiprocessing import Pool
from atd import *
from whoearned.metrics import evaluate_against
from whoearned.ground_truth import normalise_shares

import atd
RES = f'{atd.REPO}/results/mode_p/main'

def run(args):
    d, b, es, seed = args
    cell = Cell(delay=d, b=b, seed=seed, es=es)
    out = f'out/oracle/{cell.key}.json'
    if os.path.exists(out):
        return out
    stored = json.load(open(f'{RES}/{cell.key}.json'))
    cfg = cell.config(); n = cfg.n_agents
    enablers, bystanders = [0], list(cfg.bystanders)
    rows = []
    t0 = time.time()
    for idx in range(cell.n_test):
        f = factual(cell, 'test', idx)
        ev = f.event_log[0]
        row = {'episode': idx, 'return': f.total, 'pull': ev.pull_times.get(0), 'complete': ev.completion_time,
               'payout': ev.payout_time, 'open_loop_loo': stored['truth']['loo'][idx]}
        for null in ('remove', 'noop'):
            W = oracle_ablation_values(cell, 'test', idx, f, null=null)
            k = atd_credit(f.rewards, W)
            tot = k.sum(1)
            loo = f.total - W[:, 0]                       # closed-loop leave-one-out under this null, from t = 0
            rep_open = evaluate_against(tot, np.array(stored['truth']['loo'][idx]), enablers, bystanders).as_dict()
            peak = [int(np.argmax(np.abs(k[i]))) if np.abs(k[i]).max() > 1e-9 else None for i in range(n)]
            row[null] = {'credit': tot.tolist(), 'loo': loo.tolist(), 'identity_err': float(np.abs(tot - loo).max()),
                         'vs_open_loop': rep_open, 'peak_step': peak,
                         'n_nonzero_steps': [int((np.abs(k[i]) > 1e-9).sum()) for i in range(n)],
                         'min_step_credit': float(k.min())}
        rows.append(row)
    rec = {'cell': {'delay': d, 'b': b, 'es': es, 'seed': seed}, 'seconds': time.time() - t0,
           'stored_semantic_gap_noop': stored['semantic_gap']['noop'], 'stored_semantic_gap_baseline': stored['semantic_gap']['baseline'],
           'episodes': rows}
    os.makedirs('out/oracle', exist_ok=True)
    json.dump(rec, open(out, 'w'))
    return out

if __name__ == '__main__':
    grid = [(d, b, es, s) for d in (0, 10, 25, 50) for b in (0, 2, 4) for es in (0, 12) for s in range(5)]
    with Pool(2) as p:
        for i, o in enumerate(p.imap_unordered(run, grid)):
            print(i + 1, len(grid), o, flush=True)
