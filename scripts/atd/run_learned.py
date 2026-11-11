"""EXP-B/C: learned ablation critic (ATD), plus a presence-only table baseline, scored on the stored test episodes."""
import json, os, sys, time, itertools
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import Lasso
from atd import *
from whoearned import ground_truth as gt
from whoearned.ground_truth import mask_of, agents_of
from whoearned.metrics import (evaluate_against, pearson, dividend_error, identified_coalitions,
                               coalition_recovery, true_dividends)
from whoearned.replay import ground_truth_open_loop, ground_truth_closed_loop, replay

import atd
REPO = f'{atd.REPO}/results/mode_p'

def coalition_scores(div, truth):
    est = {frozenset(agents_of(s)): float(div[s]) for s in range(1, len(div))}
    tru = true_dividends(truth)
    rec = coalition_recovery(identified_coalitions(est, 0.1), list(tru))
    tot = sum(abs(v) for v in est.values())
    return {'dividend_error': dividend_error(est, tru), 'recall': rec.recall, 'precision': rec.precision,
            'interaction_mass': sum(abs(v) for g, v in est.items() if len(g) >= 2) / tot if tot else None}

def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None

def indiv(credit, target, enablers, bystanders):
    r = evaluate_against(np.asarray(credit, float), np.asarray(target, float), enablers, bystanders).as_dict()
    return r if r['valid'] else {'valid': False, 'l1': None, 'leakage': None, 'missed': None, 'spearman': None}

def summarise(rows):
    return {k: mean(rows, k) for k in ('l1', 'leakage', 'missed')} | {'n_valid': sum(r['valid'] for r in rows)}

def run(args):
    layout, d, b, es, seed = args
    cell = Cell(delay=d, b=b, seed=seed, es=es, layout=layout)
    out = f'out/learned/{cell.key}.json'
    if os.path.exists(out):
        return out
    t0 = time.time()
    cfg = cell.config(); n = cfg.n_agents; T = cfg.horizon
    enablers = sorted({e for ev in cfg.events for e in ev.enablers}); bystanders = list(cfg.bystanders)
    all_masks = list(range(1 << n))
    removed_of = lambda m: tuple(i for i in range(n) if not m >> i & 1)
    sub = 'main' if layout == 'coalition' else 'h3'
    stored, stored_h5 = None, None
    for folder in ('main', 'h5', 'h3'):
        p = f'{REPO}/{folder}/{cell.key}.json'
        if os.path.exists(p):
            stored = json.load(open(p)); break
    if os.path.exists(f'{REPO}/h5/{cell.key}.json'):
        stored_h5 = json.load(open(f'{REPO}/h5/{cell.key}.json'))
    facts = [factual(cell, 'test', i) for i in range(cell.n_test)]
    rec = {'cell': dict(layout=layout, delay=d, b=b, es=es, seed=seed), 'runs': {}}
    coalition_level = es == 0 and (layout != 'coalition' or (b == 2 and d in (0, 50)))

    # ---- truths ------------------------------------------------------------------------------
    oracle = {}
    for null in ('noop', 'remove'):
        Ws = [oracle_ablation_values(cell, 'test', i, f, null=null) for i, f in enumerate(facts)]
        oracle[null] = {'W': Ws, 'kappa': [atd_credit(f.rewards, W) for f, W in zip(facts, Ws)],
                        'loo': [f.total - W[:, 0] for f, W in zip(facts, Ws)]}
    open_loo = [np.array(x) for x in stored['truth']['loo']] if stored else None
    uni = np.ones(n)
    rec['uniform'] = {'vs_closed_noop': summarise([indiv(uni, l, enablers, bystanders) for l in oracle['noop']['loo']]),
                      'vs_closed_remove': summarise([indiv(uni, l, enablers, bystanders) for l in oracle['remove']['loo']]),
                      'vs_open': summarise([indiv(uni, l, enablers, bystanders) for l in open_loo]) if open_loo else None}
    if coalition_level:
        truth_open = [ground_truth_open_loop(cfg, f.actions) for f in facts]
        truth_closed = [ground_truth_closed_loop(cfg, (lambda i=i: cell.team('test', i))) for i in range(cell.n_test)]
        rec['uniform']['shapley_open'] = summarise([indiv(uni, t.shapley, enablers, bystanders) for t in truth_open])
        rec['uniform']['shapley_closed'] = summarise([indiv(uni, t.shapley, enablers, bystanders) for t in truth_closed])

    # ---- learned ablation critic ---------------------------------------------------------------
    plan = [('noop', 0), ('noop', 200), ('noop', 800), ('remove', 200)] + ([('remove', 800)] if coalition_level else [])
    for null, n_abl in plan:
        ta = time.time()
        X, y, info = ablation_dataset(cell, n_abl, seed=seed, null=null)
        predict = fit_critic(X, y, seed=seed)
        rows, rows_open, rows_d, kap_hat, kap_true, mass_near, top3 = [], [], [], [], [], [], []
        for i, f in enumerate(facts):
            W = learned_ablation_values(predict, f, null=null)
            k = atd_credit(f.rewards, W)
            credit = k.sum(1)                                       # = R - W_hat[:, 0]
            top3.append(float(set(np.argsort(-credit)[:3].tolist()) == {0, 1, 2}))
            v_full = float(predict(aug(f.states[0], n)[None])[0])
            rows.append(indiv(credit, oracle[null]['loo'][i], enablers, bystanders))
            rows_d.append(indiv(v_full - W[:, 0], oracle[null]['loo'][i], enablers, bystanders))
            if open_loo:
                rows_open.append(indiv(credit, open_loo[i], enablers, bystanders))
            kt = oracle[null]['kappa'][i]
            kap_hat.append(k); kap_true.append(kt)
            for a in range(n):
                if np.abs(kt[a]).max() > 1e-9:
                    tp = int(np.argmax(np.abs(kt[a])))
                    tot = np.abs(k[a]).sum()
                    mass_near.append(float(np.abs(k[a, max(tp - 1, 0):tp + 2]).sum() / tot) if tot > 0 else 0.0)
        run_rec = {'data': info, 'n_samples': int(len(y)), 'final_loss': predict.final_loss, 'seconds': time.time() - ta,
                   'vs_closed': summarise(rows), 'vs_closed_dform': summarise(rows_d),
                   'vs_open': summarise(rows_open) if rows_open else None,
                   'step_pearson': pearson(np.stack(kap_hat), np.stack(kap_true)),
                   'top3_is_coalition': float(np.mean(top3)) if layout == 'coalition' else None,
                   'mass_within_1_step_of_true_event': float(np.mean(mass_near)) if mass_near else None}
        if coalition_level and null == 'remove':
            s0 = facts[0].states[0]
            v = np.array([predict(aug(edit_state(s0, n, removed_of(m)), n, removed_of(m))[None])[0] for m in all_masks])
            v[0] = 0.0
            div = gt.harsanyi_dividends(v); phi = gt.shapley_values(v)
            for name, truths in (('open', truth_open), ('closed', truth_closed)):
                run_rec[f'coalition_{name}'] = {k: mean([coalition_scores(div, t) for t in truths], k)
                                                for k in ('dividend_error', 'recall', 'precision', 'interaction_mass')}
                run_rec[f'shapley_{name}'] = summarise([indiv(phi, t.shapley, enablers, bystanders) for t in truths])
            run_rec['top_dividends'] = sorted(((list(agents_of(s)), round(float(div[s]), 2)) for s in range(1, len(div))),
                                              key=lambda x: -abs(x[1]))[:6]
        rec['runs'][f'{null}_{n_abl}'] = run_rec

    # ---- state-conditioned low-order dividend critic (no-op null) ----------------------------------
    for n_abl, sp in ((200, 0.0), (800, 0.0), (200, 0.005)):
        ta = time.time()
        X, y, info = ablation_dataset(cell, n_abl, seed=seed, null='noop')
        dv = fit_mobius_critic(X, y, n, k=3, seed=seed, sparsity=sp)
        rows, rows_open, kap_hat, kap_true, mass_near, top3 = [], [], [], [], [], []
        for i, f in enumerate(facts):
            W = mobius_ablation_values(dv, f); k = atd_credit(f.rewards, W); credit = k.sum(1)
            top3.append(float(set(np.argsort(-credit)[:3].tolist()) == {0, 1, 2}))
            rows.append(indiv(credit, oracle['noop']['loo'][i], enablers, bystanders))
            if open_loo:
                rows_open.append(indiv(credit, open_loo[i], enablers, bystanders))
            kt = oracle['noop']['kappa'][i]; kap_hat.append(k); kap_true.append(kt)
            for a in range(n):
                if np.abs(kt[a]).max() > 1e-9:
                    tp = int(np.argmax(np.abs(kt[a]))); tot = np.abs(k[a]).sum()
                    mass_near.append(float(np.abs(k[a, max(tp - 1, 0):tp + 2]).sum() / tot) if tot > 0 else 0.0)
        run_rec = {'data': info, 'final_loss': dv.final_loss, 'seconds': time.time() - ta, 'vs_closed': summarise(rows),
                   'vs_open': summarise(rows_open) if rows_open else None,
                   'step_pearson': pearson(np.stack(kap_hat), np.stack(kap_true)),
                   'top3_is_coalition': float(np.mean(top3)) if layout == 'coalition' else None,
                   'mass_within_1_step_of_true_event': float(np.mean(mass_near)) if mass_near else None}
        if coalition_level:
            g0 = dv(facts[0].states[:1])[0]
            div = np.zeros(1 << n); phi = np.zeros(n)
            for c, val in zip(dv.subsets, g0):
                div[mask_of(c)] = val; phi[list(c)] += val / len(c)
            for name, truths in (('open', truth_open), ('closed', truth_closed)):
                run_rec[f'coalition_{name}'] = {kk: mean([coalition_scores(div, t) for t in truths], kk)
                                                for kk in ('dividend_error', 'recall', 'precision', 'interaction_mass')}
                run_rec[f'shapley_{name}'] = summarise([indiv(phi, t.shapley, enablers, bystanders) for t in truths])
            run_rec['top_dividends'] = sorted(((list(c), round(float(val), 2)) for c, val in zip(dv.subsets, g0)), key=lambda x: -abs(x[1]))[:6]
        rec['runs'][f'mob3{"s" if sp else ""}_noop_{n_abl}'] = run_rec

    # ---- presence-only table baseline on the repo's own "mix" data (same data ECE-k_mix saw) -------
    if coalition_level:
        rng = np.random.default_rng(10_000 + seed)
        P, R = [], []
        train_actions = [factual(cell, 'train', i, keep_snaps=False) for i in range(cell.n_train)]
        for f in train_actions:
            P.append((1 << n) - 1); R.append(f.total)
        for f in train_actions:
            kept = mask_of(np.flatnonzero(rng.random(n) < 0.75))
            P.append(kept); R.append(replay(cfg, f.actions, present=kept).total)
        for k in (2, 3):
            subsets = [c for o in range(1, k + 1) for c in itertools.combinations(range(n), o)]
            Xd = np.array([[float(all(m >> i & 1 for i in c)) for c in subsets] for m in P])
            beta = Lasso(alpha=0.025, fit_intercept=False, max_iter=50_000).fit(Xd, np.array(R)).coef_
            div = np.zeros(1 << n)
            for c, bta in zip(subsets, beta):
                div[mask_of(c)] = bta
            phi = np.zeros(n)
            for c, bta in zip(subsets, beta):
                phi[list(c)] += bta / len(c)
            tr = {}
            for name, truths in (('open', truth_open), ('closed', truth_closed)):
                tr[f'coalition_{name}'] = {kk: mean([coalition_scores(div, t) for t in truths], kk)
                                           for kk in ('dividend_error', 'recall', 'precision', 'interaction_mass')}
                tr[f'shapley_{name}'] = summarise([indiv(phi, t.shapley, enablers, bystanders) for t in truths])
            tr['top_dividends'] = sorted(((list(c), round(float(bta), 2)) for c, bta in zip(subsets, beta)), key=lambda x: -abs(x[1]))[:6]
            tr['n_distinct_masks'] = len(set(P))
            rec['runs'][f'table{k}_mix'] = tr
        if stored_h5:
            rec['stored'] = {a: {'summary': stored_h5['algs'][a]['summary'], 'coalition': stored_h5['algs'][a].get('summary_coalition')}
                             for a in stored_h5['algs']}
    if stored and layout == 'coalition' and 'qmix' in stored['algs']:
        rec['stored_main'] = {a: {'summary': stored['algs'][a]['summary'],
                                  'step_pearson': stored['algs'][a].get('summary_fidelity', {}).get('step_pearson')}
                              for a in stored['algs']}
    rec['seconds'] = time.time() - t0
    os.makedirs('out/learned', exist_ok=True)
    json.dump(rec, open(out, 'w'))
    return out

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'smoke':
        print(json.dumps(json.load(open(run(('coalition', 0, 2, 0, 0)))), indent=1)[:6000]); sys.exit()
    grid = []
    if which in ('h5', 'all'):
        grid += [(lay, d, b, 0, s) for lay, b in (('coalition', 2), ('overlap', 1), ('partial', 2)) for d in (0, 50) for s in range(5)]
    if which in ('main', 'all'):
        grid += [('coalition', d, b, es, s) for d in (0, 25, 50) for b in (2, 4) for es in (0, 12) for s in range(3)]
    grid = list(dict.fromkeys(grid))
    with Pool(2) as p:
        for i, o in enumerate(p.imap_unordered(run, grid)):
            print(i + 1, len(grid), o, flush=True)
