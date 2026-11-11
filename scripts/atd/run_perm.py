"""EXP-E: roles are shuffled every episode (which agent index is the enabler / a finisher / a bystander).
A presence-only table (what ECE-k reduces to on fixed layouts) cannot know who matters in a given episode;
a state-conditioned ablation critic can read it off the state."""
import json, sys, os, time, itertools
import numpy as np
from multiprocessing import Pool
from atd import *
from whoearned.envs.gridworld import EventSpec, GridConfig
from whoearned.metrics import evaluate_against, pearson
from sklearn.linear_model import LinearRegression

class EpCell:
    def __init__(self, cfg, base, eps=0.2): self.cfg, self.base, self.eps = cfg, base, eps
    def config(self): return self.cfg
    def team(self, split, index): return ScriptedTeam(self.cfg, seed=self.base + index, epsilon=self.eps)

def perm_config(b, delay, horizon, perm):
    base = coalition_layout(b, delay=delay, horizon=horizon); n = base.n_agents; ev = base.events[0]
    start = [None] * n
    for role in range(n): start[perm[role]] = base.start_cells[role]
    event = EventSpec(enablers=(int(perm[0]),), finishers=tuple(sorted((int(perm[1]), int(perm[2])))), reward=10.0,
                      lever_cells={int(perm[0]): ev.lever_cells[0]},
                      goal_cells={int(perm[1]): ev.goal_cells[1], int(perm[2]): ev.goal_cells[2]})
    return GridConfig(n_agents=n, height=5, width=10, horizon=horizon, delay=delay, start_cells=tuple(start), events=(event,))

def episode(seed, split, idx, b, delay, horizon):
    rng = np.random.default_rng([seed, 0 if split == 'train' else 1, idx])
    perm = rng.permutation(3 + b)
    cell = EpCell(perm_config(b, delay, horizon, perm), seed * 100_000 + (0 if split == 'train' else 50_000))
    return cell, perm

def run(args):
    seed, b, delay = args
    horizon = 60; n = 3 + b; n_train, n_test = 200, 30
    out = f'out/perm/d{delay}_b{b}_s{seed}.json'
    if os.path.exists(out): return out
    rng = np.random.default_rng([seed, 99])
    train = []
    for idx in range(n_train):
        cell, perm = episode(seed, 'train', idx, b, delay, horizon)
        train.append((cell, factual(cell, 'train', idx)))
    Xf = np.concatenate([np.stack([aug(s, n) for s in f.states[:-1]]) for _, f in train]); yf = np.concatenate([f.to_go()[:-1] for _, f in train])
    rec = {'cell': dict(seed=seed, b=b, delay=delay), 'runs': {}}
    tests = []
    for idx in range(n_test):
        cell, perm = episode(seed, 'test', idx, b, delay, horizon)
        f = factual(cell, 'test', idx)
        W = oracle_ablation_values(cell, 'test', idx, f, null='noop')
        tests.append((cell, perm, f, W))
    def score(credits):
        rows = []
        for (cell, perm, f, W), c in zip(tests, credits):
            loo = f.total - W[:, 0]
            byst = [int(perm[r]) for r in range(3, n)]
            r = evaluate_against(np.asarray(c, float), loo, [int(perm[0])], byst).as_dict()
            r['top3'] = float(set(np.argsort(-np.asarray(c))[:3].tolist()) == {int(perm[0]), int(perm[1]), int(perm[2])})
            rows.append(r)
        v = [r for r in rows if r['valid']]
        return {k: float(np.mean([r[k] for r in v if r[k] is not None])) for k in ('l1', 'leakage', 'missed', 'top3')} | {'n_valid': len(v)}
    rec['uniform'] = score([np.ones(n)] * n_test)
    for n_abl in (200, 800):
        X, y, P, R = [Xf], [yf], [], []
        for _ in range(n_abl):
            idx = int(rng.integers(n_train)); cell, f = train[idx]
            t0 = int(rng.integers(0, max(f.t_done, 1)))
            removed = (int(rng.integers(n)),) if rng.random() < 0.5 else tuple(int(i) for i in np.flatnonzero(rng.random(n) < rng.uniform(0.1, 0.9))) or (int(rng.integers(n)),)
            g, states, rewards = continue_ablated(cell, 'train', idx, f.snaps[t0], removed, record=True, null='noop')
            X.append(states); y.append(np.cumsum(rewards[::-1])[::-1])
            # the table gets the same number of interventions, all at t = 0 (its best case)
            g0 = continue_ablated(cell, 'train', idx, f.snaps[0], removed, null='noop')
            m = np.ones(n); m[list(removed)] = 0; P.append(m); R.append(g0)
        predict = fit_critic(np.concatenate(X), np.concatenate(y), seed=seed)
        credits, kh, kt = [], [], []
        for cell, perm, f, W in tests:
            What = learned_ablation_values(predict, f, null='noop'); k = atd_credit(f.rewards, What)
            credits.append(k.sum(1)); kh.append(k); kt.append(atd_credit(f.rewards, W))
        rec['runs'][f'atd_{n_abl}'] = score(credits) | {'step_pearson': pearson(np.stack(kh), np.stack(kt))}
        # presence-only table: full-team episodes (mask all ones) + the interventions; order <= 3 monomials
        subsets = [c for o in range(1, 4) for c in itertools.combinations(range(n), o)]
        Pm = np.concatenate([np.ones((n_train, n)), np.array(P)]); Rv = np.concatenate([[f.total for _, f in train], R])
        feats = lambda M: np.array([[float(all(m[i] for i in c)) for c in subsets] for m in M])
        lr = LinearRegression(fit_intercept=False).fit(feats(Pm), Rv)
        loo_masks = np.ones((n, n)) - np.eye(n)
        vhat = lr.predict(feats(loo_masks))
        rec['runs'][f'table3_{n_abl}'] = score([f.total - vhat for _, _, f, _ in tests])
    os.makedirs('out/perm', exist_ok=True); json.dump(rec, open(out, 'w'))
    return out

if __name__ == '__main__':
    grid = [(s, b, 25) for b in (2, 4) for s in range(5)]
    with Pool(2) as p:
        for o in p.imap_unordered(run, grid): print(o, flush=True)
