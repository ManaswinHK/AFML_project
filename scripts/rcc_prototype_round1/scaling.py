"""How many intervention episodes does the removal-conditioned critic need?
Varies the number of training episodes and the fraction that carry a removal. Reuses the test-episode
ground truth saved by run_rcc.py. Usage: python scaling.py <workers>"""
import json, os, sys, itertools, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from rcc_lib import *
from run_rcc import fit, predict, coalition_scores, rep, mean
from whoearned.envs.gridworld import PlantedCoalitionGridworld

CONFIGS = [(200, 0.1), (200, 0.25), (200, 0.5), (400, 0.5), (800, 0.5)]
CELLS = [("out_h3", Cell(delay=0, n_bystanders=1, seed=s, layout="overlap")) for s in range(3)] + \
        [("out_main", Cell(delay=25, n_bystanders=2, seed=s)) for s in range(3)] + \
        [("out_h3", Cell(delay=0, n_bystanders=2, seed=s, layout="partial")) for s in range(3)]

def job(args):
    (src, cell), (n_train, p_drop) = args
    saved = json.load(open(os.path.join(os.environ.get("RCC_RESULTS", "."), src, cell.key + ".json")))
    cfg = cell.config(); n = cfg.n_agents; full = (1 << n) - 1
    enablers = sorted({e for ev in cfg.events for e in ev.enablers}); bystanders = list(cfg.bystanders)
    X, Y, info = training_set(cell, p_drop=p_drop, n_episodes=n_train)
    m = fit(X, Y, cell.seed)
    env = PlantedCoalitionGridworld(cfg); env.reset(); s0 = env.get_state()
    q0 = np.stack([remove_from_state(s0, n, [i for i in range(n) if not s >> i & 1]) for s in range(1 << n)])
    v_hat = predict(m, q0)[:, 1]
    d_sp = sparse_dividends(v_hat, seed=cell.seed); v_sp = gt.values_from_dividends(d_sp)
    loo, shap, coal = [], [], []
    for v_true, G in zip(saved["truth"]["open_v"], saved["truth"]["returns"]):
        v_true = np.array(v_true)
        loo.append(rep(G - v_hat[[full ^ (1 << i) for i in range(n)]], gt.leave_one_out(v_true), enablers, bystanders)["l1"])
        shap.append(rep(gt.shapley_values(v_sp), gt.shapley_values(v_true), enablers, bystanders)["l1"])
        coal.append(coalition_scores(d_sp, gt.harsanyi_dividends(v_true)))
    return dict(layout=cell.layout, key=cell.key, n_train=n_train, p_drop=p_drop, interventions=info["interventions"],
                train_return=info["mean_return"], loo_l1=mean(loo), shapley_l1=mean(shap),
                **{k: mean([c[k] for c in coal]) for k in coal[0]})

if __name__ == "__main__":
    jobs = list(itertools.product(CELLS, CONFIGS))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        rows = []
        for r in p.imap_unordered(job, jobs):
            rows.append(r); print(r, flush=True)
    json.dump(rows, open("scaling.json", "w"), indent=1)
    print("\nlayout  n_train  p_drop  interventions  train_return  LOO_L1  Shapley_L1  dividend_error  interaction_mass  recall  precision")
    for lay in ("coalition", "overlap", "partial"):
        for cfgk in CONFIGS:
            sel = [r for r in rows if r["layout"] == lay and (r["n_train"], r["p_drop"]) == cfgk]
            if sel:
                a = lambda k: np.mean([r[k] for r in sel])
                print(f"{lay:9s} {cfgk[0]:5d}   {cfgk[1]:.2f}   {a('interventions'):6.0f}        {a('train_return'):.2f}        {a('loo_l1'):.3f}   {a('shapley_l1'):.3f}       {a('dividend_error'):.3f}           {a('interaction_mass'):.3f}        {a('recall'):.3f}   {a('precision'):.3f}")
