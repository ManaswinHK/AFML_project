"""Is the global state Markov in the partial-synergy layout under delay?
The state shows 'payout pending' and its countdown but not the pending AMOUNT, which in a partial event
is 5 or 10. Test: train the same critic with and without the amount appended to the state."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import rcc_lib
from rcc_lib import *
from run_rcc import fit, predict
from whoearned.metrics import pearson

def one(cell, aug):
    rcc_lib.AUG_PENDING_AMOUNT = aug
    cfg = cell.config(); n, T = cfg.n_agents, cfg.horizon
    S, Vm, R = [], [], []
    for e in range(cell.n_test):
        V, states, rewards, _ = true_persistent_values(cell, e)
        S.append(states); Vm.append(V); R.append(rewards)
    S, Vm = np.stack(S), np.stack(Vm)
    out = {}
    for name, p in (("rcc", 0.5), ("no_int", 0.0)):
        X, Y, _ = training_set(cell, p_drop=p)
        m = fit(X, Y, cell.seed)
        # full-team value fit on test episodes (what every centralised critic needs)
        vfull = predict(m, S[:, :T].reshape(-1, S.shape[-1]))[:, 0].reshape(len(S), T)
        gfull = np.stack([to_go(r, cell.gamma)[:T] for r in R])
        Vhat = np.zeros_like(Vm)
        for i in range(n):
            Vhat[:, :T, i] = predict(m, remove_from_state(S[:, :T], n, (i,)).reshape(-1, S.shape[-1]))[:, 0].reshape(len(S), T)
        c_hat = np.stack([persistent_credit(Vhat[e], R[e], cell.gamma) for e in range(len(S))])
        c_true = np.stack([persistent_credit(Vm[e], R[e], cell.gamma) for e in range(len(S))])
        out[name] = dict(full_value_rmse=float(np.sqrt(((vfull - gfull) ** 2).mean())), value_r=pearson(Vhat[:, :T], Vm[:, :T]),
                         step_r=pearson(c_hat, c_true))
    return out

if __name__ == "__main__":
    rows = []
    for d in (0, 25):
        for seed in range(3):
            cell = Cell(delay=d, n_bystanders=2, seed=seed, layout="partial")
            for aug in (False, True):
                r = one(cell, aug); rows.append(dict(d=d, seed=seed, aug=aug, **{f"{k}_{m}": v for k, vv in r.items() for m, v in vv.items()}))
                print(rows[-1], flush=True)
    json.dump(rows, open("aug_test.json", "w"), indent=1)
    for d in (0, 25):
        for aug in (False, True):
            sel = [r for r in rows if r["d"] == d and r["aug"] == aug]
            print(f"partial d={d} amount_in_state={aug}: " + ", ".join(f"{k}={np.mean([r[k] for r in sel]):.3f}" for k in sel[0] if k not in ("d", "seed", "aug")))
