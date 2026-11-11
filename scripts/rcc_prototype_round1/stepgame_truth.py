"""Coalition-level 'level 2': the TRUE per-step mixer game (what a perfect Q_tot would give).

g_t(S) = G_t(agents in S take their factual action at step t, the rest take noop at step t; the scripted
team then continues, closed loop, common random numbers) - G_t(everyone noops at step t),
with G_t the discounted return from t. Its exact Harsanyi dividends are summed over the episode, exactly as
the repo does for learned mixers (docs/step7_ports.md, "Mixer-game coalition read-out").
Question: is the per-step game additive by construction, as docs/h3_results.md argues?
"""
import sys, json, numpy as np
from rcc_lib import *
from whoearned.envs.gridworld import PlantedCoalitionGridworld, NOOP
from whoearned.replay import ground_truth_open_loop
from whoearned.metrics import dividend_error, coalition_recovery, identified_coalitions

def run_override(cfg, team, t0, keep_mask):
    env = PlantedCoalitionGridworld(cfg); env.reset(); team.reset(env)
    rewards = np.zeros(cfg.horizon)
    for t in range(cfg.horizon):
        a = list(team.act(env))
        if t == t0:
            a = [a[i] if keep_mask >> i & 1 else NOOP for i in range(cfg.n_agents)]
        _, rewards[t], _, _ = env.step(a)
    return rewards

def step_game(cell, index, gamma=0.99):
    cfg = cell.config(); n, T = cfg.n_agents, cfg.horizon
    _, rewards, actions = run(cfg, make_team(cell, "test", index))
    div_sum = np.zeros(1 << n)
    for t in range(T):
        g = np.zeros(1 << n)
        for S in range(1 << n):
            g[S] = to_go(run_override(cfg, make_team(cell, "test", index), t, S), gamma)[t]
        g -= g[0]
        div_sum += gt.harsanyi_dividends(g)
    return div_sum, actions

if __name__ == "__main__":
    layout, d, n_eps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = []
    for seed in range(int(sys.argv[4]) if len(sys.argv) > 4 else 1):
        cell = Cell(delay=d, n_bystanders=(1 if layout == "overlap" else 2), seed=seed, layout=layout)
        cfg = cell.config(); n = cfg.n_agents
        for e in range(n_eps):
            dsum, actions = step_game(cell, e)
            truth = ground_truth_open_loop(cfg, actions)
            est = {frozenset(gt.agents_of(s)): float(dsum[s]) for s in range(1, 1 << n)}
            tru = {frozenset(gt.agents_of(s)): float(truth.dividends[s]) for s in range(1, 1 << n) if abs(truth.dividends[s]) > 1e-9}
            tot = sum(abs(x) for x in est.values())
            rec = coalition_recovery(identified_coalitions(est, 0.1), list(tru))
            top = sorted(est.items(), key=lambda kv: -abs(kv[1]))[:5]
            out.append(dict(seed=seed, ep=e, interaction_mass=sum(abs(x) for g_, x in est.items() if len(g_) >= 2) / tot if tot else None,
                            dividend_error=dividend_error(est, tru), recall=rec.recall, precision=rec.precision,
                            top=[("".join("ABCDEFG"[i] for i in sorted(g_)), round(x, 3)) for g_, x in top],
                            truth={"".join("ABCDEFG"[i] for i in sorted(g_)): x for g_, x in tru.items()}))
            print(out[-1], flush=True)
    json.dump(out, open(f"stepgame_{layout}_d{d}.json", "w"), indent=1)
    for k in ("interaction_mass", "dividend_error", "recall", "precision"):
        print(k, round(float(np.mean([o[k] for o in out if o[k] is not None])), 4))
