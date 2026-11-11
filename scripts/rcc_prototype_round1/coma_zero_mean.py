"""Numerical check of the identity E_pi[ sum_t A_i^COMA ] = 0 for every agent, with a PERFECT critic.
Uses the repo's exact per-step counterfactual truth (replay.step_counterfactuals):
  baseline_effect[t, i] = G_t - sum_a pi_b(a) G_t(i does a at t)   (what COMA's advantage estimates)
  noop_effect[t, i]     = G_t - G_t(i does noop at t)              (what the Q_tot difference estimates)
Prints the mean episode-summed credit per agent with its standard error."""
import sys, numpy as np
from rcc_lib import Cell
from whoearned.replay import step_counterfactuals
rows_b, rows_n = [], []
for seed in (0, 1):
    cell = Cell(delay=0, n_bystanders=2, seed=seed)
    for e in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20):
        st = step_counterfactuals(cell.config(), lambda: cell.team("test", e), gamma=cell.gamma)
        rows_b.append(st.baseline_effect().sum(0)); rows_n.append(st.noop_effect().sum(0))
b, n = np.array(rows_b), np.array(rows_n)
se = lambda x: x.std(0, ddof=1) / np.sqrt(len(x))
print("episodes:", len(b), " agents: A (enabler), B, C (finishers), D, E (bystanders); layout d=0, b=2, slack")
print("COMA-style baseline effect, mean per agent:", np.round(b.mean(0), 3), " SE:", np.round(se(b), 3))
print("   median per agent:", np.round(np.median(b, 0), 3), "  fraction of episodes with negative credit:", np.round((b < -1e-9).mean(0), 2))
print("noop effect, mean per agent:              ", np.round(n.mean(0), 3), " SE:", np.round(se(n), 3))
