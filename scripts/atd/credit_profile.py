import numpy as np
from atd import *
from whoearned.replay import step_counterfactuals
cell = Cell(delay=25, b=2, seed=0)
f = factual(cell,'test',0)
ev=f.event_log[0]; print('pull',ev.pull_times,'complete',ev.completion_time,'payout',ev.payout_time)
st = step_counterfactuals(cell.config(), lambda: cell.team('test',0), gamma=0.99)
one = st.noop_effect().T      # (n, T)
W = oracle_ablation_values(cell,'test',0,f,null='noop'); k = atd_credit(f.rewards, W)
names='ABCDE'
for i in range(5):
    nz=np.flatnonzero(np.abs(one[i])>1e-6)
    print(names[i],'one-step: total',round(one[i].sum(),3),'steps',[(int(t),round(float(one[i,t]),3)) for t in nz][:12],
          '| ATD: total',k[i].sum(),'steps',[(int(t),float(k[i,t])) for t in np.flatnonzero(np.abs(k[i])>1e-9)])
