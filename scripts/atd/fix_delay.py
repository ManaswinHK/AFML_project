import json, sys, numpy as np
from atd import *
from whoearned.metrics import pearson, evaluate_against
res={}
for d in (25,50):
    for seed in range(3):
        cell=Cell(delay=d,b=2,seed=seed); n=5
        facts=[factual(cell,'test',i) for i in range(30)]
        Ws=[oracle_ablation_values(cell,'test',i,f,null='noop') for i,f in enumerate(facts)]
        for until in ('done','payout'):
            X,y,info=ablation_dataset(cell,200,seed=seed,null='noop',until=until)
            pr=fit_critic(X,y,seed=seed)
            kh=[];kt=[];l1=[];mass=[]
            for f,W in zip(facts,Ws):
                What=learned_ablation_values(pr,f,null='noop'); k=atd_credit(f.rewards,What); ko=atd_credit(f.rewards,W)
                kh.append(k);kt.append(ko)
                l1.append(evaluate_against(k.sum(1), f.total-W[:,0],[0],[3,4]).l1)
                for a in range(n):
                    if np.abs(ko[a]).max()>1e-9:
                        tp=int(np.argmax(np.abs(ko[a]))); mass.append(np.abs(k[a,max(tp-1,0):tp+2]).sum()/np.abs(k[a]).sum())
            res[f'd{d}_s{seed}_{until}']=dict(step_r=pearson(np.stack(kh),np.stack(kt)), l1=float(np.mean(l1)), mass=float(np.mean(mass)))
            print(d,seed,until,res[f'd{d}_s{seed}_{until}'],flush=True)
json.dump(res,open('out/fix_delay.json','w'))
