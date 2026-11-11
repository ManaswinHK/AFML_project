import json, glob, collections, numpy as np
def m(v):
    v=[x for x in v if x is not None]; return (round(float(np.mean(v)),3), len(v)) if v else None
L=[json.load(open(p)) for p in sorted(glob.glob('out/learned/*.json'))]
print('cells', len(L))
# ---- A. individual credit, coalition layout (main-grid style), grouped by b ----
print('\n== individual LOO credit (coalition layout), mean over cells; L1 vs closed-loop truth of the same null / vs repo open-loop truth ==')
for b in (2,4):
    C=[r for r in L if r['cell']['layout']=='coalition' and r['cell']['b']==b]
    print(f'b={b}: {len(C)} cells; uniform L1 closed', m([r['uniform']['vs_closed_noop']['l1'] for r in C]), 'open', m([r['uniform']['vs_open']['l1'] for r in C]))
    for run in ('noop_0','noop_200','noop_800','remove_200','mob3_noop_200','mob3_noop_800','mob3s_noop_200'):
        R=[r['runs'][run] for r in C if run in r['runs']]
        print(f'  {run:15s} L1c', m([x['vs_closed']['l1'] for x in R]), 'L1open', m([x['vs_open']['l1'] for x in R]), 'leak', m([x['vs_closed']['leakage'] for x in R]),
              'missed', m([x['vs_closed']['missed'] for x in R]), 'step r', m([x['step_pearson'] for x in R]), 'top3', m([x['top3_is_coalition'] for x in R]),
              'mass±1', m([x['mass_within_1_step_of_true_event'] for x in R]))
    for a in ('vdn','qmix','coma','mappo'):
        S=[r['stored_main'][a] for r in C if 'stored_main' in r]
        print(f'  stored {a:6s} L1open', m([x['summary']['l1'] for x in S]), 'leak', m([x['summary']['leakage'] for x in S]), 'missed', m([x['summary']['missed'] for x in S]), 'own-target step r', m([x['step_pearson'] for x in S]))
# worst cell for noop_200
C=[r for r in L if r['cell']['layout']=='coalition']
w=sorted(C,key=lambda r:-r['runs']['noop_200']['vs_closed']['l1'])[:3]
print('worst noop_200 cells', [(r['cell'], round(r['runs']['noop_200']['vs_closed']['l1'],3), round(r['runs']['noop_200']['step_pearson'],3)) for r in w])
print('by es: ', {es: m([r['runs']['noop_200']['vs_closed']['l1'] for r in C if r['cell']['es']==es]) for es in (0,12)},
      {es: m([r['runs']['noop_200']['step_pearson'] for r in C if r['cell']['es']==es]) for es in (0,12)})
print('by d: ', {d: (m([r['runs']['noop_200']['vs_closed']['l1'] for r in C if r['cell']['delay']==d]), m([r['runs']['noop_200']['step_pearson'] for r in C if r['cell']['delay']==d])) for d in (0,25,50)})
# ---- B. coalition level on the H5 layouts ----
print('\n== coalition level (H5 cells) ==')
for lay,b in (('coalition',2),('overlap',1),('partial',2)):
    for d in (0,50):
        C=[r for r in L if r['cell']['layout']==lay and r['cell']['delay']==d and r['cell']['es']==0 and r['cell']['b']==b and 'table3_mix' in r['runs']]
        if not C: continue
        print(f'{lay} d={d}: {len(C)} cells; uniform shapley L1 open', m([r['uniform']['shapley_open']['l1'] for r in C]), 'closed', m([r['uniform']['shapley_closed']['l1'] for r in C]))
        for run in ('table2_mix','table3_mix','remove_200','remove_800','mob3_noop_200','mob3s_noop_200','mob3_noop_800'):
            R=[r['runs'][run] for r in C]
            print(f'  {run:15s} divErr open', m([x['coalition_open']['dividend_error'] for x in R]), 'closed', m([x['coalition_closed']['dividend_error'] for x in R]),
                  'rec/prec open', m([x['coalition_open']['recall'] for x in R])[0], m([x['coalition_open']['precision'] for x in R])[0],
                  'shapL1 open', m([x['shapley_open']['l1'] for x in R]), 'closed', m([x['shapley_closed']['l1'] for x in R]))
        for a in ('ece2_mix','ece3_mix','na2q_mix'):
            S=[r['stored'][a] for r in C if 'stored' in r]
            print(f'  stored {a:9s} divErr open', m([x['coalition']['dividend_error'] for x in S]), 'rec/prec', m([x['coalition']['recall'] for x in S])[0], m([x['coalition']['precision'] for x in S])[0], 'L1(shapley,open)', m([x['summary']['l1'] for x in S]))
        print('   example dividends table3:', C[0]['runs']['table3_mix']['top_dividends'][:4])
