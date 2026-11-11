import json,glob,numpy as np,collections
L=[json.load(open(p)) for p in sorted(glob.glob('out/oracle/*.json'))]
print('cells',len(L),'episodes',sum(len(r['episodes']) for r in L))
for null in ('noop','remove'):
    ident=max(e[null]['identity_err'] for r in L for e in r['episodes'])
    a_at_pull=[];bc_at_done=[];a_at_done=[];neg=0;byst=0;multi=0;tot=0;zero_ret=0
    l1open=collections.defaultdict(list)
    for r in L:
        b=r['cell']['b']
        for e in r['episodes']:
            x=e[null]; tot+=1
            if e['return']==0: zero_ret+=1; continue
            if x['peak_step'][0] is not None:
                a_at_pull.append(x['peak_step'][0]==e['pull']); a_at_done.append(x['peak_step'][0]==e['complete'])
            for i in (1,2):
                if x['peak_step'][i] is not None: bc_at_done.append(x['peak_step'][i]==e['complete'])
            neg+= x['min_step_credit']<-1e-9
            byst+= any(abs(c)>1e-9 for c in x['credit'][3:])
            multi+= any(k>1 for k in x['n_nonzero_steps'])
            if x['vs_open_loop']['valid']: l1open[b].append(x['vs_open_loop']['l1'])
    print(null,'max |sum_t kappa - LOO| =',ident,'| A credit at pull step:',np.mean(a_at_pull).round(3),'at completion:',np.mean(a_at_done).round(3),'| B,C at completion:',np.mean(bc_at_done).round(3),
          '| episodes with a negative step credit:',neg,'| with nonzero bystander credit:',byst,'| with credit on >1 step:',multi,'| zero-return episodes:',zero_ret,'of',tot)
    print('   L1 of oracle credit vs repo open-loop LOO by b:',{b:round(float(np.mean(v)),3) for b,v in sorted(l1open.items())})
for es in (0,12):
    C=[r for r in L if r['cell']['es']==es]
    print('stored perfect one-step critic, es',es,': missed(noop)',np.mean([r['stored_semantic_gap_noop']['missed'] for r in C]).round(3),'L1',np.mean([r['stored_semantic_gap_noop']['l1'] for r in C]).round(3),
          '| missed(baseline)',np.mean([r['stored_semantic_gap_baseline']['missed'] for r in C]).round(3),'L1',np.mean([r['stored_semantic_gap_baseline']['l1'] for r in C]).round(3))
