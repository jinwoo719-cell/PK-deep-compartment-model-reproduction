# -*- coding: utf-8 -*-
"""논문 조건 그대로 재현: 6회 채혈 · 120명 · ζ0 · 에포크 확대, 성인/소아 분할 채점"""
exec(open('01_grid.py').read().split("print()\nprint(f'{\"채혈\"")[0])
import torch, numpy as np, time, json, torch.nn.functional as F

def run_long(strat, ntr, use_z0, seed, epochs, probe=2000):
    torch.manual_seed(seed)
    g = np.random.default_rng(100+seed); perm = g.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    lo,hi = Xt[tr].min(0).values, Xt[tr].max(0).values
    Xn = (Xt-lo)/(hi-lo); j = idx[strat]
    m = DCM([150.,2500.,150.,2000.] if use_z0 else None)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist=[]
    for e in range(epochs+1):
        opt.zero_grad()
        z = m(Xn[tr]); c = conc(z[:,0],z[:,1],z[:,2],z[:,3], dt[tr], tt)
        loss = F.mse_loss(c[:,j], Ot[tr][:,j]); loss.backward(); opt.step()
        if e % probe == 0:
            with torch.no_grad():
                zz = m(Xn); cc = conc(zz[:,0],zz[:,1],zz[:,2],zz[:,3], dt, tt)
                ad = adult[te]; te_a = te[ad]; te_c = te[~ad]
                hist.append((e, round(accuracy(cc[te],Ct[te]),1),
                             round(accuracy(cc[te_a],Ct[te_a]),1),
                             round(accuracy(cc[te_c],Ct[te_c]),1)))
    return hist, te

out={}
for seed in (0,1,2):
    t0=time.time()
    h,te = run_long('extensive',120,True,seed,30000)
    out[seed]=h
    print('seed',seed,'%.0fs'%(time.time()-t0),'최고',max(x[1] for x in h),
          '@',[x[0] for x in h if x[1]==max(y[1] for y in h)][0], flush=True)
    json.dump(out,open('repro18.json','w'))
print('시험 환자 중 성인 %d · 소아 %d'%(adult[te].sum(), (~adult[te]).sum()))
