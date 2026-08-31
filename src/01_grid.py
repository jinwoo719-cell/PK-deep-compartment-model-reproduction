import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
torch.manual_seed(0)

# ── 1~4. 데이터 500명 (앞서 확정한 설정) ─────────────────
rng = np.random.default_rng(0); n = 500
adult = rng.random(n) < 0.71
wt  = np.where(adult, np.clip(rng.normal(80,15,n),45,111),
                      np.clip(np.exp(rng.normal(np.log(19),.55,n)),5,85))
age = np.where(adult, np.clip(rng.normal(48,16,n),19,78),
                      np.clip(np.exp(rng.normal(np.log(4.3),1.),n and n),.2,17.3)) if False else \
      np.where(adult, np.clip(rng.normal(48,16,n),19,78),
                      np.clip(np.exp(rng.normal(np.log(4.3),1.,n)),.2,17.3))
bgO = (rng.random(n)<.505).astype(float)
maj = (rng.random(n)<np.where(adult,.614,.190)).astype(float)
dose = np.maximum(np.round(wt*rng.uniform(25,50,n)/250)*250, 250)

CL = 150*(wt/68)**.75*(age/40)**-.17*1.26**bgO*.93**maj
V1 = 2810*(wt/68)*(age/40)**-.09
Q  = 160*(wt/68)**.75
V2 = 1900*(wt/68)

def conc(CL,V1,Q,V2,d,t):
    k10,k12,k21 = CL/V1, Q/V1, Q/V2
    s,p = k10+k12+k21, k10*k21
    disc = torch.clamp(s*s-4*p, min=1e-12) if torch.is_tensor(s) else np.maximum(s*s-4*p,1e-12)
    r = disc**0.5
    a,b = (s+r)/2, (s-r)/2
    A,B = (a-k21)/(a-b), (k21-b)/(a-b)
    if torch.is_tensor(s):
        return (d/V1).unsqueeze(1)*(A.unsqueeze(1)*torch.exp(-a.unsqueeze(1)*t)
                                  + B.unsqueeze(1)*torch.exp(-b.unsqueeze(1)*t))
    return (d/V1)[:,None]*(A[:,None]*np.exp(-a[:,None]*t)+B[:,None]*np.exp(-b[:,None]*t))

tgrid = np.concatenate([[0.5], np.arange(1,49.)])          # 49개 시점
C_true = conc(CL,V1,Q,V2,dose,tgrid)
print(f'데이터 검증  최고 {C_true[:,0].mean():.3f} (논문 0.890)   최저 {C_true[:,-1].mean():.4f} (논문 0.0900)')

C_obs = np.maximum(C_true + rng.normal(0,0.05,C_true.shape), 0)

# ── 5. 채혈 전략 ───────────────────────────────────────
STRAT = {'extensive':[0.5,4,12,24,36,48], 'routine':[4,24,48], 'limited':[8,30], 'extreme':[24]}
idx = {k:[int(np.where(np.isclose(tgrid,x))[0][0]) for x in v] for k,v in STRAT.items()}

X = np.column_stack([wt,age,bgO,maj])
Xt = torch.tensor(X,dtype=torch.float32); dt = torch.tensor(dose,dtype=torch.float32)
Ct = torch.tensor(C_true,dtype=torch.float32); Ot = torch.tensor(C_obs,dtype=torch.float32)
tt = torch.tensor(tgrid,dtype=torch.float32)

class DCM(nn.Module):
    def __init__(self, z0=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64), nn.SiLU(), nn.Linear(64,16), nn.SiLU(), nn.Linear(16,4))
        self.z0 = None if z0 is None else torch.tensor(z0,dtype=torch.float32)
    def forward(self,x):
        o = self.net(x)
        return F.softplus(o) if self.z0 is None else self.z0*(F.celu(o,alpha=0.5)+1)

def accuracy(pred, true):
    tol = torch.where(true>=0.15, 0.05, 0.02)
    return ((pred-true).abs() <= tol).float().mean().item()*100

def run(strat, ntr, use_z0, seed, epochs=3000):
    torch.manual_seed(seed)
    g = np.random.default_rng(100+seed); perm = g.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    lo,hi = Xt[tr].min(0).values, Xt[tr].max(0).values          # train 기준 min-max
    Xn = (Xt-lo)/(hi-lo)
    j = idx[strat]
    m = DCM([150.,2500.,150.,2000.] if use_z0 else None)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, wait = 1e9, 0
    for e in range(epochs):
        opt.zero_grad()
        z = m(Xn[tr]); c = conc(z[:,0],z[:,1],z[:,2],z[:,3], dt[tr], tt)
        loss = F.mse_loss(c[:,j], Ot[tr][:,j])
        loss.backward(); opt.step()
        L = loss.item()
        if L < best-1e-7: best, wait = L, 0
        else:
            wait += 1
            if wait > 200: break
    with torch.no_grad():
        z = m(Xn); c = conc(z[:,0],z[:,1],z[:,2],z[:,3], dt, tt)
        return accuracy(c[tr],Ct[tr]), accuracy(c[te],Ct[te]), e+1

print()
print(f'{"채혈":10s} {"n":>4} {"모델":>6} {"train":>7} {"test":>7} {"논문 test":>10} {"epoch":>7} {"초":>5}')
targets = {('extensive',120,0):99.1,('extensive',120,1):99.4,
           ('routine',120,0):65.3,('routine',120,1):97.8,
           ('extreme',120,0):28.9,('extreme',120,1):76.0}
for strat,ntr in [('extensive',120),('routine',120),('extreme',120)]:
    for z0 in (0,1):
        t0=time.time(); tra,tes,ep = run(strat,ntr,bool(z0),0)
        print(f'{strat:10s} {ntr:4d} {"ζ₀형" if z0 else "표준":>6} {tra:7.1f} {tes:7.1f} {targets[(strat,ntr,z0)]:10.1f} {ep:7d} {time.time()-t0:5.0f}')
