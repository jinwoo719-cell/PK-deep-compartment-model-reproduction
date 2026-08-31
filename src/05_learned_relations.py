# -*- coding: utf-8 -*-
"""신경망이 실제로 배운 공변량-파라미터 관계를 참값과 비교"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, copy, json
torch.manual_seed(0)

# ---------- 1. 500명 재생성 (사용자 노트북과 동일, seed=0) ----------
rng = np.random.default_rng(0); n = 500
adult = rng.random(n) < 0.71
wt = np.where(adult, rng.normal(80,15,n).clip(45,111),
              np.exp(rng.normal(np.log(19),0.55,n)).clip(5,85))
age = np.where(adult, rng.normal(48,16,n).clip(19,78),
               np.exp(rng.normal(np.log(4.3),1.0,n)).clip(0.2,17.3))
ci = np.where(~adult)[0]; wt[ci[np.argsort(age[ci])]] = np.sort(wt[ci])
bg_O = (rng.random(n) < 0.505).astype(float)
major = (rng.random(n) < np.where(adult,0.614,0.190)).astype(float)
dose = np.maximum(np.round(wt*rng.uniform(25,50,n)/250)*250, 250)
CL = 150*(wt/68)**0.75*(age/40)**(-0.17)*1.26**bg_O*0.93**major
V1 = 2810*(wt/68)*(age/40)**(-0.09); Q = 160*(wt/68)**0.75; V2 = 1900*(wt/68)
t = np.linspace(0,48,49); t[0]=0.5

def conc_np(CL,V1,Q,V2,dose,t):
    k10,k12,k21 = CL/V1,Q/V1,Q/V2; s,p = k10+k12+k21,k10*k21
    r=np.sqrt(np.maximum(s*s-4*p,1e-12)); a,b=(s+r)/2,(s-r)/2
    A,B=(a-k21)/(a-b),(k21-b)/(a-b)
    return (dose/V1)[:,None]*(A[:,None]*np.exp(-a[:,None]*t)+B[:,None]*np.exp(-b[:,None]*t))
C_true = conc_np(CL,V1,Q,V2,dose,t)
C_obs  = np.maximum(C_true + rng.normal(0,0.05,C_true.shape), 0)

X = np.stack([wt,age,bg_O,major],1).astype(np.float32)
idx_ext = [np.argmin(np.abs(t-v)) for v in (0.5,4,12,24,36,48)]
perm = rng.permutation(n); tr, te = perm[:120], perm[120:]

# ---------- 2. torch PK 엔진 ----------
def conc(CL,V1,Q,V2,dose,tt):
    k10,k12,k21 = CL/V1,Q/V1,Q/V2; s,p = k10+k12+k21,k10*k21
    r=torch.sqrt(torch.clamp(s**2-4*p,min=1e-12)); a,b=(s+r)/2,(s-r)/2
    A,B=(a-k21)/(a-b),(k21-b)/(a-b)
    return (dose/V1).unsqueeze(1)*(A.unsqueeze(1)*torch.exp(-a.unsqueeze(1)*tt)
                                  +B.unsqueeze(1)*torch.exp(-b.unsqueeze(1)*tt))
class DCM(nn.Module):
    def __init__(self, z0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64), nn.SiLU(),
                                 nn.Linear(64,16), nn.SiLU(), nn.Linear(16,4))
        self.z0 = torch.tensor(z0, dtype=torch.float32)
    def forward(self,x): return self.z0*(Fn.celu(self.net(x),alpha=0.5)+1)

lo, hi = X[tr].min(0), X[tr].max(0)          # 정규화는 학습 세트에서만
nrm = lambda a: (a-lo)/(hi-lo+1e-8)
Xtr = torch.tensor(nrm(X[tr])); dtr = torch.tensor(dose[tr],dtype=torch.float32)
Ytr = torch.tensor(C_obs[np.ix_(tr,idx_ext)],dtype=torch.float32)
tt  = torch.tensor(t[idx_ext],dtype=torch.float32)

vs = int(len(tr)*0.2); vi, ti = torch.arange(vs), torch.arange(vs,len(tr))
m = DCM([150.,2500.,150.,2000.]); opt = torch.optim.Adam(m.parameters(),lr=1e-3)
best, bstate, bad = 1e9, None, 0
for ep in range(6001):
    opt.zero_grad()
    z = m(Xtr[ti]); pr = conc(z[:,0],z[:,1],z[:,2],z[:,3],dtr[ti],tt)
    loss = ((pr-Ytr[ti])**2).mean(); loss.backward(); opt.step()
    if ep % 50 == 0:
        with torch.no_grad():
            zv = m(Xtr[vi]); pv = conc(zv[:,0],zv[:,1],zv[:,2],zv[:,3],dtr[vi],tt)
            vl = ((pv-Ytr[vi])**2).mean().item()
        if vl < best-1e-9: best, bstate, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 20: break
m.load_state_dict(bstate); m.eval()
print("멈춘 에포크", ep, "· 검증손실", round(best,6))

# 테스트 정확도
with torch.no_grad():
    Xte = torch.tensor(nrm(X[te])); z = m(Xte)
    pr = conc(z[:,0],z[:,1],z[:,2],z[:,3],
              torch.tensor(dose[te],dtype=torch.float32),
              torch.tensor(t,dtype=torch.float32)).numpy()
tru = C_true[te]; tol = np.where(tru>=0.15,0.05,0.02)
print("테스트 정확도", round(float((np.abs(pr-tru)<=tol).mean()*100),1), "%")

def predict(w,a,o,mj):
    xx = np.stack([w,a,o,mj],1).astype(np.float32)
    with torch.no_grad(): return m(torch.tensor(nrm(xx))).numpy()

# ---------- 3. 스윕 ----------
out = {}
w = np.linspace(45,111,60); a = np.full(60,48.); o=np.zeros(60); mj=np.zeros(60)
out["wt"] = dict(x=w.tolist(),
                 pred=predict(w,a,o,mj)[:,0].tolist(),
                 true=(150*(w/68)**0.75*(48/40)**-0.17).tolist())
a2 = np.linspace(19,78,60); w2=np.full(60,80.)
out["age"] = dict(x=a2.tolist(),
                  pred=predict(w2,a2,np.zeros(60),np.zeros(60))[:,0].tolist(),
                  true=(150*(80/68)**0.75*(a2/40)**-0.17).tolist())
base = predict(np.array([80.]),np.array([48.]),np.array([0.]),np.array([0.]))[0,0]
oo   = predict(np.array([80.]),np.array([48.]),np.array([1.]),np.array([0.]))[0,0]
mm   = predict(np.array([80.]),np.array([48.]),np.array([0.]),np.array([1.]))[0,0]
out["cat"] = dict(base=float(base), o=float(oo), major=float(mm),
                  true_base=float(150*(80/68)**0.75*(48/40)**-0.17),
                  true_o=float(150*(80/68)**0.75*(48/40)**-0.17*1.26),
                  true_major=float(150*(80/68)**0.75*(48/40)**-0.17*0.93))
out["meta"]=dict(epoch=ep)
json.dump(out, open("/home/claude/learned.json","w"))
print("O형 배수  신경망", round(oo/base,3), " 참값 1.26")
print("대수술 배수 신경망", round(mm/base,3), " 참값 0.93")
