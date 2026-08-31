# -*- coding: utf-8 -*-
"""서비스 배포용 DCM 학습 → 가중치 JSON 내보내기"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, copy, json
torch.manual_seed(0)

# ---------- 데이터 (재현 파이프라인과 동일, seed=0) ----------
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
t = np.linspace(0,48,49); t[0] = 0.5

def conc_np(CL,V1,Q,V2,dose,t):
    k10,k12,k21 = CL/V1,Q/V1,Q/V2; s,p = k10+k12+k21,k10*k21
    r=np.sqrt(np.maximum(s*s-4*p,1e-12)); a,b=(s+r)/2,(s-r)/2
    A,B=(a-k21)/(a-b),(k21-b)/(a-b)
    return (dose/V1)[:,None]*(A[:,None]*np.exp(-a[:,None]*t)+B[:,None]*np.exp(-b[:,None]*t))
C_true = conc_np(CL,V1,Q,V2,dose,t)
C_obs  = np.maximum(C_true + rng.normal(0,0.05,C_true.shape), 0)
X = np.stack([wt,age,bg_O,major],1).astype(np.float32)
idx = [int(np.argmin(np.abs(t-v))) for v in (0.5,4,12,24,36,48)]

perm = rng.permutation(n); tr, te = perm[:400], perm[400:]

def conc(CL,V1,Q,V2,dose,tt):
    k10,k12,k21 = CL/V1,Q/V1,Q/V2; s,p = k10+k12+k21,k10*k21
    r=torch.sqrt(torch.clamp(s**2-4*p,min=1e-12)); a,b=(s+r)/2,(s-r)/2
    A,B=(a-k21)/(a-b),(k21-b)/(a-b)
    return (dose/V1).unsqueeze(1)*(A.unsqueeze(1)*torch.exp(-a.unsqueeze(1)*tt)
                                  +B.unsqueeze(1)*torch.exp(-b.unsqueeze(1)*tt))
Z0=[150.,2500.,150.,2000.]
class DCM(nn.Module):
    def __init__(s_):
        super().__init__()
        s_.net = nn.Sequential(nn.Linear(4,64), nn.SiLU(),
                               nn.Linear(64,16), nn.SiLU(), nn.Linear(16,4))
        s_.z0 = torch.tensor(Z0, dtype=torch.float32)
    def forward(s_,x): return s_.z0*(Fn.celu(s_.net(x),alpha=0.5)+1)

lo, hi = X[tr].min(0), X[tr].max(0)
nrm = lambda a: (a-lo)/(hi-lo+1e-8)
Xtr = torch.tensor(nrm(X[tr])); dtr = torch.tensor(dose[tr],dtype=torch.float32)
Ytr = torch.tensor(C_obs[np.ix_(tr,idx)],dtype=torch.float32)
tt  = torch.tensor(t[idx],dtype=torch.float32)
vs = int(len(tr)*0.2); vi, ti = torch.arange(vs), torch.arange(vs,len(tr))

m = DCM(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
best, bstate, bad = 1e9, None, 0
for ep in range(20001):
    opt.zero_grad()
    z = m(Xtr[ti]); pr = conc(z[:,0],z[:,1],z[:,2],z[:,3],dtr[ti],tt)
    ((pr-Ytr[ti])**2).mean().backward(); opt.step()
    if ep % 50 == 0:
        with torch.no_grad():
            zv=m(Xtr[vi]); pv=conc(zv[:,0],zv[:,1],zv[:,2],zv[:,3],dtr[vi],tt)
            vl=((pv-Ytr[vi])**2).mean().item()
        if vl < best-1e-9: best,bstate,bad = vl,copy.deepcopy(m.state_dict()),0
        else:
            bad += 1
            if bad>=25: break
m.load_state_dict(bstate); m.eval()

with torch.no_grad():
    zt = m(torch.tensor(nrm(X[te])))
    prt = conc(zt[:,0],zt[:,1],zt[:,2],zt[:,3],
               torch.tensor(dose[te],dtype=torch.float32),
               torch.tensor(t,dtype=torch.float32)).numpy()
tol = np.where(C_true[te]>=0.15,0.05,0.02)
acc = float((np.abs(prt-C_true[te])<=tol).mean()*100)
zt = zt.numpy(); true_z = np.stack([CL[te],V1[te],Q[te],V2[te]],1)
mape = np.abs(zt-true_z)/true_z*100
print(f"에포크 {ep} · 테스트 정확도 {acc:.1f}% · 학습 {len(tr)}명 / 시험 {len(te)}명")
mm=mape.mean(0); print("파라미터 평균 절대오차율(퍼센트)  CL {:.1f}  V1 {:.1f}  Q {:.1f}  V2 {:.1f}".format(*mm))

sd = m.state_dict()
out = {
  "meta":{"trained_on":len(tr),"tested_on":len(te),"test_accuracy":round(acc,1),
          "epochs":int(ep),"sampling":"0.5·4·12·24·36·48 h",
          "param_mape":{k:round(float(v),1) for k,v in
                        zip(["CL","V1","Q","V2"],mape.mean(0))}},
  "norm":{"lo":lo.tolist(),"hi":hi.tolist()},
  "zeta0":Z0,
  "iiv":{"CL":0.37,"V1":0.27},          # Hazendonk 2016 Table 4
  "sigma_add":0.05,
  "layers":[{"W":sd["net.0.weight"].tolist(),"b":sd["net.0.bias"].tolist()},
            {"W":sd["net.2.weight"].tolist(),"b":sd["net.2.bias"].tolist()},
            {"W":sd["net.4.weight"].tolist(),"b":sd["net.4.bias"].tolist()}]
}
json.dump(out, open("/home/claude/svc/model.json","w"))

# 검증용: 대표 환자 몇 명의 ζ를 파이썬 쪽에서 뽑아 저장 (JS 대조용)
cases=[[80,45,1,1],[68,40,0,0],[19,4,0,0],[100,60,1,0],[45,19,0,1]]
with torch.no_grad():
    zc = m(torch.tensor(nrm(np.array(cases,dtype=np.float32)))).numpy()
json.dump({"cases":cases,"zeta":zc.tolist()}, open("/home/claude/svc/check.json","w"))
for c,z in zip(cases,zc):
    print(f"  {c} -> CL {z[0]:7.1f}  V1 {z[1]:8.1f}  Q {z[2]:6.1f}  V2 {z[3]:8.1f}")
