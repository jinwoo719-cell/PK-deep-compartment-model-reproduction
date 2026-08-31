import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time, copy
rng=np.random.default_rng(0); n=500
adult=rng.random(n)<0.71
wt=np.where(adult, rng.normal(80,15,n).clip(45,111), np.exp(rng.normal(np.log(19),.55,n)).clip(5,85))
age=np.where(adult, rng.normal(48,16,n).clip(19,78), np.exp(rng.normal(np.log(4.3),1.,n)).clip(.2,17.3))
ci=np.where(~adult)[0]; wt[ci[np.argsort(age[ci])]]=np.sort(wt[ci])
bgO=(rng.random(n)<.505).astype(float); maj=(rng.random(n)<np.where(adult,.614,.19)).astype(float)
dose=np.maximum(np.round(wt*rng.uniform(25,50,n)/250)*250,250)
CL=150*(wt/68)**.75*(age/40)**-.17*1.26**bgO*.93**maj
V1=2810*(wt/68)*(age/40)**-.09; Q=160*(wt/68)**.75; V2=1900*(wt/68)
k,k2,k3=CL/V1,Q/V1,Q/V2; s_,p_=k+k2+k3,k*k3
r=np.sqrt(s_*s_-4*p_); a_,b_=(s_+r)/2,(s_-r)/2; A_,B_=(a_-k3)/(a_-b_),(k3-b_)/(a_-b_)
t=np.concatenate([[.5],np.arange(1,49.)])
C_true=(dose/V1)[:,None]*(A_[:,None]*np.exp(-a_[:,None]*t)+B_[:,None]*np.exp(-b_[:,None]*t))
C_obs=np.maximum(C_true+rng.normal(0,.05,C_true.shape),0)
perm=rng.permutation(n)
def conc(CL,V1,Q,V2,dose,tt):
    k10,k12,k21=CL/V1,Q/V1,Q/V2; s,p=k10+k12+k21,k10*k21
    d=torch.clamp(s**2-4*p,min=1e-12); r=torch.sqrt(d)
    al,be=(s+r)/2,(s-r)/2; A,B=(al-k21)/(al-be),(k21-be)/(al-be)
    return (dose/V1).unsqueeze(1)*(A.unsqueeze(1)*torch.exp(-al.unsqueeze(1)*tt)+B.unsqueeze(1)*torch.exp(-be.unsqueeze(1)*tt))
class DCM(nn.Module):
    def __init__(s_,z0=None):
        super().__init__(); s_.net=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU(),nn.Linear(16,4))
        s_.z0=None if z0 is None else torch.tensor(z0,dtype=torch.float32)
    def forward(s_,x):
        o=s_.net(x); return F.softplus(o) if s_.z0 is None else s_.z0*(F.celu(o,alpha=.5)+1)
Xt=torch.tensor(np.column_stack([wt,age,bgO,maj]),dtype=torch.float32)
dt=torch.tensor(dose,dtype=torch.float32); tt=torch.tensor(t,dtype=torch.float32)
Ot=torch.tensor(C_obs,dtype=torch.float32); Ct=torch.tensor(C_true,dtype=torch.float32)
def acc(p_,tr_):
    tol=torch.where(tr_>=0.15,0.05,0.02); return ((p_-tr_).abs()<=tol).float().mean().item()*100
j=[0,4,12,24,36,48]
pool=perm[:120]; test=perm[120:]; val,tr=pool[:24],pool[24:]
lo,hi=Xt[tr].min(0).values,Xt[tr].max(0).values; Xn=(Xt-lo)/(hi-lo)
torch.manual_seed(0); m=DCM(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
hist=[]; t0=time.time()
for e in range(200001):
    opt.zero_grad(); z=m(Xn[tr]); C=conc(z[:,0],z[:,1],z[:,2],z[:,3],dt[tr],tt)
    loss=F.mse_loss(C[:,j],Ot[tr][:,j]); loss.backward(); opt.step()
    if e%10000==0:
        with torch.no_grad():
            zv=m(Xn[val]); Cv=conc(zv[:,0],zv[:,1],zv[:,2],zv[:,3],dt[val],tt)
            vl=F.mse_loss(Cv[:,j],Ot[val][:,j]).item()
            zz=m(Xn); cc=conc(zz[:,0],zz[:,1],zz[:,2],zz[:,3],dt,tt)
            hist.append((e,round(loss.item(),5),round(vl,5),round(acc(cc[test],Ct[test]),1),
                         [round(v,1) for v in zz.mean(0).tolist()]))
            json.dump({'hist':hist,'sec':round(time.time()-t0)},open('longstd.json','w'))
print('done',round(time.time()-t0))
