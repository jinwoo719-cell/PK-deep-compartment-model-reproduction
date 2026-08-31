exec(open('02_convergence_standard.py').read().split("torch.manual_seed(0); m=DCM()")[0])
import torch, numpy as np, torch.nn.functional as F, time
torch.manual_seed(0); m=DCM(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
t0=time.time()
for e in range(40001):
    opt.zero_grad(); z=m(Xn[tr]); Cc=conc(z[:,0],z[:,1],z[:,2],z[:,3],dt[tr],tt)
    loss=F.mse_loss(Cc[:,j],Ot[tr][:,j]); loss.backward(); opt.step()
print("학습", round(time.time()-t0), "초")
with torch.no_grad():
    zz=m(Xn); cc=conc(zz[:,0],zz[:,1],zz[:,2],zz[:,3],dt,tt)
    a=acc(cc[test],Ct[test]); print("시험 정확도 %.1f %%"%a)
Z=zz.numpy()
np.save('std40k_z.npy',Z)
lab=["CL","V1","Q","V2"]; TRUE=[CL,V1,Q,V2]
print("\n%-4s %10s %10s %10s %10s | %10s"%("","5%","중앙값","95%","평균","참값 중앙값"))
for i in range(4):
    q=np.percentile(Z[:,i],[5,50,95])
    print("%-4s %10.2f %10.2f %10.2f %10.2f | %10.1f"%(lab[i],q[0],q[1],q[2],Z[:,i].mean(),np.median(TRUE[i])))
k10=Z[:,0]/Z[:,1]; k12=Z[:,2]/Z[:,1]; k21=Z[:,2]/Z[:,3]
s=k10+k12+k21; p=k10*k21; r=np.sqrt(np.maximum(s*s-4*p,1e-12))
al,be=(s+r)/2,(s-r)/2; A=(al-k21)/(al-be); B=(k21-be)/(al-be)
amp1=(dose/Z[:,1])*A; amp2=(dose/Z[:,1])*B
print("\n관측되는 4가지 (곡선을 실제로 결정하는 값)")
print("%-24s %10s %10s | %10s"%("","중앙값","평균","참값 중앙값"))
kt10=CL/V1; kt12=Q/V1; kt21=Q/V2
st=kt10+kt12+kt21; pt=kt10*kt21; rt=np.sqrt(st*st-4*pt)
alt,bet=(st+rt)/2,(st-rt)/2; At=(alt-kt21)/(alt-bet); Bt=(kt21-bet)/(alt-bet)
for nm,v,vt in [("빠른 상 진폭 (IU/mL)",amp1,(dose/V1)*At),("빠른 상 반감기 (h)",np.log(2)/al,np.log(2)/alt),
                ("느린 상 진폭 (IU/mL)",amp2,(dose/V1)*Bt),("느린 상 반감기 (h)",np.log(2)/be,np.log(2)/bet)]:
    print("%-24s %10.3f %10.3f | %10.3f"%(nm,np.median(v),v.mean(),np.median(vt)))
print("\n첫 채혈(0.5h) 전에 빠른 상이 몇 % 사라지나 :  ① %.1f%%   참값 %.1f%%"
      %(np.median(1-np.exp(-al*0.5))*100, np.median(1-np.exp(-alt*0.5))*100))
