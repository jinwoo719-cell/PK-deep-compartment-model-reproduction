exec(open('02_convergence_standard.py').read().split("torch.manual_seed(0); m=DCM()")[0])
import torch, numpy as np, torch.nn.functional as F, json, time
torch.manual_seed(0); m=DCM([150.,2500.,150.,2000.]); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
hist=[]; t0=time.time()
for e in range(200001):
    opt.zero_grad(); z=m(Xn[tr]); Cc=conc(z[:,0],z[:,1],z[:,2],z[:,3],dt[tr],tt)
    loss=F.mse_loss(Cc[:,j],Ot[tr][:,j]); loss.backward(); opt.step()
    if e%10000==0:
        with torch.no_grad():
            zv=m(Xn[val]); Cv=conc(zv[:,0],zv[:,1],zv[:,2],zv[:,3],dt[val],tt)
            vl=F.mse_loss(Cv[:,j],Ot[val][:,j]).item()
            zz=m(Xn); cc=conc(zz[:,0],zz[:,1],zz[:,2],zz[:,3],dt,tt)
            hist.append((e,round(loss.item(),5),round(vl,5),round(acc(cc[test],Ct[test]),1),
                         [round(v,1) for v in np.median(zz.numpy(),0).tolist()]))
            json.dump({'hist':hist,'sec':round(time.time()-t0)},open('z0_long.json','w'))
            if e%40000==0: np.save('z0_long_%d.npy'%e, zz.numpy())
print('done',round(time.time()-t0))
