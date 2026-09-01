# -*- coding: utf-8 -*-
"""
06 · 격차의 정체 — 시험 환자를 성인/소아로 나눠 채점
    채혈 6회 · 학습 120명 · ζ0 초기화 · 5회 반복
    학습 프로토콜은 01_grid.py 와 동일 (검증 손실 조기 종료 + 최적 가중치 복원)

실행 : python src/06_subgroup_adult_child.py      결과 : results/subgroup.json
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F

_g = {}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_grid.py")).read()
     .split('# ── 7. 전체 격자')[0], _g)
globals().update({k: v for k, v in _g.items() if not k.startswith("__")})


def run_split(seed, epochs=6000, probe=50, patience=20):
    torch.manual_seed(seed)
    g = np.random.default_rng(100 + seed); perm = g.permutation(n)
    tr, te = perm[:120], perm[120:]
    lo, hi = Xt[tr].min(0).values, Xt[tr].max(0).values
    Xn = (Xt - lo) / (hi - lo); j = idx['extensive']
    vs = int(len(tr) * 0.2); vi, ti = tr[:vs], tr[vs:]
    m = DCM([150., 2500., 150., 2000.])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, bstate, bad = 1e9, None, 0
    for e in range(epochs + 1):
        opt.zero_grad()
        z = m(Xn[ti]); c = conc(z[:, 0], z[:, 1], z[:, 2], z[:, 3], dt[ti], tt)
        F.mse_loss(c[:, j], Ot[ti][:, j]).backward(); opt.step()
        if e % probe == 0:
            with torch.no_grad():
                zv = m(Xn[vi]); cv = conc(zv[:, 0], zv[:, 1], zv[:, 2], zv[:, 3], dt[vi], tt)
                vl = F.mse_loss(cv[:, j], Ot[vi][:, j]).item()
            if vl < best - 1e-9:
                best, bstate, bad = vl, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= patience: break
    m.load_state_dict(bstate)
    with torch.no_grad():
        z = m(Xn); c = conc(z[:, 0], z[:, 1], z[:, 2], z[:, 3], dt, tt)
    ad = adult[te]
    return dict(all=round(accuracy(c[te], Ct[te]), 1),
                adult=round(accuracy(c[te[ad]], Ct[te[ad]]), 1),
                child=round(accuracy(c[te[~ad]], Ct[te[~ad]]), 1),
                epoch=e, train_adult=int(adult[tr].sum()), train_child=int((~adult[tr]).sum()),
                test_adult=int(ad.sum()), test_child=int((~ad).sum()))


if __name__ == "__main__":
    R = [run_split(sd) for sd in range(5)]
    A = np.array([[r["all"], r["adult"], r["child"]] for r in R])
    print("seed  전체   성인   소아   멈춤   학습(성인/소아)   시험(성인/소아)")
    for i, r in enumerate(R):
        print("  %d  %5.1f %6.1f %6.1f %6d      %3d/%-3d        %3d/%-3d"
              % (i, r["all"], r["adult"], r["child"], r["epoch"],
                 r["train_adult"], r["train_child"], r["test_adult"], r["test_child"]))
    print("\n5회 평균   전체 %.1f ± %.1f   성인 %.1f ± %.1f   소아 %.1f ± %.1f"
          % (A[:, 0].mean(), A[:, 0].std(), A[:, 1].mean(), A[:, 1].std(),
             A[:, 2].mean(), A[:, 2].std()))
    print("성인 − 소아 %.1f %%p · 논문(99.4) 대비 성인 %.1f %%p"
          % (A[:, 1].mean() - A[:, 2].mean(), A[:, 1].mean() - 99.4))
    RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(RES, exist_ok=True)
    json.dump({"per_seed": R,
               "mean": {"all": round(float(A[:, 0].mean()), 1),
                        "adult": round(float(A[:, 1].mean()), 1),
                        "child": round(float(A[:, 2].mean()), 1)},
               "sd": {"all": round(float(A[:, 0].std()), 1),
                      "adult": round(float(A[:, 1].std()), 1),
                      "child": round(float(A[:, 2].std()), 1)}},
              open(os.path.join(RES, "subgroup.json"), "w"), ensure_ascii=False, indent=1)
    print("results/subgroup.json 저장")
