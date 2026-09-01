# -*- coding: utf-8 -*-
"""
01 · 12조건 격자 재현
    가상 환자 500명 → 채혈 4종 × 학습 20/60/120명 × (표준 DCM / ζ0 초기화) × 5회 반복

조기 종료 : 학습셋의 20%를 검증으로 떼고 50 에포크마다 검증 손실을 본다.
            20회 연속 나아지지 않으면 멈추고, 검증 손실이 가장 낮았던 가중치를 되살린다.
            (논문 "Models were trained until MSE stopped improving"에 대응)

실행 : python src/01_grid.py      결과 : results/grid.json
"""
import os, json, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# ── 1~4. 데이터 500명 ────────────────────────────────────────
rng = np.random.default_rng(0); n = 500
adult = rng.random(n) < 0.71
wt  = np.where(adult, np.clip(rng.normal(80, 15, n), 45, 111),
                      np.clip(np.exp(rng.normal(np.log(19), .55, n)), 5, 85))
age = np.where(adult, np.clip(rng.normal(48, 16, n), 19, 78),
                      np.clip(np.exp(rng.normal(np.log(4.3), 1., n)), .2, 17.3))
bgO = (rng.random(n) < .505).astype(float)
maj = (rng.random(n) < np.where(adult, .614, .190)).astype(float)
dose = np.maximum(np.round(wt * rng.uniform(25, 50, n) / 250) * 250, 250)

CL = 150 * (wt/68)**.75 * (age/40)**-.17 * 1.26**bgO * .93**maj
V1 = 2810 * (wt/68) * (age/40)**-.09
Q  = 160 * (wt/68)**.75
V2 = 1900 * (wt/68)

def conc(CL, V1, Q, V2, d, t):
    k10, k12, k21 = CL/V1, Q/V1, Q/V2
    s, p = k10 + k12 + k21, k10 * k21
    disc = torch.clamp(s*s - 4*p, min=1e-12) if torch.is_tensor(s) else np.maximum(s*s - 4*p, 1e-12)
    r = disc**0.5
    a, b = (s + r)/2, (s - r)/2
    A, B = (a - k21)/(a - b), (k21 - b)/(a - b)
    if torch.is_tensor(s):
        return (d/V1).unsqueeze(1) * (A.unsqueeze(1)*torch.exp(-a.unsqueeze(1)*t)
                                    + B.unsqueeze(1)*torch.exp(-b.unsqueeze(1)*t))
    return (d/V1)[:, None] * (A[:, None]*np.exp(-a[:, None]*t) + B[:, None]*np.exp(-b[:, None]*t))

tgrid = np.concatenate([[0.5], np.arange(1, 49.)])          # 49개 시점
C_true = conc(CL, V1, Q, V2, dose, tgrid)
C_obs  = np.maximum(C_true + rng.normal(0, 0.05, C_true.shape), 0)   # 논문과 동일한 σ=0.05

# ── 5. 채혈 전략 ─────────────────────────────────────────────
STRAT = {'extensive': [0.5, 4, 12, 24, 36, 48], 'routine': [4, 24, 48],
         'limited': [8, 30], 'extreme': [24]}
idx = {k: [int(np.where(np.isclose(tgrid, x))[0][0]) for x in v] for k, v in STRAT.items()}

X = np.column_stack([wt, age, bgO, maj])
Xt = torch.tensor(X, dtype=torch.float32); dt = torch.tensor(dose, dtype=torch.float32)
Ct = torch.tensor(C_true, dtype=torch.float32); Ot = torch.tensor(C_obs, dtype=torch.float32)
tt = torch.tensor(tgrid, dtype=torch.float32)

class DCM(nn.Module):
    def __init__(self, z0=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(),
                                 nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 4))
        self.z0 = None if z0 is None else torch.tensor(z0, dtype=torch.float32)
    def forward(self, x):
        o = self.net(x)
        return F.softplus(o) if self.z0 is None else self.z0 * (F.celu(o, alpha=0.5) + 1)

def accuracy(pred, true):
    tol = torch.where(true >= 0.15, 0.05, 0.02)
    return ((pred - true).abs() <= tol).float().mean().item() * 100

# ── 6. 한 조건 학습 ──────────────────────────────────────────
def run(strat, ntr, use_z0, seed, epochs=6000, probe=50, patience=20):
    """검증 손실 기준 조기 종료 + 최적 가중치 복원. (test 정확도, 멈춘 에포크) 반환"""
    torch.manual_seed(seed)
    g = np.random.default_rng(100 + seed); perm = g.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    lo, hi = Xt[tr].min(0).values, Xt[tr].max(0).values      # 정규화는 학습셋 기준
    Xn = (Xt - lo) / (hi - lo); j = idx[strat]
    vs = max(1, int(len(tr) * 0.2)); vi, ti = tr[:vs], tr[vs:]
    m = DCM([150., 2500., 150., 2000.] if use_z0 else None)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, bstate, bad, e = 1e9, None, 0, 0
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
    if bstate: m.load_state_dict(bstate)
    with torch.no_grad():
        z = m(Xn); c = conc(z[:, 0], z[:, 1], z[:, 2], z[:, 3], dt, tt)
        return accuracy(c[te], Ct[te]), e

# ── 7. 전체 격자 ─────────────────────────────────────────────
if __name__ == "__main__":
    LAB = {"extreme": "채혈 1회", "limited": "채혈 2회",
           "routine": "채혈 3회", "extensive": "채혈 6회"}
    PAPER = {  # Janssen 2022 Table 1 (test)
        ("extreme",20): (32.2,72.9), ("extreme",60): (29.4,65.2), ("extreme",120): (28.9,76.0),
        ("limited",20): (61.2,76.5), ("limited",60): (73.6,83.0), ("limited",120): (76.1,90.3),
        ("routine",20): (59.1,90.1), ("routine",60): (59.5,94.8), ("routine",120): (65.3,97.8),
        ("extensive",20): (84.4,88.7), ("extensive",60): (93.0,97.9), ("extensive",120): (99.1,99.4)}
    out, t0 = {}, time.time()
    print("%-9s %5s %6s | %7s %7s | %7s" % ("채혈", "n", "방식", "재현", "논문", "멈춘 에포크"))
    for strat in ("extreme", "limited", "routine", "extensive"):
        for ntr in (20, 60, 120):
            for z0 in (0, 1):
                a, eps = [], []
                for sd in range(5):
                    acc, e = run(strat, ntr, bool(z0), sd)
                    a.append(round(acc, 1)); eps.append(e)
                key = "%s|%d|%s" % (strat, ntr, "z0" if z0 else "std")
                out[key] = dict(acc=a, mean=round(float(np.mean(a)), 1),
                                sd=round(float(np.std(a)), 1), epochs=eps)
                print("%-9s %5d %6s | %7.1f %7.1f | %d~%d" %
                      (LAB[strat], ntr, "ζ0" if z0 else "표준", out[key]["mean"],
                       PAPER[(strat, ntr)][z0], min(eps), max(eps)), flush=True)
    RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "grid.json"), "w"), ensure_ascii=False, indent=1)
    print("\n완료 %d초 · results/grid.json 저장" % round(time.time() - t0))
