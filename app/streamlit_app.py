# -*- coding: utf-8 -*-
"""
응고인자 투여 설계 — Streamlit 버전
실행:  pip install streamlit numpy scipy matplotlib
       streamlit run streamlit_app.py
"""
import json, numpy as np, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.optimize import minimize

st.set_page_config(page_title="응고인자 투여 설계", layout="wide")

# ---------- 한글 폰트 ----------
import glob

def _setup_korean_font():
    cands = []
    for pat in ("/usr/share/fonts/truetype/nanum/Nanum*.ttf",
                "/usr/share/fonts/**/NanumGothic*.ttf",
                "/usr/share/fonts/**/NotoSansCJK*.ot[fc]",
                "/usr/share/fonts/**/NotoSansKR*.[ot]tf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "C:/Windows/Fonts/malgun.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
        cands += sorted(glob.glob(pat, recursive=True))
    for _p in cands:
        try:
            fm.fontManager.addfont(_p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
            return True
        except Exception:
            continue
    return False

_setup_korean_font()
plt.rcParams["axes.unicode_minus"] = False

# ---------- 색 (다크 모드) ----------
BG   = "#1F1418"     # 페이지 바탕
PANEL= "#2A1D22"     # 카드 바탕
INK  = "#EFE7E4"     # 본문 글자
MUTE = "#B9A9A6"     # 흐린 글자
OX   = "#E0808E"     # 예측 농도 곡선
TEAL = "#5AA3A6"     # 목표 범위
AMB  = "#D9A24A"
GRAY = "#9A8F88"     # 보정 전 집단 예측
LINE = "#4A3138"     # 구분선

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTE, "axes.edgecolor": LINE,
    "xtick.color": MUTE, "ytick.color": MUTE, "grid.color": LINE,
})

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(_HERE, "model.json"), encoding="utf-8"))

PERIODS = [
    dict(name="수술 직후", span="0 – 24 시간",   t0=0.0,   t1=24.0,  lo=0.80, hi=1.00),
    dict(name="1 – 5 일",  span="24 – 120 시간", t0=24.0,  t1=120.0, lo=0.50, hi=0.80),
    dict(name="5 일 이후", span="120 시간 이후", t0=120.0, t1=1e4,   lo=0.30, hi=0.50)]

# ---------- 신경망 ----------
def silu(x): return x / (1 + np.exp(-x))
def celu(x, a=0.5): return np.where(x > 0, x, a * (np.exp(x / a) - 1))

def predict_zeta(wt, age, bg, mj):
    lo, hi = np.array(M["norm"]["lo"]), np.array(M["norm"]["hi"])
    h = (np.array([wt, age, bg, mj], float) - lo) / (hi - lo + 1e-8)
    for i, L in enumerate(M["layers"]):
        h = np.array(L["W"]) @ h + np.array(L["b"])
        if i < 2: h = silu(h)
    return np.array(M["zeta0"]) * (celu(h) + 1)

# ---------- 2구획 약동학 ----------
def pk(z):
    CL, V1, Q, V2 = z
    k10, k12, k21 = CL/V1, Q/V1, Q/V2
    s, p = k10+k12+k21, k10*k21
    r = np.sqrt(max(s*s - 4*p, 1e-12))
    al, be = (s+r)/2, (s-r)/2
    return dict(CL=CL, V1=V1, Q=Q, V2=V2, al=al, be=be,
                A=(al-k21)/(al-be), B=(k21-be)/(al-be))

def conc(P, doses, infs, t):
    t = np.atleast_1d(np.asarray(t, float)); c = np.zeros_like(t)
    for d in doses:
        u = t - d["t"]; m = u >= 0
        c[m] += d["amt"]/P["V1"]*(P["A"]*np.exp(-P["al"]*u[m]) + P["B"]*np.exp(-P["be"]*u[m]))
    for f in infs:
        dt = t - f["t0"]; u = np.minimum(t, f["t1"]) - f["t0"]; m = u > 0
        if not m.any(): continue
        c[m] += (f["rate"]/P["V1"])*(
            P["A"]/P["al"]*(np.exp(-P["al"]*(dt[m]-u[m])) - np.exp(-P["al"]*dt[m])) +
            P["B"]/P["be"]*(np.exp(-P["be"]*(dt[m]-u[m])) - np.exp(-P["be"]*dt[m])))
    return c

# ---------- 처방 설계 ----------
TAUS = [2, 3, 4, 6, 8, 12, 24]

def build(z, mode, tau_sel, horizon, load_in):
    P = pk(z); doses = []; infs = []; plan = []
    p0 = PERIODS[0]
    load = load_in if load_in > 0 else round(p0["hi"]*P["V1"]/50)*50
    doses.append(dict(t=0.0, amt=float(load), label="로딩"))

    if mode == "bolus":
        for per in PERIODS:
            if per["t0"] >= horizon: break
            t1 = min(per["t1"], horizon); c_avg = (per["lo"]+per["hi"])/2
            cands = TAUS if tau_sel == "auto" else [int(tau_sel)]
            best = None
            for tau in cands:
                amt = max(50, round(P["CL"]*c_avg*tau/50)*50)
                dd = list(doses); t = tau if per["t0"] == 0 else per["t0"]
                while t < t1:
                    if conc(P, dd, [], t)[0] < per["hi"]*0.98: dd.append(dict(t=float(t), amt=float(amt)))
                    t += tau
                grid = np.arange(per["t0"], t1+1e-9, 0.25)
                cc = conc(P, dd, [], grid)
                frac = float(((cc >= per["lo"]*0.97) & (cc <= per["hi"]*1.05)).mean())
                if best is None or frac > best["frac"]+1e-9 or (abs(frac-best["frac"]) < 1e-9 and tau > best["tau"]):
                    best = dict(tau=tau, amt=amt, frac=frac, doses=dd)
            doses = best["doses"]
            starts = [d["t"] for d in doses if per["t0"] <= d["t"] < t1 and d.get("label") != "로딩"]
            plan.append(dict(per=per, tau=best["tau"], amt=best["amt"], frac=best["frac"],
                             feasible=best["frac"] > 0.98, start=min(starts) if starts else per["t0"]))
        doses.sort(key=lambda d: d["t"])
    else:
        BOOST = 8.0
        for per in PERIODS:
            if per["t0"] >= horizon: break
            t1 = min(per["t1"], horizon); c_avg = (per["lo"]+per["hi"])/2
            rate = max(1, round(P["CL"]*c_avg))
            t0 = per["t0"]
            if per["t0"] > 0:
                while t0 < t1 and conc(P, doses, infs, t0)[0] > c_avg: t0 += 0.25
            if t0 >= t1:
                plan.append(dict(per=per, rate=rate, feasible=True, start=t0, hold=t0-per["t0"], boost=None)); continue
            boost = None
            if per["t0"] == 0:
                b_end = min(BOOST, t1)
                def dip(rb):
                    test = [dict(t0=0.0, t1=b_end, rate=float(rb)), dict(t0=b_end, t1=t1, rate=float(rate))]
                    return conc(P, doses, test, np.arange(0, t1+1e-9, 0.25)).min()
                if dip(rate) < per["lo"]:
                    a, b = rate, rate*8
                    if dip(b) >= per["lo"]:
                        for _ in range(24):
                            m = (a+b)/2
                            if dip(m) >= per["lo"]: b = m
                            else: a = m
                    boost = dict(t0=0.0, t1=b_end, rate=float(round(b)))
            if boost:
                infs.append(boost); infs.append(dict(t0=b_end, t1=t1, rate=float(rate)))
            else:
                infs.append(dict(t0=float(t0), t1=float(t1), rate=float(rate)))
            plan.append(dict(per=per, rate=rate, boost=boost["rate"] if boost else None,
                             boost_end=boost["t1"] if boost else None,
                             feasible=True, start=t0, hold=t0-per["t0"]))
    return P, doses, infs, plan, load

# ---------- MAP 베이지안 개인 보정 ----------
def map_fit(z_pop, doses, infs, obs):
    w_cl, w_v1, sg = M["iiv"]["CL"], M["iiv"]["V1"], M["sigma_add"]
    ts = np.array([o[0] for o in obs]); cs = np.array([o[1] for o in obs])
    def nll(e):
        z = [z_pop[0]*np.exp(e[0]), z_pop[1]*np.exp(e[1]), z_pop[2], z_pop[3]]
        r = conc(pk(z), doses, infs, ts) - cs
        return float((r**2).sum()/sg**2 + e[0]**2/w_cl**2 + e[1]**2/w_v1**2)
    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-9, maxiter=2000))
    e = r.x
    return np.array([z_pop[0]*np.exp(e[0]), z_pop[1]*np.exp(e[1]), z_pop[2], z_pop[3]]), e

# ---------- UI ----------
st.title("응고인자 투여 설계")
st.caption(f"혈우병 A 수술기 8인자 · Deep Compartment Model 재현 · 시험 정확도 {M['meta']['test_accuracy']}%")

c1, c2 = st.columns([1, 2.1], gap="large")
with c1:
    st.subheader("1 · 환자")
    a, b = st.columns(2)
    wt = a.number_input("체중 (kg)", 3.0, 150.0, 80.0, 0.5)
    age = b.number_input("나이 (세)", 0.1, 95.0, 45.0, 0.5)
    a, b = st.columns(2)
    bg = 1.0 if a.selectbox("혈액형", ["O형 아님", "O형"], 0) == "O형" else 0.0
    mj = 1.0 if b.selectbox("수술 종류", ["대수술", "소수술"], 0) == "대수술" else 0.0

    st.subheader("2 · 투여 계획")
    mode = "bolus" if st.radio("투여 방식", ["반복 볼루스", "지속 주입"], horizontal=True) == "반복 볼루스" else "inf"
    a, b = st.columns(2)
    load_in = a.number_input("로딩 용량 (IU) · 0이면 자동", 0, 20000, 0, 250)
    tau_sel = b.selectbox("투여 간격 (시간)", ["auto", "2", "3", "4", "6", "8", "12", "24"],
                          0, format_func=lambda v: "자동 (구간별 최적)" if v == "auto" else f"{v} 시간",
                          disabled=(mode != "bolus"))
    horizon = float(st.selectbox("관찰 기간 (시간)", [72, 168, 240], 1))

    st.subheader("3 · 실측 농도")
    st.caption("투여 후 실제로 잰 농도를 넣으면 MAP 베이지안으로 개인 보정합니다.")
    n_obs = st.number_input("채혈 건수", 0, 10, 0)
    obs = []
    for i in range(int(n_obs)):
        a, b = st.columns(2)
        t_i = a.number_input(f"시간 {i+1} (h)", 0.0, 500.0, 12.0*(i+1), 0.5, key=f"t{i}")
        c_i = b.number_input(f"농도 {i+1} (IU/mL)", 0.0, 5.0, 0.50, 0.01, key=f"c{i}")
        obs.append((t_i, c_i))

z_pop = predict_zeta(wt, age, bg, mj)
P0, d0, i0, _, _ = build(z_pop, mode, tau_sel, horizon, load_in)
if obs:
    z, eta = map_fit(z_pop, d0, i0, obs)
else:
    z, eta = z_pop, None
P, doses, infs, plan, load = build(z, mode, tau_sel, horizon, load_in)

grid = np.linspace(0, horizon, 900)
cs = conc(P, doses, infs, grid)
cp = conc(P0, d0, i0, grid) if obs else None

with c2:
    cols = st.columns(3)
    for col, per in zip(cols, [p for p in PERIODS if p["t0"] < horizon]):
        t1 = min(per["t1"], horizon)
        pl = next(p for p in plan if p["per"] is per)
        settle = pl["start"] + 2 if per["t0"] > 0 else per["t0"]
        m_all = (grid >= per["t0"]) & (grid <= t1)
        m_set = m_all & (grid >= settle)
        if not m_set.any(): m_set = m_all
        def verdict(mask):
            mn, mx = cs[mask].min(), cs[mask].max()
            if mn < per["lo"]*0.97: return "목표 미달", mn, mx
            if mx > per["hi"]*1.15: return "목표 초과", mn, mx
            return "범위 유지", mn, mx
        s_all, _, _ = verdict(m_all); s_set, mn, mx = verdict(m_set)
        state = s_set if (s_all != "범위 유지" and s_set == "범위 유지") else s_all
        if state != "범위 유지": mn, mx = cs[m_all].min(), cs[m_all].max()
        col.metric(f"{per['name']} · 목표 {per['lo']:.2f}–{per['hi']:.2f}",
                   state, f"예측 {mn:.2f} – {mx:.2f} IU/mL", delta_color="off")

    fig, ax = plt.subplots(figsize=(11, 3.9))
    for per in PERIODS:
        if per["t0"] >= horizon: continue
        t1 = min(per["t1"], horizon)
        ax.fill_between([per["t0"], t1], per["lo"], per["hi"], color=TEAL, alpha=.20, lw=0)
        ax.hlines([per["lo"], per["hi"]], per["t0"], t1, color=TEAL, lw=1.1, alpha=.55)
        if per["t0"] > 0: ax.axvline(per["t0"], color=LINE, lw=1, ls=(0, (3, 3)))
        ax.text((per["t0"]+t1)/2, 1.19, per["name"], ha="center", fontsize=10, color=TEAL)
    if cp is not None: ax.plot(grid, cp, color=GRAY, lw=1.5, ls=(0, (5, 4)), label="보정 전 집단 예측")
    ax.plot(grid, cs, color=OX, lw=2.2, label="예측 농도")
    if obs: ax.scatter([o[0] for o in obs], [o[1] for o in obs], s=42, facecolor=BG,
                       edgecolor=INK, linewidth=1.4, zorder=5, label="실측 농도")
    ax.set_xlim(0, horizon); ax.set_ylim(0, max(1.25, cs.max()*1.12))
    ax.set_xlabel("투여 시작 후 시간 (h)"); ax.set_ylabel("농도 (IU/mL)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    lg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t_ in lg.get_texts(): t_.set_color(MUTE)
    st.pyplot(fig, use_container_width=True)

    st.subheader("권장 투여 계획")
    st.markdown(f"**로딩 용량** {load:,.0f} IU · {load/wt:.1f} IU/kg")
    for p in plan:
        per = p["per"]
        if mode == "bolus":
            body = f"**{p['amt']:,.0f} IU** 를 **{p['tau']}시간**마다"
            if not p["feasible"]: body += f"  ·  범위 유지율 {p['frac']*100:.0f}% — 지속 주입 권장"
        else:
            body = (f"처음 {p['boost_end']:.0f}시간 **{p['boost']:,.0f} IU/h** → 이후 **{p['rate']:,.0f} IU/h**"
                    if p.get("boost") else f"**{p['rate']:,.0f} IU/h** 지속 주입"
                    + (f"  ·  {p['hold']:.0f}시간 휴약 후 시작" if p.get("hold", 0) > 0.2 else ""))
        st.markdown(f"- {per['name']} · {per['span']} · 목표 {per['lo']:.2f}–{per['hi']:.2f} — {body}")

    st.subheader("추정 파라미터")
    k = st.columns(4)
    for col, lab, val, u in zip(k, ["CL", "V1", "Q", "V2"], z, ["mL/h", "mL", "mL/h", "mL"]):
        col.metric(lab, f"{val:,.1f}", u, delta_color="off")
    st.caption(f"분포 반감기 t½α {np.log(2)/P['al']:.1f} h · 소실 반감기 t½β {np.log(2)/P['be']:.1f} h"
               + (f"  ·  개인 보정 폭 CL {(np.exp(eta[0])-1)*100:+.0f}% · V1 {(np.exp(eta[1])-1)*100:+.0f}%"
                  if eta is not None else ""))

st.divider()
st.caption("시뮬레이션 데이터로 학습한 재현 연구용 도구입니다. 실제 진료 판단에 사용할 수 없습니다. · "
           "Janssen et al. CPT:PSP 2022;11:934–945 · Hazendonk et al. Thromb Haemost 2016;116(4):639–650")
