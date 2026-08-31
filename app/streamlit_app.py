# -*- coding: utf-8 -*-
"""
응고인자 투여 설계 — Streamlit 버전
실행:  pip install streamlit numpy scipy pandas plotly
       streamlit run streamlit_app.py
"""
import json, os, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="응고인자 투여 설계", layout="wide")

# 그래프는 브라우저가 그리므로 서버에 한글 폰트가 없어도 글자가 깨지지 않습니다.
KRFONT = "Malgun Gothic, Apple SD Gothic Neo, NanumGothic, Noto Sans KR, sans-serif"

# ---------- 색 (다크 모드) ----------
BG   = "#1F1418"     # 페이지 바탕
PANEL= "#2A1D22"     # 카드 바탕
INK  = "#EFE7E4"     # 본문 글자
MUTE = "#B9A9A6"     # 흐린 글자
OX   = "#E0808E"     # 예측 농도 곡선
TEAL = "#5AA3A6"     # 목표 범위
AMB  = "#D9A24A"
GRAY = "#9A8F88"     # 흐린 글자
GHOST= "#6E625E"     # 보정 전 집단 예측 (유령선)
LINE = "#4A3138"     # 구분선



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

def period_at(t):
    for per in PERIODS:
        if per["t0"] <= t < per["t1"]: return per
    return PERIODS[-1]

def envelope(P, doses, infs, horizon):
    """투여 간격마다 최고점·최저점을 뽑아 두 곡선으로 잇는다."""
    ts = sorted({0.0, float(horizon)} | {d["t"] for d in doses if d["t"] < horizon})
    if len(ts) < 2: return None
    pt, pc, tt, tc = [], [], [], []
    for a, b in zip(ts[:-1], ts[1:]):
        seg = np.linspace(a, b, 80)
        cc = conc(P, doses, infs, seg)
        pt.append(float(seg[cc.argmax()])); pc.append(float(cc.max()))
        tt.append(float(seg[cc.argmin()])); tc.append(float(cc.min()))
    return np.array(pt), np.array(pc), np.array(tt), np.array(tc)

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

    # ---------- 보기 구간 ----------
    opts = ["전체"] + [p_["name"] for p_ in PERIODS if p_["t0"] < horizon] + ["직접 지정"]
    pick = st.radio("보기 구간", opts, horizontal=True, label_visibility="collapsed")
    if pick == "직접 지정":
        xr = st.slider("보기 범위 (시간)", 0.0, float(horizon),
                       (0.0, min(24.0, float(horizon))), 0.5, label_visibility="collapsed")
    elif pick == "전체":
        xr = (0.0, float(horizon))
    else:
        pr = next(p_ for p_ in PERIODS if p_["name"] == pick)
        xr = (float(pr["t0"]), float(min(pr["t1"], horizon)))
    st.caption("끌어서 확대 · 아래 미니맵으로 이동 · 두 번 눌러 원래대로")

    env = envelope(P, doses, infs, horizon) if mode == "bolus" else None
    ymax = max(1.25, float(cs.max()) * 1.12)
    # 좁게 볼수록 실제 톱니를, 넓게 볼수록 최고–최저 띠를 앞세운다
    zoomed = (xr[1] - xr[0]) <= 48

    fig = go.Figure()
    for per in PERIODS:
        if per["t0"] >= horizon: continue
        t1 = min(per["t1"], horizon)
        fig.add_shape(type="rect", x0=per["t0"], x1=t1, y0=per["lo"], y1=per["hi"],
                      fillcolor=TEAL, opacity=.15, line_width=0, layer="below")
        for y in (per["lo"], per["hi"]):
            fig.add_shape(type="line", x0=per["t0"], x1=t1, y0=y, y1=y,
                          line=dict(color=TEAL, width=1.1), opacity=.55, layer="below")
        if per["t0"] > 0:
            fig.add_shape(type="line", x0=per["t0"], x1=per["t0"], y0=0, y1=1, yref="paper",
                          line=dict(color=LINE, width=1, dash="dot"), layer="below")
        fig.add_annotation(x=(per["t0"] + t1) / 2, y=0.94, yref="paper", yanchor="top",
                           text=f"{per['name']} · 목표 {per['lo']:.2f}–{per['hi']:.2f}",
                           showarrow=False, font=dict(color=TEAL, size=11, family=KRFONT))

    if env is not None:
        pt, pc, tt, tc = env
        fig.add_trace(go.Scatter(x=np.concatenate([pt, tt[::-1]]),
                                 y=np.concatenate([pc, tc[::-1]]),
                                 fill="toself", fillcolor="rgba(224,128,142,0.13)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=pt, y=pc, mode="lines", name="투여 직후 최고",
                                 line=dict(color=OX, width=1.2), opacity=.45 if zoomed else .8,
                                 hovertemplate="최고 %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=tt, y=tc, mode="lines", name="투여 직전 최저",
                                 line=dict(color=OX, width=1.8 if zoomed else 2.8),
                                 opacity=.6 if zoomed else 1.0,
                                 hovertemplate="최저 %{y:.2f}<extra></extra>"))
        bad = [(t_, c_) for t_, c_ in zip(tt, tc) if c_ < period_at(t_)["lo"] * 0.97]
        if bad:
            fig.add_trace(go.Scatter(x=[b[0] for b in bad], y=[b[1] for b in bad],
                                     mode="markers", name="목표 미달 시점",
                                     marker=dict(color=AMB, size=9, symbol="triangle-down",
                                                 line=dict(color=BG, width=1.5)),
                                     hovertemplate="목표 미달 %{y:.2f}<extra></extra>"))

    fig.add_trace(go.Scatter(x=grid, y=cs, mode="lines",
                             name="실제 곡선" if env is not None else "예측 농도",
                             line=dict(color=OX,
                                       width=(2.0 if zoomed else 1.1) if env is not None else 2.2),
                             opacity=(0.95 if zoomed else 0.40) if env is not None else 1.0,
                             hovertemplate="실제 %{y:.2f} IU/mL<extra></extra>"))
    if cp is not None:
        fig.add_trace(go.Scatter(x=grid, y=cp, mode="lines", name="보정 전 집단 예측",
                                 line=dict(color=GHOST, width=1.6, dash="dash"),
                                 hovertemplate="보정 전 %{y:.2f}<extra></extra>"))
    if obs:
        fig.add_trace(go.Scatter(x=[o[0] for o in obs], y=[o[1] for o in obs],
                                 mode="markers", name="실측 농도",
                                 marker=dict(color=INK, size=10, line=dict(color=BG, width=2)),
                                 hovertemplate="실측 %{y:.2f}<extra></extra>"))

    fig.update_layout(height=440, margin=dict(l=10, r=10, t=52, b=10),
                      paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family=KRFONT, color=MUTE, size=12),
                      hovermode="x unified",
                      hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE,
                                      font=dict(color=INK, family=KRFONT, size=12)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=MUTE, size=11, family=KRFONT)))
    fig.update_xaxes(title_text="투여 시작 후 시간 (h)", range=list(xr), gridcolor=LINE,
                     zeroline=False, showspikes=True, spikecolor=MUTE, spikethickness=1,
                     spikedash="dot", spikemode="across", ticksuffix=" h",
                     hoverformat=".1f",
                     rangeslider=dict(visible=True, thickness=0.10, bgcolor=PANEL,
                                      bordercolor=LINE, borderwidth=1, range=[0, horizon]))
    fig.update_yaxes(title_text="농도 (IU/mL)", range=[0, ymax], gridcolor=LINE,
                     zeroline=False, hoverformat=".2f")
    st.plotly_chart(fig, use_container_width=True,
                    config=dict(displaylogo=False, scrollZoom=True, doubleClick="reset",
                                modeBarButtonsToRemove=["select2d", "lasso2d"]))

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

    st.subheader("투여 시각과 용량")
    if mode == "bolus":
        ds = [d for d in doses if d["t"] <= horizon]
        rows = []
        for i, d in enumerate(ds):
            if i == 0:
                trough, judge = None, "—"
            else:
                seg = np.linspace(ds[i-1]["t"], d["t"], 80)
                trough = float(conc(P, doses, infs, seg).min())
                per = period_at(d["t"])
                judge = "⚠ 미달" if trough < per["lo"]*0.97 else "유지"
            rows.append({"회차": "로딩" if d.get("label") == "로딩" else str(i),
                         "투여 시각": f"{d['t']:.0f} h", 
                         "경과": f"{int(d['t'])//24}일 {int(d['t'])%24}시간",
                         "용량 (IU)": f"{d['amt']:,.0f}",
                         "IU/kg": f"{d['amt']/wt:.1f}",
                         "직전 최저 농도": "—" if trough is None else f"{trough:.2f}",
                         "판정": judge})
        df = pd.DataFrame(rows)
        total = sum(d["amt"] for d in ds)
    else:
        rows = [{"구간": f"{d['t']:.0f} h · 로딩", "속도 (IU/h)": "—",
                 "구간 총량 (IU)": f"{d['amt']:,.0f}"} for d in doses if d["t"] <= horizon]
        for f in infs:
            if f["t0"] >= horizon: continue
            t1 = min(f["t1"], horizon)
            rows.append({"구간": f"{f['t0']:.0f} – {t1:.0f} h",
                         "속도 (IU/h)": f"{f['rate']:,.0f}",
                         "구간 총량 (IU)": f"{f['rate']*(t1-f['t0']):,.0f}"})
        df = pd.DataFrame(rows)
        total = (sum(d["amt"] for d in doses if d["t"] <= horizon)
                 + sum(f["rate"]*(min(f["t1"], horizon)-f["t0"]) for f in infs if f["t0"] < horizon))
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(430, 36*len(df) + 42))
    st.caption(f"관찰 {horizon:.0f}시간 총 투여량 **{total:,.0f} IU** · {total/wt:,.0f} IU/kg"
               f"  ·  약값 어림 {total*627/10000:,.0f}만 원 (IU당 627원)")
    st.download_button("투여 계획 내려받기 (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="투여계획.csv", mime="text/csv")

    with st.expander("농도 표로 보기 (1시간 간격)"):
        th = np.arange(0, horizon + 1e-9, 1.0)
        ch = conc(P, doses, infs, th)
        st.dataframe(pd.DataFrame({
            "시간 (h)": th.astype(int), "농도 (IU/mL)": np.round(ch, 3),
            "구간": [period_at(t)["name"] for t in th],
            "판정": ["미달" if c < period_at(t)["lo"]*0.97 else
                     ("초과" if c > period_at(t)["hi"]*1.15 else "유지") for t, c in zip(th, ch)]}),
            use_container_width=True, hide_index=True, height=320)

    st.subheader("추정 파라미터")
    k = st.columns(4)
    for col, lab, val, u in zip(k, ["CL", "V1", "Q", "V2"], z, ["mL/h", "mL", "mL/h", "mL"]):
        col.metric(lab, f"{val:,.1f}", u, delta_color="off")
    st.caption(f"분포 반감기 t½α {np.log(2)/P['al']:.1f} h · 소실 반감기 t½β {np.log(2)/P['be']:.1f} h"
               + (f"  ·  개인 보정 폭 CL {(np.exp(eta[0])-1)*100:+.0f}% · V1 {(np.exp(eta[1])-1)*100:+.0f}%"
                  if eta is not None else ""))

st.divider()
st.caption("시뮬레이션 데이터로 학습한 재현 연구용 도구입니다. 실제 진료 판단에 사용할 수 없습니다. · "
           "Janssen et al. CPT:PSP 2022;11:934–945 · Hazendonk et al. Haematologica 2016;101(10):1159–1169")
