#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         theory_pf_tools.py
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Section "The equations as tools" of the PF theory view: one tab per equation of the field network, each with the
equation, its symbols, two or three sliders, one chart, a caption that quotes the numbers and a short interpretation.
The tabs follow the derivation of the network from POD (same order as the paper):
    1 space × time (POD)   2 one lag (ZOH)   3 chips add up   4 physics residual   5 steady-state anchor
    6 why moving chips are hard for POD
Calculations: src/py/tools/theoryPF.py (nominal IMS board of the parametric study, 3453 nodes). Widget keys 'thPF|t…'.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import show, table, fmt, color, MUTED
from gui.views.tools import _common as cm
from src.py.tools import theoryPF as tp

NAMES = ['T1 (IGBT)', 'D1 (diode)', 'M1 (SiC)']
BWR = [[0, '#00549f'], [0.5, '#ffffff'], [1, '#cc071e']]


#######################################################################################################################
# Cached calculations
#######################################################################################################################
@st.cache_resource(show_spinner="Simulating 40 s of the board and computing its POD …")
def _pod():
    return tp.pod()


@st.cache_data(show_spinner=False, max_entries=32)
def _lag(tau, Ts, period):
    return tp.lag(tau, Ts, period)


@st.cache_data(show_spinner="Solving the steady state per chip …", max_entries=16)
def _sup(P):
    return tp.superposition(np.array(P))


@st.cache_data(show_spinner=False, max_entries=32)
def _res(eps, ftau):
    return tp.residual(eps, ftau)


@st.cache_data(show_spinner=False, max_entries=32)
def _drift(tauSlow, tauCap, amp):
    return tp.drift(tauSlow, tauCap, amp)


@st.cache_data(show_spinner="Solving the steady state of the sampled boards …", max_entries=8)
def _nw(n):
    return tp.nwidth(n)


def _heat(M, title, sym=False, zmax=None):
    z = np.asarray(M, float)
    kw = dict(colorscale=BWR, zmid=0) if sym else dict(colorscale='Viridis', zmin=0, zmax=zmax)
    fig = go.Figure(go.Heatmap(z=z, x=np.arange(z.shape[1]), y=np.arange(z.shape[0]), colorbar=dict(thickness=10),
                               **kw))
    fig.update_layout(title=title, xaxis_title='x (mm)', yaxis_title='y (mm)', yaxis=dict(scaleanchor='x'),
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def _note(lines):
    st.markdown("\n".join(f"- {s}" for s in lines))


#######################################################################################################################
# Section
#######################################################################################################################
def tools(n):
    cm.section(n, "The equations as tools",
               "Each tab takes one equation of the field network, lets you change its inputs and shows what it does on "
               "the IMS board of the parametric study (30 × 20 mm, three chips, 3453 nodes, cooling h = 1500 "
               "W/(m²K)). The tabs follow the derivation: POD splits the field into space × time, the time part is a "
               "set of lags, the chips add up, and the physics losses replace the simulation.")
    tabs = st.tabs(["① Space × time (POD)", "② One lag", "③ Chips add up", "④ Physics residual",
                    "⑤ Steady-state anchor", "⑥ Moving chips vs POD"])
    for tab, fn in zip(tabs, (_tPod, _tLag, _tSup, _tRes, _tDrift, _tNw)):
        with tab:
            fn()


# ==============================================================================
# 1 POD
# ==============================================================================
def _tPod():
    st.latex(r"\boldsymbol\theta(t)\approx\sum_{k=1}^{K}\boldsymbol\phi_k\,a_k(t),\qquad "
             r"\mathbf C_r\dot{\boldsymbol a}=-\mathbf G_r\boldsymbol a+\mathbf B_r\boldsymbol P,\quad "
             r"\mathbf C_r=\boldsymbol\Phi^\top\mathbf C\boldsymbol\Phi,\ \mathbf G_r=\boldsymbol\Phi^\top\mathbf G"
             r"\boldsymbol\Phi")
    st.caption("A temperature field that changes in time is written as a few fixed **patterns in space** φ_k "
               "(the POD modes, from the singular value decomposition of simulated snapshots) times **amplitudes "
               "over time** a_k(t). Projecting the finite-volume model onto the patterns (Galerkin) gives K equations "
               "instead of N = 3453. This is the starting point of the field network: it keeps the space × time form.")
    d = _pod()
    c = st.columns(2)
    K = int(c[0].slider("Modes K", 1, len(d['K']), 4, key='thPF|t1|K', help="Number of POD modes in the reduced "
                        "model. More modes: smaller error, more states."))
    k = int(c[1].slider("Show mode k", 1, K, min(int(st.session_state.get('thPF|t1|k', 1)), K), key='thPF|t1|k',
                        help="Which mode to draw (map and amplitude).")) if K > 1 else 1
    i = K - 1
    cm.tiles([("Captured energy E_K", f"{100 * d['E'][i]:.4f} %", "Σ σ_i² of the kept modes / Σ of all."),
              ("Best possible RMSE", f"{fmt(d['proj'][i])} K", "Projection of the simulated field onto the K modes "
               "(no model error): the limit of any model with these patterns."),
              ("Galerkin RMSE", f"{fmt(d['gal'][i])} K", "Reduced model with K states, all nodes, 40 s."),
              ("Max. junction error", f"{fmt(d['galJ'][i])} K", f"Largest junction error of the Galerkin model; the "
               f"maximum rise is {fmt(d['rise'])} K.")])
    c = st.columns([1, 1])
    with c[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d['K'], y=d['proj'], name='best possible (projection)', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=d['K'], y=d['gal'], name='Galerkin model', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=[K], y=[d['gal'][i]], mode='markers', marker=dict(color='red', size=12),
                                 name='current K'))
        fig.update_layout(xaxis_title='modes K', yaxis=dict(type='log', title='RMSE over all nodes (K)'))
        show(fig, height=330, key='thPF|t1|err', info="**Error over the number of modes** (log scale). The blue "
             "line is the best any model with K patterns can do, the orange line the Galerkin model. Both fall "
             "quickly: the field of three chips needs only a few patterns.")
    with c[1]:
        show(_heat(tp.layerMap(d['B'], d['Phi'][:, k - 1]), f"mode φ_{k} (copper layer)", sym=True), height=330,
             key='thPF|t1|map', info="**Spatial pattern of mode k** on the copper layer (red positive, blue "
             "negative, arbitrary scale). Mode 1 is the mean heating of all chips; higher modes describe how the "
             "chips heat differently and how heat spreads.")
    fig = go.Figure(go.Scatter(x=d['t'], y=d['a'][:, k - 1], name=f'a_{k}(t)'))
    fig.update_layout(xaxis_title='t (s)', yaxis_title=f'a_{k}(t) (K)')
    show(fig, height=220, key='thPF|t1|a', info="**Amplitude of mode k over time** for the random loss profile of "
         "the three chips (holds of 0.5–8 s).")
    st.caption(f"With K = {K} of {len(d['t'])} snapshot patterns the field keeps {100 * d['E'][i]:.3f} % of its "
               f"energy; the Galerkin model with {K} states reaches {fmt(d['gal'][i])} K RMSE over all 3453 nodes "
               f"and at most {fmt(d['galJ'][i])} K at a junction (maximum rise {fmt(d['rise'])} K). "
               f"Mode {k} has the singular value σ_{k}/σ₁ = {d['sv'][k - 1]:.2e}.")
    _note([f"**Few patterns suffice:** {int(np.argmax(d['gal'] < 0.5) + 1)} modes bring the error below 0.5 K. "
           "The heat equation smooths the field, so its singular values fall by decades.",
           "**But the patterns belong to this board.** A new cooling, material or chip position changes them — tab ⑥. "
           "The field network therefore learns the patterns as functions φ_k(x, z_c, μ) of the chip and the condition."])
    cm.refs(['Holmes2012'])


# ==============================================================================
# 2 Lag
# ==============================================================================
def _tLag():
    st.latex(r"\dot b=\lambda\,(\hat p-b),\qquad b[n]=a\,b[n-1]+(1-a)\,\hat p[n],\quad a=e^{-\lambda T_s},\quad"
             r"\tau=1/\lambda")
    st.caption("Diagonalising the Galerkin model turns every mode into a first-order lag of the chip loss: b follows "
               "the scaled loss p̂ = P/P_ref with the time constant τ. For a loss that is constant within each "
               "sample (zero-order hold, ZOH) the recursion on the right is **exact** for any T_s, and it is stable "
               "because 0 < a < 1. The network uses it for every space function, so a 30 min profile is as accurate "
               "as 40 s.")
    c = st.columns(3)
    tau = float(c[0].select_slider("Time constant τ (s)", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], 0.5,
                                   key='thPF|t2|tau', help="τ = 1/λ. Chip-level modes: 0.01–0.5 s, base plate and "
                                   "cooling: 1–10 s."))
    Ts = float(c[1].select_slider("Sampling time T_s (s)", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 0.05,
                                  key='thPF|t2|Ts', help="Time step of the loss profile (50 ms in the study)."))
    per = float(c[2].slider("Pulse period (s)", 1.0, 8.0, 4.0, 0.5, key='thPF|t2|per', help="On/off loss, 50 % duty."))
    d = _lag(tau, Ts, per)
    cm.tiles([("Ratio T_s/τ", fmt(Ts / tau), "Above 2, explicit Euler is unstable."),
              ("ZOH factor a", fmt(d['a']), "e^(−T_s/τ): always between 0 and 1."),
              ("Euler factor", fmt(d['aEuler']), "1 − T_s/τ: below −1 the Euler recursion grows without bound."),
              ("Max. error ZOH / Euler", f"{fmt(d['errZoh'])} / {fmt(d['errEuler']) if abs(d['errEuler']) < 1e3 else '∞'}",
               "Against the continuous lag (scale 0…1).")])
    eu = np.clip(d['euler'], -1.5, 2.5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['t'], y=d['p'], name='loss p̂', line=dict(color=MUTED, shape='hv')))
    fig.add_trace(go.Scatter(x=d['tf'], y=d['ref'], name='continuous lag', line=dict(color=color(0), width=3)))
    fig.add_trace(go.Scatter(x=d['t'], y=d['zoh'], name='exact ZOH (network)', mode='markers',
                             marker=dict(color=color(1), size=5)))
    fig.add_trace(go.Scatter(x=d['t'], y=eu, name='explicit Euler', line=dict(color=color(3), dash='dot')))
    fig.update_layout(xaxis_title='t (s)', yaxis=dict(title='b (–)', range=[-0.3, 1.4]))
    show(fig, height=330, key='thPF|t2|fig', info="**One lag over a pulse train.** Grey: the loss, blue: the exact "
         "continuous response, orange dots: the ZOH recursion of the network at the samples, red: explicit Euler "
         "(clipped to −1.5…2.5 when it diverges).")
    st.caption(f"τ = {fmt(tau)} s and T_s = {fmt(Ts)} s give a = {d['a']:.4f}. The ZOH samples lie on the continuous "
               f"curve (max. {fmt(d['errZoh'])}); explicit Euler deviates by "
               + (f"{fmt(d['errEuler'])}." if d['stable'] else "an unbounded amount: T_s > 2τ makes it unstable."))
    _note(["**Fast modes are no problem:** the network has space functions with τ = 0.02 s (die) next to 40 s (base), "
           "sampled with 50 ms. Only an exact integrator handles both with the same T_s.",
           "**Parallel in time:** the recursion is a convolution with (1 − a)·aⁿ and is evaluated with an FFT over the "
           "whole profile at once."])


# ==============================================================================
# 3 Superposition
# ==============================================================================
def _tSup():
    st.latex(r"\boldsymbol\theta_{ss}=\mathbf G^{-1}\mathbf B_p\boldsymbol P=\sum_{c=1}^{n_c}\boldsymbol\Theta_c\,"
             r"P_c,\qquad\boldsymbol\Theta=\mathbf G^{-1}\mathbf B_p\ \text{(K/W)}")
    st.caption("The board is linear in the losses: the field of all chips is the sum of the fields of each chip "
               "alone. The network uses this — every chip gets its own space functions and lags, and the sum Σ_c "
               "runs over the chips. A board with four chips needs no new network, only one more term.")
    c = st.columns(3)
    P = tuple(float(c[i].slider(f"Loss {NAMES[i]} (W)", 0.0, 60.0, float(tp.P_NOM[i]), 1.0, key=f'thPF|t3|P{i}'))
              for i in range(3))
    d = _sup(P)
    zmax = float(np.max(d['sum']))
    cm.tiles([(f"T_j {NAMES[i]}", f"{fmt(d['tj'][i])} K", "Steady junction rise (hottest die node) with all "
               "chips on.") for i in range(3)] + [("Sum − direct solve", f"{d['diff']:.1e} K", "Largest difference "
                                                    "between Σ Θ_c·P_c and G⁻¹·Bp·P over all nodes.")])
    cols = st.columns(4)
    for i in range(3):
        with cols[i]:
            show(_heat(d['maps'][i], f"{NAMES[i]} alone", zmax=zmax), height=240, key=f'thPF|t3|m{i}',
                 info="**Steady rise of one chip alone** (hottest node of every column, same colour scale for all).")
    with cols[3]:
        show(_heat(d['sum'], "sum = all chips", zmax=zmax), height=240, key='thPF|t3|sum',
             info="**Sum of the three maps** — identical to solving the board with all losses at once.")
    st.markdown("**Coupling matrix R (K/W)** — junction rise of every chip (row) per watt of every chip (column):")
    table(pd.DataFrame(np.round(d['R'], 3), index=NAMES, columns=NAMES))
    T1 = d['tjPart'][0]
    st.caption(f"T1 heats itself by {fmt(T1[0])} K; D1 and M1 add {fmt(d['tjPart'][1][0])} K and "
               f"{fmt(d['tjPart'][2][0])} K at T1. The sum of the maps equals the direct solve within "
               f"{d['diff']:.0e} K (rounding).")
    _note(["**Neighbours matter:** the off-diagonal entries of R are the share of a chip's temperature caused by "
           "the others; they fall with the distance — Theory & Tools › Design effects moves the chips.",
           "**Junction = maximum over the die:** the sum of the per-chip maxima is an upper bound, because the "
           "hottest node of each part need not coincide."])


# ==============================================================================
# 4 Residual
# ==============================================================================
def _tRes():
    st.latex(r"\boldsymbol r=\mathbf G^{-1}\mathbf C\,\dot{\hat{\boldsymbol\theta}}+\hat{\boldsymbol\theta}-"
             r"\boldsymbol\Theta\boldsymbol P=\boldsymbol e+\mathbf G^{-1}\mathbf C\,\dot{\boldsymbol e},\qquad"
             r"\boldsymbol e=\hat{\boldsymbol\theta}-\boldsymbol\theta")
    st.caption("The physics loss checks the heat balance of the finite-volume model for the prediction θ̂ — it needs "
               "the matrices of the board, but no simulated temperature. Multiplied by G⁻¹ the residual is a "
               "temperature: where the prediction is steady, r is the error itself; during transients it adds the "
               "error of the rate of change. Here the exact solution is disturbed on purpose.")
    c = st.columns(2)
    eps = float(c[0].slider("Amplitude error ε", -0.2, 0.2, 0.05, 0.01, key='thPF|t4|eps', help="θ̂ = (1 + ε)·θ: "
                            "wrong space functions (e.g. a wrong steady state)."))
    ft = float(c[1].slider("Time-constant factor", 0.5, 2.0, 1.0, 0.05, key='thPF|t4|ft', help="All time constants "
                           "× factor: wrong rates (correct steady state)."))
    d = _res(eps, ft)
    cm.tiles([("RMS error e", f"{fmt(d['rmsE'])} K", "θ̂ − θ at the hottest die node of T1."),
              ("RMS residual r", f"{fmt(d['rmsR'])} K", "Physics residual at the same node — computed without θ."),
              ("Slowest τ", f"{fmt(d['tau'][0])} s", "Slowest time constant of the board (nominal).")])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['t'], y=d['th'], name='exact θ', line=dict(color=color(0), width=3)))
    fig.add_trace(go.Scatter(x=d['t'], y=d['thh'], name='prediction θ̂', line=dict(color=color(1), dash='dash')))
    fig.add_trace(go.Scatter(x=d['t'], y=d['e'], name='error e', line=dict(color=color(3)), yaxis='y2'))
    fig.add_trace(go.Scatter(x=d['t'], y=d['r'], name='residual r', line=dict(color=color(2), dash='dot'),
                             yaxis='y2'))
    fig.update_layout(xaxis_title='t (s)', yaxis_title='rise at T1 (K)',
                      yaxis2=dict(title='e, r (K)', overlaying='y', side='right', showgrid=False))
    show(fig, height=330, key='thPF|t4|fig', info="**A 40 W pulse train on T1** (period 8 s). Left axis: exact and "
         "disturbed rise at T1; right axis: their difference e and the physics residual r, both in kelvin.")
    st.caption(f"ε = {_sgn(eps)} and time constants × {fmt(ft)} give an RMS error of {fmt(d['rmsE'])} K and an RMS "
               f"residual of {fmt(d['rmsR'])} K. "
               + ("A pure amplitude error appears in r almost one to one at the end of each pulse." if ft == 1.0 else
                  "Wrong rates leave the steady state right; r is large during the transients, where dθ̂/dt is "
                  "wrong."))
    _note(["**Residual ≈ error:** the loss r² is small exactly when the temperature error is small — without "
           "knowing θ. This is why boards without any simulation can train the network.",
           "**Exact solution → r = 0:** with ε = 0 and factor 1 the residual is zero up to rounding."])


def _sgn(eps):
    return f"{eps:+.2f}"


# ==============================================================================
# 5 Drift
# ==============================================================================
def _tDrift():
    st.latex(r"\boldsymbol r_{ss,c}=\sum_{k=1}^{K}\boldsymbol\varphi_k(\boldsymbol z_c,\boldsymbol\mu)-"
             r"\boldsymbol\Theta_c\frac{P_{ref}}{\Delta T},\qquad\lambda_k\ge\frac{1}{\tau_{cap}}")
    st.caption("The training profiles last 40 s. A space function with a time constant of several minutes barely "
               "moves in 40 s, so neither the data nor the physics loss can see whether its amplitude is right — "
               "but over a 30 min mission profile it builds up fully and the prediction drifts. Two fixes: the "
               "steady-state loss pins the sum of all space functions to the exact steady field Θ, and the bound "
               "τ_cap keeps every time constant visible within training.")
    c = st.columns(3)
    ts = float(c[0].slider("Time constant of the wrong mode (s)", 20.0, 1000.0, 300.0, 10.0, key='thPF|t5|ts'))
    cap = float(c[1].slider("Bound τ_cap (s)", 10.0, 1000.0, 80.0, 10.0, key='thPF|t5|cap', help="80 s in the study."))
    amp = float(c[2].slider("Wrong amplitude (K at 40 W)", 0.0, 10.0, 5.0, 0.5, key='thPF|t5|amp'))
    d = _drift(ts, cap, amp)
    cm.tiles([("Effective τ", f"{fmt(d['tau'])} s", "min(τ, τ_cap)."),
              ("Visible in 40 s", f"{100 * d['visible']:.0f} %", "1 − e^(−40 s/τ): how far the mode moves during a "
               "training profile."),
              ("RMS error, no anchor", f"{fmt(d['allNo'])} K", f"30 min; within the first 40 s only "
               f"{fmt(d['trainNo'])} K."),
              ("RMS error, with anchor", f"{fmt(d['allAn'])} K", f"30 min; within the first 40 s "
               f"{fmt(d['trainAn'])} K (visible to the data loss).")])
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=40, fillcolor=MUTED, opacity=0.25, line_width=0, annotation_text="training length",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=d['t'], y=d['noAnchor'], name='no steady-state loss', line=dict(color=color(3))))
    fig.add_trace(go.Scatter(x=d['t'], y=d['anchor'], name='with steady-state loss', line=dict(color=color(0))))
    fig.update_layout(xaxis_title='t (s)', yaxis_title='error at T1 (K)')
    show(fig, height=320, key='thPF|t5|fig', info="**Error that one wrong slow space function adds** over a 30 min "
         "profile (random levels held 10–120 s). Grey band: the length of a training profile. Red: the error builds "
         "up over minutes. Blue: with the steady-state anchor a fast space function carries the opposite amplitude, "
         "so the error shows up early (and is trained away) and vanishes in steady phases.")
    st.caption(f"With τ = {fmt(d['tau'])} s only {100 * d['visible']:.0f} % of the wrong amplitude shows within 40 s "
               f"(RMS {fmt(d['trainNo'])} K), but over 30 min the RMS error is {fmt(d['allNo'])} K. With the anchor it "
               f"is {fmt(d['allAn'])} K.")
    _note(["**Measured on the board family:** without the steady-state loss the 30 min profile drifts to an RMSE of "
           "8.5 K; with it 0.57 K (section 7 of this page).",
           "**τ_cap is a guard rail, not a model of the board:** the slowest true time constant here is 8 s at "
           "h = 500 W/(m²K), so a bound of 80 s costs no accuracy."])


# ==============================================================================
# 6 n-width
# ==============================================================================
def _tNw():
    st.latex(r"\min_{\boldsymbol\Phi}\ \frac{\lVert\mathbf S-\boldsymbol\Phi\boldsymbol\Phi^\top\mathbf S\rVert}"
             r"{\lVert\mathbf S\rVert}=\sqrt{\frac{\sum_{i>K}\sigma_i^2}{\sum_i\sigma_i^2}}")
    st.caption("A parametric reduced model (pMOR) uses one POD basis for all designs. That works when a parameter "
               "only scales entries of G or C (cooling, conductivities): the fields change their amplitude, not "
               "their shape. A chip that moves carries its hot spot with it — the fields have a different shape for "
               "every position, and a linear basis needs many patterns (slow decay of the Kolmogorov n-width).")
    n = int(st.slider("Number of designs", 8, 40, 24, 4, key='thPF|t6|n', help="Boards per sweep: h log-spaced over "
                      "500–5000 W/(m²K), or T1 at random valid positions."))
    d = _nw(n)
    cm.tiles([("Modes for 1 % error, cooling h", str(d['h']['k1']), "Steady fields of T1 (40 W) for the h sweep."),
              ("Modes for 1 % error, position", str(d['x']['k1']), f"Steady fields with T1 at {n} positions."),
              ("Modes for 0.1 %, h / position", f"{d['h']['k01']} / {d['x']['k01']}", "Relative projection error "
               "(Frobenius norm).")])
    fig = go.Figure()
    k = np.arange(1, n + 1)
    fig.add_trace(go.Scatter(x=k, y=d['h']['err'], name='cooling h varies', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=k, y=d['x']['err'], name='chip T1 moves', mode='lines+markers'))
    fig.add_hline(y=0.01, line=dict(color=MUTED, dash='dot'), annotation_text='1 %')
    fig.update_layout(xaxis_title='modes K', yaxis=dict(type='log', title='relative error of the best K-mode basis'))
    show(fig, height=320, key='thPF|t6|fig', info="**How many POD modes a set of designs needs.** Each curve is the "
         "error of the best K-mode basis for its snapshot set (log scale). Fast decay: one small basis fits all "
         "designs (pMOR works). Slow decay: every design needs its own patterns.")
    st.caption(f"{n} designs each: the cooling sweep needs {d['h']['k1']} modes for 1 %, the moving chip "
               f"{d['x']['k1']} — almost one per design.")
    _note(["**Why the network uses coordinates relative to the chip:** with x − x_c as input, the space functions "
           "move with the chip instead of being re-learned for every position.",
           "**Krylov MOR** avoids the problem by building a new basis from the matrices of every design — exact, but "
           "it needs the operator and a factorisation each time; the network needs neither at prediction time."])
    cm.refs(['Benner2015', 'Ohlberger2016', 'Odabasioglu1998'])
