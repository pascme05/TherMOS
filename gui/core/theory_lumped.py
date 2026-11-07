#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         theory_lumped.py
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Theory views of the lumped models: RC (Foster network fitted with the toolkit's output-error fit rcFit._fitSim and
simulated with rcSim), SS (state space identified with fitN4sid from a known 3-node network) and SF (nodal Foster
model C·dT/dt = P − G·T with self- and cross-heating). Each view: question, equations with symbols, an explorer on a
known reference system (so the identified model can be compared with the truth) and an interpretation.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import expm

from gui.core.style import table, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import sketch
from src.py.model.RC.rcSim import rcSim
from src.py.model.RC.rcFit import _fitSim
from src.py.numeric.fitN4sid import fitN4sid

#######################################################################################################################
# Settings
#######################################################################################################################
R_TRUE = np.array([0.02, 0.05, 0.08, 0.10])                                      # K/W, reference heat path
TAU_TRUE = np.array([0.01, 0.1, 1.0, 10.0])                                      # s


#######################################################################################################################
# Helpers
#######################################################################################################################
def num(c, label, value, key, **kw):
    """Number input of a theory explorer (key 'th…', not bound to the config)."""
    f = kw.pop('format', '%g')
    cast = int if f == '%d' else float
    kw = {k: cast(v) if k in ('min_value', 'max_value', 'step') else v for k, v in kw.items()}
    return c.number_input(label, value=cast(value), key=key, format=f, **kw)


def intro(text):
    st.caption(text + " The explorer uses a known reference system so that the identified model can be compared "
                      "with the truth; it does not change the configuration.")


def _get(cfg, *path, default=None):
    d = cfg
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return default if d is None else d


#######################################################################################################################
# RC
#######################################################################################################################
@st.cache_data(show_spinner=False)
def _rc_fit(kind, noise, Kmax):
    Ts, tEnd = 0.005, 40.0
    t = np.arange(0, tEnd, Ts)
    P = np.full(len(t), 10.0) if kind == 'step' else np.where((t % 8.0) < 4.0, 10.0, 0.0)
    T, _ = rcSim(R_TRUE, TAU_TRUE, P, Ts)
    Tm = T + np.random.default_rng(42).normal(0, noise, len(t))
    out = {}
    for K in range(1, Kmax + 1):
        p, _ = _fitSim([P], [Tm], Ts, K, 0.0, 400, 1e-10)
        Tf, _ = rcSim(p[0:K], p[K:], P, Ts)
        out[K] = {'R': p[0:K], 'tau': p[K:], 'T': Tf, 'rmse': float(np.sqrt(np.mean((Tf - T) ** 2)))}
    return t, P, T, Tm, out


def theory_rc(cfg):
    intro("RC mode: a Foster network of K parallel RC pairs in series is fitted to the measured temperature rise.")
    cm.question("How many RC pairs does the heat path need, and what do the fitted R_i and τ_i mean?")
    sketch.show(sketch.heat_path(int(_get(cfg, 'Mdl', 'rc', 'K', default=4))))
    cm.equations([
        ("Foster network", r"Z_{th}(t) = \sum_{k=1}^{K} R_k\left(1 - e^{-t/\tau_k}\right),\quad \tau_k = R_k C_k",
         "Thermal impedance: temperature rise per watt after a loss step."),
        ("Response to any loss profile (rcSim)", r"T_k(i) = \frac{2\tau_k - T_s}{2\tau_k + T_s}T_k(i-1) + "
                                                 r"\frac{R_k T_s}{2\tau_k + T_s}\left(P(i) + P(i-1)\right),\quad "
                                                 r"T(i) = T_{ref}(i) + \sum_k T_k(i)",
         "Trapezoidal (Tustin) update of every RC pair, exact for piecewise-linear losses."),
        ("Fit (rcFit, legacy = 0)", r"\min_{\log R_k,\,\log\tau_k}\; \sum_{profiles}\sum_i \left(T_{sim}(i) - "
                                    r"\Delta T_{meas}(i)\right)^2",
         "Output-error fit on the measured temperature rise; log parameters keep R and τ positive."),
    ], sym=[('R_k, τ_k, C_k', 'resistance, time constant, capacitance of RC pair k', 'K/W, s, Ws/K'),
            ('K', 'number of RC pairs (Mdl › rc › K)', '–'), ('T_s', 'sampling time', 's'),
            ('P', 'power loss', 'W'), ('T_ref', 'reference temperature (e.g. coolant)', '°C')], expanded=True)

    cm.section(None, "Explorer: fit a Foster network to a known heat path")
    c = st.columns(3)
    kind = c[0].segmented_control("Loss profile", ['step', 'pulses'], default='step', key='thRC|kind') or 'step'
    noise = num(c[1], "Measurement noise σ (K)", 0.02, 'thRC|noise', min_value=0.0, max_value=1.0, step=0.01)
    K = c[2].slider("RC pairs K", 1, 6, int(min(max(_get(cfg, 'Mdl', 'rc', 'K', default=3), 1), 6)), key='thRC|K')
    with st.spinner("Fitting K = 1 … 6 …"):
        t, P, T, Tm, fits = _rc_fit(kind, float(noise), 6)
    f = fits[K]
    cm.tiles([("R_th fitted", f"{fmt(f['R'].sum())} K/W", f"true {fmt(R_TRUE.sum())} K/W"),
              ("RMSE of the fit", f"{fmt(f['rmse'])} K", f"noise σ = {fmt(noise)} K"),
              ("Slowest τ", f"{fmt(f['tau'].max())} s", f"true {fmt(TAU_TRUE.max())} s")])
    tabs = st.tabs(["Z_th(t)", "Temperature", "Error over K", "Parameters"])
    tt = np.logspace(-3, 2, 300)
    zt = np.sum(R_TRUE[None, :] * (1 - np.exp(-tt[:, None] / TAU_TRUE[None, :])), axis=1)
    zf = np.sum(f['R'][None, :] * (1 - np.exp(-tt[:, None] / f['tau'][None, :])), axis=1)
    with tabs[0]:
        fig = go.Figure([line(tt, zt, 'true (4 RC pairs)', 0, width=3), line(tt, zf, f'fitted, K = {K}', 1,
                                                                              dash='dot')])
        fig.update_layout(xaxis=dict(type='log', title='t (s)'), yaxis_title='Z_th (K/W)')
        show(fig, key='thRC|z')
    with tabs[1]:
        n = slice(0, len(t), max(len(t) // 2000, 1))
        fig = go.Figure([line(t[n], Tm[n], 'measured (with noise)', 0, width=1),
                         line(t[n], f['T'][n], f'Foster K = {K}', 1, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='ΔT (K)')
        show(fig, key='thRC|T')
    with tabs[2]:
        e = [fits[k]['rmse'] for k in fits]
        fig = go.Figure(line(list(fits), e, 'RMSE vs. truth', 0, mode='lines+markers'))
        fig.update_layout(xaxis_title='RC pairs K', yaxis=dict(type='log', title='RMSE (K)'))
        show(fig, key='thRC|eK')
    with tabs[3]:
        table(pd.DataFrame({'pair': np.arange(1, K + 1), 'R_k (K/W)': f['R'], 'τ_k (s)': f['tau'],
                            'C_k = τ_k/R_k (Ws/K)': f['tau'] / f['R']}), hide_index=True)
    e = [fits[k]['rmse'] for k in fits]
    Kb = int(np.argmin(np.array(e) > 1.2 * min(e)) + 1)
    cm.interpret([
        f"The reference heat path has 4 time constants from {fmt(TAU_TRUE.min())} s to {fmt(TAU_TRUE.max())} s. With "
        f"K = {K} RC pairs the fit reaches an RMSE of {fmt(f['rmse'])} K; from K ≈ {Kb} on, more pairs do not "
        f"improve the fit noticeably (the rest is noise or time constants the profile does not excite).",
        "The total R_th is well determined by any K, the split into pairs is not unique: a Foster network is a fit, "
        "its inner nodes have no physical meaning (see Theory & Tools › Thermal networks for the Cauer form).",
        "A pulse profile excites the fast time constants better than a step; profiles shorter than the slowest τ "
        "cannot identify it. The RC mode fits the first input/output pair only."])


#######################################################################################################################
# SS
#######################################################################################################################
def _chain(n=3):
    """3-node thermal chain: inputs at node 1 and node 3, outputs T1 and T3, cooling at node 3."""
    Cn = np.array([2.0, 8.0, 40.0])
    g = np.array([0.5, 0.8])
    G = np.zeros((n, n))
    for i, gi in enumerate(g):
        G[i, i] += gi
        G[i + 1, i + 1] += gi
        G[i, i + 1] -= gi
        G[i + 1, i] -= gi
    G[2, 2] += 1.5
    A = -G / Cn[:, None]
    B = np.zeros((n, 2))
    B[0, 0], B[2, 1] = 1 / Cn[0], 1 / Cn[2]
    C = np.zeros((2, n))
    C[0, 0], C[1, 2] = 1.0, 1.0
    return A, B, C


def _disc(A, B, Ts):
    n = len(A)
    M = expm(np.block([[A, B], [np.zeros((B.shape[1], n + B.shape[1]))]]) * Ts)
    return M[0:n, 0:n], M[0:n, n:]


def _simd(A, B, C, u):
    x = np.zeros(len(A))
    y = np.zeros((len(u), C.shape[0]))
    for i in range(len(u)):
        y[i] = C @ x
        x = A @ x + B @ u[i]
    return y


@st.cache_data(show_spinner=False)
def _ss_id(order, noise):
    Ac, Bc, Cc = _chain()
    Ts = 1.0
    Ad, Bd = _disc(Ac, Bc, Ts)
    rng = np.random.default_rng(42)
    u = np.repeat(rng.uniform(0, 10, (60, 2)), 25, axis=0)
    y = _simd(Ad, Bd, Cc, u) + rng.normal(0, noise, (len(u), 2))
    sysA = fitN4sid([u], [y], list(range(1, 7)))
    sys = fitN4sid([u], [y], int(order))
    ut = np.zeros((600, 2))
    ut[50:, 0], ut[300:, 1] = 10.0, 5.0
    return {'Ad': Ad, 'Bd': Bd, 'C': Cc, 'sv': np.asarray(sysA['sv']), 'nAuto': int(sysA['n']), 'sys': sys,
            'ut': ut, 'yt': _simd(Ad, Bd, Cc, ut), 'yp': _simd(sys['A'], sys['B'], sys['C'], ut), 'Ts': Ts}


def theory_ss(cfg):
    intro("SS mode: a linear state-space model of order n is identified from the losses and temperatures (subspace "
          "method N4SID/MOESP, optionally refined by an output-error fit).")
    cm.question("Which model order does the data support, and how do the poles of the identified model relate to "
                "the thermal time constants?")
    cm.equations([
        ("Discrete state space", r"x(k+1) = A\,x(k) + B\,u(k),\qquad y(k) = C\,x(k)",
         "u: losses (all inputs), y: temperatures (all outputs), x: n internal states. Cross-coupling is included."),
        ("Subspace identification", r"Y_f = \Gamma_n X_f + H_n U_f,\quad \Gamma_n = \begin{bmatrix} C \\ CA \\ "
                                    r"\vdots \end{bmatrix},\quad \text{SVD}\Rightarrow n,\;\Gamma_n\Rightarrow A, C",
         "The singular values of the projected Hankel matrix show the order: a gap after σ_n."),
        ("Poles and time constants", r"z_i = \text{eig}(A),\quad \tau_i = -\frac{T_s}{\ln z_i}",
         "A stable thermal model has real poles 0 < z_i < 1."),
    ], sym=[('A, B, C', 'state, input and output matrix', '–'), ('n', 'model order (Mdl › ss › K)', '–'),
            ('Γ_n', 'extended observability matrix', '–'), ('z_i, τ_i', 'discrete poles, time constants', '–, s'),
            ('T_s', 'sampling time', 's')], expanded=True)
    cm.section(None, "Explorer: identify a 3-node chain (2 heat sources, 2 sensors)")
    c = st.columns(2)
    n = c[0].slider("Model order n", 1, 6, int(min(max(_get(cfg, 'Mdl', 'ss', 'K', default=3), 1), 6)), key='thSS|n')
    noise = num(c[1], "Sensor noise σ (K)", 0.05, 'thSS|noise', min_value=0.0, max_value=2.0, step=0.05)
    with st.spinner("Identifying …"):
        r = _ss_id(n, float(noise))
    zt = np.sort(np.linalg.eigvals(r['Ad']).real)
    zp = np.linalg.eigvals(r['sys']['A'])
    err = np.abs(r['yp'] - r['yt'])
    cm.tiles([("Order from the SVD gap", f"{r['nAuto']}", "true order 3"),
              ("Max. error (test steps)", f"{fmt(err.max())} K", f"rise {fmt(r['yt'].max())} K"),
              ("Time constants (true)", ', '.join(fmt(-r['Ts'] / np.log(z)) for z in zt) + " s", None)])
    tabs = st.tabs(["Singular values", "Poles", "Step test"])
    with tabs[0]:
        sv = r['sv'][0:12]
        fig = go.Figure(go.Bar(x=np.arange(1, len(sv) + 1), y=sv))
        fig.update_layout(xaxis_title='index i', yaxis=dict(type='log', title='singular value σ_i'))
        show(fig, key='thSS|sv')
    with tabs[1]:
        th = np.linspace(0, np.pi / 2, 50)
        fig = go.Figure([line(np.cos(th), np.sin(th), 'unit circle', 6, width=1, dash='dot'),
                         go.Scatter(x=zt, y=0 * zt, mode='markers', name='true', marker=dict(size=12,
                                                                                            symbol='circle-open')),
                         go.Scatter(x=zp.real, y=zp.imag, mode='markers', name=f'identified, n = {n}',
                                    marker=dict(size=9, symbol='x'))])
        fig.update_layout(xaxis_title='Re z', yaxis_title='Im z')
        show(fig, key='thSS|poles')
    with tabs[2]:
        t = np.arange(len(r['ut'])) * r['Ts']
        fig = go.Figure([line(t, r['yt'][:, j], f'true T{2 * j + 1}', j, width=3) for j in range(2)] +
                        [line(t, r['yp'][:, j], f'model T{2 * j + 1}', j, dash='dot') for j in range(2)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='ΔT (K)')
        show(fig, key='thSS|step')
    cm.interpret([
        f"The singular values drop after index {r['nAuto']}: the data support a model of this order (the reference "
        f"chain has 3 nodes, i.e. 3 time constants). With n = {n} the step test is reproduced with a maximum error "
        f"of {fmt(err.max())} K.",
        "Too small an order merges time constants (the slow tail or the fast rise is missed); too large an order "
        "fits noise and can produce complex or unstable poles. Poles close to 1 are slow modes, poles near 0 fast "
        "ones.",
        "Unlike the RC mode, one state-space model covers all inputs and outputs, so cross-heating (P2 heats T1) is "
        "part of the model."])


#######################################################################################################################
# SF
#######################################################################################################################
def theory_sf(cfg):
    intro("SF mode: nodal Foster model of M coupled heat sources; self-heating from Foster fits, coupling from the "
          "steady-state resistance matrix. Experimental (see the README, F-F1).")
    cm.question("How strongly do neighbouring heat sources heat each other, and what does the resistance matrix "
                "tell?")
    cm.equations([
        ("Nodal network", r"C\,\dot T = P(t) - G\,(T - T_{ref}),\quad G = R_{th}^{-1}",
         "One node per heat source; C from the Foster fits of the self-heating curves."),
        ("Resistance matrix", r"R_{th,ik} = \frac{\Delta T_i(\infty)}{P_k}\Big|_{\text{only } k \text{ heated}},"
                              r"\qquad \Delta T(\infty) = R_{th}\,P",
         "Diagonal: self-heating, off-diagonal: cross-heating. Needs one single-source experiment per node."),
    ], sym=[('T, P', 'node temperatures, losses', '°C, W'), ('C', 'node capacitances', 'Ws/K'),
            ('G, R_th', 'conductance and resistance matrix', 'W/K, K/W')], expanded=True)
    cm.section(None, "Explorer: two chips on a common heat sink")
    c = st.columns(4)
    R11 = num(c[0], "R_th,11 (K/W)", 0.5, 'thSF|R11', min_value=0.01)
    R22 = num(c[1], "R_th,22 (K/W)", 0.8, 'thSF|R22', min_value=0.01)
    k = c[2].slider("Coupling R_12/√(R_11 R_22)", 0.0, 0.9, 0.35, 0.05, key='thSF|k')
    tau = num(c[3], "Time constant τ (s)", 5.0, 'thSF|tau', min_value=0.01)
    R12 = k * np.sqrt(R11 * R22)
    Rth = np.array([[R11, R12], [R12, R22]])
    G = np.linalg.inv(Rth)
    Cn = tau * np.diag(G)
    t = np.linspace(0, 6 * tau, 400)
    P = np.array([10.0, 0.0])
    lam, V = np.linalg.eig(-G / Cn[:, None])
    x = np.array([V @ np.diag((np.exp(lam * ti) - 1) / lam) @ np.linalg.solve(V, P / Cn) for ti in t]).real
    cm.tiles([("Self-heating ΔT_1", f"{fmt(Rth[0, 0] * 10)} K", "P_1 = 10 W, P_2 = 0"),
              ("Cross-heating ΔT_2", f"{fmt(Rth[1, 0] * 10)} K", f"{100 * Rth[1, 0] / Rth[0, 0]:.0f} % of ΔT_1"),
              ("Both at 10 W: ΔT_1", f"{fmt((Rth @ [10, 10])[0])} K", "superposition R_th·P")])
    c = st.columns([2, 1])
    with c[0]:
        fig = go.Figure([line(t, x[:, 0], 'ΔT_1 (heated)', 0), line(t, x[:, 1], 'ΔT_2 (cross-heating)', 1)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='ΔT (K)')
        show(fig, key='thSF|T')
    with c[1]:
        table(pd.DataFrame(Rth, index=['T_1', 'T_2'], columns=['P_1', 'P_2']).round(4))
        st.caption("R_th (K/W): column k = temperatures when only source k is heated with 1 W.")
    cm.interpret([
        f"Heating chip 1 alone with 10 W raises chip 2 by {fmt(Rth[1, 0] * 10)} K — the coupling through the common "
        f"heat sink. With both chips at 10 W, chip 1 sees {fmt((Rth @ [10, 10])[0])} K instead of "
        f"{fmt(Rth[0, 0] * 10)} K.",
        "The cross-heating curve starts flat: the heat needs time to travel to the neighbour. The SF mode reproduces "
        "this with one Foster network per node and the coupling of G; the RC mode (single input) cannot.",
        "Note: in the implementation the branch time constants follow from C and G, so the dynamics can differ from "
        "the Foster fit (F-F1). The steady state R_th·P is exact."])
