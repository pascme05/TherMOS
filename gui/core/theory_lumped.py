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
@st.cache_data(show_spinner=False, max_entries=16)
def _sf_case(Rj, Rn, Rd, Rhs, exc, K, noise):
    """
    Reference: four chips of a full bridge (2 x 2) on a DBC, each with a 3-stage Cauer ladder (chip, DBC, heat sink);
    junction couplings Rn (neighbours) and Rd (diagonal), heat sink couplings Rhs. Profiles: single-chip steps or a
    full-bridge operation (high side and low side of each leg alternate); the coupled network with K stages is fitted
    (sfFit) to the steps or to the first 70 % of the operation and tested on the rest.
    """
    from src.py.model.SF.sfNet import sfNet
    from src.py.model.SF.sfFit import _fitNet
    M, Nr = 4, 3
    Ts, N = 0.1, 1200
    R = np.tile(np.array([0.25, 0.35, 0.4]) * Rj, (M, 1))
    tau = np.array([0.05, 0.8, 12.0])
    C = tau[None, :] / R
    pair = [(i, j) for i in range(M) for j in range(i + 1, M)]
    diag = {(0, 3), (1, 2)}
    g = []
    for k in range(Nr):
        for i, j in pair:
            if k == 0:
                g.append(1 / (Rd if (i, j) in diag else Rn))
            elif k == Nr - 1:
                g.append(1 / Rhs)
            else:
                g.append(1e-9)
    p = np.log(np.concatenate([R.ravel(), C.ravel(), np.asarray(g)]))
    rng = np.random.default_rng(42)
    t = np.arange(N) * Ts
    if exc == 'single-chip steps':
        P2 = []
        for i in range(M):
            P = np.zeros((N // M, M))
            P[:, i] = 20.0
            P2.append(P)
    else:
        P = np.zeros((N, M))
        per = int(4 / Ts)
        on = (np.arange(N) // per) % 2 == 0
        amp = 20 * (1 + 0.5 * np.sin(2 * np.pi * t / 60))
        P[:, 0], P[:, 3] = np.where(on, amp, 2), np.where(on, amp, 2)
        P[:, 1], P[:, 2] = np.where(on, 2, amp * 0.8), np.where(on, 2, amp * 0.8)
        P2 = [P[:int(0.7 * N)], P[int(0.7 * N):]]
    T0 = [None] * len(P2)
    Y2 = sfNet(p, M, Nr, P2[0:1], Ts)
    if exc != 'single-chip steps':
        [G, Cv, E] = sfNet(p, M, Nr)
        Yall = sfNet(p, M, Nr, [P], Ts)[0]
        Y2 = [Yall[:int(0.7 * N)], Yall[int(0.7 * N):]]
        T0 = [None, Yall[int(0.7 * N)]]
    else:
        Y2 = sfNet(p, M, Nr, P2, Ts)
    Ym = [y + noise * rng.standard_normal(y.shape) for y in Y2]
    n = len(P2) if exc == 'single-chip steps' else 1
    mdl = _fitNet(P2[0:n], Ym[0:n], [np.zeros_like(y) for y in Ym[0:n]], Ts, M, int(K), 400, 1e4, 1500)
    Yf = sfNet(mdl['p'], M, int(K), P2, Ts, [None if v is None else v for v in T0])
    [G, _, _] = sfNet(p, M, Nr)
    Rtrue = np.linalg.inv(G)[0::Nr][:, 0::Nr]
    return {'P2': P2, 'Y2': Y2, 'Ym': Ym, 'Yf': Yf, 'Rtrue': Rtrue, 'Rfit': mdl['Rth'], 'Ts': Ts, 'mdl': mdl,
            'test': 1 if exc != 'single-chip steps' else None}


def theory_sf(cfg):
    intro("SF mode: thermal network of M coupled heat sources (chips). Every chip has an RC ladder to the heat sink, "
          "the chips heat each other through coupling resistances.")
    cm.question("How strongly do neighbouring chips heat each other, and how is a coupled network found from "
                "measured or simulated profiles?")
    K = int(min(max(_get(cfg, 'Mdl', 'rc', 'K', default=2), 1), 3))
    sketch.show(sketch.coupled(4, K, ['T1', 'T2', 'T3', 'T4']),
                "Left: the four switches of a full bridge (high side HS and low side LS of the legs A and B) on one "
                "substrate; R_ij are the coupling resistances (neighbours solid, diagonal dashed). Right: the coupled "
                f"Cauer network of the SF mode with K = {K} stage(s) per chip (Mdl › rc › K). Stage 1 is the "
                "junction; the couplings of deeper stages describe heat that spreads through the substrate and the "
                "heat sink and reaches the neighbour later.")
    cm.equations([
        ("Nodal network", r"C\,\dot T = E\,P(t) - G\,T,\qquad T_{j} = [T]_{\text{junctions}},\qquad "
                          r"\Delta T_j(\infty) = R_{th}\,P,\quad R_{th} = [G^{-1}]_{jj}",
         "C diagonal (one capacitance per node), G symmetric: ladders plus couplings. E puts the loss of chip i into "
         "its junction node."),
        ("Conductance matrix", r"G = \sum_{i,k} \frac{1}{R'_{ik}}\,(e_{ik}-e_{i,k+1})(e_{ik}-e_{i,k+1})^T + "
                               r"\sum_{i<j,\,k} \frac{1}{R_{ij,k}}\,(e_{ik}-e_{jk})(e_{ik}-e_{jk})^T",
         "Every resistor adds its conductance to the two nodes it connects (the last ladder stage connects to "
         "T_ref). Positive R and C → G positive definite → the network is passive and stable."),
        ("Coupled Cauer network (selSF = 2): output-error fit with bounds",
         r"\min_{p}\sum_n \lVert T(n) - \hat T(n;p)\rVert^2,\quad p = \log[R'_{ik},\,C'_{ik},\,1/R_{ij,k}],"
         r"\quad p_0 - \ln b \le p \le p_0 + \ln b",
         "Works with any profiles (single-chip steps or all chips loaded). The log-parameters keep every R and C "
         "positive; the bounds b (Mdl › sf › bnd) stop poorly excited parameters from drifting. Couplings that end "
         "at the lower bound are negligible."),
        ("Exact simulation (sfNet)", r"S=C^{1/2},\; S^{-1}GS^{-1}=V\Lambda V^T:\quad z_k[n+1] = e^{-\lambda_k T_s}"
                                     r"z_k[n] + \frac{1-e^{-\lambda_k T_s}}{\lambda_k}\,b_k[n]",
         "Decoupled modes, exact for losses that are constant within a sample (zero-order hold)."),
        ("Nodal Foster (selSF = 1, MATLAB)", r"R_{th,ik} = \frac{\Delta T_i(\infty)}{P_k}\Big|_{\text{only } k "
                                             r"\text{ heated}}",
         "Needs one single-source experiment per chip; the ODE time constants differ from the fitted Foster pairs "
         "(F-F1). With fewer experiments than chips the coupled network is used."),
    ], sym=[('T, P', 'node temperature rises, chip losses', 'K, W'),
            ('C, G', 'capacitance and conductance matrix', 'Ws/K, W/K'),
            ("R'_ik, C'_ik", 'resistance / capacitance of stage k of chip i', 'K/W, Ws/K'),
            ('R_ij,k', 'coupling resistance between stage k of chips i and j', 'K/W'),
            ('R_th', 'steady-state resistance matrix of the junctions', 'K/W'),
            ('b', 'bound factor (Mdl › sf › bnd)', '–')], expanded=True)

    cm.section(None, "Explorer: full bridge with four chips — fit the coupled network to a known reference")
    c = st.columns(5)
    Rj = num(c[0], "R_th chip → heat sink (K/W)", 1.0, 'thSF|Rj', min_value=0.05)
    Rn = num(c[1], "Coupling neighbours R_n (K/W)", 4.0, 'thSF|Rn', min_value=0.1)
    Rd = num(c[2], "Coupling diagonal R_d (K/W)", 12.0, 'thSF|Rd', min_value=0.1)
    Rhs = num(c[3], "Coupling via heat sink (K/W)", 2.0, 'thSF|Rhs', min_value=0.1)
    noise = num(c[4], "Noise σ (K)", 0.02, 'thSF|n', min_value=0.0, max_value=1.0, step=0.01)
    c = st.columns([2, 1])
    exc = c[0].segmented_control("Training profiles", ['single-chip steps', 'full-bridge operation'],
                                 default='full-bridge operation', key='thSF|exc') or 'full-bridge operation'
    Kf = c[1].slider("Fitted stages K", 1, 3, 2, key='thSF|K')
    with st.spinner("Fitting the coupled network …"):
        r = _sf_case(float(Rj), float(Rn), float(Rd), float(Rhs), exc, int(Kf), float(noise))
    Rt, Rf = r['Rtrue'], r['Rfit']
    ev = r['test'] if r['test'] is not None else 0
    Y, Yf = r['Y2'][ev], r['Yf'][ev]
    mae = float(np.mean(np.abs(Y - Yf)))
    cm.tiles([("Self-heating R_th,11", f"{fmt(Rt[0, 0])} K/W", f"fitted {fmt(Rf[0, 0])} K/W"),
              ("Cross-heating R_th,21", f"{fmt(Rt[1, 0])} K/W", f"{100 * Rt[1, 0] / Rt[0, 0]:.0f} % of R_th,11 · "
                                                                 f"fitted {fmt(Rf[1, 0])}"),
              ("MAE " + ("test part" if r['test'] is not None else "chip 1 step"), f"{fmt(mae)} K",
               f"network with K = {Kf} stage(s)")])
    tabs = st.tabs(["Temperatures", "R_th matrix", "Fitted network", "Losses"])
    names = ['T1 (HS A)', 'T2 (LS A)', 'T3 (HS B)', 'T4 (LS B)']
    tt = np.arange(len(Y)) * r['Ts']
    with tabs[0]:
        fig = go.Figure()
        for i in range(4):
            fig.add_trace(line(tt, Y[:, i], names[i] + ' reference', i, width=2.5))
            fig.add_trace(line(tt, Yf[:, i], names[i] + ' network', i, dash='dot', showlegend=False))
        fig.update_layout(xaxis_title='t (s)', yaxis_title='ΔT (K)')
        show(fig, key='thSF|T', info="**Junction temperature rises.** Solid: reference, dotted: fitted coupled "
             "network. With single-chip steps the plot shows the step of chip 1: the other chips only heat up via "
             "the couplings (cross-heating starts flat).")
    with tabs[1]:
        cc = st.columns(2)
        with cc[0]:
            st.caption("Reference R_th (K/W): column k = temperature rises for 1 W in chip k")
            table(pd.DataFrame(Rt, index=names, columns=['P1', 'P2', 'P3', 'P4']).round(4))
        with cc[1]:
            st.caption("Fitted R_th = [G⁻¹]_jj of the coupled network (K/W)")
            table(pd.DataFrame(Rf, index=names, columns=['P1', 'P2', 'P3', 'P4']).round(4))
    with tabs[2]:
        m = r['mdl']
        table(pd.DataFrame(np.column_stack([m['Rlad'], m['Clad']]), index=names,
                           columns=[f"R'{k + 1} (K/W)" for k in range(Kf)] + [f"C'{k + 1} (Ws/K)" for k in range(Kf)])
              .round(4))
        rows = []
        for k in range(Kf):
            for i in range(4):
                for j in range(i + 1, 4):
                    rows.append({'stage': k + 1, 'pair': f"{i + 1}-{j + 1}", 'R_ij (K/W)': m['Rcpl'][k, i, j]})
        st.caption("Coupling resistances per stage (large values = negligible coupling)")
        table(pd.DataFrame(rows).round(3), hide_index=True)
    with tabs[3]:
        P = r['P2'][ev]
        fig = go.Figure([line(tt, P[:, i], names[i], i) for i in range(4)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)')
        show(fig, key='thSF|P')
    cm.interpret([
        f"Heating chip 1 with 1 W raises chip 2 by {fmt(Rt[1, 0])} K ({100 * Rt[1, 0] / Rt[0, 0]:.0f} % of its "
        f"self-heating): with all four chips loaded, chip 1 sees ΔT = Σ_k R_th,1k·P_k, i.e. "
        f"{fmt(Rt[0].sum() / Rt[0, 0])}× the self-heating for equal losses.",
        f"The fitted network reproduces the reference with MAE = {fmt(mae)} K. In full-bridge operation all chips are "
        "loaded at the same time — the Foster method of the MATLAB version cannot use these data, the coupled "
        "network can, because it fits all couplings at once (output error).",
        ("Full-bridge operation: the diagonal switches (HS A with LS B) always carry the same loss, so the data only "
         "show sums such as R_th,11 + R_th,14. The network predicts this operation well, but its individual R_th "
         f"entries are not unique (fitted R_th,11 = {fmt(Rf[0, 0])} vs. {fmt(Rt[0, 0])} K/W). To identify every "
         "coupling, add single-chip steps (one profile per chip) to the training data."
         if exc != 'single-chip steps' else
         "Single-chip steps excite every chip on its own: self- and cross-heating are separated and the fitted "
         f"R_th matrix matches the reference (R_th,21 = {fmt(Rf[1, 0])} vs. {fmt(Rt[1, 0])} K/W)."),
        "One stage (K = 1) gives a single time constant per chip; the fast chip and the slow heat sink dynamics need "
        "K ≥ 2. Couplings at deeper stages delay the cross-heating — the reason why the curves of the neighbours "
        "start flat."])
