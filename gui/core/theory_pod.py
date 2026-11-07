#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         theory_pod.py
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
Theory views of the field models: PO (POD-Galerkin) and PS (POD state space). The explorer builds a small board with
the board designer's toolkit functions (genBoard, calcBoardMat, simBoard, romBoard), trains the POD on random loss
levels and tests on a pulse profile, so that both reduced models can be compared with the full finite-volume model.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st

from gui.core.style import table, show, line, fmt, SERIES
from gui.core.theory_lumped import intro, _get, _simd
from gui.views.tools import _common as cm
from gui.views.tools import sketch
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.genProfile import genProfile
from src.py.board.romBoard import romBoard, simRom
from src.py.numeric.fitN4sid import fitN4sid

#######################################################################################################################
# Settings
#######################################################################################################################
LAYOUT = {'view': 'xy', 'W': 40, 'L': 30, 'board': 'DBC Al₂O₃ on 3 mm Cu baseplate', 'h': 1000, 'Ta': 25, 'dx': 1.0,
          'chips': [{'name': 'T1', 'tech': 'Si IGBT', 'x': 12, 'y': 15, 'P': 40},
                    {'name': 'D1', 'tech': 'Si diode', 'x': 30, 'y': 12, 'P': 15}]}
PROTOCOLS = pd.DataFrame([
    (1, 'unweighted sum over the nodes', 'legacy only'),
    (2, 'Gauss-Legendre quadrature of the projection (pure Galerkin)', 'physics only'),
    (3, 'as 2 (legacy: relative element size)', 'physics only'),
    (4, '2 + steady-state load correction α', 'physics + 1 calibration'),
    (5, 'C_r and G_r fitted to the training data + α', 'data-calibrated (default)'),
    (6, 'as 5 with a prior on the projected matrices', 'data-calibrated, regularised'),
], columns=['protocol', 'reduced matrices', 'kind'])


#######################################################################################################################
# Helpers
#######################################################################################################################
@st.cache_data(show_spinner=False)
def _board():
    D = genBoard(LAYOUT)
    C, G, Bp = calcBoardMat(D)
    Ts = 1.0
    t = np.arange(0, 600.0, Ts)
    Ptr = np.column_stack([genProfile('random', c['P'], t, {'hold': 30}, seed=3 + i) for i, c in enumerate(D['chips'])])
    Pte = np.column_stack([genProfile('pulse', c['P'], t, {'period': 120 + 80 * i, 'duty': 0.5}, seed=i)
                           for i, c in enumerate(D['chips'])])
    Ttr, _ = simBoard(C, G, Bp, Ptr, Ts, 25, 2)
    Tte, _ = simBoard(C, G, Bp, Pte, Ts, 25, 2)
    V = D['wA'] * D['dx'] * D['dy']
    G0 = (G - sp.diags(D['hv'] * V)).tocsc()
    return {'D': D, 'C': C, 'G': G, 'G0': G0, 'Bp': Bp, 't': t, 'Ptr': Ptr, 'Pte': Pte, 'Ttr': Ttr, 'Tte': Tte,
            'Ts': Ts}


@st.cache_data(show_spinner=False)
def _po(K):
    b = _board()
    rom = romBoard(b['C'], b['G'], b['Bp'], b['Ttr'], 25, K)
    Phi = rom['Phi']
    th = b['Tte'] - 25
    proj = (th @ Phi) @ Phi.T
    Tp = simRom(rom, b['Pte'], b['Ts'], 25, 2)
    T0 = simRom(dict(rom, Gr=Phi.T @ (b['G0'] @ Phi)), b['Pte'], b['Ts'], 25, 2)
    return {'rom': rom, 'e': np.abs(Tp - b['Tte']).max(), 'e0': np.abs(T0 - b['Tte']).max(),
            'eP': np.abs(proj - th).max(), 'Tp': Tp}


@st.cache_data(show_spinner=False)
def _ps(K, Ks):
    b = _board()
    rom = romBoard(b['C'], b['G'], b['Bp'], b['Ttr'], 25, K)
    Phi = rom['Phi']
    atr = (b['Ttr'] - 25) @ Phi
    ahead = lambda P: np.vstack([P[1:], P[-1:]])                    # T(k) responds to P(k) (implicit time step)
    sys = fitN4sid([ahead(b['Ptr'])], [atr], int(Ks))
    ap = _simd(sys['A'], sys['B'], sys['C'], ahead(b['Pte']))
    lamG = np.sort(np.linalg.eigvals(np.linalg.solve(rom['Cr'], rom['Gr'])).real)
    z = np.linalg.eigvals(sys['A'])
    return {'e': np.abs(ap @ Phi.T - (b['Tte'] - 25)).max(), 'ap': ap, 'at': (b['Tte'] - 25) @ Phi,
            'tauG': 1 / lamG, 'tauS': np.sort(-b['Ts'] / np.log(np.clip(np.abs(z), 1e-12, 1 - 1e-12)))[::-1]}


def _mode_fig(D, phi, title):
    ny, nx = D['shape']
    fig = go.Figure(go.Heatmap(z=phi.reshape(ny, nx), x=D['x'] * 1e3, y=D['y'] * 1e3, colorscale='RdBu_r',
                               zmid=0, showscale=False))
    fig.update_layout(title=title, xaxis_title='x (mm)', yaxis_title='y (mm)', margin=dict(l=40, r=10, t=40, b=40))
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


#######################################################################################################################
# PO
#######################################################################################################################
def theory_po(cfg):
    intro("PO mode: POD modes from field snapshots, heat equation projected onto them (Galerkin) and optionally "
          "calibrated with the training data.")
    cm.question("How many spatial modes describe a temperature field, and why must the projection contain the "
                "cooling?")
    sketch.show(sketch.mor(41 * 31, int(_get(cfg, 'Mdl', 'pod', 'K', default=5)),
                           'random'))
    cm.equations([
        ("Heat equation (2D, board)", r"\rho c\,\partial_t T = \partial_x(k\,\partial_x T) + \partial_y(k_y\,"
                                      r"\partial_y T) - h_v\,(T - T_a) + Q", "h_v: distributed cooling of a board "
                                                                            "(W/m³K), k_y: anisotropic conductivity."),
        ("POD of the snapshots", r"\Theta = [\theta(t_1) \dots \theta(t_N)] = U\Sigma V^T,\quad \Phi = U_{:,1..K},"
                                 r"\quad E_K = \frac{\sum_{i\le K}\sigma_i^2}{\sum_i\sigma_i^2}", None),
        ("Galerkin projection (weak form, quadrature)",
         [r"C_{r,ij} = \int_\Omega \rho c\,\phi_i\phi_j\,d\Omega,\quad G_{r,ij} = \int_\Omega \left(k\,\partial_x"
          r"\phi_i\partial_x\phi_j + k_y\,\partial_y\phi_i\partial_y\phi_j + h_v\,\phi_i\phi_j\right) d\Omega",
          r"C_r\,\dot a = -G_r\,a + q,\quad q_i = \int_\Omega Q\,\phi_i\,d\Omega,\quad T = T_0 + \Phi a"],
         "The integrals are evaluated with Gauss-Legendre quadrature (calcPodQuad, calcPodProj)."),
    ], sym=[('θ, T_0', 'temperature rise at the nodes, reference field', 'K, °C'),
            ('Φ, φ_i', 'POD modes', '–'), ('a', 'modal coordinates', 'K'), ('σ_i', 'singular values', 'K'),
            ('C_r, G_r', 'reduced capacitance and conductance', 'Ws/K, W/K'),
            ('h_v', 'distributed heat sink', 'W/(m³ K)'), ('Q, q', 'heat source, projected heat source', 'W/m³, W')],
                 expanded=True)
    st.markdown("**Protocols** (Mdl › pod › protocol)")
    pr = int(_get(cfg, 'Mdl', 'pod', 'protocol', default=5))
    table(PROTOCOLS.assign(current=np.where(PROTOCOLS['protocol'] == pr, '◀ current', '')), hide_index=True)

    cm.section(None, "Explorer: POD-Galerkin on a DBC board with two chips")
    K = st.slider("Modes K", 1, 20, int(min(max(_get(cfg, 'Mdl', 'pod', 'K', default=5), 1), 20)), key='thPO|K')
    with st.spinner("Simulating the board and projecting …"):
        b = _board()
        r = _po(K)
    rise = (b['Tte'] - 25).max()
    rom = r['rom']
    cm.tiles([("Full model", f"{b['D']['geo'].shape[0]} nodes", "finite volumes, 1 mm grid"),
              ("Energy E_K", f"{100 * rom['E'][K - 1]:.4f} %", None),
              ("Galerkin error (test)", f"{fmt(r['e'])} K", f"{100 * r['e'] / rise:.2f} % of the rise"),
              ("Without the h_v term", f"{fmt(r['e0'])} K", "projection without the cooling")])
    tabs = st.tabs(["Modes", "Energy", "Error over K", "Test response"])
    with tabs[0]:
        c = st.columns(3)
        for i in range(min(3, K)):
            with c[i]:
                show(_mode_fig(b['D'], rom['Phi'][:, i], f"φ_{i + 1}"), height=260, key=f'thPO|m{i}')
    with tabs[1]:
        n = 20
        fig = go.Figure(go.Bar(x=np.arange(1, n + 1), y=rom['sig'][0:n] ** 2 / np.sum(rom['sig'] ** 2),
                               marker_color=SERIES[0]))
        fig.update_layout(xaxis_title='mode i', yaxis=dict(type='log', title='σ_i² / Σσ²'))
        show(fig, key='thPO|en')
    with tabs[2]:
        Ks = [1, 2, 3, 5, 8, 12, 16, 20]
        es = [_po(k) for k in Ks]
        fig = go.Figure([line(Ks, [e['e'] for e in es], 'Galerkin (with h_v)', 0, mode='lines+markers'),
                         line(Ks, [e['eP'] for e in es], 'best possible (projection of the truth)', 1,
                              mode='lines+markers', dash='dot'),
                         line(Ks, [e['e0'] for e in es], 'Galerkin without h_v', 3, mode='lines+markers')])
        fig.update_layout(xaxis_title='modes K', yaxis=dict(type='log', title='max. error on test (K)'))
        show(fig, key='thPO|eK')
    with tabs[3]:
        D = b['D']
        j = int(np.argmax(b['Tte'][-1]))
        fig = go.Figure([line(b['t'], b['Tte'][:, j], 'full model (hottest node)', 0, width=3),
                         line(b['t'], r['Tp'][:, j], f'POD-Galerkin K = {K}', 1, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='T (°C)')
        show(fig, key='thPO|resp')
        st.caption(f"Training: random loss levels; test: pulse trains of {', '.join(c['name'] for c in D['chips'])}.")
    cm.interpret([
        f"The first {K} modes hold {100 * rom['E'][K - 1]:.4f} % of the snapshot energy; the Galerkin model reproduces "
        f"the unseen pulse test with {fmt(r['e'])} K maximum error, close to the best a basis of {K} modes can do "
        f"({fmt(r['eP'])} K).",
        f"Without the cooling term h_v in G_r the error is {fmt(r['e0'])} K: the projected model has no heat sink and "
        "the temperature keeps rising. Board data exported from the board designer therefore store h_v, and the PO "
        "projection includes it (legacy = 0).",
        "Mode 1 is the mean heating pattern, the next modes describe how the chips heat differently. Calibrated "
        "protocols (4–6) compensate projection errors with the training data but can then not be extrapolated as "
        "safely as the pure Galerkin model (2)."])


#######################################################################################################################
# PS
#######################################################################################################################
def theory_ps(cfg):
    intro("PS mode: the same POD modes, but the dynamics of the modal coordinates are identified from data as a "
          "state-space model instead of being projected.")
    cm.question("When is an identified modal model better than the Galerkin projection, and how many states does "
                "it need?")
    cm.equations([
        ("Modal coordinates", r"a(k) = \Phi^T\left(T(k) - T_0\right),\quad q_i(k) = \int_\Omega Q(k)\,\phi_i\,d\Omega",
         None),
        ("Identified dynamics (fitN4sid + fitPem)", r"x(k+1) = A\,x(k) + B\,q(k),\quad a(k) = C\,x(k),\quad "
                                                    r"T(k) = T_0 + \Phi\,C\,x(k)",
         "K modes, K_s states; no material data or boundary conditions are needed, only snapshots and loads."),
    ], sym=[('a', 'modal coordinates (K)', 'K'), ('q', 'projected heat sources', 'W'), ('x', 'K_s states', '–'),
            ('A, B, C', 'identified matrices', '–')], expanded=True)
    cm.section(None, "Explorer: identify the modal dynamics of the DBC board")
    c = st.columns(2)
    K = c[0].slider("Modes K", 1, 12, int(min(max(_get(cfg, 'Mdl', 'pod', 'K', default=5), 1), 12)), key='thPS|K')
    Ks = c[1].slider("States K_s", 1, 10, int(min(max(_get(cfg, 'Mdl', 'ss', 'K', default=4), 1), 10)), key='thPS|Ks')
    with st.spinner("Identifying …"):
        b = _board()
        r = _ps(K, Ks)
        g = _po(K)
    rise = (b['Tte'] - 25).max()
    cm.tiles([("PS error (test)", f"{fmt(r['e'])} K", f"{100 * r['e'] / rise:.2f} % of the rise"),
              ("PO error (same K)", f"{fmt(g['e'])} K", "Galerkin projection"),
              ("States", f"{Ks} for {K} modes", None)])
    st.caption("Board with a 3 mm Cu baseplate, training on random loss levels, test on pulse trains. The fast modes "
               "(τ ≈ 0.1–1 s) settle within one sample of T_s = 1 s, so the temperature at sample k already responds "
               "to the loss of sample k; the input is therefore aligned one sample ahead for the identification "
               "(state-space model without feedthrough).")
    tabs = st.tabs(["Modal coordinates", "Time constants"])
    with tabs[0]:
        fig = go.Figure([line(b['t'], r['at'][:, i], f'a_{i + 1} true', i, width=3) for i in range(min(3, K))] +
                        [line(b['t'], r['ap'][:, i], f'a_{i + 1} PS', i, dash='dot') for i in range(min(3, K))])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='a (K)')
        show(fig, key='thPS|a')
    with tabs[1]:
        n = max(len(r['tauG']), len(r['tauS']))
        table(pd.DataFrame({'i': np.arange(1, n + 1),
                            'τ Galerkin (s)': np.r_[np.sort(r['tauG'])[::-1], [np.nan] * (n - len(r['tauG']))],
                            'τ identified (s)': np.r_[r['tauS'], [np.nan] * (n - len(r['tauS']))]}), hide_index=True)
    cm.interpret([
        f"With K = {K} modes and K_s = {Ks} states the identified model reaches {fmt(r['e'])} K on the pulse test "
        f"(Galerkin: {fmt(g['e'])} K).",
        "The identified model needs no material data and absorbs model errors of the snapshots (e.g. FEM meshes, "
        "unknown cooling), but it only knows the dynamics the training loads excited. Too few states merge time "
        "constants; more states than modes × time constants fit noise.",
        "The slowest time constants of both models should agree: they are set by the heat capacity of the board and "
        "the cooling."])
