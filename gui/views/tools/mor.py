#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         mor.py
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
MOR & POD: the model order reduction families of TherMOS and an interactive comparison of POD-Galerkin and modal
truncation on a 1D rod (full model from the Numerics page): mode shapes, singular value decay, error over the number
of modes for a training and a different test load.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools.numerics import rod
from src.py.tools.toolData import MATERIALS

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|mor"
FAMILIES = pd.DataFrame([
    ('RC', 'Foster network', 'one RC chain per output, fitted to a step response', 'lumped, physical', '1D'),
    ('SS', 'State space', 'identified linear system (A, B, C, D) of order K', 'black box, linear', '1D'),
    ('SF', 'Nodal Foster', 'matrix of Foster networks (self- and cross-heating)', 'lumped, physical', '1D'),
    ('PO', 'POD-Galerkin', 'POD modes of field snapshots, heat equation projected on them', 'projection', '2D/3D'),
    ('PS', 'POD state space', 'POD modes + identified modal dynamics', 'projection + data', '2D/3D'),
    ('ML', 'Machine learning', 'regression of T on (rolled) features of P', 'data driven', '1D'),
    ('DL', 'Deep learning', 'CNN/LSTM/… on windows of P', 'data driven', '1D'),
    ('PI', 'Physics-informed NN', 'network with an RC residual in the loss', 'hybrid', '1D'),
], columns=['mode', 'name', 'reduced model', 'kind', 'data'])


#######################################################################################################################
# Functions
#######################################################################################################################
def _sim(C, G, B, P, h):
    """Implicit Euler for C dx/dt = −G x + B P (P: Nt)."""
    Ainv = np.linalg.inv(np.diag(C) / h + G if C.ndim == 1 else C / h + G)
    Cm = np.diag(C) if C.ndim == 1 else C
    x = np.zeros(len(B))
    out = np.zeros((len(P), len(B)))
    for i in range(1, len(P)):
        x = Ainv @ (Cm / h @ x + B * P[i])
        out[i] = x
    return out


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("MOR & POD", "Model order reduction: from N node temperatures to a few states — the idea behind all "
                              "modes of TherMOS.")
    cm.question("How can a thermal model with thousands of nodes be replaced by a handful of states, and how many "
                "are needed?")
    cm.section(1, "Reduction families in TherMOS", "Every mode of TherMOS is one way of building the reduced model.",
               divider=False)
    table(FAMILIES, hide_index=True)

    cm.section(2, "POD-Galerkin vs. modal truncation", "1D rod (heated at x = 0, cooled at x = L) with N nodes. "
                                                       "POD modes are learnt from a training load, modal truncation "
                                                       "keeps the slowest eigenmodes of the model; both are tested "
                                                       "on a different load.")
    c = st.columns(4)
    mat = c[0].selectbox("Material", ['Cu', 'Al alloy (6063)', 'Si', 'AlN'], key=f"{K}|mat")
    N = cm.num(c[1], "Nodes N", 101, 'mor|N', format='%d', min_value=11, max_value=401)
    train = c[2].selectbox("Training load", ['step', 'pulse train', 'random'], key=f"{K}|tr")
    test = c[3].selectbox("Test load", ['pulse train', 'step', 'random', 'fast pulses'], key=f"{K}|te")
    k, rho, cp = MATERIALS[mat][0:3]
    L = 0.02
    C, G, b, x = rod(N, L, k, rho * cp, 5000.0)
    Ci = 1 / np.sqrt(C)
    lam, V = np.linalg.eigh(Ci[:, None] * G * Ci[None, :])
    tau = 1 / lam
    h = tau[0] / 200
    t = np.arange(0, 3 * tau[0], h)
    rng = np.random.default_rng(1)

    def load(kind):
        if kind == 'step':
            return np.full(len(t), 10.0)
        if kind == 'pulse train':
            return np.where((t % (tau[0] / 2)) < tau[0] / 4, 10.0, 0.0)
        if kind == 'fast pulses':
            return np.where((t % (tau[0] / 40)) < tau[0] / 80, 10.0, 0.0)
        return 10.0 * rng.uniform(0, 1, int(len(t) / 20) + 1)[np.arange(len(t)) // 20]

    Ptr, Pte = load(train), load(test)
    Xtr = _sim(C, G, b, Ptr, h)
    Xte = _sim(C, G, b, Pte, h)
    U, sig, _ = np.linalg.svd(Xtr.T, full_matrices=False)
    Vm = Ci[:, None] * V                                                   # eigenmodes of C⁻¹G (C-orthonormal)
    Ks = np.arange(1, 13)
    rise = max(np.abs(Xte).max(), 1e-12)
    rows = []
    for Kr in Ks:
        res = {}
        for lab, Phi in (('POD-Galerkin', U[:, 0:Kr]), ('Modal truncation', Vm[:, 0:Kr])):
            Cr, Gr, br = Phi.T @ (C[:, None] * Phi), Phi.T @ G @ Phi, Phi.T @ b
            Xr = _sim(Cr, Gr, br, Pte, h) @ Phi.T
            res[lab] = np.abs(Xr - Xte).max()
        rows.append({'K': Kr, **res})
    err = pd.DataFrame(rows)

    cm.section(None, "Result")
    E = np.cumsum(sig ** 2) / np.sum(sig ** 2)
    K99 = int(np.argmax(E >= 0.9999) + 1)
    kp = int(err['POD-Galerkin'].idxmin())
    cm.tiles([("Full model", f"{N} states", None), ("POD modes for 99.99 % energy", f"{K99}", None),
              ("POD error, K = 3", f"{fmt(err['POD-Galerkin'][2])} K", f"{100 * err['POD-Galerkin'][2] / rise:.2f} % "
                                                                       f"of the max. rise {fmt(rise)} K"),
              ("Modal error, K = 3", f"{fmt(err['Modal truncation'][2])} K", None)])
    tabs = st.tabs(["Error over K", "Mode shapes", "Singular values", "Test response"])
    with tabs[0]:
        fig = go.Figure([line(err['K'], err['POD-Galerkin'], 'POD-Galerkin', 0),
                         line(err['K'], err['Modal truncation'], 'Modal truncation', 1)])
        fig.update_layout(xaxis_title='number of modes K', yaxis=dict(type='log', title='max. error on test (K)'))
        show(fig, key=f"{K}|err")
    with tabs[1]:
        n = st.slider("Modes shown", 1, 6, 3, key=f"{K}|nm")
        fig = go.Figure([line(x * 1e3, U[:, i] * np.sign(U[0, i]), f"POD {i + 1}", i) for i in range(n)] +
                        [line(x * 1e3, Vm[:, i] / np.abs(Vm[:, i]).max() * np.abs(U[:, i]).max() * np.sign(Vm[0, i]),
                              f"eigen {i + 1}", i, dash='dot', width=1) for i in range(n)])
        fig.update_layout(xaxis_title='x (mm)', yaxis_title='mode shape (normalised)')
        show(fig, key=f"{K}|modes")
    with tabs[2]:
        fig = go.Figure(go.Bar(x=np.arange(1, 21), y=sig[0:20] ** 2 / np.sum(sig ** 2), marker_color=SERIES[0]))
        fig.update_layout(xaxis_title='mode i', yaxis=dict(type='log', title='energy σ_i² / Σσ²'))
        show(fig, key=f"{K}|sv")
    with tabs[3]:
        Kr = st.slider("K", 1, 12, 3, key=f"{K}|Kr")
        Phi = U[:, 0:Kr]
        Xr = _sim(Phi.T @ (C[:, None] * Phi), Phi.T @ G @ Phi, Phi.T @ b, Pte, h) @ Phi.T
        fig = go.Figure([line(t, Xte[:, 0], 'full model, x = 0', 0, width=3), line(t, Xr[:, 0], f'POD K = {Kr}', 1,
                                                                                    dash='dot'),
                         line(t, Xte[:, N // 2], 'full model, x = L/2', 2, width=3),
                         line(t, Xr[:, N // 2], f'POD K = {Kr}, L/2', 3, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='θ (K)')
        show(fig, key=f"{K}|resp")

    cm.equations([
        ("Projection", r"\theta \approx \Phi a,\quad \Phi^T C \Phi\,\dot a = -\Phi^T G \Phi\,a + \Phi^T b\,P",
         "Any basis Φ (N × K) reduces N equations to K; the choice of Φ is the reduction method."),
        ("POD basis", r"\min_\Phi \sum_t \lVert\theta(t) - \Phi\Phi^T\theta(t)\rVert^2 \;\Rightarrow\; \Phi = "
                      r"U_{:,1..K},\;\; \Theta = U\Sigma V^T", "Optimal for the training snapshots (energy E_K)."),
        ("Modal basis", r"G v_i = \lambda_i C v_i,\quad \Phi = [v_1 \dots v_K],\; \lambda_1 < \lambda_2 < \dots",
         "Slowest physical modes, independent of the load."),
    ], sym=[('θ', 'nodal temperature rise', 'K'), ('Φ', 'reduction basis', '–'), ('a', 'reduced states', 'K'),
            ('U, Σ', 'singular vectors and values of the snapshot matrix Θ', '–, K'),
            ('λ_i, v_i', 'eigenvalues (1/τ_i) and eigenvectors', '1/s, –')])
    cm.interpret([
        f"The POD needs **{K99} modes** for 99.99 % of the snapshot energy of the {train} load; with K = 3 the "
        f"error on the {test} load is {fmt(err['POD-Galerkin'][2])} K ({100 * err['POD-Galerkin'][2] / rise:.2f} % "
        f"of the rise), compared to {fmt(err['Modal truncation'][2])} K for the 3 slowest eigenmodes.",
        "POD is optimal for the loads it has seen: a training load that excites the fast dynamics (random, pulses) "
        "gives modes that also work for other loads, a pure step does not. Modal truncation needs no data but "
        "wastes modes on dynamics the load never excites. The smallest error is reached at K = "
        f"{err['K'][kp]}; beyond that the error is limited by the time step.",
        "In TherMOS the PO mode uses exactly this projection on 2D/3D FEM fields, the PS mode identifies the reduced "
        "matrices from data instead (robust to model mismatch), and the board designer produces the training data."])
