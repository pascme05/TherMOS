#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         numerics.py
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
Numerics: spatial discretisation and time integration of the heat equation on a 1D rod (finite volumes), with the
stability limit of the explicit Euler method, the accuracy of implicit Euler and Crank-Nicolson (trapezoidal rule,
used by the RC mode's rcSim) against the exact solution of the semi-discrete system, the stiffness of the system and
the convergence with the grid.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import sketch
from src.py.tools.toolData import MATERIALS

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|num"
METHODS = {'Explicit Euler': 0.0, 'Implicit Euler': 1.0, 'Crank-Nicolson': 0.5}


#######################################################################################################################
# Functions
#######################################################################################################################
def rod(N, L, k, rc, h, A=1e-4):
    """
    Vertex-centred FV model of a rod (length L, cross-section A): heat P enters at x = 0, convection h at x = L.
    Returns C (N), G (N x N), b (N), x (N).
    """
    x = np.linspace(0, L, N)
    dx = L / (N - 1)
    w = np.ones(N)
    w[[0, -1]] = 0.5
    C = rc * A * dx * w
    g = k * A / dx
    G = np.diag(np.r_[g, 2 * g * np.ones(N - 2), g]) - g * np.eye(N, k=1) - g * np.eye(N, k=-1)
    G[-1, -1] += h * A
    b = np.zeros(N)
    b[0] = 1.0
    return C, G, b, x


def exact(C, G, b, P, t):
    """Exact solution of C dθ/dt = −G θ + b P (step P at t = 0) by modal decomposition."""
    Ci = 1 / np.sqrt(C)
    lam, V = np.linalg.eigh(Ci[:, None] * G * Ci[None, :])
    beta = V.T @ (Ci * b) * P
    z = beta[None, :] / lam[None, :] * (1 - np.exp(-np.outer(t, lam)))
    return (z @ V.T) * Ci[None, :], lam


def theta(C, G, b, P, h, n, th):
    """θ-method: (C/h + θG) x+ = (C/h − (1−θ)G) x + b P."""
    A = np.diag(C / h) + th * G
    Bm = np.diag(C / h) - (1 - th) * G
    Ainv = np.linalg.inv(A)
    x = np.zeros(len(C))
    out = np.zeros((n + 1, len(C)))
    for i in range(1, n + 1):
        x = Ainv @ (Bm @ x + b * P)
        out[i] = x
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e8:
            out[i:] = np.nan
            break
    return out


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Numerics", "Spatial discretisation and time integration of the heat equation — why implicit "
                             "solvers are used and how fine the grid has to be.")
    cm.question("Which time step can I use, which integration method is accurate enough, and how many nodes does a "
                "thermal model need?")

    cm.section(1, "Rod model", "1D rod, heated with P at x = 0, cooled with h at x = L, insulated otherwise "
                               "(cross-section 1 cm²).", divider=False)
    c = st.columns(5)
    mat = c[0].selectbox("Material", ['Cu', 'Al alloy (6063)', 'Si', 'AlN', 'Al2O3 (96 %)', 'FR4 (in-plane)'],
                         key=f"{K}|mat")
    L = cm.num(c[1], "Length L (mm)", 20.0, 'num|L', min_value=1.0) * 1e-3
    h = cm.num(c[2], "h at x = L (W/m²K)", 5000.0, 'num|h', min_value=1.0)
    P = cm.num(c[3], "Power P (W)", 10.0, 'num|P', min_value=0.0)
    N = cm.num(c[4], "Nodes N", 21, 'num|N', format='%d', min_value=3, max_value=201)
    k, rho, cp = MATERIALS[mat][0:3]
    sketch.show(sketch.rod(N, L * 1e3, mat, h, P), "The heat equation ρc·∂T/∂t = k·∂²T/∂x² is discretised "
                "with finite volumes: every node i owns the volume around it (shaded half volumes at the ends), which "
                "stores heat (C_i) and exchanges it with its neighbours through the conductance G.")
    C, G, b, x = rod(N, L, k, rho * cp, h)
    lamMax = float(np.max(np.linalg.eigvalsh((1 / np.sqrt(C))[:, None] * G * (1 / np.sqrt(C))[None, :])))
    hCrit = 2 / lamMax
    tEnd = 3 * rho * cp * 1e-4 * L / (h * 1e-4) + 3 * rho * cp * L ** 2 / k
    tEnd = min(tEnd, 1e5)

    cm.section(2, "Time integration", "Same semi-discrete model, three integrators with step h; the reference is the "
                                      "exact modal solution.")
    c = st.columns([2, 1])
    f = c[0].select_slider("Time step h / h_crit (explicit Euler)", [0.25, 0.5, 0.9, 0.99, 1.01, 1.2, 2, 5, 20, 100],
                           value=0.9, key=f"{K}|f", help="h_crit = 2/λ_max is the stability limit of explicit Euler.")
    hstep = f * hCrit
    n = int(min(np.ceil(tEnd / hstep), 4000))
    t = np.arange(n + 1) * hstep
    ref, lam = exact(C, G, b, P, t)
    rows = []
    figs = [line(t, ref[:, 0], 'exact', 0, width=3)]
    for i, (name, th) in enumerate(METHODS.items(), 1):
        y = theta(C, G, b, P, hstep, n, th)
        e = np.nanmax(np.abs(y[:, 0] - ref[:, 0])) if np.all(np.isfinite(y)) else np.inf
        rows.append({'method': name, 'θ': th, 'stable': 'yes' if np.isfinite(e) else 'no (diverges)',
                     'max error at x = 0 (K)': e, 'error (% of rise)': 100 * e / max(ref[-1, 0], 1e-12)})
        figs.append(line(t, y[:, 0], name, i, dash='dot'))
    c[1].metric("h_crit = 2/λ_max", f"{fmt(hCrit)} s")
    c[1].metric("Steps to steady state", f"{n}", f"h = {fmt(hstep)} s", delta_color='off')
    fig = go.Figure(figs)
    fig.update_layout(xaxis_title='t (s)', yaxis_title='θ(x = 0) (K)',
                      yaxis_range=[-0.1 * ref.max(), 1.3 * ref.max()])
    show(fig, key=f"{K}|ti")
    table(pd.DataFrame(rows), hide_index=True)

    cm.section(3, "Stiffness and grid", "Eigenvalues of the semi-discrete system and convergence of θ(x = 0) at "
                                        "t = τ_slowest with the number of nodes.")
    c = st.columns(2)
    with c[0]:
        fig = go.Figure(go.Scatter(x=np.arange(1, len(lam) + 1), y=1 / np.sort(lam), mode='markers+lines',
                                   name='τ_i = 1/λ_i'))
        fig.update_layout(xaxis_title='mode i', yaxis=dict(type='log', title='time constant τ_i (s)'))
        show(fig, key=f"{K}|ev")
        st.caption(f"Stiffness ratio λ_max/λ_min = {fmt(lam.max() / lam.min())}: the fastest mode limits the "
                   f"explicit step, the slowest sets the simulated time.")
    with c[1]:
        tau1 = 1 / lam.min()
        Ns = np.array([3, 5, 9, 17, 33, 65, 129])
        v = np.array([exact(*rod(m, L, k, rho * cp, h)[0:3], P, np.array([tau1]))[0][0, 0] for m in Ns])
        vr = exact(*rod(513, L, k, rho * cp, h)[0:3], P, np.array([tau1]))[0][0, 0]
        dxs = L / (Ns - 1)
        err = np.abs(v - vr)
        fig = go.Figure([line(dxs * 1e3, err, '|θ_N − θ_ref|', 0),
                         line(dxs * 1e3, err[-1] * (dxs / dxs[-1]) ** 2, 'slope 2 (dx²)', 2, dash='dash')])
        fig.update_layout(xaxis=dict(type='log', title='dx (mm)'), yaxis=dict(type='log', title='error (K)'))
        show(fig, key=f"{K}|cv")

    cm.equations([
        ("Semi-discrete heat equation", r"C\,\dot\theta = -G\,\theta + b\,P", "Finite volumes in space: one ODE per "
                                                                             "node."),
        ("θ-method", r"\left(\tfrac{C}{h} + \vartheta G\right)\theta_{n+1} = \left(\tfrac{C}{h} - (1-\vartheta)G"
                     r"\right)\theta_n + bP", "ϑ = 0 explicit Euler, 1 implicit Euler, ½ Crank–Nicolson "
                                              "(trapezoidal/Tustin, as in rcSim)."),
        ("Stability of explicit Euler", r"h < h_{crit} = \frac{2}{\lambda_{max}(C^{-1}G)} \approx "
                                        r"\frac{\rho c\,\Delta x^2}{2k}", "Halving dx divides the allowed step by 4."),
        ("Accuracy", r"e_{explicit}, e_{implicit} = \mathcal{O}(h),\quad e_{CN} = \mathcal{O}(h^2),\quad "
                     r"e_{space} = \mathcal{O}(\Delta x^2)", None),
    ], sym=[('θ', 'temperature rise at the nodes', 'K'), ('C, G', 'capacitance vector, conductance matrix',
                                                          'Ws/K, W/K'),
            ('h', 'time step', 's'), ('λ', 'eigenvalues of C⁻¹G (1/time constants)', '1/s'),
            ('ϑ', 'implicitness of the θ-method', '–'), ('Δx', 'node distance', 'm')])
    cm.refs(['Incropera2007'])
    ok = [r['method'] for r in rows if r['stable'] == 'yes']
    cm.interpret([
        f"With {N} nodes in {mat} the fastest time constant is {fmt(1 / lamMax)} s while the rod needs "
        f"{fmt(1 / lam.min())} s (slowest τ) to settle: the system is stiff (ratio {fmt(lam.max() / lam.min())}). "
        f"Explicit Euler would need steps below {fmt(hCrit)} s, i.e. about {fmt(tEnd / hCrit)} steps for this run.",
        f"At h = {fmt(f)} h_crit the stable methods are: {', '.join(ok) if ok else 'none'}. Implicit Euler is stable "
        f"for any step but damps fast transients (first order), Crank–Nicolson is second order but can ring for very "
        f"large steps. TherMOS therefore uses implicit or exact (modal/Tustin) updates in all reduced models.",
        "Model order reduction attacks exactly this problem: the POD/SS/Foster models keep only the few slow modes "
        "that carry the energy, so the stiffness disappears and large steps become accurate."])
