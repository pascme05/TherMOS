#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Thermal Modelling, Model Order Reduction
# File:         numerics_rom
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
Numerics of the reduced models (Tools › Numerics, steps 4-6):
4) Quadrature weights W of the Galerkin projection (calcPodQuad): Gauss-Legendre points and weights per axis, their
   product in 2D/3D, and the accuracy compared to the node grid (trapezoidal rule)
5) Relative element size J (jacobianMesh, legacy = 1 only): what MATLAB multiplied into the integrals and why the
   corrected version uses J = 1
6) Calibration of C_th and G_th (optPodMat protocol 5 / 6, SF bounds): which entries the data determine, how the
   truncation rcond, the prior weight λ and the bounds keep the fit physical
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gui.core.style import table, show, line, fmt, color, TEMP, INK2
from gui.views.tools import _common as cm
from src.py.numeric.gaussLegendre import gaussLegendre
from src.py.numeric.jacobianMesh import jacobianMesh

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|numr"


#######################################################################################################################
# Functions
#######################################################################################################################
def _mode(x, L, i, kind):
    """1D test modes on [0, L]: cosine modes (Neumann) or a localised hot spot mode."""
    if kind == 'smooth (cosine modes)':
        return np.cos(i * np.pi * x / L)
    return np.exp(-((x - 0.3 * L) / (0.04 * L * (1 + 0.5 * i))) ** 2)


def _calib(a, q, Ts, C0, G0, mode, rcond, lam):
    """Equation-error fit of C, G as optPodMat (protocol 5: min-norm correction, 6: prior λ)."""
    da = np.gradient(a, Ts, axis=0)
    Phi = np.hstack([da, a])
    X0 = np.vstack([C0.T, G0.T])
    if mode == 5:
        X = X0 + np.linalg.lstsq(Phi, q - Phi @ X0, rcond=rcond)[0]
    else:
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        X = np.linalg.solve(A, Phi.T @ q + lam * X0)
    n = C0.shape[0]
    s = np.linalg.svd(Phi, compute_uv=False)                          # what lstsq truncates with rcond
    return [X[:n].T, X[n:].T, s]


def _sim(C, G, q, Ts):
    """Implicit Euler of C·da/dt + G·a = q."""
    M = np.linalg.inv(C / Ts + G)
    a = np.zeros((len(q), C.shape[0]))
    for i in range(1, len(q)):
        a[i] = M @ (C / Ts @ a[i - 1] + q[i])
    return a


#######################################################################################################################
# Sections
#######################################################################################################################
def quadrature():
    cm.section(4, "Quadrature weights W of the Galerkin projection",
               "The reduced matrices are integrals of products of modes. POD (PO mode) evaluates them with "
               "Gauss–Legendre quadrature on the rectangle/cuboid of the data (calcPodQuad).")
    cm.equations([
        ("Reduced matrices", [r"C_{r,ij} = \int_\Omega \rho c\,\varphi_i\varphi_j\,d\Omega,\qquad "
                              r"G_{r,ij} = \int_\Omega k\,\nabla\varphi_i\cdot\nabla\varphi_j\,d\Omega,\qquad "
                              r"q_{r,i}(t) = \int_\Omega q\,\varphi_i\,d\Omega"],
         "Galerkin projection of ρc·∂T/∂t = ∇·(k∇T) + q onto the modes φ_i."),
        ("Quadrature", r"\int_\Omega f\,d\Omega \approx \sum_q W_q\,f(\mathbf{x}_q),\qquad W_q = w_{x,i}\,w_{y,j}"
                       r"\,(w_{z,l}),\qquad \sum_q W_q = |\Omega|",
         "One Gauss–Legendre rule per axis with as many points as grid lines; W is the product of the 1D weights."),
        ("Mapping to the points", r"f(\mathbf{x}_q) = R_q\,M_g\,f_{nodes}",
         "M_g: scattered nodes → regular grid (linear), R_q: grid → quadrature points (multilinear)."),
    ], sym=[('φ_i', 'POD mode i', '–'), ('W_q', 'quadrature weight of point q (area / volume it represents)',
                                          'm², m³'),
            ('w_x,i', '1D Gauss–Legendre weight', 'm'), ('ρc, k', 'volumetric heat capacity, conductivity',
                                                         'J/m³K, W/mK')])
    c = st.columns(4)
    L = cm.num(c[0], "Length L (mm)", 30.0, 'numr|L', min_value=1.0)
    n = cm.num(c[1], "Points n per axis", 8, 'numr|n', format='%d', min_value=2, max_value=60)
    kind = c[2].selectbox("Mode shape", ['smooth (cosine modes)', 'localised hot spot'], key=f"{K}|kind")
    ij = c[3].selectbox("Integral", ['C₁₂ = ∫φ₁φ₂', 'C₂₂ = ∫φ₂²', 'G₂₂ = ∫(φ₂′)²'], key=f"{K}|ij")
    x, w = gaussLegendre(int(n), 0.0, L)
    xf = np.linspace(0, L, 20001)

    def f(xx):
        if ij.startswith('G'):
            d = 1e-4 * L
            return ((_mode(xx + d, L, 2, kind) - _mode(xx - d, L, 2, kind)) / (2 * d)) ** 2
        i1 = 1 if ij.startswith('C₁₂') else 2
        return _mode(xx, L, i1, kind) * _mode(xx, L, 2, kind)
    ref = float(np.trapezoid(f(xf), xf))
    rows = []
    for m in range(2, 41):
        xg, wg = gaussLegendre(m, 0.0, L)
        xu = np.linspace(0, L, m)
        rows.append({'n': m, 'Gauss–Legendre': abs(float(np.sum(wg * f(xg))) - ref),
                     'node grid (trapezoid)': abs(float(np.trapezoid(f(xu), xu)) - ref)})
    df = pd.DataFrame(rows)
    sc = max(abs(ref), 1e-12)
    cc = st.columns([3, 2])
    with cc[0]:
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(line(xf, f(xf), 'integrand f(x)', 0), secondary_y=False)
        fig.add_trace(go.Bar(x=x, y=w, name='weights w_i (mm)', marker_color=color(1), opacity=0.5,
                             width=L / n / 6), secondary_y=True)
        fig.add_trace(go.Scatter(x=x, y=f(x), mode='markers', name='points x_i', marker=dict(color=color(1), size=9)),
                      secondary_y=False)
        fig.update_xaxes(title_text='x (mm)')
        fig.update_yaxes(title_text='f(x)', secondary_y=False)
        fig.update_yaxes(title_text='w (mm)', secondary_y=True, rangemode='tozero')
        show(fig, height=330, key=f"{K}|gl", info="**Gauss–Legendre points and weights.** The points cluster towards "
             "the ends, the weights (bars) are smaller there; Σw = L. An n-point rule integrates polynomials up to "
             "degree 2n − 1 exactly.")
    with cc[1]:
        fig = go.Figure([line(df['n'], np.maximum(df['Gauss–Legendre'] / sc, 1e-16), 'Gauss–Legendre', 1,
                              mode='lines+markers'),
                         line(df['n'], np.maximum(df['node grid (trapezoid)'] / sc, 1e-16), 'node grid (trapezoid)',
                              0, mode='lines+markers')])
        fig.add_vline(x=n, line=dict(color=INK2, dash='dot'))
        fig.update_layout(xaxis_title='points per axis n', yaxis=dict(type='log', title='relative error'))
        show(fig, height=330, key=f"{K}|glerr", info="**Error of the integral** relative to a fine reference. "
             "Gauss–Legendre converges exponentially for smooth modes; for a localised hot spot both need points "
             "inside the spot.")
    e1 = float(df.loc[df['n'] == n, 'Gauss–Legendre'].iloc[0]) / sc
    e2 = float(df.loc[df['n'] == n, 'node grid (trapezoid)'].iloc[0]) / sc
    W2 = np.outer(w, w)
    cm.tiles([("Exact value", fmt(ref), None), ("Error GL / grid", f"{e1:.1e} / {e2:.1e}", None),
              ("Σ W_q in 2D", f"{fmt(W2.sum())} mm² = L²", "Product weights of a square L × L."),
              ("W_max / W_min (2D)", fmt(W2.max() / W2.min()), "Corner points get the smallest weight.")])
    cm.interpret([
        f"With n = {n} points the {kind.split(' (')[0]} integral {ij.split(' =')[0]} has a relative error of "
        f"{e1:.1e} with Gauss–Legendre and {e2:.1e} on an equally spaced grid. The weights W_q are the area (2D) or "
        f"volume (3D) that every quadrature point represents, so ρc and k are weighted by the space they occupy.",
        "TherMOS uses one point per grid line (dx, dy, dz of the data): finer data grids give both more points and "
        "better integrals. A localised hot spot needs a grid that resolves it — otherwise the modes are already "
        "wrong, whatever the quadrature."])


def element_size():
    cm.section(5, "Relative element size J (legacy only)",
               "The MATLAB version multiplied the integrands with the relative size of the FE elements around each "
               "quadrature point (jacobian2D/3D). It is not the Jacobian of a coordinate transformation.")
    c = st.columns(4)
    N = cm.num(c[0], "FE nodes", 600, 'numr|N', format='%d', min_value=50, max_value=4000)
    ref = cm.num(c[1], "Refinement at the chip (×)", 4.0, 'numr|ref', min_value=1.0, max_value=20.0)
    th = cm.num(c[2], "Threshold th", 0.001, 'numr|th', min_value=0.0, max_value=0.5)
    nq = cm.num(c[3], "Quadrature points per axis", 30, 'numr|nq', format='%d', min_value=5, max_value=80)
    rng = np.random.default_rng(1)
    n1 = int(N / (1 + ref * 0.25))
    p = np.vstack([rng.uniform(0, 1, (n1, 2)),
                   0.5 + rng.normal(0, 0.08, (int(N) - n1, 2))]).clip(0, 1) * [40e-3, 30e-3]
    p = np.vstack([p, [[0, 0], [40e-3, 0], [0, 30e-3], [40e-3, 30e-3]]])
    qx, _ = gaussLegendre(int(nq), 0.0, 40e-3)
    qy, _ = gaussLegendre(int(nq), 0.0, 30e-3)
    J = jacobianMesh(p, [qx, qy], th)
    cc = st.columns(2)
    with cc[0]:
        fig = go.Figure(go.Scatter(x=p[:, 0] * 1e3, y=p[:, 1] * 1e3, mode='markers', marker=dict(size=3,
                                                                                              color=color(0))))
        fig.update_layout(title="FE nodes of the data", xaxis_title='x (mm)', yaxis_title='y (mm)',
                          yaxis=dict(scaleanchor='x'))
        show(fig, height=330, key=f"{K}|mesh", info="**FE mesh** of a field data set, refined at a chip in the "
             "centre (as COMSOL/ANSYS meshes are).")
    with cc[1]:
        fig = go.Figure(go.Heatmap(z=J, x=qx * 1e3, y=qy * 1e3, colorscale='RdBu_r', zmid=1,
                                   colorbar=dict(title='J')))
        fig.update_layout(title="Relative element size J at the quadrature points", xaxis_title='x (mm)',
                          yaxis_title='y (mm)', yaxis=dict(scaleanchor='x'))
        show(fig, height=330, key=f"{K}|J", info="**J = element area / median element area.** Blue (J < 1) where "
             "the mesh is fine, red where it is coarse. Values within 1 ± th are set to one.")
    st.latex(r"C_{r,ij}^{legacy} = \sum_q W_q\,J_q\,\rho c\,\varphi_i\varphi_j \quad\text{vs.}\quad "
             r"C_{r,ij} = \sum_q W_q\,\rho c\,\varphi_i\varphi_j")
    cm.interpret([
        f"J ranges from {fmt(np.min(J))} to {fmt(np.max(J))}: with J the integrals depend on how the FE program "
        f"meshed the part, not only on the physics — the refined chip region (where the heat is) gets a weight of "
        f"about {fmt(np.median(J[J < 1]) if np.any(J < 1) else 1)}, i.e. its heat capacity is under-counted.",
        "The quadrature weights W already account for the area of each point, so the corrected version uses J = 1 "
        "(finding F-P8). J is only computed for legacy = 1 with protocol > 2 to reproduce the MATLAB results."])


def calibration():
    cm.section(6, "Calibrating C_th and G_th: bounds and regularisation",
               "PO protocols 5/6 fit the reduced matrices to the projected training data (optPodMat); the SF mode "
               "fits a nodal network within bounds. Both must stay physical when the data cannot determine every "
               "entry.")
    cm.equations([
        ("Equation error", r"\min_{C,G}\ \big\| q - C\,\dot a - G\,a \big\|^2,\qquad \Phi = [\dot a,\ a]",
         "Linear least squares per row; a: modal coordinates, q: projected losses."),
        ("Protocol 5: minimum-norm correction", r"X = X_0 + \Phi^{+}_{rcond}\,(q - \Phi X_0)",
         "Starts at the projected matrices X₀ = [C₀; G₀] and only changes what the data determine: singular values of "
         "Φ below rcond·σ_max are dropped."),
        ("Protocol 6: prior", r"\min_X\ \|q - \Phi X\|^2 + \lambda\,\|X - X_0\|^2 \;\Rightarrow\; "
                              r"X = (\Phi^T\Phi + \lambda I)^{-1}(\Phi^T q + \lambda X_0)",
         "λ = 1e-3 in TherMOS: the projected matrices act as a soft bound."),
        ("Stability check", r"\max\,\Re\,\lambda(-C^{-1}G) < 0",
         "Otherwise the projected matrices are kept (WARN)."),
        ("SF network bounds", r"\ln p_0 - \ln b \;\le\; \ln p \;\le\; \ln p_0 + \ln b",
         "Every R, C and coupling g in log space within a factor b (Mdl › sf › bnd) of its initial value; trust-region "
         "least squares (TRF)."),
    ], sym=[('a, ȧ', 'modal coordinates and their time derivative', 'K, K/s'),
            ('q', 'projected heat generation', 'W'), ('C, G', 'reduced capacitance and conductance matrices',
                                                     'Ws/K, W/K'),
            ('Φ⁺_rcond', 'pseudo-inverse with truncated small singular values', '–'),
            ('λ', 'weight of the prior', '–'), ('b', 'bound factor of the SF fit', '–')])
    c = st.columns(5)
    exc = c[0].selectbox("Excitation", ['single step', 'random steps (PRBS)'], key=f"{K}|exc")
    noise = cm.num(c[1], "Noise σ (% of a)", 0.1, 'numr|noise', min_value=0.0, max_value=20.0)
    err0 = cm.num(c[2], "Error of C₀, G₀ (%)", 30.0, 'numr|err0', min_value=0.0, max_value=200.0)
    rc = c[3].select_slider("rcond (protocol 5)", [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-3, key=f"{K}|rc",
                            format_func=lambda v: f"{v:g}")
    lam = c[4].select_slider("λ (protocol 6)", [1e-6, 1e-4, 1e-3, 1e-2, 1, 100], value=1e-3, key=f"{K}|lam",
                             format_func=lambda v: f"{v:g}")
    # Two-mode reference system
    Ct = np.diag([200.0, 20.0])
    Gt = np.array([[2.0, -0.4], [-0.4, 4.0]])
    Ts, nt = 0.5, 1200
    rng = np.random.default_rng(3)
    if exc == 'single step':
        q = np.zeros((nt, 2))
        q[20:] = [10.0, 3.0]
    else:
        q = np.repeat(rng.uniform(0, 10, (nt // 60 + 1, 2)), 60, axis=0)[:nt]
    a = _sim(Ct, Gt, q, Ts)
    an = a + rng.normal(0, noise / 100 * np.abs(a).max(), a.shape)
    C0 = Ct * (1 + err0 / 100 * np.array([[1, 0], [0, -0.5]]))
    G0 = Gt * (1 - err0 / 100 * np.array([[0.6, 1], [1, -0.4]]))
    res = {'true': (Ct, Gt), 'projected C₀, G₀': (C0, G0)}
    [C5, G5, s] = _calib(an, q, Ts, C0, G0, 5, rc, lam)
    [C6, G6, _] = _calib(an, q, Ts, C0, G0, 6, rc, lam)
    res['protocol 5'] = (C5, G5)
    res['protocol 6'] = (C6, G6)
    rows, sims = [], {}
    for nm, (Cm, Gm) in res.items():
        try:
            ev = np.linalg.eigvals(-np.linalg.solve(Cm, Gm))
            stable = bool(np.max(np.real(ev)) < 0)
            tau = np.sort(-1 / np.real(ev))[::-1]
        except np.linalg.LinAlgError:
            stable, tau = False, [np.nan, np.nan]
        sims[nm] = _sim(Cm, Gm, q, Ts) if stable else None
        rows.append({'model': nm, 'C₁₁': Cm[0, 0], 'C₂₂': Cm[1, 1], 'C₁₂': Cm[0, 1], 'G₁₁': Gm[0, 0],
                     'G₂₂': Gm[1, 1], 'G₁₂': Gm[0, 1], 'τ₁ (s)': tau[0], 'τ₂ (s)': tau[1],
                     'stable': 'yes' if stable else 'no',
                     'MAE a₁ (K)': np.mean(np.abs(sims[nm][:, 0] - a[:, 0])) if stable else np.nan})
    table(pd.DataFrame(rows), hide_index=True)
    cc = st.columns([3, 2])
    t = np.arange(nt) * Ts
    with cc[0]:
        fig = go.Figure([line(t, an[:, 0], 'data a₁ (noisy)', 0, width=1)])
        for i, nm in enumerate(['projected C₀, G₀', 'protocol 5', 'protocol 6'], 1):
            if sims[nm] is not None:
                fig.add_trace(line(t, sims[nm][:, 0], nm, i, dash='dash' if i == 1 else None))
        fig.update_layout(xaxis_title='t (s)', yaxis_title='a₁ (K)')
        show(fig, height=330, key=f"{K}|cal", info="**Simulation with the calibrated matrices** (first mode). "
             "Protocol 5/6 should follow the data where C₀, G₀ are off.")
    with cc[1]:
        fig = go.Figure(go.Bar(x=[f"σ{i + 1}" for i in range(len(s))], y=s / s.max(), marker_color=[
            color(0) if v / s.max() >= rc else '#b8b8b8' for v in s]))
        fig.add_hline(y=rc, line=dict(color=INK2, dash='dot'), annotation_text='rcond')
        fig.update_layout(yaxis=dict(type='log', title='σ / σ_max'), xaxis_title='singular values of Φ')
        show(fig, height=330, key=f"{K}|sv", info="**What the data determine.** Each singular value of the "
             "regressor matrix Φ = [ȧ, a] is one direction in the (C, G) space; grey ones are below rcond and keep "
             "the value of C₀, G₀ in protocol 5.")
    nDrop = int(np.sum(s / s.max() < rc))
    cm.interpret([
        f"With a {exc} the regressor matrix has a condition number of {fmt(s.max() / s.min())}; {nDrop} of "
        f"{len(s)} directions fall below rcond = {rc:g} and are not changed by protocol 5. A single step excites "
        f"only the exponentials of one response, so C and G cannot be separated (small σ) — random steps make them "
        f"identifiable. Noise is amplified by the time derivative ȧ and biases C towards zero (try 1 %): then only "
        f"a strong prior (large λ) or a larger rcond keeps the model physical.",
        f"Protocol 6 pulls every entry towards C₀, G₀ with λ = {lam:g}: large λ keeps the projected (physical) "
        f"matrices, small λ follows the data and the noise. A very small rcond in protocol 5 fits the noise in the "
        f"unidentifiable directions (unstable or ill-conditioned C); the stability check then falls back to C₀, G₀.",
        "The SF network fit uses hard bounds instead: every resistance, capacitance and coupling may move by a factor "
        "b = Mdl › sf › bnd from its start value; couplings that end at the lower bound are reported as not needed."])
