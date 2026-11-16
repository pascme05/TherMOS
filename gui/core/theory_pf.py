#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         theory_pf.py
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Theory view of the PF mode (physics-informed field network for 2D/3D temperature fields): structure of both variants,
the loss with all symbols, an explorer of the source encoding and of the sensor placement on the training data of the
current configuration (toolkit functions encodeSrc, selSensors), the meaning of every parameter and measured reference
results of the tutorial board.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core import dataset
from gui.core.style import table, show, fmt, color, MUTED
from gui.core.theory_lumped import _get
from gui.views.tools import _common as cm
from src.py.model.PF.pfData import pfPara, encodeSrc, selSensors, geoScale, toHat, PHYS

#######################################################################################################################
# Settings
#######################################################################################################################
# Measured on the tutorial board (setup/tutorial/genBoardPF.py, 3453 nodes, 3 chips, 4 training profiles of 20 s,
# test: designed pulse/sine profile of 40 s, maximum rise 107 K), RTX 3070
# 2D: the same board exported as a lumped 2D field (one node per column of the stack), maximum rise 79 K
COLS = ['model', 'physics', 'temperatures used', 'RMSE (K)', 'MAE (K)', 'MAX (K)', 'training (s)']
REFERENCE = pd.DataFrame([
    ('1 · Operator network', '0 · none (data only)', 'training profiles, all nodes', 8.94, 8.46, 18.9, 22),
    ('1 · Operator network', '1 · automatic differentiation, β = 0.01', 'training profiles, all nodes', 31.9, 30.8,
     66.3, 66),
    ('1 · Operator network', '2 · finite-volume operator', 'training profiles, all nodes', 0.81, 0.58, 19.8, 159),
    ('1 · Operator network', '2 · finite-volume operator, col = 0', 'training profiles, all nodes', 2.32, 2.11, 12.8,
     108),
    ('1 · Operator network', '2 · finite-volume operator, α = 0', 'none', 1.98, 1.20, 29.2, 82),
    ('POD state space (PS mode, K = 6)', '–', 'training profiles, all nodes', 1.09, 0.37, 30.6, 26),
    ('2 · Coordinate network', '0 · none (data only)', 'testing profile, 50 sensors', 5.64, 3.12, 44.9, 42),
    ('2 · Coordinate network', '2 · finite-volume operator', 'testing profile, 50 sensors', 3.18, 2.31, 43.9, 245),
], columns=COLS)
REFERENCE2D = pd.DataFrame([
    ('1 · Operator network', '0 · none (data only)', 'training profiles, all nodes', 7.06, 6.45, 14.5, 11),
    ('1 · Operator network', '1 · automatic differentiation', 'training profiles, all nodes', 2.31, 1.81, 10.4, 83),
    ('1 · Operator network', '2 · finite-volume operator', 'training profiles, all nodes', 0.65, 0.47, 6.6, 56),
    ('POD state space (PS mode, K = 6)', '–', 'training profiles, all nodes', 0.44, 0.24, 7.6, 11),
    ('2 · Coordinate network', '0 · none (data only)', 'testing profile, 30 sensors', 3.99, 2.88, 20.0, 33),
    ('2 · Coordinate network', '1 · automatic differentiation', 'testing profile, 30 sensors', 0.30, 0.23, 2.0, 492),
    ('2 · Coordinate network', '2 · finite-volume operator', 'testing profile, 30 sensors', 2.08, 1.60, 14.9, 202),
], columns=COLS)

DOT = '''digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.45;
  node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", fillcolor="#f4f3ee", color="#898781",
        margin="0.14,0.07"];
  edge [arrowsize=0.7, color="#52514e", fontname="Helvetica", fontsize=9, fontcolor="#52514e"];
  %s
}'''
DOT1 = DOT % '''
  Q [label=<<B>Heat sources</B><BR/>Q(x, t) (W/m³)>, shape=note, fillcolor="#dbe9f6", color="#4a7fb5"];
  enc [label=<<B>Source encoding</B><BR/>Q ≈ Σ U<SUB>j</SUB>(x)·p<SUB>j</SUB>(t)>];
  br [label=<<B>Branch</B> (LSTM)<BR/>p(t) → b<SUB>1…K</SUB>(t)>];
  x [label=<<B>Position</B><BR/>x, y(, z)>, shape=note, fillcolor="#dbe9f6", color="#4a7fb5"];
  tr [label=<<B>Trunk</B> (MLP)<BR/>x → φ<SUB>1…K</SUB>(x)>];
  T [label=<<B>Field</B><BR/>T̂ = T<SUB>ref</SUB> + ΔT·Σ b<SUB>k</SUB>(t)·φ<SUB>k</SUB>(x)>, fillcolor="#dcefdc", color="#4d9a4d"];
  L [label=<<B>Loss</B><BR/>α·data + β·physics + γ·init>, fillcolor="#fbe6cf", color="#d28a2e"];
  Q -> enc; enc -> br [label="p(t)"]; br -> T; x -> tr; tr -> T; T -> L;
  enc -> L [label="U, p", style=dashed];
'''
DOT2 = DOT % '''
  xt [label=<<B>Position and time</B><BR/>x, y(, z), t>, shape=note, fillcolor="#dbe9f6", color="#4a7fb5"];
  net [label=<<B>Network</B> (MLP)<BR/>(x, t) → θ̂/ΔT>];
  T [label=<<B>Field of one profile</B><BR/>T̂ = T<SUB>ref</SUB> + ΔT·f(x, t)>, fillcolor="#dcefdc", color="#4d9a4d"];
  S [label=<<B>Sensors</B><BR/>T at n<SUB>Sens</SUB> nodes>, shape=note, fillcolor="#dbe9f6", color="#4a7fb5"];
  Q [label=<<B>Heat sources</B><BR/>Q(x, t)>, shape=note, fillcolor="#dbe9f6", color="#4a7fb5"];
  L [label=<<B>Loss</B><BR/>α·data + β·physics + γ·init>, fillcolor="#fbe6cf", color="#d28a2e"];
  xt -> net; net -> T; T -> L; S -> L [label="data"]; Q -> L [label="physics"];
'''
PARAMS = [
    ('Model variant', 'Run › selPF', "1: operator network — trained on several profiles, predicts other load "
     "profiles. 2: coordinate network — reconstructs the field of the testing profile from its sensors."),
    ('Physics', 'pinf › phys', "0: none. 1: heat equation with automatic differentiation (any data). 2: "
     "finite-volume operator of the data file (board designer export), the most accurate."),
    ('α, β, γ', 'pinf › alpha, beta, gamma', "Weights of the data, physics and initial-condition loss. All three "
     "losses are relative to ΔT², so β = 1 weights data and physics equally. α = 0 trains without temperatures."),
    ('Basis functions K', 'pinf › K', "Operator network: number of products b_k(t)·φ_k(x). 8–32; more than the "
     "field needs costs time, too few limit the accuracy (compare the POD energy of the PS mode)."),
    ('Sensors', 'pinf › nSens', "Nodes with measured temperatures in the data loss. 0 = all nodes. With physics, a "
     "few tens of sensors can be enough."),
    ('Warm-up, window', 'pinf › warm, dl › W', "Operator network: the branch has no thermal state at the start of "
     "a window. warm ≈ 3·τ_max/Ts samples are excluded from the loss, the window W should be at least twice as "
     "long."),
    ('Collocation profiles', 'pinf › col', "Operator network: the physics loss is also evaluated for the sources of "
     "the validation and testing profiles (their temperatures are not used). This is what lets the network follow "
     "loads it has no temperature data for."),
    ('Network size', 'pinf › width, depth, fourier', "Fully connected network (trunk / coordinate network): neurons "
     "per layer, layers and Fourier frequencies of the coordinates (sharp chip edges need 3–6)."),
    ('Training', 'dl › epoch, initLr, lrAdapt, valPat, batch', "Adam, learning-rate schedule and early stopping "
     "as in the DL and PI modes; gen › dev selects the GPU."),
]


#######################################################################################################################
# Helper
#######################################################################################################################
@st.cache_data(show_spinner="Encoding the heat sources …", max_entries=4)
def _encode(name, tol, _m):
    d = dataset.field(name)
    geo = np.atleast_2d(np.asarray(d['geo'], dtype=float))
    [U, sv] = encodeSrc([np.atleast_2d(d['X'])], tol)
    th = np.atleast_2d(d['y']) - np.atleast_2d(d.get('r', 0.0))
    return geo, U, sv, int(np.argmax(np.max(np.abs(th), axis=0))), float(np.max(th))


@st.cache_data(show_spinner="Placing the sensors …", max_entries=8)
def _sensors(name, n, first, _m):
    geo = np.atleast_2d(np.asarray(dataset.field(name)['geo'], dtype=float))
    xh = toHat(geo, geoScale(geo))
    idx = selSensors(xh, n, 42, first)
    step = max(len(geo) // 4000, 1)
    d = np.sqrt(np.min(np.sum((xh[::step, None, :] - xh[None, idx, :]) ** 2, axis=2), axis=1))
    return idx, float(np.mean(d)), float(np.max(d))


def _scatter(geo, ax, groups, title):
    """Nodes projected on two axes (mm): groups = [(name, index or mask, colour, size)]."""
    lab = 'xyz'
    fig = go.Figure()
    for name, sel, col, size in groups:
        g = geo[sel]
        if len(g) > 6000:
            g = g[::int(np.ceil(len(g) / 6000))]
        fig.add_trace(go.Scattergl(x=g[:, ax[0]] * 1e3, y=g[:, ax[1]] * 1e3, mode='markers', name=name,
                                   marker=dict(color=col, size=size)))
    fig.update_layout(title=title, xaxis_title=f"{lab[ax[0]]} (mm)", yaxis_title=f"{lab[ax[1]]} (mm)")
    return fig


#######################################################################################################################
# View
#######################################################################################################################
def theory_pf(cfg):
    G = cfg.get('Run') or {}
    var = int(G.get('selPF', 1) or 1)
    pp = pfPara({'Mdl': cfg.get('Mdl') or {}})
    st.caption("PF mode: a physics-informed neural network (PINN) for 2D and 3D temperature fields with heat sources "
               "that vary in space and time. It learns from finite-element or finite-volume temperature data and, at "
               "the same time, has to satisfy the heat equation. The views below explain the two variants, the loss "
               "and the data preparation on the training data of the current configuration; they do not change it.")

    # ==============================================================================
    # Structure
    # ==============================================================================
    cm.section(1, "The two variants", "Variant %d is selected (Parameters › Model variant)." % var, divider=False)
    c = st.columns(2)
    with c[0]:
        st.markdown("**1 · Operator network** — one model for many load profiles")
        st.graphviz_chart(DOT1, width='content')
        st.caption("The field is a sum of K products: a time function b_k(t) from the branch (a sequence network "
                   "that sees the source amplitudes and carries the thermal memory) times a space function φ_k(x) "
                   "from the trunk (a fully connected network of the position). The trunk plays the role of the POD "
                   "modes, the branch the role of the state-space model of the PS mode — here both are learned "
                   "together and the physics loss ties them to the heat equation.")
    with c[1]:
        st.markdown("**2 · Coordinate network** — the field of one profile from a few sensors")
        st.graphviz_chart(DOT2, width='content')
        st.caption("The classical PINN of Cai et al.: one network maps position and time to the temperature of ONE "
                   "profile (the testing profile). It only sees the temperatures at the sensor nodes; the heat "
                   "equation fills in the field between them. A new load profile needs a new training.")

    # ==============================================================================
    # Equations
    # ==============================================================================
    cm.equations([
        ("Source encoding (both variants)",
         r"Q(x, t) \approx \sum_{j=1}^{S} U_j(x)\,p_j(t),\qquad p = Q\,U,\qquad U^{T}U = I",
         "Singular value decomposition of the heat generation of the training data. For chips with fixed "
         "footprints S is the number of independently driven chips and the encoding is exact."),
        ("Operator network (variant 1)",
         r"\hat T(x, t) = T_{ref}(x) + \Delta T \sum_{k=1}^{K} b_k(t)\,\varphi_k(x),\qquad b = "
         r"\mathrm{branch}(p),\quad \varphi = \mathrm{trunk}(\hat x)", None),
        ("Coordinate network (variant 2)", r"\hat T(x, t) = T_{ref}(x) + \Delta T\, f(\hat x, \hat t)", None),
        ("Heat equation for the rise θ = T − T_ref",
         r"\rho c\,\frac{\partial\theta}{\partial t} = \sum_i \frac{\partial}{\partial x_i}\!\left(k_i "
         r"\frac{\partial\theta}{\partial x_i}\right) - h_v\,\theta + Q", None),
        ("Physics 1: residual with automatic differentiation, at collocation points",
         r"r = \frac{\rho c\,\partial_t\hat\theta - \sum_i k_i\,\partial^2_{x_i}\hat\theta + h_v\hat\theta - Q}"
         r"{g\,\Delta T},\qquad g = \sum_i \frac{2 k_i}{\ell_i^2} + h_v",
         "Mesh-free. The material of a point is that of its nearest node; the jump of k at material interfaces and "
         "the boundary conditions are not part of the residual (they are learned from the data)."),
        ("Physics 2: residual of the finite-volume operator, at the nodes",
         r"r = \frac{G^{-1}\left(C\,\dot{\hat\theta} + G\,\hat\theta - V Q\right)}{\Delta T} = "
         r"\frac{G^{-1} C\,\dot{\hat\theta} + \hat\theta - \theta_{ss}}{\Delta T}",
         "Needs the operator in the data file (board designer). Multiplying the heat balance by G⁻¹ turns it into "
         "the temperature error it causes: r = 0.01 is an error of 1 % of ΔT."),
        ("Loss", [r"L = \alpha\,L_{data} + \beta\,L_{phys} + \gamma\,L_{init}",
                  r"L_{data} = \overline{\left(\tfrac{\hat\theta - \theta}{\Delta T}\right)^2}\ \text{(sensors)},"
                  r"\quad L_{phys} = \overline{r^2},\quad L_{init} = \overline{\left(\tfrac{\hat\theta}"
                  r"{\Delta T}\right)^2}\ (t \le 0)"],
         "All three parts are dimensionless and relative to ΔT²: L = 10⁻⁴ is an RMS error of 1 % of ΔT."),
    ], sym=[
        ('T, T̂', 'measured (data file, y) and predicted temperature at position x and time t', '°C'),
        ('T_ref', 'reference temperature: ambient of the data file (r), or the removed offset (Exp › init)', '°C'),
        ('θ, θ̂', 'measured and predicted temperature rise T − T_ref', 'K'),
        ('ΔT', 'temperature scale: 99 % quantile of the measured rise of the training data (computed)', 'K'),
        ('Q', 'heat generation per volume (data file, X)', 'W/m³'),
        ('U_j, p_j', 'footprint of source j (N values, orthonormal) and its amplitude over time; j = 1 … S', '–, W/m³'),
        ('S', 'number of source amplitudes (singular values above pinf › srcTol)', '–'),
        ('b_k, φ_k', 'coefficient over time (branch) and basis function in space (trunk); k = 1 … K (pinf › K)', '–'),
        ('x̂, t̂', 'position and time scaled to [−1, 1]', '–'),
        ('ρc', 'heat capacity per volume ρ·Cp (data file, rho and Cp)', 'J/(m³ K)'),
        ('k_i', 'thermal conductivity in direction i = x, y, z (data file: k, ky, kz)', 'W/(m K)'),
        ('h_v', 'distributed heat sink of a lumped board (data file, hv); 0 for FEM data', 'W/(m³ K)'),
        ('ℓ_i', 'half the extent of the domain in direction i', 'm'),
        ('C, G, V', 'capacitances, conductance matrix incl. cooling and control volumes of the finite-volume model '
                    '(data file: Cn, Gi/Gj/Gv, Vn)', 'Ws/K, W/K, m³'),
        ('θ_ss', 'steady-state rise for the present sources, G⁻¹·V·Q', 'K'),
        ('α, β, γ', 'weights of the data, physics and initial-condition loss (pinf › alpha, beta, gamma)', '–'),
    ], title="Equations and symbols", expanded=True)

    # ==============================================================================
    # Explorer
    # ==============================================================================
    fmtD = G.get('format', 'mat')
    names = [G.get('teFile')] if var == 2 else list(G.get('trFile') or [])
    name = names[0] if names else None
    kind = dataset.info(name, fmtD)['kind'] if name and fmtD == 'mat' else 'missing'
    cm.section(2, "Your data: heat sources and sensors",
               f"Data set **{name}** ({'testing' if var == 2 else 'first training'} file of the configuration).")
    if kind not in ('2D', '3D'):
        st.info("The PF mode needs 2D or 3D field data (.mat with X, y, geo). Load the example of this mode "
                "(Parameters › Load example) after generating its data with `python setup/tutorial/genBoardPF.py`, "
                "or export a board from Theory & Tools › Board designer.", icon=':material/info:')
    else:
        m = dataset._mtime(name, 'mat')
        [geo, U, sv, hot, rise] = _encode(name, pp['srcTol'], m)
        d, N, S = geo.shape[1], len(geo), U.shape[1]
        tabs = st.tabs(["How many sources are there?", "Where are the sensors?"])
        with tabs[0]:
            c = st.columns([2, 3])
            with c[0]:
                n = min(len(sv), max(S + 4, 6))
                fig = go.Figure(go.Bar(x=np.arange(1, n + 1), y=np.maximum(sv[:n], 1e-18),
                                       marker_color=[color(0) if i < S else MUTED for i in range(n)]))
                fig.add_hline(y=pp['srcTol'], line=dict(color=color(1), dash='dot'))
                fig.update_layout(xaxis_title='source amplitude j', yaxis=dict(type='log', title='σ_j / σ_1'))
                show(fig, height=320, key='thPF|sv', info="**Singular values of the heat generation** relative to "
                     "the largest one (log scale). Coloured bars are kept (above the dotted tolerance pinf › "
                     "srcTol), grey ones are numerical noise. Every kept bar is one input p_j(t) of the network.")
            with c[1]:
                act = [np.abs(U[:, j]) > 1e-3 * np.max(np.abs(U[:, j])) for j in range(S)]
                show(_scatter(geo, (0, 1), [('nodes', slice(None), '#d9d8d2', 3)] +
                              [(f"footprint U_{j + 1}", act[j], color(j), 5) for j in range(min(S, 8))],
                              "Footprints of the sources (top view)"), height=320, key='thPF|foot',
                     info="**Where the sources act.** Every colour marks the nodes of one footprint U_j. The "
                          "footprints come from the SVD, so two chips that are always driven together appear as "
                          "one footprint.")
            st.caption(f"The {N} nodes of this file hold **S = {S} independent source(s)**: the heat generation of "
                       f"every time step is described exactly by {S} number(s) instead of {N}. "
                       + (f"The first dropped singular value is {fmt(float(sv[S]))}·σ₁. " if len(sv) > S else "")
                       + "The model can only predict loads that are combinations of these footprints; the test "
                       "run prints the encoding error of the testing profile.")
        with tabs[1]:
            nS = int(st.slider("Number of sensors n_Sens", 0, int(min(N, 500)), int(min(pp['nSens'], min(N, 500))),
                               key='thPF|ns', help="Value of Mdl › pinf › nSens to try here (0 = all nodes). The "
                                                   "configuration is not changed."))
            if nS == 0:
                st.caption(f"n_Sens = 0: the temperatures of all {N} nodes are used in the data loss.")
            else:
                [idx, dm, dx] = _sensors(name, nS, hot, m)
                ext = np.ptp(geo, axis=0) * 1e3
                views = [((0, 1), "Top view")] + ([((0, 2), "Side view")] if d == 3 else [])
                for col, (ax, ttl) in zip(st.columns(len(views)), views):
                    with col:
                        show(_scatter(geo, ax, [('nodes', slice(None), '#d9d8d2', 3), ('sensors', idx, color(1), 7),
                                                ('hottest node', [hot], color(3), 11)], ttl), height=320,
                             key=f'thPF|sens{ax[1]}')
                st.caption(f"**{nS} sensors** = {100 * nS / N:.2f} % of the {N} nodes. Farthest-point sampling "
                           f"starts at the hottest node (rise {fmt(rise)} K) and always adds the node farthest from "
                           f"all sensors chosen so far. In scaled coordinates (every axis −1 … 1) a node is on "
                           f"average {fmt(dm)} and at most {fmt(dx)} away from the nearest sensor; the domain is "
                           f"{' × '.join(fmt(float(e)) for e in ext)} mm, so thin directions are covered as "
                           f"densely as wide ones.")

    # ==============================================================================
    # Parameters and reference
    # ==============================================================================
    cm.section(3, "What the settings do")
    table(pd.DataFrame(PARAMS, columns=['setting', 'parameter', 'effect']), hide_index=True,
          column_config={'effect': st.column_config.TextColumn(width='large')})
    st.caption(f"Current configuration: variant {var}, physics {pp['phys']} ({PHYS.get(pp['phys'], '?')}), α = "
               f"{fmt(pp['alpha'])}, β = {fmt(pp['beta'])}, γ = {fmt(pp['gamma'])}, K = {pp['K']}, n_Sens = "
               f"{pp['nSens']}, warm-up {pp['warm']} samples, window W = {_get(cfg, 'Mdl', 'dl', 'W', default='?')}.")
    cm.section(4, "What to expect: measured on the tutorial board",
               "3D board with three chips (3453 nodes), four random training profiles of 20 s, tested on a designed "
               "pulse/sine profile of 40 s with a maximum rise of 107 K (tutorial *PF: field PINN of a board*).")
    tabs = st.tabs(["3D board, layer-resolved (maximum rise 107 K)", "Same board as a lumped 2D field (79 K)"])
    with tabs[0]:
        table(REFERENCE, hide_index=True)
    with tabs[1]:
        table(REFERENCE2D, hide_index=True)
    st.caption("Columns: RMSE, MAE and MAX are the root-mean-square, mean absolute and largest error of the testing "
               "profile over all nodes and time steps (the error at nodes without a sensor included). *Temperatures "
               "used* are the data in the loss; the physics loss needs none. Training time on an RTX 3070; operator "
               "network up to 600 epochs, coordinate network 150 epochs, both with early stopping. Runs on a GPU "
               "differ by a few hundredths of a kelvin.")
    cm.interpret([
        "**Data alone do not generalise.** The operator network without physics reaches 2.1 K on the validation "
        "profiles but 8.9 K (8 % of the rise) on the testing profile, whose loads (all chips at full power at "
        "once) lie at the edge of the random training loads: a sequence network does not know that the heat "
        "equation is linear.",
        "**The physics loss closes this gap**: 8.9 K → 0.81 K with the finite-volume operator. Two thirds of the "
        "remaining gain come from the collocation profiles (col = 0: 2.3 K), i.e. from evaluating the heat balance "
        "for the sources of the testing profile without a single temperature of it. Without any temperature data "
        "(α = 0) the network still reaches 2.0 K.",
        "**Which physics?** Use 2 whenever the data come from the board designer: the operator is exact at material "
        "interfaces and its residual is a true temperature error. Physics 1 works for any mesh but neglects the "
        "jump of the conductivity between materials and knows no boundary conditions. On the lumped 2D board "
        "(conductivity varies smoothly) it helps: 7.1 K → 2.3 K, and 4.0 K → 0.30 K for the coordinate network. On "
        "the layered 3D board (copper next to dielectric, ratio 195) it is harmful even with β = 0.01: 31.9 K. "
        "The run prints a warning when the conductivity ratio is above 10.",
        "**Few sensors (variant 2).** 30 sensors (5 % of the nodes) plus the heat equation give the 2D field with "
        "0.30 K RMSE, better than the POD model that saw all nodes of four profiles (0.44 K). In 3D, 50 sensors "
        "(1.4 % of 3453 nodes) reach 3.2 K with physics 2 against 5.6 K without; the largest error (44 K) is in the "
        "chips, where no sensor sits and the field is steepest — add sensors there or more epochs.",
        "**PINN or POD?** For a linear problem with full field data the PS mode is 2–6 times faster and about as "
        "accurate (3D: 1.09 K vs 0.81 K RMSE, 2D: 0.44 K vs 0.65 K). The PF mode pays off when temperatures are "
        "known at few sensors only, when no temperature data exist for the load of interest, or as the basis for "
        "effects that the linear reduced models cannot represent.",
    ], title="How to read these results")
    cm.refs(['Cai2021', 'Raissi2019', 'Lu2021', 'Wang2021'])
