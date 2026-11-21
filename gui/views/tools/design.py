#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         design.py
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
Theory & Tools › Design effects: what a change of the cooling, of a material or of a chip position does to the
temperatures of an IMS board with three chips (3D finite-volume model of the parametric PF study, calculations in
src/py/tools/designCalc.py). The design is set once at the top; one tab per question:
    1) Where does a change act?        which entries of C, G, Bp change (and why pMOR handles some but not others)
    2) Cooling & materials             steady junction temperatures, field, sweep and sensitivities
    3) Move a chip                     junction temperatures over the position, coupling (R_th matrix)
    4) Transients & time constants     step response of the current design against the nominal board
    5) Parametric network              prediction of the trained network (results/pfparam) and the study results
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, show, line, fmt, TEMP
from gui.views.tools import _common as cm
from gui.views.tools import design_art as art
from src.py.tools import designCalc as dc
from src.py.model.PF.pfParam import CHIPS, NOM, MU, W_MM, L_MM, MARGIN

#######################################################################################################################
# Settings
#######################################################################################################################
K = 'de'                                                                   # key prefix (cm.num adds 'tl|')
TABS = ["Where does a change act?", "Cooling & materials", "Move a chip", "Transients & time constants",
        "Parametric network"]
SWEEP = {'h': np.geomspace(250, 8000, 15), 'kDiel': np.geomspace(0.5, 12, 15), 'sAtt': np.geomspace(0.3, 3, 13),
         'sRc': np.linspace(0.5, 1.5, 11)}
HELP = {'h': "Heat transfer coefficient of the cold plate at the bottom of the Al base. Typical: 500 (weak air / "
             "poor contact), 1500 (liquid cold plate via grease), 5000 (direct liquid cooling). Training range of "
             "the network 500–5000.",
        'kDiel': "Conductivity of the 100 µm IMS dielectric. Standard IMS 1–3, high-performance 5–8, ceramic-filled "
                 "up to 12. It sits in series with the whole heat path, so it acts on the steady temperature.",
        'sAtt': "Factor on the conductivity of the die attach (SAC305 58 W/(mK) for T1/D1, sintered Ag 200 W/(mK) "
                "for M1). 0.5 ≈ voided or aged solder, 2 ≈ better sinter.",
        'sRc': "Factor on the heat capacity ρ·c of the Al base (e.g. a different alloy or thickness equivalent). It "
               "changes only the transient (time constants), never the steady state."}
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                   'results', 'pfparam')


def _head(n, title, text):
    c = st.columns([3, 2], vertical_alignment='center')
    with c[0]:
        cm.section(n, title, text, divider=False)
    return c[1]


def _pic(slot, svg):
    slot.markdown(svg, unsafe_allow_html=True)


def _key(mu, pos):
    return tuple(sorted(mu.items())), tuple(tuple(p) for p in pos)


#######################################################################################################################
# Cached calculations
#######################################################################################################################
@st.cache_data(show_spinner=False, max_entries=64)
def _steady(mk, pk, P):
    B = dc.board(dict(mk), list(pk))
    th = dc.steady(B, P)
    x, y, M = dc.topMap(B, th)
    return dc.junction(B, th), (x, y, M), dc.timeConstants(B, 3)


@st.cache_data(show_spinner=False, max_entries=32)
def _sweepParam(mk, pk, P, key):
    return dc.sweepParam(dict(mk), list(pk), P, key, SWEEP[key])


@st.cache_data(show_spinner=False, max_entries=32)
def _sweepPos(mk, pk, P, chip, axis, values):
    return dc.sweepPos(dict(mk), list(pk), P, chip, axis, np.asarray(values))


@st.cache_data(show_spinner=False, max_entries=32)
def _sens(mk, pk, P):
    return dc.sensitivity(dict(mk), list(pk), P)


@st.cache_data(show_spinner=False, max_entries=32)
def _coupling(mk, pk):
    return dc.coupling(dict(mk), list(pk))


@st.cache_data(show_spinner=False, max_entries=32)
def _change(mk, pk, key, chip, dx, dy):
    return dc.operatorChange(dict(mk), list(pk), key, 0.2, (dx, dy), chip)


@st.cache_data(show_spinner=False, max_entries=32)
def _step(mk, pk, P, tOn, tEnd, Ts):
    t = np.arange(0.0, tEnd + Ts * 1e-9, Ts)
    PP = np.where((t > 0) & (t <= tOn), 1.0, 0.0)[:, None] * np.asarray(P)[None, :]
    B = dc.board(dict(mk), list(pk))
    t0 = time.time()
    y = dc.transient(B, PP, Ts)
    return t, PP, y, time.time() - t0


@st.cache_resource(show_spinner=False)
def _net(path, mtime):
    return dc.loadNet(path)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Design effects", "Effect of the cooling, the materials and the chip positions on the temperatures "
                                   "of an IMS board (30 × 20 mm, Si IGBT T1, Si diode D1, SiC MOSFET M1, 3D "
                                   "finite-volume model with 3453 nodes). Every result is computed from the model "
                                   "itself in milliseconds; the last tab compares it with the parametric network.")
    cm.question("What happens to the junction temperatures when the cooling, a material or a chip position changes — "
                "and where does each change enter the thermal model?",
                "Set the design once below. Each tab answers one question for this design, the charts move one "
                "parameter at a time and mark the current design in red.")
    d = _design()
    tabs = st.tabs(TABS)
    for t, f in zip(tabs, [_where, _materials, _move, _transient, _network]):
        with t:
            if not d['ok']:
                st.warning("The layout is not valid (overlap or margin). Change the chip positions above.",
                           icon=':material/warning:')
                continue
            f(d)


def _design():
    pic = _head(1, "Design of the board", "Condition of the board (cooling and materials), losses of the chips and "
                "chip centres. Positions are on the 1 mm grid of the model; the chips must keep 1 mm to the edges "
                "and 1 mm to each other.")
    c = st.columns(4)
    mu = {}
    for i, (k, (lab, unit, _)) in enumerate(dc.PARAMS.items()):
        lo = {'h': 100.0, 'kDiel': 0.2, 'sAtt': 0.1, 'sRc': 0.3}[k]
        hi = {'h': 20000.0, 'kDiel': 30.0, 'sAtt': 5.0, 'sRc': 2.0}[k]
        mu[k] = cm.num(c[i], f"{lab} ({unit})", NOM[k], f"{K}|mu|{k}", min_value=lo, max_value=hi, help=HELP[k])
    c = st.columns(4)
    P = []
    for i, ch in enumerate(CHIPS):
        P.append(cm.num(c[i], f"Loss {ch['name']} (W)", ch['P'], f"{K}|P|{i}", min_value=0.0, max_value=200.0,
                        help=f"Constant loss of {ch['name']} ({ch['tech']}, {ch['w']} × {ch['l']} mm) for the "
                             f"steady state; the transient tab switches it on and off. Rated {ch['P']} W."))
    c[3].caption("Total loss: **%s W**" % fmt(sum(P)))
    c = st.columns(6)
    pos = []
    for i, ch in enumerate(CHIPS):
        x = cm.num(c[2 * i], f"x {ch['name']} (mm)", ch['x'], f"{K}|x|{i}", format='%d', step=1,
                   min_value=int(np.ceil(MARGIN + ch['w'] / 2)), max_value=int(np.floor(W_MM - MARGIN - ch['w'] / 2)),
                   help=f"Centre of {ch['name']} along the 30 mm side (nominal {ch['x']} mm).")
        y = cm.num(c[2 * i + 1], f"y {ch['name']} (mm)", ch['y'], f"{K}|y|{i}", format='%d', step=1,
                   min_value=int(np.ceil(MARGIN + ch['l'] / 2)), max_value=int(np.floor(L_MM - MARGIN - ch['l'] / 2)),
                   help=f"Centre of {ch['name']} along the 20 mm side (nominal {ch['y']} mm).")
        pos.append((float(x), float(y)))
    ok = dc.valid(pos)
    _pic(pic, art.top(pos, CHIPS, ok, None, mu['h']))
    out = dc.inRange(mu)
    if out:
        st.caption(":material/info: Outside the training range of the parametric network: " +
                   ", ".join(f"{dc.PARAMS[k][0]} = {fmt(v)} (range {fmt(lo)}–{fmt(hi)})" for k, (v, lo, hi)
                             in out.items()) + ". The model results are exact; the network extrapolates.")
    return {'mu': mu, 'P': tuple(P), 'pos': pos, 'ok': ok, 'mk': _key(mu, pos)[0], 'pk': _key(mu, pos)[1]}


#######################################################################################################################
# 1) Where does a change act?
#######################################################################################################################
def _where(d):
    names = {'h': "Cooling h", 'kDiel': "Dielectric k", 'sAtt': "Die attach k", 'sRc': "Base ρc", 'pos': "Chip position"}
    key = st.segmented_control("Change", list(names), format_func=names.get, default='h', key=f"{K}|wh|k") or 'h'
    chip, dx, dy = 0, 0.0, 0.0
    if key == 'pos':
        c = st.columns(4)
        chip = c[0].selectbox("Chip", range(3), format_func=lambda i: CHIPS[i]['name'], key=f"{K}|wh|c")
        dx = cm.num(c[1], "Δx (mm)", 2.0, f"{K}|wh|dx", step=1.0, help="Shift of the chip along x.")
        dy = cm.num(c[2], "Δy (mm)", 0.0, f"{K}|wh|dy", step=1.0, help="Shift of the chip along y.")
    pic = _head(2, "Which entries of the model change?", "The board is the FV model C·dθ/dt = −G·θ + Bp·P. The tab "
                "raises the selected parameter by 20 % (or shifts the chip) and compares the matrices entry by entry.")
    r = _change(d['mk'], d['pk'], key, chip, dx, dy)
    if not r.get('sameNodes', False):
        st.info("The shifted layout has a different node set (or is invalid). Choose another shift.")
        return
    _pic(pic, art.stack(r['rows'], key))
    cm.section(None, "Result")
    cm.tiles([("G diagonal", f"{r['nDiag']} of {r['N']}", "Changed diagonal entries G_ii = Σ_j g_ij + h·A_i (the "
               "sum of all conductances at node i plus the cooling)."),
              ("G off-diagonal", f"{r['nOff']}", "Changed series conductances −g_ij between neighbouring nodes "
               "(conduction through a layer or between layers)."),
              ("C", f"{r['nC']} of {r['N']}", "Changed capacitances C_i = ρ·c·V_i (heat storage)."),
              ("median |ΔG/G|", "–" if not r['nDiag'] + r['nOff'] else
               ("new links" if not np.isfinite(r['relG']) else f"{fmt(100 * r['relG'])} %"),
               "Relative change of the changed G entries for a 20 % change of the parameter. Small values mean "
               "that the parameter is one term of a sum (e.g. h·A next to the conduction at the bottom nodes).")])
    rows = [n for n, _ in art.ROWS]
    fig = go.Figure(go.Bar(x=r['rows'], y=rows, orientation='h', marker_color=SERIES[1],
                           text=[f"{int(a)} / {int(b)}" for a, b in zip(r['rows'], r['nodesRow'])],
                           textposition='auto'))
    fig.update_layout(xaxis_title='nodes with a changed entry', yaxis=dict(categoryorder='array', categoryarray=rows))
    show(fig, height=280, key=f"{K}|wh|bar", info="Number of nodes per node row (bottom → top of the stack) that "
         "own a changed entry of C, G or Bp; the label gives changed / all nodes of the row.")
    where = dc.WHERE[key]
    st.caption(f"{names[key]} acts on the {where}: {r['nDiag']} diagonal and {r['nOff']} off-diagonal entries of G "
               f"(of {r['nnzG']} non-zeros) and {r['nC']} entries of C change.")
    cm.equations([("Finite-volume model", r"\mathbf C\,\dot{\boldsymbol\theta}=-\mathbf G\,\boldsymbol\theta+"
                   r"\mathbf B_p\,\boldsymbol P(t)", "One equation per node: storage = conduction + cooling + loss."),
                  ("Series conductance between nodes i and j (layer half-cells in series)",
                   r"g_{ij}=\frac{A_{ij}}{\dfrac{d_i}{2k_i}+\dfrac{d_j}{2k_j}},\qquad G_{ij}=-g_{ij}",
                   "A conductivity k changes every g_ij that crosses its layer — off-diagonal entries and, through the "
                   "sum, the diagonal."),
                  ("Diagonal and cooling", r"G_{ii}=\sum_j g_{ij}+h\,A_i^{\mathrm{bottom}}",
                   "h only adds to the diagonal of the bottom nodes."),
                  ("Storage and loss", [r"C_i=\rho\,c\,V_i", r"\mathbf B_{p,ic}=\frac{V_i}{V_{\mathrm{die},c}}\;"
                                        r"\text{for the die cells } i \text{ of chip } c"],
                   "ρc changes C only. A chip position decides which nodes receive the loss and to which copper "
                   "nodes the die and attach cells are linked.")],
                 sym=[("θ", "temperature rise of the nodes above ambient", "K"),
                      ("C, C_i", "capacitance matrix (diagonal), heat capacity of node i", "J/K"),
                      ("G, G_ij, g_ij", "conductance matrix, entry, series conductance between nodes i and j", "W/K"),
                      ("B_p", "distribution of the chip losses to the die cells (column c sums to 1)", "-"),
                      ("P", "losses of the chips (index c = T1, D1, M1)", "W"),
                      ("A_ij, d_i, k_i", "face area between i and j, thickness and conductivity of cell i",
                       "m², m, W/(mK)"),
                      ("h, A_i^bottom", "heat transfer coefficient, cooled area of a bottom node", "W/(m²K), m²"),
                      ("ρ, c, V_i", "density, specific heat, volume of node i", "kg/m³, J/(kgK), m³"),
                      ("i, j", "node indices (N = 3453 nodes)", "-")])
    cm.refs(['Incropera2007', 'Benner2015', 'Odabasioglu1998'])
    aff = key != 'pos'
    cm.interpret([
        f"**{names[key]} changes {r['nDiag'] + r['nOff'] + r['nC']} matrix entries in "
        f"{int(np.sum(r['rows'] > 0))} of 7 node rows** — {where}.",
        ("Each entry depends linearly on the parameter (G = G₀ + h·G_h, C = C₀ + s·C_ρc, …). This *affine* form is "
         "what parametric model order reduction (pMOR) exploits: the reduced matrices are precomputed once per term "
         "and only re-weighted for a new value." if aff else
         "A position change does not scale existing entries — it moves the loss and rewires the links between the die "
         "and the copper. The temperature field moves with the chip, and no fixed set of modes describes all "
         "positions well (the 'Kolmogorov barrier' of projection methods). This is why the parametric network uses "
         "coordinates relative to the chip."),
        ("Lever: the cooling and the dielectric act on every node row below the chips, the die attach only on the "
         "small area under each chip; see the tab *Cooling & materials* for how much each moves T_j."),
        "Caveats: constant material properties (linear model); the lateral faces are adiabatic."])


#######################################################################################################################
# 2) Cooling & materials
#######################################################################################################################
def _materials(d):
    mu, P = d['mu'], d['P']
    tj, (x, y, M), tau = _steady(d['mk'], d['pk'], P)
    tj0 = _steady(_key(NOM, d['pos'])[0], d['pk'], P)[0]
    pic = _head(2, "Steady junction temperatures of the design", "Constant losses until the board is in steady state "
                "(θ = G⁻¹·Bp·P). The difference to the nominal condition (h 1500, k_diel 2, factors 1) shows the "
                "effect of the current cooling and materials.")
    _pic(pic, art.top(d['pos'], CHIPS, True, int(np.argmax(tj)), mu['h']))
    cm.section(None, "Result")
    cm.tiles([(f"T_j {c['name']}", f"{fmt(v)} K", f"Junction rise of {c['name']} above ambient (hottest die node); "
               f"{v - v0:+.1f} K against the nominal condition.") for c, v, v0 in zip(CHIPS, tj, tj0)] +
             [("slowest τ", f"{fmt(tau[0])} s", "Slowest time constant of the board (1/λ of G·v = λ·C·v): after "
               "≈ 3τ the steady state is reached within 5 %.")])
    t1, t2, t3 = st.tabs(["Temperature map", "Sweep of one parameter", "Which parameter acts most?"])
    with t1:
        fig = go.Figure(go.Heatmap(x=x, y=y, z=M, colorscale=TEMP, colorbar=dict(title='θ (K)'),
                                   hovertemplate='x %{x} mm, y %{y} mm: %{z:.1f} K<extra></extra>'))
        for (px, py), c in zip(d['pos'], CHIPS):
            fig.add_shape(type='rect', x0=px - c['w'] / 2, x1=px + c['w'] / 2, y0=py - c['l'] / 2,
                          y1=py + c['l'] / 2, line=dict(color='black', width=1))
        fig.update_layout(xaxis_title='x (mm)', yaxis_title='y (mm)', yaxis=dict(scaleanchor='x'))
        show(fig, height=360, key=f"{K}|ma|map", info="Hottest node of every (x, y) column of the board in steady "
             "state (red hot, blue cold); black frames are the chips.")
        st.caption(f"Hottest point {fmt(np.max(M))} K; the coldest column of the board is at {fmt(np.min(M))} K "
                   f"above ambient.")
    with t2:
        key = st.segmented_control("Parameter", list(dc.PARAMS), format_func=lambda k: dc.PARAMS[k][0],
                                   default='h', key=f"{K}|ma|sw") or 'h'
        T, Tau = _sweepParam(d['mk'], d['pk'], P, key)
        v = SWEEP[key]
        fig = go.Figure([line(v, T[:, c], CHIPS[c]['name'], c) for c in range(3)])
        fig.add_trace(go.Scatter(x=v, y=Tau, name='slowest τ', yaxis='y2', mode='lines',
                                 line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=[mu[key]] * 3, y=tj, mode='markers', name='current', marker=dict(color='red',
                                                                                                     size=9)))
        lo, hi = MU[key][0], MU[key][1]
        fig.add_vrect(x0=lo, x1=hi, fillcolor='gray', opacity=0.08, line_width=0)
        fig.update_layout(xaxis=dict(title=f"{dc.PARAMS[key][0]} ({dc.PARAMS[key][1]})",
                                     type='log' if key != 'sRc' else 'linear'),
                          yaxis_title='T_j rise (K)', yaxis2=dict(title='τ (s)', overlaying='y', side='right'))
        show(fig, key=f"{K}|ma|swf", info="Steady junction rise of the three chips over the parameter (solid, left "
             "axis) and the slowest time constant (dotted, right axis); red: current design; gray band: training "
             "range of the parametric network.")
        i0, i1 = 0, len(v) - 1
        st.caption(f"From {fmt(v[i0])} to {fmt(v[i1])} {dc.PARAMS[key][1]}: T_j of T1 {fmt(T[i0, 0])} → "
                   f"{fmt(T[i1, 0])} K, slowest τ {fmt(Tau[i0])} → {fmt(Tau[i1])} s.")
    with t3:
        S = _sens(d['mk'], d['pk'], P)
        keys = list(dc.PARAMS)
        fig = go.Figure([go.Bar(x=[dc.PARAMS[k][0] for k in keys], y=[S[k][c] / 10 for k in keys],
                                name=CHIPS[c]['name'], marker_color=SERIES[c]) for c in range(3)])
        fig.update_layout(barmode='group', yaxis_title='ΔT_j for +10 % (K)')
        show(fig, height=320, key=f"{K}|ma|sens", info="Change of the steady junction rise when one parameter is "
             "raised by 10 % (linearised: dT_j/d ln μ · 0.1). Negative: the chip gets cooler.")
        top_ = max(keys, key=lambda k: abs(S[k][0]))
        st.caption(f"+10 % {dc.PARAMS[top_][0]} changes T_j of T1 by {S[top_][0] / 10:+.2f} K — the largest lever "
                   f"of the four; the base ρc factor has no steady effect ({S['sRc'][0] / 10:+.2f} K).")
    cm.equations([("Steady state", r"\boldsymbol\theta_{ss}=\mathbf G^{-1}\mathbf B_p\boldsymbol P,\qquad "
                   r"\theta_{j,c}=\max_{i\in\mathrm{die}\,c}\theta_{ss,i}"),
                  ("Sensitivity", r"\frac{\partial\theta_{j,c}}{\partial\ln\mu}\approx\frac{\theta_{j,c}(\mu(1+"
                   r"\epsilon))-\theta_{j,c}(\mu(1-\epsilon))}{\ln\frac{1+\epsilon}{1-\epsilon}}",
                   "Central difference with ε = 1 %; per 10 % multiply by ≈ 0.1."),
                  ("Time constants", r"\mathbf G\,\boldsymbol v=\lambda\,\mathbf C\,\boldsymbol v,\qquad\tau=1/\lambda")],
                 sym=[("θ_ss", "steady temperature rise of all nodes", "K"),
                      ("θ_j,c", "junction rise of chip c (hottest die node)", "K"),
                      ("μ", "parameter (h, k_diel, s_att, s_ρc)", "see inputs"),
                      ("ε", "relative step of the central difference (1 %)", "-"),
                      ("λ, v, τ", "eigenvalue, eigenvector (mode) and time constant", "1/s, -, s")])
    cm.refs(['Incropera2007'])
    dT = tj - tj0
    cm.interpret([
        f"**T_j of T1 is {fmt(tj[0])} K above ambient** ({dT[0]:+.1f} K against the nominal condition); the hottest "
        f"chip is {CHIPS[int(np.argmax(tj))]['name']}.",
        "Why: in steady state the heat of every chip must leave through the dielectric and the cold plate. These "
        "two resistances are in series with the whole board, so h and k_diel act on all chips almost equally, while "
        "the die attach only acts under each chip and ρc not at all.",
        f"Lever: {dc.PARAMS[top_][0]} (+10 % → {S[top_][0] / 10:+.2f} K at T1). Cooling gains saturate once the "
        "dielectric dominates — compare the slope of the sweep at the two ends.",
        "Caveats: constant material properties, uniform loss over the die, adiabatic board edges."])


#######################################################################################################################
# 3) Move a chip
#######################################################################################################################
def _move(d):
    c = st.columns(4)
    chip = c[0].selectbox("Chip to move", range(3), format_func=lambda i: CHIPS[i]['name'], key=f"{K}|mv|c")
    axis = c[1].segmented_control("Direction", [0, 1], format_func=lambda a: 'x (30 mm side)' if a == 0 else
                                  'y (20 mm side)', default=0, key=f"{K}|mv|a")
    axis = 0 if axis is None else axis
    ch = CHIPS[chip]
    half = (ch['w'] if axis == 0 else ch['l']) / 2
    vals = tuple(np.arange(np.ceil(MARGIN + half), np.floor((W_MM if axis == 0 else L_MM) - MARGIN - half) + 1))
    pic = _head(2, f"Junction temperatures while {ch['name']} moves along {'x' if axis == 0 else 'y'}",
                "All other chips stay at their positions. Positions where the chips would overlap or leave the 1 mm "
                "margin are skipped.")
    _pic(pic, art.top(d['pos'], CHIPS, True, chip, d['mu']['h']))
    T = _sweepPos(d['mk'], d['pk'], d['P'], chip, axis, vals)
    tj = _steady(d['mk'], d['pk'], d['P'])[0]
    R = _coupling(d['mk'], d['pk'])
    ok = np.isfinite(T[:, chip])
    Tmax = np.where(ok, np.max(np.where(np.isfinite(T), T, -np.inf), axis=1), np.inf)
    best = vals[int(np.argmin(Tmax))] if ok.any() else np.nan
    cur = d['pos'][chip][axis]
    cm.section(None, "Result")
    cm.tiles([(f"T_j {ch['name']} now", f"{fmt(tj[chip])} K", "Steady junction rise at the current position."),
              (f"range over {'x' if axis == 0 else 'y'}", f"{fmt(np.nanmin(T[:, chip]))}–{fmt(np.nanmax(T[:, chip]))} K",
               "Lowest and highest junction rise of the moved chip over all valid positions."),
              ("best position", f"{fmt(best)} mm", "Position with the lowest maximum junction rise of all three "
               "chips (the hottest chip limits the design)."),
              ("self / max. coupling", f"{fmt(R[chip, chip])} / {fmt(np.max(np.delete(R[chip], chip)))} K/W",
               "R_th,cc: rise of the chip per W of its own loss; R_th,cj: largest rise per W caused by another chip "
               "(heating through the board).")])
    t1, t2 = st.tabs(["Temperature over the position", "Coupling between the chips"])
    with t1:
        fig = go.Figure([line(vals, T[:, c], CHIPS[c]['name'], c) for c in range(3)])
        fig.add_trace(go.Scatter(x=[cur] * 3, y=tj, mode='markers', name='current', marker=dict(color='red', size=9)))
        fig.update_layout(xaxis_title=f"{'x' if axis == 0 else 'y'} of {ch['name']} (mm)", yaxis_title='T_j rise (K)')
        show(fig, key=f"{K}|mv|f", info="Steady junction rise of all chips over the position of the moved chip; gaps "
             "are invalid layouts; red: current position. Near an edge the heat can spread to one side only (the "
             "edges are adiabatic); near another chip the two heat each other.")
        i = int(np.nanargmin(T[:, chip]))
        st.caption(f"{ch['name']} is coolest at {fmt(vals[i])} mm ({fmt(T[i, chip])} K) and hottest at "
                   f"{fmt(vals[int(np.nanargmax(T[:, chip]))])} mm ({fmt(np.nanmax(T[:, chip]))} K).")
    with t2:
        df = pd.DataFrame(R, columns=[f"from {c['name']}" for c in CHIPS], index=[f"T_j {c['name']}" for c in CHIPS])
        table(df.reset_index().rename(columns={'index': 'rise of'}), hide_index=True)
        st.caption("R_th matrix (K/W) of the current layout: row = chip whose junction rises, column = chip whose "
                   "loss of 1 W causes it. The diagonal is the self-heating, the off-diagonal the heating through "
                   f"the board. With the current losses, {CHIPS[chip]['name']} gets "
                   f"{fmt(float(R[chip] @ np.asarray(d['P']) - R[chip, chip] * d['P'][chip]))} K from its neighbours.")
    cm.equations([("Superposition of the chips", r"\theta_{j,c}=\sum_{c'}R_{th,cc'}\,P_{c'},\qquad "
                   r"R_{th,cc'}=\max_{i\in\mathrm{die}\,c}\bigl(\mathbf G^{-1}\mathbf B_{p}\bigr)_{ic'}",
                   "Exact for the linear board (max taken per source)."),
                  ("Position sensitivity", r"\frac{\partial\theta_{j,c}}{\partial x_{c'}}\approx"
                   r"\frac{\theta_{j,c}(x_{c'}+1\,\mathrm{mm})-\theta_{j,c}(x_{c'}-1\,\mathrm{mm})}{2\,\mathrm{mm}}")],
                 sym=[("R_th,cc'", "rise of chip c per W of chip c' (self-heating for c = c')", "K/W"),
                      ("P_c'", "loss of chip c'", "W"), ("x_c'", "centre of chip c' along x (or y)", "mm"),
                      ("c, c'", "chip indices T1, D1, M1", "-")])
    nb = int(np.argmax(np.where(np.arange(3) == chip, -1, R[chip])))
    cm.interpret([
        f"**Moving {ch['name']} changes its junction rise by {fmt(np.nanmax(T[:, chip]) - np.nanmin(T[:, chip]))} K "
        f"over the valid range** (current {fmt(tj[chip])} K); the best position for the hottest chip is "
        f"{fmt(best)} mm.",
        "Why: the self-heating is set mostly by the stack under the chip (dielectric, cooling), which does not move. "
        f"Position acts through two smaller effects: spreading (an edge blocks one side) and coupling — "
        f"{CHIPS[nb]['name']} adds {fmt(R[chip, nb])} K per W to {ch['name']}.",
        "Lever: keep the strongest sources apart and away from the edges; for a systematic search, the parametric "
        "network gives smooth gradients of T_j over the positions (layout optimisation in the PF study).",
        "Caveats: positions on the 1 mm grid of the model; the loss is uniform over the die."])


#######################################################################################################################
# 4) Transients
#######################################################################################################################
def _transient(d):
    c = st.columns(4)
    tOn = cm.num(c[0], "Loss on for (s)", 10.0, f"{K}|tr|on", min_value=0.1, max_value=200.0,
                 help="All chips switch to their losses at t = 0 and off after this time (heating and cooling curve).")
    tEnd = cm.num(c[1], "Simulated time (s)", 20.0, f"{K}|tr|end", min_value=0.5, max_value=600.0,
                  help="Length of the profile.")
    Ts = cm.num(c[2], "Sampling time (s)", 0.05, f"{K}|tr|ts", min_value=0.005, max_value=1.0,
                help="Sample of the loss profile (zero-order hold).")
    pic = _head(2, "Heating and cooling of the junctions", "Exact solution of a block Krylov reduction of the FV model "
                "(18 states, junction error < 0.05 K) for the current design and for the nominal condition at the "
                "same positions.")
    tau = _steady(d['mk'], d['pk'], d['P'])[2]
    tau0 = _steady(_key(NOM, d['pos'])[0], d['pk'], d['P'])[2]
    _pic(pic, art.top(d['pos'], CHIPS, True, None, d['mu']['h']))
    t, PP, y, tm = _step(d['mk'], d['pk'], d['P'], tOn, tEnd, Ts)
    y0 = _step(_key(NOM, d['pos'])[0], d['pk'], d['P'], tOn, tEnd, Ts)[2]
    i63 = np.argmax(y[:, 0] >= 0.632 * np.max(y[t <= tOn, 0])) if np.max(y[:, 0]) > 0 else 0
    cm.section(None, "Result")
    cm.tiles([("τ₁ / τ₂ / τ₃", " / ".join(fmt(v) for v in tau) + " s", "Three slowest time constants of the board "
               "(nominal: " + " / ".join(fmt(v) for v in tau0) + " s)."),
              ("peak T_j T1", f"{fmt(np.max(y[:, 0]))} K", f"Highest junction rise of T1 (nominal condition "
               f"{fmt(np.max(y0[:, 0]))} K)."),
              ("63 % of the peak after", f"{fmt(t[i63])} s", "Time until T1 reaches 63 % of its highest rise during "
               "the on-phase (one 'apparent' time constant)."),
              ("solve time", f"{fmt(1e3 * tm)} ms", f"Krylov model for {len(t)} samples (FV with 4 sub-steps per "
               "sample: ≈ 2 ms per sample).")])
    fig = go.Figure([line(t, y[:, c], CHIPS[c]['name'], c) for c in range(3)] +
                    [line(t, y0[:, c], f"{CHIPS[c]['name']} nominal", c, dash='dot', width=1) for c in range(3)])
    fig.update_layout(xaxis_title='t (s)', yaxis_title='T_j rise (K)')
    show(fig, key=f"{K}|tr|f", info="Junction rises of the current design (solid) and of the nominal condition "
         "(dotted). The fast first rise (ms) is the die and attach, the slow part (seconds) the base and the "
         "cooling.")
    st.caption(f"After {fmt(tOn)} s T1 is at {fmt(y[t <= tOn, 0][-1])} K of a steady {fmt(_steady(d['mk'], d['pk'], d['P'])[0][0])} K; "
               f"the slowest time constant is {fmt(tau[0])} s (nominal {fmt(tau0[0])} s).")
    cm.equations([("Modal solution of the linear board", r"\boldsymbol\theta(t)=\sum_m\boldsymbol v_m\boldsymbol v_m^"
                   r"\top\mathbf B_p\int_0^t e^{-\lambda_m(t-\tau)}\boldsymbol P(\tau)\,d\tau"),
                  ("Krylov reduction (moment matching at s = 0)", r"\mathbf V=\mathrm{orth}\bigl[\mathbf G^{-1}"
                   r"\mathbf B_p,\;(\mathbf G^{-1}\mathbf C)\mathbf G^{-1}\mathbf B_p,\dots\bigr],\quad"
                   r"\boldsymbol\theta\approx\mathbf V\boldsymbol x",
                   "6 blocks × 3 chips = 18 states; the steady state and the slow response are exact.")],
                 sym=[("v_m, λ_m", "mode m and its decay rate (C-orthonormal)", "-, 1/s"),
                      ("V, x", "reduced basis and its states", "-, K"), ("τ", "integration variable", "s")])
    cm.refs(['Odabasioglu1998', 'Benner2015'])
    cm.interpret([
        f"**The slowest time constant is {fmt(tau[0])} s** ({'longer' if tau[0] > tau0[0] else 'shorter'} than the "
        f"nominal {fmt(tau0[0])} s); T1 peaks at {fmt(np.max(y[:, 0]))} K.",
        "Why: the slow mode is the heat capacity of the Al base discharging through the cold plate, τ ≈ ρ·c·t_base/h "
        "for a thin base. Lower h or higher ρc slow it down; the dielectric and die attach shape the first second.",
        "Lever: for short pulses the die attach and the copper matter (fast modes), for long loads the cooling.",
        "Caveats: zero-order hold of the losses; constant material properties."])


#######################################################################################################################
# 5) Parametric network
#######################################################################################################################
def _network(d):
    files = sorted(glob.glob(os.path.join(RES, '*.pt')))
    cm.section(2, "Prediction of the parametric network", "One network for all cooling, material and position "
               "variants, trained once (setup/tutorial/studyPFParam.py). It predicts the same step test as the tab "
               "*Transients* without any matrix solve.", divider=False)
    if not files:
        st.info("No trained network found in results/pfparam/. Create it with\n\n"
                "`python setup/tutorial/genBoardPFParam.py` (board families, ≈ 5 min) and\n"
                "`python setup/tutorial/studyPFParam.py base --out results/pfparam` (training, ≈ 20 min on a GPU).",
                icon=':material/info:')
        return
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    run = st.selectbox("Trained network", names, index=names.index('base') if 'base' in names else 0,
                       key=f"{K}|nn|run", help="Runs of studyPFParam: base = 24 labelled + 200 physics-only boards; "
                       "physAll = physics boards at any position; physExt = physics boards over the extended range; "
                       "nophys = without physics; n3/n6/n12 = fewer labelled boards; K8/K16 = fewer space functions.")
    f = files[names.index(run)]
    nd = _net(f, os.path.getmtime(f))
    tOn, tEnd, Ts = 10.0, 20.0, 0.05
    t, PP, y, tm = _step(d['mk'], d['pk'], d['P'], tOn, tEnd, Ts)
    B = dc.board(d['mu'], d['pos'])
    t0 = time.time()
    yn = dc.netJunction(nd, B, PP, Ts)
    tn = time.time() - t0
    e = yn - y
    out = dc.inRange(d['mu'])
    cm.tiles([("T_j RMSE", f"{fmt(np.sqrt(np.mean(e ** 2)))} K", "Root mean squared error of the three junction "
               "rises against the exact model over the 20 s step test."),
              ("max. error", f"{fmt(np.max(np.abs(e)))} K", "Largest junction error; typically at the switching "
               "instants, where the die responds within milliseconds."),
              ("network time", f"{fmt(1e3 * tn)} ms", "Prediction on the CPU (the GPU is faster for long profiles)."),
              ("in training range", "yes" if not out else "no", "Outside the training range the network "
               "extrapolates: " + (", ".join(dc.PARAMS[k][0] for k in out) if out else "all parameters inside."))])
    fig = go.Figure([line(t, y[:, c], f"{CHIPS[c]['name']} model", c) for c in range(3)] +
                    [line(t, yn[:, c], f"{CHIPS[c]['name']} network", c, dash='dash') for c in range(3)])
    fig.update_layout(xaxis_title='t (s)', yaxis_title='T_j rise (K)')
    show(fig, key=f"{K}|nn|f", info="Junction rises of the exact model (solid) and of the network (dashed) for the "
         "current design; change the cooling, materials or positions above and the network follows without "
         "retraining.")
    st.caption(f"Network '{run}': junction RMSE {fmt(np.sqrt(np.mean(e ** 2)))} K for this design "
               f"({'inside' if not out else 'outside'} the training range).")
    js = sorted(glob.glob(os.path.join(RES, '*.json')))
    rows = []
    for j in js:
        try:
            r = json.load(open(j))
        except (OSError, ValueError):
            continue
        if not all(k in r for k in ('run', 'train', 'phys', 't1', 't2', 't3', 't4')):
            continue
        rows.append({'run': r['run'], 'labelled': r['train'], 'physics-only': r['phys'],
                     'T1 (K)': r['t1']['sum']['rmse'], 'T2 extrap. (K)': r['t2']['sum']['rmse'],
                     'T3 positions (K)': r['t3']['sum']['rmse'], 'T4 30 min (K)': r['t4']['rmse'],
                     'training (min)': r['time'] / 60})
    if rows:
        cm.section(3, "Results of the study", "RMSE over all nodes and time steps of the testing sets "
                   "(setup/tutorial/studyPFParam.py).")
        df = pd.DataFrame(rows).sort_values('T1 (K)')
        table(df, hide_index=True)
        st.caption("labelled: boards with FV temperatures · physics-only: boards with only C, G, Bp (no simulation) · "
                   "T1: new conditions and positions in the training range · T2: cooling and dielectric outside the "
                   "range · T3: chip centres in a band never seen in training · T4: 30 min mission profile.")
    cm.interpret([
        f"**The network '{run}' reproduces the junction rises of this design with "
        f"{fmt(np.sqrt(np.mean(e ** 2)))} K RMSE** in {fmt(1e3 * tn)} ms, without retraining.",
        "Why: it keeps the structure of the exact modal solution — space functions that move with the chip and "
        "first-order lags with learned time constants — and was trained on the operators of hundreds of boards.",
        "Lever: outside the training range add physics-only boards there (no simulation needed, see run physExt).",
        "Caveats: linear board, 3 chip types; the Krylov model is more accurate for a single known design — the "
        "network pays off for many designs, long profiles and gradient-based layout optimisation."])
