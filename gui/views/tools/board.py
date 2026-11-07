#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         board.py
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
Board designer: chips of a selected technology are placed on a board of a selected technology (position table), the
board is simulated with a finite-volume model in the top view (x-y, lumped stack-up) or a cross-section (x-z, every
layer resolved), a POD-Galerkin reduced model is built from the result and the data are exported in the 2D format of
TherMOS (data/board/<name>_train/_test.mat + config/gui/board_<name>.json) to train the PO/PS modes on them.
Toolkit functions: src/py/board/{genBoard, calcBoardMat, simBoard, genProfile, romBoard, saveBoard, boardLib}.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, SEQ, INK2, MUTED, show, line, fmt
from gui.views.tools import _common as cm
from src.py.board.boardLib import BOARDS, CHIPS, COOL, H_TOP
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.genProfile import genProfile
from src.py.board.romBoard import romBoard, simRom
from src.py.board.saveBoard import saveBoard

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|board"
PROFILES = ['step', 'pulse', 'sine', 'random']
CHIP_COLS = ['name', 'tech', 'x [mm]', 'y [mm]', 'w [mm]', 'l [mm]', 'P [W]', 'profile', 'period [s]']
DEFAULT_CHIPS = [['T1', 'Si IGBT', 15.0, 15.0, 10.0, 10.0, 60.0, 'pulse', 40.0],
                 ['D1', 'Si diode', 30.0, 15.0, 6.0, 8.0, 20.0, 'pulse', 40.0],
                 ['M1', 'SiC MOSFET', 45.0, 28.0, 4.0, 5.0, 25.0, 'sine', 60.0]]


#######################################################################################################################
# Functions
#######################################################################################################################
def _layout(view):
    """Layout dictionary of genBoard from the widgets."""
    s = st.session_state
    df = s.get(f"{K}|chips_df")
    chips = []
    for _, r in (df if df is not None else pd.DataFrame(DEFAULT_CHIPS, columns=CHIP_COLS)).iterrows():
        if pd.isna(r['tech']) or r['tech'] not in CHIPS or pd.isna(r['x [mm]']) or pd.isna(r['y [mm]']):
            continue
        t = CHIPS[r['tech']]
        chips.append({'name': str(r['name']) if not pd.isna(r['name']) else r['tech'], 'tech': r['tech'],
                      'x': float(r['x [mm]']), 'y': float(r['y [mm]']),
                      'w': float(t['w'] if pd.isna(r['w [mm]']) else r['w [mm]']),
                      'l': float(t['l'] if pd.isna(r['l [mm]']) else r['l [mm]']),
                      'P': float(t['P'] if pd.isna(r['P [W]']) else r['P [W]']),
                      'profile': r['profile'] if r['profile'] in PROFILES else 'step',
                      'period': float(40.0 if pd.isna(r['period [s]']) else r['period [s]'])})
    return {'view': view, 'W': s.get(f"{K}|W", 60.0), 'L': s.get(f"{K}|L", 40.0),
            'board': s.get(f"{K}|tech", 'DBC Al₂O₃ (0.38 mm)'), 'h': s.get(f"{K}|h", 5000.0),
            'hTop': s.get(f"{K}|hTop", H_TOP), 'Ta': s.get(f"{K}|Ta", 25.0), 'dx': s.get(f"{K}|dx", 1.0),
            'yCut': s.get(f"{K}|yCut", 15.0), 'chips': chips}


def _profiles(lay, D, t, kind=None, seed=0):
    """Loss profiles (Nt x nChips) of the chips used in D (designed profiles, or random ones for training)."""
    P = []
    for i, c in enumerate(D['chips']):
        k = kind or c.get('profile', 'step')
        P.append(genProfile(k, c['P'], t, {'period': c.get('period', 40.0), 'duty': 0.5, 'amp': 0.5,
                                         'hold': max(c.get('period', 40.0) / 4, t[1] - t[0])}, seed=seed + i))
    return np.column_stack(P) if P else np.zeros((len(t), 0))


def _tj(D, T, P):
    """Junction temperature of every chip (Nt x nChips): hottest board node under the chip + P R_jb (top view) or
    hottest die node (cross-section)."""
    T = np.atleast_2d(T)
    out = np.zeros((T.shape[0], len(D['chips'])))
    for i in range(len(D['chips'])):
        m = np.asarray(D['B'])[:, i] > 0
        out[:, i] = T[:, m].max(axis=1) + (np.atleast_2d(P)[:, i] * D['Rjb'][i] if D['view'] == 'xy' else 0.0)
    return out


def _heat(F, x, y, unit, title, equal=True, zlim=None, scale=SEQ):
    fig = go.Figure(go.Heatmap(z=F, x=x * 1e3, y=y * 1e3, colorscale=scale, zmin=None if zlim is None else zlim[0],
                               zmax=None if zlim is None else zlim[1], colorbar=dict(title=unit, len=0.9),
                               hovertemplate='x %{x:.1f} mm<br>%{y:.2f} mm<br>%{z:.3g} ' + unit + '<extra></extra>'))
    fig.update_layout(title=title, xaxis_title='x (mm)', yaxis_title='y (mm)' if equal else 'z (mm)')
    if equal:
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


def _top_fig(lay):
    """Top view of the layout: board outline, chips, cut line of the cross-section."""
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=lay['W'], y1=lay['L'], line=dict(color=INK2, width=2),
                  fillcolor='rgba(180,170,140,0.25)')
    for i, c in enumerate(lay['chips']):
        col = SERIES[i % len(SERIES)]
        fig.add_shape(type='rect', x0=c['x'] - c['w'] / 2, x1=c['x'] + c['w'] / 2, y0=c['y'] - c['l'] / 2,
                      y1=c['y'] + c['l'] / 2, line=dict(color=col, width=2), fillcolor=col, opacity=0.55)
        fig.add_annotation(x=c['x'], y=c['y'], text=f"<b>{c['name']}</b><br>{fmt(c['P'])} W", showarrow=False,
                           font=dict(size=11))
    fig.add_trace(go.Scatter(x=[0, lay['W']], y=[lay['yCut']] * 2, mode='lines', name='cross-section cut',
                             line=dict(color=MUTED, dash='dash')))
    fig.update_xaxes(range=[-2, lay['W'] + 2], title='x (mm)')
    fig.update_yaxes(range=[-2, lay['L'] + 2], title='y (mm)', scaleanchor='x', scaleratio=1)
    fig.update_layout(title='Top view (x-y)', showlegend=True)
    return fig


def _cut_fig(lay):
    """Cross-section of the stack-up at y = yCut (thickness not to scale with x)."""
    fig = go.Figure()
    z = 0.0
    for i, l in enumerate(reversed(BOARDS[lay['board']])):
        fig.add_shape(type='rect', x0=0, x1=lay['W'], y0=z, y1=z + l[2], line=dict(width=0.5, color=INK2),
                      fillcolor='#c98b4b' if l[1] in ('Cu',) else ('#b8c2cc' if 'Al' in l[1] else '#9fc39b'),
                      opacity=0.8)
        fig.add_annotation(x=lay['W'] * 1.01, y=z + l[2] / 2, text=f"{l[0]} · {l[2]} mm · k={fmt(l[4])}",
                           showarrow=False, xanchor='left', font=dict(size=10))
        z += l[2]
    for i, c in enumerate(lay['chips']):
        if abs(c['y'] - lay['yCut']) > c['l'] / 2:
            continue
        t = CHIPS[c['tech']]
        fig.add_shape(type='rect', x0=c['x'] - c['w'] / 2, x1=c['x'] + c['w'] / 2, y0=z, y1=z + t['t_att'],
                      fillcolor='#8a8a8a', line=dict(width=0.5))
        fig.add_shape(type='rect', x0=c['x'] - c['w'] / 2, x1=c['x'] + c['w'] / 2, y0=z + t['t_att'],
                      y1=z + t['t_att'] + t['t'], fillcolor=SERIES[i % len(SERIES)], line=dict(width=0.5))
        fig.add_annotation(x=c['x'], y=z + t['t_att'] + t['t'], text=c['name'], showarrow=False, yanchor='bottom')
    fig.add_annotation(x=lay['W'] / 2, y=-0.08 * z, text=f"coolant h = {fmt(lay['h'])} W/m²K", showarrow=False)
    fig.update_xaxes(range=[-2, lay['W'] * 1.45], title='x (mm)')
    fig.update_yaxes(range=[-0.2 * z, z * 1.6 + 0.3], title='z (mm)')
    fig.update_layout(title=f"Cross-section (x-z) at y = {fmt(lay['yCut'])} mm (z not to scale)")
    return fig


#######################################################################################################################
# Steps
#######################################################################################################################
def board_inputs():
    cm.section(1, "Board and cooling", "Board technology (stack-up from the material library), size, cooling of the "
                                        "bottom side and ambient temperature.", divider=False)
    c = st.columns(4)
    tech = c[0].selectbox("Board technology", list(BOARDS), index=list(BOARDS).index('DBC Al₂O₃ (0.38 mm)'),
                          key=f"{K}|tech", help="Layer stack-up of the board (top → bottom), see the table below.")
    cm.num(c[1], "Width W (mm)", 60.0, 'board|W', min_value=5.0, max_value=500.0, step=5.0)
    cm.num(c[2], "Length L (mm)", 40.0, 'board|L', min_value=5.0, max_value=500.0, step=5.0)
    cm.num(c[3], "Ambient / coolant T_a (°C)", 25.0, 'board|Ta', min_value=-40.0, max_value=150.0)
    c = st.columns(4)
    cool = c[0].selectbox("Cooling of the bottom side", list(COOL), index=list(COOL).index(
        'Liquid cold plate (via grease)'), key=f"{K}|cool", help="Typical heat transfer coefficients (toolData).")
    if st.session_state.get(f"{K}|cool_last") != cool:                       # the preset sets h, the user may edit
        st.session_state[f"{K}|h"] = float(COOL[cool])
        st.session_state[f"{K}|cool_last"] = cool
    c[1].number_input("h bottom (W/m²K)", min_value=0.1, key=f"{K}|h", format='%g', help="Heat transfer "
                      "coefficient of the cooling on the bottom of the board (incl. the interface to the heat sink).")
    cm.num(c[2], "h top (W/m²K)", H_TOP, 'board|hTop', min_value=0.0, help="Natural convection on the top side.")
    cm.num(c[3], "Grid spacing dx (mm)", 1.0, 'board|dx', min_value=0.1, max_value=10.0, step=0.25,
           help="Node spacing of the finite-volume grid (dx = dy). Chips should span at least 3 nodes.")
    rows = [{'layer': l[0], 'material': l[1], 't (mm)': l[2], 'k in-plane (W/mK)': l[3], 'k through (W/mK)': l[4],
             'ρ (kg/m³)': l[5], 'c (J/kgK)': l[6]} for l in BOARDS[tech]]
    with st.expander(f"Stack-up of '{tech}' (top → bottom)", icon=':material/layers:'):
        table(pd.DataFrame(rows), hide_index=True)
        cm.refs(['Incropera2007', 'Wintrich2015'])


def chip_inputs():
    cm.section(2, "Chips", "One row per chip: technology, centre position (x, y from the lower left corner of the "
                           "board), footprint, loss and loss profile. Empty w, l or P take the values of the "
                           "technology. Rows can be added at the bottom and deleted with the row menu.")
    df0 = pd.DataFrame(DEFAULT_CHIPS, columns=CHIP_COLS)
    df = st.data_editor(df0, key=f"{K}|chips", num_rows='dynamic', hide_index=True, width='stretch', column_config={
        'tech': st.column_config.SelectboxColumn('tech', options=list(CHIPS), required=True,
                                                 help="Chip technology (die material, thickness, die attach)."),
        'profile': st.column_config.SelectboxColumn('profile', options=PROFILES, help="Loss profile P(t): step, "
                                                    "pulse (50 % duty), sine (between P/3 and P), random levels."),
        'x [mm]': st.column_config.NumberColumn(format='%.1f'), 'y [mm]': st.column_config.NumberColumn(format='%.1f'),
        'P [W]': st.column_config.NumberColumn(format='%.1f', help="Loss of the chip (peak of the profile)."),
        'period [s]': st.column_config.NumberColumn(format='%.0f', help="Period of pulse/sine, 4× hold of random.")})
    st.session_state[f"{K}|chips_df"] = df
    with st.expander("Chip technologies", icon=':material/memory:'):
        table(pd.DataFrame([{'technology': n, 'die': c['mat'], 't die (mm)': c['t'], 'w × l (mm)': f"{c['w']} × "
                             f"{c['l']}", 'die attach': c['attach'], 't attach (mm)': c['t_att'], 'P typ. (W)': c['P'],
                             'T_j,max (°C)': c['Tjmax'], 'note': c['note']} for n, c in CHIPS.items()]),
              hide_index=True)

    lay = _layout('xy')
    out = [c['name'] for c in lay['chips'] if c['x'] - c['w'] / 2 < 0 or c['x'] + c['w'] / 2 > lay['W']
           or c['y'] - c['l'] / 2 < 0 or c['y'] + c['l'] / 2 > lay['L']]
    if out:
        st.warning(f"Chips outside the board: {', '.join(out)} (they are clipped at the edge).", icon=':material/warning:')
    c = st.columns([3, 2])
    with c[0]:
        show(_top_fig(lay), height=420, key=f"{K}|topfig")
    with c[1]:
        cm.num(st, "Cut of the cross-section y (mm)", 15.0, 'board|yCut', min_value=0.0, max_value=float(lay['L']))
        show(_cut_fig(_layout('xz')), height=330, key=f"{K}|cutfig")


def simulation():
    cm.section(3, "Simulation", "Finite-volume model of the board, transient run with the loss profiles of the chips "
                                "and steady state for the last losses.")
    c = st.columns(4)
    view = c[0].segmented_control("View", ['Top view (x-y)', 'Cross-section (x-z)'], default='Top view (x-y)',
                                  key=f"{K}|view", help="Top view: board lumped over its thickness, all chips. "
                                  "Cross-section: every layer resolved, only the chips cut by y = y_cut.") \
        or 'Top view (x-y)'
    cm.num(c[1], "Duration t_end (s)", 200.0, 'board|tEnd', min_value=1.0, max_value=1e5)
    cm.num(c[2], "Sampling time T_s (s)", 1.0, 'board|Ts', min_value=1e-3, max_value=100.0)
    cm.num(c[3], "Sub-steps per T_s", 2, 'board|nSub', format='%d', min_value=1, max_value=50,
           help="Implicit Euler steps per sampling period (accuracy of fast transients).")
    v = 'xy' if view.startswith('Top') else 'xz'
    lay = _layout(v)
    s = st.session_state
    tEnd, Ts, nSub = s[f"{cm.KEY}|board|tEnd"], s[f"{cm.KEY}|board|Ts"], s[f"{cm.KEY}|board|nSub"]
    try:
        D = genBoard(lay)
    except Exception as e:
        st.error(f"Board could not be built: {e}", icon=':material/error:')
        return None
    st.caption(f"Grid {D['shape'][1]} × {D['shape'][0]} = {D['geo'].shape[0]} nodes, dx = {fmt(D['dx'] * 1e3)} mm, "
               f"{'dz' if v == 'xz' else 'dy'} = {fmt(D['dy'] * 1e3)} mm, {len(D['chips'])} chips"
               + (", z exaggerated in the plots" if v == 'xz' else ""))
    if not D['chips']:
        st.info("No chip in this view (cross-section: move the cut through a chip).", icon=':material/info:')
        return None
    if D['geo'].shape[0] > 60000:
        st.warning("More than 60 000 nodes: increase dx.", icon=':material/warning:')
        return None
    sig = json.dumps([lay, tEnd, Ts, nSub], sort_keys=True, default=str)
    res = s.get(f"{K}|res")
    if st.button("Simulate", icon=':material/play_arrow:', type='primary', key=f"{K}|run") or \
            (res is not None and res['sig'] != sig and s.get(f"{K}|auto", False)):
        with st.spinner("Simulating the finite-volume model …"):
            t0 = time.perf_counter()
            C, G, Bp = calcBoardMat(D)
            t = np.arange(0.0, tEnd + Ts * 1e-9, Ts)
            P = _profiles(lay, D, t)
            T, Tss = simBoard(C, G, Bp, P, Ts, lay['Ta'], nSub)
            s[f"{K}|res"] = res = {'sig': sig, 'lay': lay, 'D': D, 't': t, 'P': P, 'T': T, 'Tss': Tss, 'C': C,
                                   'G': G, 'Bp': Bp, 'nSub': nSub, 'time': time.perf_counter() - t0}
    if res is None:
        st.info("Press *Simulate* to run the board model.", icon=':material/info:')
        return None
    if res['sig'] != sig:
        st.caption(":material/history: The results belong to the previous inputs — press *Simulate* to update.")
    return res


def results(res):
    D, t, P, T, Tss, lay = res['D'], res['t'], res['P'], res['T'], res['Tss'], res['lay']
    ny, nx = D['shape']
    xy = D['view'] == 'xy'
    Tj = _tj(D, T, P)
    Tjs = _tj(D, Tss[None, :], P[-1:])[0]
    names = [c['name'] for c in D['chips']]
    cm.section(None, "Result")
    cm.tiles([("T_max board (steady)", f"{fmt(Tss.max())} °C", "Hottest node of the steady state for the last "
               "losses."), ("T_j,max (transient)", f"{fmt(Tj.max())} °C", "Highest junction temperature of the run."),
              ("Total loss (last)", f"{fmt(P[-1].sum())} W", None),
              ("Simulation time", f"{fmt(res['time'])} s", f"{D['geo'].shape[0]} nodes × {len(t)} steps")])
    rows = []
    for i, c in enumerate(D['chips']):
        m = c['Tjmax'] - Tj[:, i].max()
        lvl = 'good' if m > 25 else ('warning' if m > 0 else 'critical')
        rows.append({'chip': c['name'], 'technology': c['tech'], 'P last (W)': P[-1, i],
                     'T_j steady (°C)': Tjs[i], 'T_j max run (°C)': Tj[:, i].max(), 'T_j,max (°C)': c['Tjmax'],
                     'margin (K)': m, 'R_jb (K/W)': D['Rjb'][i] if xy else np.nan,
                     'status': {'good': 'OK', 'warning': 'tight', 'critical': 'too hot'}[lvl]})
    table(pd.DataFrame(rows), hide_index=True)
    st.caption("T_j: hottest board node under the chip + P·R_jb (die and die attach, top view) or hottest node of the "
               "die (cross-section). Margin = T_j,max − T_j max of the run; OK > 25 K, tight 0–25 K.")
    tabs = st.tabs(["Steady state", "Transient field", "Junction temperatures", "Losses"])
    with tabs[0]:
        show(_heat(Tss.reshape(ny, nx), D['x'], D['y'], '°C', 'Steady-state temperature', equal=xy),
             height=430 if xy else 330, key=f"{K}|ss")
    with tabs[1]:
        i = st.slider("Time step", 0, len(t) - 1, len(t) - 1, key=f"{K}|tsl")
        st.caption(f"t = {fmt(t[i])} s")
        show(_heat(T[i].reshape(ny, nx), D['x'], D['y'], '°C', f"Temperature at t = {fmt(t[i])} s", equal=xy,
                   zlim=[lay['Ta'], max(T.max(), lay['Ta'] + 1e-6)]), height=430 if xy else 330, key=f"{K}|tr")
    with tabs[2]:
        fig = go.Figure([line(t, Tj[:, i], n, i) for i, n in enumerate(names)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='T_j (°C)')
        show(fig, key=f"{K}|tj")
    with tabs[3]:
        fig = go.Figure([line(t, P[:, i], n, i) for i, n in enumerate(names)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)')
        show(fig, key=f"{K}|p")
    return Tj


def pod(res):
    cm.section(4, "POD reduced model", "Proper orthogonal decomposition of the simulated field and Galerkin "
                                       "projection of the finite-volume model onto the first K modes.")
    D, t, T, lay = res['D'], res['t'], res['T'], res['lay']
    ny, nx = D['shape']
    Kmax = int(min(30, len(t)))
    c = st.columns([1, 3])
    Kr = c[0].slider("Modes K", 1, Kmax, min(5, Kmax), key=f"{K}|K")
    rom = romBoard(res['C'], res['G'], res['Bp'], T, lay['Ta'], Kr)
    t0 = time.perf_counter()
    Tr = simRom(rom, res['P'], t[1] - t[0] if len(t) > 1 else 1.0, lay['Ta'], res['nSub'])
    tr = time.perf_counter() - t0
    err = np.abs(Tr - T)
    rise = max(T.max() - lay['Ta'], 1e-9)
    c[0].metric("Captured energy E_K", f"{100 * rom['E'][Kr - 1]:.4f} %")
    c[0].metric("Max. error", f"{fmt(err.max())} K", f"{100 * err.max() / rise:.2f} % of the rise", delta_color='off')
    c[0].metric("Unknowns", f"{Kr} instead of {D['geo'].shape[0]}", f"{fmt(res['time'] / max(tr, 1e-9))}× faster",
                delta_color='off')
    with c[1]:
        tabs = st.tabs(["Energy", "Mode shapes", "Error over time"])
        with tabs[0]:
            n = min(len(rom['sig']), 40)
            fig = go.Figure([go.Bar(x=np.arange(1, n + 1), y=rom['sig'][0:n] ** 2 / np.sum(rom['sig'] ** 2),
                                    name='energy of mode i', marker_color=SERIES[0]),
                             go.Scatter(x=np.arange(1, n + 1), y=rom['E'][0:n], name='captured E_K', yaxis='y2',
                                        line=dict(color=SERIES[1]))])
            fig.update_layout(xaxis_title='mode i', yaxis=dict(title='σ_i² / Σσ²', type='log'),
                              yaxis2=dict(title='E_K', overlaying='y', side='right', range=[0, 1.02]))
            show(fig, key=f"{K}|en")
        with tabs[1]:
            k = st.slider("Mode", 1, Kr, 1, key=f"{K}|mode")
            show(_heat(rom['Phi'][:, k - 1].reshape(ny, nx), D['x'], D['y'], '–', f"POD mode {k}",
                       equal=D['view'] == 'xy', scale='RdBu_r'), height=380, key=f"{K}|ph")
        with tabs[2]:
            fig = go.Figure([line(t, err.max(axis=1), 'max |T_ROM − T_FV|', 0),
                             line(t, err.mean(axis=1), 'mean |T_ROM − T_FV|', 1)])
            fig.update_layout(xaxis_title='t (s)', yaxis_title='error (K)')
            show(fig, key=f"{K}|err")
    cm.equations([
        ("Snapshots and modes", [r"\Theta = [\theta(t_1), \dots, \theta(t_{N_t})] = U\Sigma V^T,\quad "
                                 r"\Phi = U_{:,1..K},\quad E_K = \frac{\sum_{i\le K}\sigma_i^2}{\sum_i \sigma_i^2}"],
         "θ = T − T_a at all N nodes; the columns of U are orthonormal spatial modes ordered by their energy σ_i²."),
        ("Galerkin projection", [r"\theta \approx \Phi a,\quad \Phi^T C \Phi\,\dot a = -\Phi^T G \Phi\, a + "
                                 r"\Phi^T B\,P"], "K ordinary differential equations instead of N."),
    ], sym=[('θ', 'temperature rise over ambient at the nodes', 'K'), ('Φ', 'POD modes (N × K)', '–'),
            ('a', 'modal coordinates', 'K'), ('C, G, B', 'capacitance, conductance and input matrix of the FV model',
                                               'Ws/K, W/K, –'), ('P', 'chip losses', 'W'),
            ('σ_i', 'singular values of the snapshot matrix', 'K')])
    return rom, err


def export(res):
    cm.section(5, "Export to TherMOS", "Training data (random loss levels of all chips, independent excitation) and "
                                       "test data (the designed profiles) in the 2D format of TherMOS, plus a ready-"
                                       "made config for the PO mode (pure Galerkin projection, protocol 2).")
    c = st.columns([2, 1, 1, 1], vertical_alignment='bottom')
    name = c[0].text_input("Board name", 'myBoard', key=f"{K}|name", help="data/board/<name>_train.mat, "
                           "<name>_test.mat and config/gui/board_<name>.json").strip().replace(' ', '_')
    cm.num(c[1], "Training duration (s)", 2 * float(res['t'][-1]), 'board|tTrain', min_value=1.0)
    cm.num(c[2], "Modes K of the config", 8, 'board|Kexp', format='%d', min_value=1, max_value=50)
    go_ = c[3].button("Export", icon=':material/save:', key=f"{K}|exp", width='stretch')
    if go_ and name:
        from gui.core import config as conf
        from gui.core import forms
        from gui import adapter as A
        D, lay = res['D'], res['lay']
        Ts = float(res['t'][1] - res['t'][0]) if len(res['t']) > 1 else 1.0
        tTr = np.arange(0.0, st.session_state[f"{cm.KEY}|board|tTrain"] + Ts * 1e-9, Ts)
        Ptr = _profiles(lay, D, tTr, kind='random', seed=7)
        Ttr, _ = simBoard(res['C'], res['G'], res['Bp'], Ptr, Ts, lay['Ta'], res['nSub'])
        folder = os.path.join(A.ROOT, 'data', 'board')
        saveBoard(D, tTr, Ptr, Ttr, os.path.join(folder, f"{name}_train.mat"))
        saveBoard(D, res['t'], res['P'], res['T'], os.path.join(folder, f"{name}_test.mat"))
        with open(os.path.join(folder, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(lay, f, indent=2)
        cfg = conf.read_config('tutorial_PO')
        cfg['Run'].update({'name': f"board_{name}", 'config': '/gui/_gui_run', 'mode': 'PO', 'format': 'mat',
                           'datDim': 2, 'datSep': 1, 'trFile': [f"{name}_train"], 'vlFile': [f"{name}_train"],
                           'teFile': f"{name}_test", 'train': 1, 'test': 1})
        cfg['Dat']['gen']['dOut'] = 2
        cfg['Mdl']['pod'].update({'K': int(st.session_state[f"{cm.KEY}|board|Kexp"]), 'protocol': 2})
        conf.save_config(cfg, f"board_{name}")
        st.session_state[f"{K}|exported"] = f"board_{name}"
        st.toast(f"Exported board '{name}'", icon=':material/check:')
    done = st.session_state.get(f"{K}|exported")
    if done:
        st.success(f"Written: data/board/{done[6:]}_train.mat, _test.mat, {done[6:]}.json (layout) and "
                   f"config/gui/{done}.json.", icon=':material/check_circle:')
        if st.button(f"Load config '{done}' in the GUI", icon=':material/upload:', key=f"{K}|load"):
            from gui.core import config as conf
            from gui.core import forms
            forms.set_config(conf.read_config(done), done)
            st.toast(f"Config '{done}' loaded — open Modes › PO to train the model", icon=':material/check:')
    st.caption("The data store the distributed heat sink h_v and the anisotropic conductivity k_y of the board, so "
               "that the Galerkin projection of the PO mode includes the cooling (protocol 2/3, corrected version). "
               "Loading the config replaces the current GUI configuration.")


def interpretation(res, Tj, rom, err):
    D, lay = res['D'], res['lay']
    i = int(np.argmax(Tj.max(axis=0) - np.array([c['Tjmax'] for c in D['chips']])))
    c = D['chips'][i]
    m = c['Tjmax'] - Tj[:, i].max()
    p = [f"The hottest chip relative to its limit is **{c['name']}** ({c['tech']}): T_j reaches "
         f"**{fmt(Tj[:, i].max())} °C**, {'a margin of ' + fmt(m) + ' K' if m >= 0 else fmt(-m) + ' K above'} its "
         f"maximum junction temperature of {fmt(c['Tjmax'])} °C."]
    if D['view'] == 'xy':
        p.append(f"The board conducts heat sideways with k_eff = {fmt(D['kEff'])} W/(m K) over its {fmt(D['tb'] * 1e3)}"
                 f" mm thickness and removes it through the bottom with h_eff = {fmt(D['hEff'])} W/(m² K) (stack-up "
                 f"resistance {fmt(D['Rz'] * 1e6)} K mm²/W in series with the cooling). A thicker copper layer or a "
                 f"better ceramic spreads the heat (larger k_eff); better cooling raises h_eff. The spreading length "
                 f"√(k_eff t_b / h_eff) = {fmt(np.sqrt(D['kEff'] * D['tb'] / D['hEff']) * 1e3)} mm tells how far the "
                 f"heat of a chip spreads — chips closer than about twice this length heat each other.")
    else:
        p.append("The cross-section resolves every layer, so the temperature drop across the die attach, the "
                 "ceramic and the cooling interface is visible directly. It neglects the spreading along y, so it "
                 "over-estimates T_j of small chips (use the top view for the spreading).")
    p.append(f"The POD model with K = {rom['K']} modes reproduces the full model with a maximum error of "
             f"{fmt(err.max())} K. Export the data to train the PO/PS modes of TherMOS on this board.")
    cm.interpret(p)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Board designer", "Place chips on a board, select the board and chip technologies and the cooling, "
                                   "simulate the temperature field (top view or cross-section), build a POD reduced "
                                   "model and export the data for the PO/PS modes.")
    cm.question("How hot do the chips on my board get, how strongly do they heat each other, and how many POD modes "
                "describe the temperature field?")
    board_inputs()
    chip_inputs()
    res = simulation()
    if res is None:
        return
    Tj = results(res)
    rom, err = pod(res)
    export(res)
    cm.equations([
        ("Top view: lumped board", [r"k_{eff} = \frac{\sum_l k_{xy,l} t_l}{t_b},\quad (\rho c)_{eff} = "
                                    r"\frac{\sum_l \rho_l c_l t_l}{t_b},\quad h_{eff} = \frac{1}{\sum_l t_l/k_{z,l} + "
                                    r"1/h} + h_{top}"], "Stack-up of thickness t_b lumped into one plane."),
        ("Heat equation", [r"(\rho c)\,\partial_t T = \nabla\cdot(k\nabla T) - h_v (T - T_a) + Q,\quad "
                           r"h_v = h_{eff}/t_b,\quad Q = \frac{P}{A_{chip} t_b}"], None),
        ("Finite volumes", [r"C\,\dot\theta = -G\,\theta + B\,P,\quad C_i = (\rho c)_i V_i,\quad G_{ij} = "
                            r"-\frac{A_{ij}}{d/2k_i + d/2k_j}"], "Vertex-centred control volumes, implicit Euler."),
        ("Junction temperature", [r"T_j = \max_{chip} T + P\,R_{jb},\quad R_{jb} = \frac{t_{die}/k_{die} + "
                                  r"t_{att}/k_{att}}{w\,l}"], "Top view; the cross-section resolves the die."),
    ], sym=[('k_xy, k_z', 'in-plane and through-plane conductivity of layer l', 'W/(m K)'),
            ('t_l, t_b', 'layer and board thickness', 'm'), ('h, h_top', 'heat transfer coefficient bottom, top',
                                                             'W/(m² K)'),
            ('h_v', 'distributed heat sink', 'W/(m³ K)'), ('Q', 'volumetric heat source', 'W/m³'),
            ('V_i, A_ij, d', 'control volume, face area, node distance', 'm², m, m (per m depth)'),
            ('R_jb', 'junction-to-board resistance of die and die attach', 'K/W')], title="Model equations")
    interpretation(res, Tj, rom, err)
