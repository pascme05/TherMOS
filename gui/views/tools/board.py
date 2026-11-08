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
import io
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, SEQ, TEMP, INK2, MUTED, show, line, fmt
from gui.views.tools import _common as cm
from src.py.board.boardLib import BOARDS, BELOW, CHIPS, COOL, H_TOP, LAYER_MATS, VIA_FILL, layerProps, viaK
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.genProfile import genProfile
from src.py.board.genMission import genMission
from src.py.board.romBoard import romBoard, simRom
from src.py.board.saveBoard import saveBoard

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|board"
PROFILES = ['step', 'pulse', 'sine', 'random']
CHIP_COLS = ['name', 'tech', 'x [mm]', 'y [mm]', 'w [mm]', 'l [mm]', 'P [W]', 'profile', 'period [s]', 'α [%/K]']
DEFAULT_CHIPS = [['T1', 'Si IGBT', 15.0, 15.0, 10.0, 10.0, 60.0, 'pulse', 40.0, None],
                 ['D1', 'Si diode', 30.0, 15.0, 6.0, 8.0, 20.0, 'pulse', 40.0, None],
                 ['M1', 'SiC MOSFET', 45.0, 28.0, 4.0, 5.0, 25.0, 'sine', 60.0, None]]
STACK_COLS = ['layer', 'material', 't [µm]', 'Cu coverage [%]', 'k_xy [W/mK]', 'k_z [W/mK]']
BELOW_COLS = ['layer', 'material', 't [mm]', 'k [W/mK]']
CHIPLIB_COLS = ['technology', 'die', 't die [mm]', 'w [mm]', 'l [mm]', 'die attach', 't attach [mm]', 'P typ. [W]',
                'T_j,max [°C]', 'note']
SEMI = ['Si', 'SiC (4H)', 'GaN']
ATTACH = ['SAC305 solder', 'SnPb solder', 'High-Pb solder', 'Sintered Ag']
VIA_MODES = {'none': 'No vias', 'chips': 'Under the chips', 'all': 'Whole board'}
STAR = '★ '
PCB_COLS = ['name', 'layer', 'x [mm]', 'y [mm]', 'w [mm]', 'l [mm]', 'P [W]']


#######################################################################################################################
# Library (config/gui/board_lib.json: custom chips and stack-ups)
#######################################################################################################################
def _lib_file():
    from gui import adapter as A
    return os.path.join(A.ROOT, 'config', 'gui', 'board_lib.json')


def lib_load():
    try:
        with open(_lib_file(), 'r', encoding='utf-8') as f:
            d = json.load(f)
        return {'chips': d.get('chips', {}), 'stacks': d.get('stacks', {})}
    except (OSError, ValueError):
        return {'chips': {}, 'stacks': {}}


def lib_save(chips=None, stacks=None):
    d = lib_load()
    d['chips'].update(chips or {})
    d['stacks'].update(stacks or {})
    os.makedirs(os.path.dirname(_lib_file()), exist_ok=True)
    with open(_lib_file(), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def _chip_lib():
    """All chip technologies: built-in + custom (session, initialised from the library file)."""
    return {**CHIPS, **st.session_state.get(f"{K}|chipLib", lib_load()['chips'])}


def _presets():
    return {**BOARDS, **{STAR + n: v for n, v in lib_load()['stacks'].items()}}


#######################################################################################################################
# Functions
#######################################################################################################################
def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v))


def _layout(view):
    """Layout dictionary of genBoard from the widgets."""
    s = st.session_state
    df = s.get(f"{K}|chips_df")
    lib = _chip_lib()
    chips = []
    for _, r in (df if df is not None else pd.DataFrame(DEFAULT_CHIPS, columns=CHIP_COLS)).iterrows():
        if pd.isna(r['tech']) or r['tech'] not in lib or pd.isna(r['x [mm]']) or pd.isna(r['y [mm]']):
            continue
        t = lib[r['tech']]
        chips.append({'name': str(r['name']) if not pd.isna(r['name']) else r['tech'], 'tech': r['tech'],
                      'x': float(r['x [mm]']), 'y': float(r['y [mm]']),
                      'w': float(t['w'] if pd.isna(r['w [mm]']) else r['w [mm]']),
                      'l': float(t['l'] if pd.isna(r['l [mm]']) else r['l [mm]']),
                      'P': float(t['P'] if pd.isna(r['P [W]']) else r['P [W]']),
                      'profile': r['profile'] if r['profile'] in PROFILES else 'step',
                      'period': float(40.0 if pd.isna(r['period [s]']) else r['period [s]']),
                      'alpha': float(t.get('alpha', 0.0)) if _nan(r.get('α [%/K]')) else float(r['α [%/K]']) / 100})
    board = s.get(f"{K}|tech", 'DBC Al₂O₃ (0.38 mm)')
    custom = {n: v for n, v in s.get(f"{K}|chipLib", lib_load()['chips']).items()
              if n in {c['tech'] for c in chips}}
    vias = {'mode': s.get(f"{K}|vmode") or 'none', 'd': s.get(f"{K}|vd", 0.3), 'tpl': s.get(f"{K}|vtpl", 25.0),
            'pitch': s.get(f"{K}|vpitch", 1.0), 'fill': s.get(f"{K}|vfill", 'open (air)'),
            'margin': s.get(f"{K}|vmargin", 1.0)}
    return {'view': view, 'W': s.get(f"{K}|W", 60.0), 'L': s.get(f"{K}|L", 40.0), 'board': board,
            'layers': s.get(f"{K}|layers") or _presets().get(board, BOARDS['DBC Al₂O₃ (0.38 mm)']),
            'h': s.get(f"{K}|h", 5000.0), 'hTop': s.get(f"{K}|hTop", H_TOP), 'Ta': s.get(f"{K}|Ta", 25.0),
            'dx': s.get(f"{K}|dx", 1.0), 'yCut': s.get(f"{K}|yCut", 15.0), 'chips': chips, 'chipLib': custom,
            'vias': vias, 'below': s.get(f"{K}|below") or [],
            'pcb': (s.get(f"{K}|pcb") or []) if s.get(f"{K}|pcbOn", False) else []}


def _via_regions(lay):
    """Rectangles (x0, x1, y0, y1 in mm) of the via region."""
    v = lay.get('vias') or {}
    if v.get('mode') == 'all':
        return [(0, lay['W'], 0, lay['L'])]
    if v.get('mode') == 'chips':
        g = float(v.get('margin', 0.0))
        return [(max(c['x'] - c['w'] / 2 - g, 0), min(c['x'] + c['w'] / 2 + g, lay['W']),
                 max(c['y'] - c['l'] / 2 - g, 0), min(c['y'] + c['l'] / 2 + g, lay['L'])) for c in lay['chips']]
    return []


def _has_fr4(lay):
    return any(l['mat'] == 'FR4' for l in lay['layers'])


def _profiles(lay, D, t, kind=None, seed=0, mission=None):
    """Loss profiles (Nt x nChips) of the chips used in D (designed profiles, a mission profile, or random ones for
    training)."""
    if mission is not None and kind is None:
        return genMission(mission['seq'], D['chips'], t, mission.get('repeat', 1))[0]
    P = []
    for i, c in enumerate(D['chips']):
        k = kind or c.get('profile', 'step')
        P.append(genProfile(k, c['P'], t, {'period': c.get('period', 40.0), 'duty': 0.5, 'amp': 0.5,
                                         'hold': max(c.get('period', 40.0) / 4, t[1] - t[0])}, seed=seed + i))
    return np.column_stack(P) if P else np.zeros((len(t), 0))


def _eta(bar, t0, what):
    """Progress callback with the expected remaining time."""
    def f(x):
        el = time.perf_counter() - t0
        bar.progress(min(float(x), 1.0), text=f"{what}: {100 * x:.0f} % · {fmt(el)} s elapsed · about "
                                              f"{fmt(el * (1 - x) / max(x, 1e-9))} s left")
    return f


def _cpl(D, Bp):
    """Electro-thermal coupling of the chips (die-average junction temperature, α of the chip table)."""
    return {'W': Bp, 'alpha': np.array([float(c.get('alpha', 0.0)) for c in D['chips']]), 'Tref': 25.0}


def _peff(res, T=None):
    """Effective losses with the electro-thermal coupling (nominal losses without)."""
    P, cpl = res['P'], res.get('cpl')
    if cpl is None:
        return P
    T = res['T'] if T is None else T
    Tj = (T - res['lay']['Ta']) @ cpl['W'] + res['lay']['Ta']
    return P * np.maximum(1 + cpl['alpha'] * (Tj - cpl['Tref']), 0)


def _tj(D, T, P):
    """Junction temperature of every chip (Nt x nChips): hottest board node under the chip + P R_jb (top view) or
    hottest die node (cross-section)."""
    T = np.atleast_2d(T)
    out = np.zeros((T.shape[0], len(D['chips'])))
    for i in range(len(D['chips'])):
        m = np.asarray(D['B'])[:, i] > 0
        out[:, i] = T[:, :m.size][:, m].max(axis=1) + (np.atleast_2d(P)[:, i] * D['Rjb'][i]
                                                      if D['view'] == 'xy' else 0.0)
    return out


def _disp(D, Tv, view, row=None):
    """
    Display of the state of the layer-resolved model: top view (x-y field of one row of the stack, NaN outside the
    footprint of partial rows such as the dies) or cross-section (x-z slice at y_cut, NaN = air).
    Returns [F (nt x a x b), x, y, geo of the cells (m)].
    """
    Tv = np.atleast_2d(Tv)
    ix = D['idx'][D['nBoard'] - 1 if row is None else row] if view == 'xy' else D['sel']
    F = np.where(ix[None] >= 0, Tv[:, np.maximum(ix, 0)], np.nan)
    x, y = (D['xg'], D['yg']) if view == 'xy' else (D['x'], D['y'])
    X, Y = np.meshgrid(x, y)
    return [F, x, y, np.column_stack([X.ravel(), Y.ravel()])]


def _heat(F, x, y, unit, title, equal=True, zlim=None, scale=TEMP):
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
    if _has_fr4(lay):
        for j, (x0, x1, y0, y1) in enumerate(_via_regions(lay)):
            fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color=MUTED, width=1, dash='dot'),
                          fillcolor='rgba(184,115,51,0.18)')
        if _via_regions(lay):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='thermal vias (averaged)',
                                     marker=dict(symbol='square', size=12, color='rgba(184,115,51,0.4)')))
    for j, p in enumerate(lay.get('pcb') or []):
        x0, x1, y0, y1 = _pcb_rect(p, lay)
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color='#c0392b', width=1.5, dash='dash'),
                      fillcolor='rgba(192,57,43,0.10)')
        fig.add_annotation(x=x0, y=y1, text=f"{p['name']} {fmt(p['P'])} W", showarrow=False, xanchor='left',
                           yanchor='top', font=dict(size=10, color='#c0392b'))
    if lay.get('pcb'):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='board losses',
                                 marker=dict(symbol='square-open', size=12, color='#c0392b')))
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
    fig.update_layout(title='Top view (x-y)', showlegend=True, margin=dict(t=50, b=40),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, x=0))
    return fig


def _pcb_rect(p, lay):
    """Rectangle (x0, x1, y0, y1 in mm) of an additional loss region (w or l <= 0: whole board)."""
    w, l = float(p.get('w') or 0.0), float(p.get('l') or 0.0)
    if w <= 0 or l <= 0:
        return 0.0, lay['W'], 0.0, lay['L']
    return (max(p['x'] - w / 2, 0), min(p['x'] + w / 2, lay['W']), max(p['y'] - l / 2, 0),
            min(p['y'] + l / 2, lay['L']))


def _q(D, N=None):
    """Constant additional losses of the board in W per node (padded to N for the heat sink plane), None if none."""
    q = D.get('q')
    if q is None or not np.any(q):
        return None
    q = np.asarray(q, dtype=float)
    return q if N is None or N == q.size else np.concatenate([q, np.zeros(N - q.size)])


def _cut_fig(lay):
    """Cross-section of the stack-up at y = yCut (thickness not to scale with x)."""
    fig = go.Figure()
    z = 0.0
    vr = [(x0, x1) for x0, x1, y0, y1 in _via_regions(lay) if y0 <= lay['yCut'] <= y1]
    layers = [l for l in lay['layers'] if float(l['t']) > 0]
    tMin = 0.04 * sum(float(l['t']) for l in layers)                    # thin layers drawn with a minimum height
    below = [l for l in lay.get('below') or [] if float(l['t']) > 0]
    zb = 0.0
    for l in below:                                                     # top → bottom: TIM, then the carrier
        th = max(min(float(l['t']), 4 * sum(float(m['t']) for m in layers)) * 0.35, tMin)
        k = layerProps(l)[1]
        zb -= th
        fig.add_shape(type='rect', x0=-1, x1=lay['W'] + 1, y0=zb, y1=zb + th, line=dict(width=0.5, color=INK2),
                      fillcolor='#b8c2cc' if 'Al' in l['mat'] or l['mat'] == 'Cu' else '#eda100', opacity=0.55)
        fig.add_annotation(x=lay['W'] * 1.01, y=zb + th / 2, text=f"{l.get('layer', '')} · {fmt(float(l['t']))} mm"
                           f" · k={fmt(k)}", showarrow=False, xanchor='left', font=dict(size=10))
    for i, l in enumerate(reversed(layers)):
        th = max(float(l['t']), tMin)
        kxy, kz = layerProps(l)[0:2]
        fig.add_shape(type='rect', x0=0, x1=lay['W'], y0=z, y1=z + th, line=dict(width=0.5, color=INK2),
                      fillcolor='#c98b4b' if l['mat'] == 'Cu' else ('#b8c2cc' if 'Al' in l['mat'] else '#9fc39b'),
                      opacity=0.35 + 0.6 * (float(l.get('cov') or 100) / 100 if l['mat'] == 'Cu' else 0.75))
        if l['mat'] == 'FR4':
            for x0, x1 in vr:
                fig.add_shape(type='rect', x0=x0, x1=x1, y0=z, y1=z + th, line=dict(width=0),
                              fillcolor='rgba(184,115,51,0.45)')
        li = len(layers) - 1 - i                                       # index in the stack-up (0 = top)
        for p in lay.get('pcb') or []:
            x0, x1, y0, y1 = _pcb_rect(p, lay)
            if int(p.get('layer', 0)) == li and y0 <= lay['yCut'] <= y1:
                fig.add_shape(type='rect', x0=x0, x1=x1, y0=z, y1=z + th, line=dict(color='#c0392b', width=1.5,
                              dash='dash'), fillcolor='rgba(192,57,43,0.15)')
        fig.add_annotation(x=lay['W'] * 1.01, y=z + th / 2, text=f"{l.get('layer', '')} · {fmt(float(l['t']) * 1e3)} µm"
                           f" · k_z={fmt(kz)}", showarrow=False, xanchor='left', font=dict(size=10))
        z += th
    lib = _chip_lib()
    for i, c in enumerate(lay['chips']):
        if abs(c['y'] - lay['yCut']) > c['l'] / 2:
            continue
        t = lib.get(c['tech'], CHIPS['Si MOSFET'])
        fig.add_shape(type='rect', x0=c['x'] - c['w'] / 2, x1=c['x'] + c['w'] / 2, y0=z, y1=z + t['t_att'],
                      fillcolor='#8a8a8a', line=dict(width=0.5))
        fig.add_shape(type='rect', x0=c['x'] - c['w'] / 2, x1=c['x'] + c['w'] / 2, y0=z + t['t_att'],
                      y1=z + t['t_att'] + t['t'], fillcolor=SERIES[i % len(SERIES)], line=dict(width=0.5))
        fig.add_annotation(x=c['x'], y=z + t['t_att'] + t['t'], text=c['name'], showarrow=False, yanchor='bottom')
    fig.add_annotation(x=lay['W'] / 2, y=zb - 0.08 * z, text=f"coolant h = {fmt(lay['h'])} W/m²K", showarrow=False)
    fig.update_xaxes(range=[-2, lay['W'] * 1.45], title='x (mm)')
    fig.update_yaxes(range=[zb - 0.2 * z, z * 1.6 + 0.3], title='z (mm)')
    fig.update_layout(title=f"Cross-section (x-z) at y = {fmt(lay['yCut'])} mm (z not to scale)",
                      margin=dict(t=50, b=40))
    return fig


#######################################################################################################################
# Steps
#######################################################################################################################
def board_inputs():
    cm.section(1, "Board and cooling", "Board technology (stack-up from the material library), size, cooling of the "
                                        "bottom side and ambient temperature.", divider=False)
    c = st.columns(4)
    pre = _presets()
    tech = c[0].selectbox("Board technology", list(pre), index=list(pre).index('DBC Al₂O₃ (0.38 mm)'),
                          key=f"{K}|tech", help="Preset of the layer stack-up (top → bottom). It fills the stack-up "
                                                "table below, which can be edited. ★: stack-ups from your library.")
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
                      "coefficient of the cooling on the bottom of the board, or on the bottom of the last layer "
                      "below the board (heat sink) if layers below are defined.")
    cm.num(c[2], "h top (W/m²K)", H_TOP, 'board|hTop', min_value=0.0, help="Natural convection on the top side.")
    cm.num(c[3], "Grid spacing dx (mm)", 1.0, 'board|dx', min_value=0.1, max_value=10.0, step=0.25,
           help="Node spacing of the finite-volume grid (dx = dy). Chips should span at least 3 nodes.")
    stack_inputs(tech, pre[tech])
    below_inputs()
    via_inputs()
    pcb_inputs()


def stack_inputs(tech, preset):
    """Editable stack-up (top → bottom), starting from the preset."""
    s = st.session_state
    ver = s.get(f"{K}|stackver", 0)
    df0 = pd.DataFrame([{'layer': l.get('layer', ''), 'material': l['mat'], 't [µm]': float(l['t']) * 1e3,
                         'Cu coverage [%]': float(l.get('cov', 100.0)) if l['mat'] == 'Cu' else None,
                         'k_xy [W/mK]': l.get('kxy'), 'k_z [W/mK]': l.get('kz')} for l in preset], columns=STACK_COLS)
    with st.container(border=True):
        st.markdown(f"**:material/layers: Stack-up** (top → bottom), from '{tech}'")
        df = st.data_editor(df0, key=f"{K}|stack|{tech}|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                            column_config={
                                'material': st.column_config.SelectboxColumn('material', options=LAYER_MATS,
                                                                             required=True, help="FR4: glass-epoxy "
                                                                             "laminate (k_xy 0.8, k_z 0.3 W/mK)."),
                                't [µm]': st.column_config.NumberColumn(min_value=0.0, format='%.0f'),
                                'Cu coverage [%]': st.column_config.NumberColumn(
                                    min_value=0.0, max_value=100.0, format='%.0f', help="Copper area of a patterned "
                                    "Cu layer (signal ≈ 30–50 %, plane ≈ 90 %), the rest is laminate. Empty = 100 %."),
                                'k_xy [W/mK]': st.column_config.NumberColumn(format='%g', help="Optional override of "
                                                                             "the in-plane conductivity."),
                                'k_z [W/mK]': st.column_config.NumberColumn(format='%g', help="Optional override of "
                                                                            "the through-plane conductivity.")})
        layers = []
        for _, r in df.iterrows():
            if _nan(r['material']) or r['material'] not in LAYER_MATS or _nan(r['t [µm]']) or r['t [µm]'] <= 0:
                continue
            layers.append({'layer': '' if _nan(r['layer']) else str(r['layer']), 'mat': r['material'],
                           't': float(r['t [µm]']) * 1e-3,
                           'cov': 100.0 if _nan(r['Cu coverage [%]']) else float(r['Cu coverage [%]']),
                           'kxy': None if _nan(r['k_xy [W/mK]']) else float(r['k_xy [W/mK]']),
                           'kz': None if _nan(r['k_z [W/mK]']) else float(r['k_z [W/mK]'])})
        if not layers:
            st.warning("The stack-up is empty — the preset is used.", icon=':material/warning:')
            layers = [dict(l) for l in preset]
        s[f"{K}|layers"] = layers
        props = [layerProps(l) for l in layers]
        t = np.array([l['t'] for l in layers]) * 1e-3
        tb = t.sum()
        Rz = float(np.sum(t / np.array([p[1] for p in props])))
        c = st.columns([1, 1, 1, 2], vertical_alignment='bottom')
        c[0].metric("Thickness t_b", f"{fmt(tb * 1e3)} mm")
        c[1].metric("k_eff in-plane", f"{fmt(np.sum([p[0] * ti for p, ti in zip(props, t)]) / tb)} W/mK",
                    help="Thickness-weighted in-plane conductivity Σ k_xy,l t_l / t_b (spreading).")
        c[2].metric("R_z through (K·mm²/W)", f"{fmt(Rz * 1e6)}", help="Σ t_l / k_z,l: resistance of 1 mm² of board "
                                                                       "from top to bottom (without vias).")
        with c[3]:
            b = st.columns(2)
            if b[0].button("Reset to preset", icon=':material/restart_alt:', key=f"{K}|stackreset", width='stretch'):
                s[f"{K}|stackver"] = ver + 1
                st.rerun()
            with b[1].popover("Save to library", icon=':material/bookmark_add:', width='stretch'):
                nm = st.text_input("Name of the stack-up", tech.replace(STAR, '') + ' (custom)', key=f"{K}|stackname")
                if st.button("Save", key=f"{K}|stacksave", icon=':material/save:') and nm.strip():
                    lib_save(stacks={nm.strip(): layers})
                    st.toast(f"Stack-up '{nm.strip()}' saved to config/gui/board_lib.json", icon=':material/check:')
        with st.expander("Layer properties (computed)", icon=':material/calculate:'):
            table(pd.DataFrame([{'layer': l['layer'], 'material': l['mat'], 't (µm)': l['t'] * 1e3,
                                 'k_xy (W/mK)': p[0], 'k_z (W/mK)': p[1], 'ρ (kg/m³)': p[2], 'c (J/kgK)': p[3],
                                 't/k_z (K·mm²/W)': l['t'] * 1e-3 / p[1] * 1e6} for l, p in zip(layers, props)]),
                  hide_index=True)
            st.caption("Patterned Cu layers: k_xy = c·k_Cu + (1 − c)·k_FR4,xy (parallel), k_z = 1/(c/k_Cu + "
                       "(1 − c)/k_FR4,z) (series) with the coverage c [Shabany 2010].")
            cm.refs(['Incropera2007', 'Wintrich2015'])


def below_inputs():
    """Layers below the board (interface material, heat sink), top → bottom; the cooling acts on the last one."""
    s = st.session_state
    with st.container(border=True):
        st.markdown("**:material/vertical_align_bottom: Below the board** (top → bottom: interface material, heat "
                    "sink / housing; the cooling h acts on the bottom of the last layer)")
        c = st.columns([2, 1], vertical_alignment='bottom')
        pre = c[0].selectbox("Preset", list(BELOW), index=0, key=f"{K}|belowpre",
                             help="Fills the table, which can be edited. 'none': the cooling acts directly on the "
                                  "bottom of the board.")
        ver = s.get(f"{K}|belowver", 0)
        if c[1].button("Reset to preset", icon=':material/restart_alt:', key=f"{K}|belowreset", width='stretch'):
            s[f"{K}|belowver"] = ver + 1
            st.rerun()
        df0 = pd.DataFrame([{'layer': l.get('layer', ''), 'material': l['mat'], 't [mm]': float(l['t']),
                             'k [W/mK]': l.get('kz')} for l in BELOW[pre]], columns=BELOW_COLS)
        df = st.data_editor(df0, key=f"{K}|below|{pre}|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                            column_config={
                                'material': st.column_config.SelectboxColumn('material', options=LAYER_MATS,
                                                                             required=True),
                                't [mm]': st.column_config.NumberColumn(min_value=0.0, format='%g'),
                                'k [W/mK]': st.column_config.NumberColumn(format='%g', help="Optional override of "
                                                                          "the conductivity (e.g. of the gap filler "
                                                                          "datasheet).")})
        below = []
        for _, r in df.iterrows():
            if _nan(r['material']) or r['material'] not in LAYER_MATS or _nan(r['t [mm]']) or r['t [mm]'] <= 0:
                continue
            k = None if _nan(r['k [W/mK]']) else float(r['k [W/mK]'])
            below.append({'layer': '' if _nan(r['layer']) else str(r['layer']), 'mat': r['material'],
                          't': float(r['t [mm]']), 'cov': 100.0, 'kxy': k, 'kz': k})
        s[f"{K}|below"] = below
        if below:
            props = [layerProps(l) for l in below]
            t = np.array([l['t'] for l in below]) * 1e-3
            Rif = float(np.sum(t[:-1] / np.array([p[1] for p in props[:-1]]))) if len(below) > 1 else 0.0
            c = st.columns(3)
            c[0].metric("R interface (K·mm²/W)", fmt(Rif * 1e6), help="Σ t/k of all layers except the last one: "
                        "resistance of 1 mm² of the interface (gap filler, grease, foil).")
            c[1].metric("Spreading k·t of the heat sink (W/K)", fmt(float(np.sum([p[0] * ti for p, ti in
                                                                                  zip(props, t)]))),
                        help="Σ k_i t_i: lateral conductance of the layers below. Large values = the heat sink "
                             "spreads the heat of the chips over the whole board.")
            c[2].metric("1/h (K·mm²/W)", fmt(1e6 / max(float(s.get(f'{K}|h', 5000.0)), 1e-9)),
                        help="Resistance of the cooling on the bottom of the heat sink for 1 mm².")
            st.caption("Top view: the layers below form a second plane (heat sink temperature T₂) that is coupled to "
                       "the board through the board stack-up and the interface layers and cooled by h from below. "
                       "Cross-section: every layer is resolved.")


def via_inputs():
    """Averaged thermal vias through the FR4 layers."""
    s = st.session_state
    with st.container(border=True):
        st.markdown("**:material/blur_on: Thermal vias** (averaged over the via array, through the FR4 layers)")
        mode = st.segmented_control("Vias", list(VIA_MODES), format_func=VIA_MODES.get, default='none',
                                      key=f"{K}|vmode", help="Under the chips: footprint of every chip plus the "
                                      "margin. Whole board: the via array covers the board.") or 'none'
        on = mode != 'none'
        c = st.columns([1, 1, 1, 1.4, 1])
        c = [None] + c                                                # keep the indices of the inputs below
        cm.num(c[1], "Drill d (mm)", 0.3, 'board|vd', min_value=0.05, max_value=2.0, step=0.05, disabled=not on)
        cm.num(c[2], "Plating (µm)", 25.0, 'board|vtpl', min_value=5.0, max_value=100.0, step=5.0, disabled=not on)
        cm.num(c[3], "Pitch (mm)", 1.0, 'board|vpitch', min_value=0.2, max_value=10.0, step=0.1, disabled=not on,
               help="Distance of the vias in a square array.")
        c[4].selectbox("Fill", list(VIA_FILL), key=f"{K}|vfill", disabled=not on, help="Open vias conduct only "
                       "through the copper barrel; filled vias also through the filling.")
        cm.num(c[5], "Margin (mm)", 1.0, 'board|vmargin', min_value=0.0, max_value=20.0, disabled=not on or mode != 'chips',
               help="Extension of the via region beyond the chip footprint.")
        if on:
            v = {'d': s[f"{K}|vd"], 'tpl': s[f"{K}|vtpl"], 'pitch': s[f"{K}|vpitch"], 'fill': s[f"{K}|vfill"]}
            if v['d'] >= v['pitch']:
                st.warning("The drill is larger than the pitch.", icon=':material/warning:')
            r, ri = v['d'] / 2, max(v['d'] / 2 - v['tpl'] * 1e-3, 0)
            phi = np.pi * (r ** 2 - ri ** 2) / v['pitch'] ** 2
            layers = s.get(f"{K}|layers") or []
            if not any(l['mat'] == 'FR4' for l in layers):
                st.info("The stack-up has no FR4 layer — vias have no effect (ceramic and IMS boards conduct through "
                        "the substrate).", icon=':material/info:')
            else:
                st.caption(f"Copper area of the barrels φ_Cu = {100 * phi:.2f} % → k_z of FR4 in the via region "
                           f"{fmt(viaK(v, 0.3))} W/mK instead of 0.3 W/mK (×{fmt(viaK(v, 0.3) / 0.3)}). The in-plane "
                           f"conductivity is not changed.")


def pcb_inputs():
    """Additional losses in the board (traces, shunts, passives): rectangular regions on a layer, constant."""
    s = st.session_state
    layers = s.get(f"{K}|layers") or []
    opts = [f"{i}: {l.get('layer') or l['mat']}" for i, l in enumerate(layers)] or ['0: top']
    with st.container(border=True):
        c = st.columns([2, 3], vertical_alignment='center')
        on = c[0].toggle("**:material/bolt: Additional losses in the board**", key=f"{K}|pcbOn",
                         help="Losses that are not in the chips: conduction losses of the copper traces and planes "
                              "(I²R), shunts, gate drivers, passives. They heat the board directly.")
        c[1].caption("One row per region: layer of the stack-up, centre x, y, size w × l (empty or 0 = whole board) "
                     "and loss P. The loss is constant over the run and spread uniformly over the volume of the "
                     "region in that layer." if on else "Off: only the chips are heat sources.")
        if not on:
            return
        df0 = s.get(f"{K}|pcb_base")
        if df0 is None:
            W, L = s_get('board|W', 60.0), s_get('board|L', 40.0)
            df0 = pd.DataFrame([['traces', opts[0], W / 2, L / 2, None, None, 5.0]], columns=PCB_COLS)
            s[f"{K}|pcb_base"] = df0
        df0 = df0.copy()
        df0['layer'] = [v if v in opts else opts[0] for v in df0['layer']]
        df = st.data_editor(df0, key=f"{K}|pcbed|{'|'.join(opts)}", num_rows='dynamic', hide_index=True,
                            width='stretch', column_config={
            'layer': st.column_config.SelectboxColumn('layer', options=opts, required=True, help="Layer of the "
                                                      "stack-up (index 0 = top) that carries the loss."),
            'x [mm]': st.column_config.NumberColumn(format='%.1f', help="Centre of the region."),
            'y [mm]': st.column_config.NumberColumn(format='%.1f'),
            'w [mm]': st.column_config.NumberColumn(min_value=0.0, format='%.1f', help="Width; empty = whole board."),
            'l [mm]': st.column_config.NumberColumn(min_value=0.0, format='%.1f', help="Length; empty = whole board."),
            'P [W]': st.column_config.NumberColumn(format='%g', help="Loss of the region (constant).")})
        rows = []
        for _, r in df.iterrows():
            if _nan(r['P [W]']):
                continue
            rows.append({'name': '' if _nan(r['name']) else str(r['name']),
                         'layer': int(str(r['layer']).split(':')[0]) if not _nan(r['layer']) else 0,
                         'x': 0.0 if _nan(r['x [mm]']) else float(r['x [mm]']),
                         'y': 0.0 if _nan(r['y [mm]']) else float(r['y [mm]']),
                         'w': 0.0 if _nan(r['w [mm]']) else float(r['w [mm]']),
                         'l': 0.0 if _nan(r['l [mm]']) else float(r['l [mm]']), 'P': float(r['P [W]'])})
        s[f"{K}|pcb"] = rows
        st.caption(f"Total additional loss {fmt(sum(r['P'] for r in rows))} W. It is a constant heat source in the "
                   "finite-volume model, the POD model and (as a steady offset) the Cauer network; the junction "
                   "temperatures include it. The step responses and the R_th matrix describe the chips only, the "
                   "export to the PO mode contains the chip losses only.")


def s_get(key, default):
    return st.session_state.get(f"{cm.KEY}|{key}", default)


def layout_view():
    """Top view and cross-section of the layout, side by side with the same height."""
    lay = _layout('xy')
    out = [c['name'] for c in lay['chips'] if c['x'] - c['w'] / 2 < 0 or c['x'] + c['w'] / 2 > lay['W']
           or c['y'] - c['l'] / 2 < 0 or c['y'] + c['l'] / 2 > lay['L']]
    if out:
        st.warning(f"Chips outside the board: {', '.join(out)} (they are clipped at the edge).", icon=':material/warning:')
    with st.container(border=True):
        c = st.columns([3, 2], vertical_alignment='bottom')
        c[0].markdown("**:material/developer_board: Layout** — top view and cross-section; the dashed line in the top "
                      "view marks the cut.")
        cm.num(c[1], "Cut of the cross-section y (mm)", 15.0, 'board|yCut', min_value=0.0, max_value=float(lay['L']))
        c = st.columns(2)
        with c[0]:
            show(_top_fig(lay), height=480, key=f"{K}|topfig")
        with c[1]:
            show(_cut_fig(_layout('xz')), height=480, key=f"{K}|cutfig")


def chip_inputs():
    cm.section(2, "Chips", "One row per chip: technology, centre position (x, y from the lower left corner of the "
                           "board), footprint, loss and loss profile. Empty w, l or P take the values of the "
                           "technology. Rows can be added at the bottom and deleted with the row menu.")
    s = st.session_state
    lib = _chip_lib()
    ver = s.get(f"{K}|chipver", 0)
    df0 = s.get(f"{K}|chips_base")
    if df0 is None:
        df0 = pd.DataFrame(DEFAULT_CHIPS, columns=CHIP_COLS)
    with st.container(border=True):
        c = st.columns([2, 1, 1, 1, 1], vertical_alignment='bottom')
        nt = c[0].selectbox("Add a chip", list(lib), key=f"{K}|addtech", help="Technology of the new chip.")
        ax = cm.num(c[1], "x (mm)", 10.0, 'board|addx', min_value=0.0)
        ay = cm.num(c[2], "y (mm)", 10.0, 'board|addy', min_value=0.0)
        ap = cm.num(c[3], "P (W)", float(lib[nt]['P']), f'board|addp|{nt}', min_value=0.0)
        if c[4].button("Add chip", icon=':material/add_circle:', key=f"{K}|add", width='stretch'):
            cur = s.get(f"{K}|chips_df", df0)
            pre = ''.join(w[0] for w in nt.split()[-1:]).upper() or 'C'
            n = 1 + sum(str(x).startswith(pre) for x in cur['name'])
            row = pd.DataFrame([[f"{pre}{n}", nt, ax, ay, lib[nt]['w'], lib[nt]['l'], ap, 'step', 40.0, None]],
                               columns=CHIP_COLS)
            s[f"{K}|chips_base"] = pd.concat([cur, row], ignore_index=True)
            s[f"{K}|chipver"] = ver + 1
            st.rerun()
    df = st.data_editor(df0, key=f"{K}|chips|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                        column_config={
        'tech': st.column_config.SelectboxColumn('tech', options=list(lib), required=True,
                                                 help="Chip technology (die material, thickness, die attach)."),
        'profile': st.column_config.SelectboxColumn('profile', options=PROFILES, help="Loss profile P(t): step, "
                                                    "pulse (50 % duty), sine (between P/3 and P), random levels."),
        'x [mm]': st.column_config.NumberColumn(format='%.1f'), 'y [mm]': st.column_config.NumberColumn(format='%.1f'),
        'P [W]': st.column_config.NumberColumn(format='%.1f', help="Loss of the chip (peak of the profile)."),
        'period [s]': st.column_config.NumberColumn(format='%.0f', help="Period of pulse/sine, 4× hold of random."),
        'α [%/K]': st.column_config.NumberColumn(format='%g', help="Temperature coefficient of the losses for the "
                                                 "electro-thermal coupling, P = P₀·(1 + α·(T_j − 25 °C)). Empty = "
                                                 "value of the technology (Si MOSFET 0.7, IGBT 0.3, SiC 0.3, diode "
                                                 "0.2, GaN 0.8 %/K; estimates).")})
    st.session_state[f"{K}|chips_df"] = df
    mission_inputs(df)
    layout_view()
    with st.expander("Chip technologies", icon=':material/memory:'):
        st.markdown("**Built-in**")
        table(pd.DataFrame([{'technology': n, 'die': c['mat'], 't die (mm)': c['t'], 'w × l (mm)': f"{c['w']} × "
                             f"{c['l']}", 'die attach': c['attach'], 't attach (mm)': c['t_att'], 'P typ. (W)': c['P'],
                             'T_j,max (°C)': c['Tjmax'], 'note': c['note']} for n, c in CHIPS.items()]),
              hide_index=True)
        chip_lib_editor()


MIS_DIR = os.path.join('data', 'board', 'mission')
MIS_COLS = ['segment', 'duration [s]', 'ramp [s]', 'scale [-]']


def mission_inputs(chips):
    """Load of the chips: the profiles of the chip table or a mission profile (sequence of operating segments)."""
    s = st.session_state
    with st.container(border=True):
        c = st.columns([2, 3])
        mode = c[0].segmented_control("Load", ['Chip profiles', 'Mission profile'], default='Chip profiles',
                                      key=f"{K}|loadMode", help="Chip profiles: profile and period of the chip table. "
                                      "Mission profile: a sequence of operating segments (e.g. a drive cycle) "
                                      "for all chips.") or 'Chip profiles'
        if mode != 'Mission profile':
            c[1].caption("The losses follow the profile column of the chip table.")
            return
        c[1].caption("Sequence of segments: duration, ramp to the new level, scale × P of the chip table, or the loss "
                     "of a chip in W in its own column '<name> [W]' (empty = scale × P). Template and example: "
                     f"`{MIS_DIR.replace(os.sep, '/')}`.")
        os.makedirs(MIS_DIR, exist_ok=True)
        files = sorted(f[:-4] for f in os.listdir(MIS_DIR) if f.endswith('.csv'))
        c = st.columns([2, 2, 1], vertical_alignment='bottom')
        name = c[0].selectbox("Sequence file", files, index=files.index('example_driveCycle') if
                              'example_driveCycle' in files else 0, key=f"{K}|misfile")
        up = c[1].file_uploader("or upload a CSV", type=['csv'], key=f"{K}|misup")
        rept = cm.num(c[2], "Repeat", 1, 'board|misrep', format='%d', min_value=1, max_value=100)
        try:
            df0 = pd.read_csv(up) if up is not None else pd.read_csv(os.path.join(MIS_DIR, f"{name}.csv"))
        except Exception as e:
            st.error(f"Sequence could not be read: {e}", icon=':material/error:')
            return
        names = [str(n) for n in chips['name'] if not _nan(n)]
        for n in names:                                                  # one optional column per chip
            if f"{n} [W]" not in df0.columns:
                df0[f"{n} [W]"] = np.nan
        for col in MIS_COLS:
            if col not in df0.columns:
                df0[col] = {'segment': '', 'duration [s]': 10.0, 'ramp [s]': 0.0, 'scale [-]': 1.0}[col]
        df0 = df0[MIS_COLS + [k for k in df0.columns if k not in MIS_COLS]]
        df = st.data_editor(df0, key=f"{K}|mis|{name}|{getattr(up, 'name', '')}", num_rows='dynamic',
                            hide_index=True, width='stretch', column_config={
                                'duration [s]': st.column_config.NumberColumn(min_value=0.0, format='%g'),
                                'ramp [s]': st.column_config.NumberColumn(min_value=0.0, format='%g', help="Linear "
                                                                          "transition from the previous level."),
                                'scale [-]': st.column_config.NumberColumn(format='%g', help="Loss = scale × P of the "
                                                                           "chip table (if the chip column is empty)."),
                                **{f"{n} [W]": st.column_config.NumberColumn(format='%g', help=f"Loss of {n} in this "
                                   "segment (overrides the scale).") for n in names}})
        seq = df.to_dict('records')
        Pn = [{'name': str(r['name']), 'P': 0.0 if _nan(r['P [W]']) else float(r['P [W]'])} for _, r in
              chips.iterrows() if not _nan(r['name'])]
        tEnd = float(df['duration [s]'].fillna(0).clip(lower=0).sum()) * int(rept)
        if tEnd <= 0:
            st.warning("The sequence has no segment with a duration > 0.", icon=':material/warning:')
            return
        t = np.linspace(0.0, tEnd, 1201)
        P, _ = genMission(seq, Pn, t, rept)
        fig = go.Figure([line(t, P[:, i], c['name'], i) for i, c in enumerate(Pn)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)')
        show(fig, height=260, key=f"{K}|misfig", info="**Mission profile** of the chips: the simulation runs over "
             "the whole (repeated) sequence.")
        c = st.columns([2, 1], vertical_alignment='bottom')
        nm = c[0].text_input("Save the sequence as", name if up is None else up.name[:-4], key=f"{K}|misname")
        if c[1].button("Save", icon=':material/save:', key=f"{K}|missave", width='stretch') and nm.strip():
            df.to_csv(os.path.join(MIS_DIR, f"{nm.strip().replace(' ', '_')}.csv"), index=False)
            st.toast(f"Saved {MIS_DIR}/{nm.strip()}.csv", icon=':material/check:')
        s[f"{K}|mission"] = {'seq': seq, 'repeat': int(rept), 't': np.array([0.0, tEnd]),
                             'sig': json.dumps([seq, int(rept)], default=str)}


def chip_lib_editor():
    """Own chip technologies (session + library file)."""
    s = st.session_state
    st.markdown("**Own technologies** — add rows; they can be used in the chip table right away and saved to the "
                "library (config/gui/board_lib.json).")
    own = s.get(f"{K}|chipLib", lib_load()['chips'])
    df0 = pd.DataFrame([[n, c['mat'], c['t'], c['w'], c['l'], c['attach'], c['t_att'], c['P'], c['Tjmax'],
                         c.get('note', '')] for n, c in own.items()], columns=CHIPLIB_COLS)
    if df0.empty:
        df0 = pd.DataFrame({c: pd.Series(dtype=str if c in ('technology', 'die', 'die attach', 'note') else float)
                            for c in CHIPLIB_COLS})
    df = st.data_editor(df0, key=f"{K}|chiplib", num_rows='dynamic', hide_index=True, width='stretch', column_config={
        'die': st.column_config.SelectboxColumn('die', options=SEMI, required=True),
        'die attach': st.column_config.SelectboxColumn('die attach', options=ATTACH, required=True)})
    lib = {}
    for _, r in df.iterrows():
        if _nan(r['technology']) or str(r['technology']).strip() in CHIPS or any(
                _nan(r[k]) for k in CHIPLIB_COLS[1:9]):
            continue
        lib[str(r['technology']).strip()] = dict(mat=r['die'], t=float(r['t die [mm]']), w=float(r['w [mm]']),
                                                 l=float(r['l [mm]']), attach=r['die attach'],
                                                 t_att=float(r['t attach [mm]']), P=float(r['P typ. [W]']),
                                                 Tjmax=float(r['T_j,max [°C]']),
                                                 note='' if _nan(r['note']) else str(r['note']))
    if lib != own:
        s[f"{K}|chipLib"] = lib
        st.rerun()
    if st.button("Save own technologies to the library", icon=':material/bookmark_add:', key=f"{K}|chipsave"):
        lib_save(chips=lib)
        st.toast(f"{len(lib)} chip technologies saved to config/gui/board_lib.json", icon=':material/check:')


def simulation():
    cm.section(3, "Simulation", "Layer-resolved finite-volume model of the whole board (x, y and every layer in z, die "
                                "attach and die on top), transient run with the loss profiles of the chips and steady "
                                "state for the last losses. Top view and cross-section are two views of this one "
                                "result.")
    c = st.columns(4)
    s = st.session_state
    mis = s.get(f"{K}|mission") if s.get(f"{K}|loadMode") == 'Mission profile' else None
    cm.num(c[0], "Duration t_end (s)", 200.0, 'board|tEnd', min_value=1.0, max_value=1e5,
           disabled=mis is not None, help="With a mission profile the duration is the length of the sequence.")
    cm.num(c[1], "Sampling time T_s (s)", 1.0, 'board|Ts', min_value=1e-3, max_value=100.0)
    cm.num(c[2], "Sub-steps per T_s", 2, 'board|nSub', format='%d', min_value=1, max_value=50,
           help="Implicit Euler steps per sampling period (accuracy of fast transients).")
    c[3].toggle("Re-run on changes", key=f"{K}|auto", help="Simulate again whenever an input changes.")
    cpl_on = st.toggle("Electro-thermal coupling", key=f"{K}|cpl", help="The losses of every chip rise linearly "
                       "with its junction temperature, P = P₀·(1 + α·(T_j − 25 °C)) with α of the chip table (column "
                       "α). Off: the losses are independent of the temperature (linear thermal model).")
    lay = _layout('xz')
    tEnd, Ts, nSub = s[f"{cm.KEY}|board|tEnd"], s[f"{cm.KEY}|board|Ts"], s[f"{cm.KEY}|board|nSub"]
    if mis is not None:
        tEnd = float(mis['t'][-1])
    try:
        D = genBoard(lay)
        D2 = genBoard(_layout('xy'))
    except Exception as e:
        st.error(f"Board could not be built: {e}", icon=':material/error:')
        return None
    st.caption(f"{D['N3']:,} nodes: {len(D['xg'])} × {len(D['yg'])} in x-y (dx = {fmt(D['dx'] * 1e3)} mm) × "
               f"{len(D['rowName'])} rows in z ({D['nBelow']} below the board, {D['nBoard'] - D['nBelow']} board, "
               f"{len(D['rowName']) - D['nBoard']} die attach + die), {len(D['chips'])} chips")
    if not D['chips']:
        st.info("Place at least one chip.", icon=':material/info:')
        return None
    if D['N3'] > 150000:
        st.warning("More than 150 000 nodes: increase dx.", icon=':material/warning:')
        return None
    sig = json.dumps([lay, tEnd, Ts, nSub, None if mis is None else mis['sig'], bool(cpl_on)], sort_keys=True,
                     default=str)
    res = s.get(f"{K}|res")
    nStep = int(tEnd / Ts + 1) * int(nSub)
    rate = s.get(f"{K}|rate")                                           # seconds per node and implicit step
    st.caption(f":material/schedule: Expected duration ≈ {fmt(rate * D['N3'] * nStep)} s ({nStep:,} implicit steps × "
               f"{D['N3']:,} nodes, from the speed of the last run)" if rate else
               f":material/schedule: {nStep:,} implicit steps × {D['N3']:,} nodes — the expected duration is shown "
               f"after the first run (and as a progress bar while it runs).")
    if st.button("Simulate", icon=':material/play_arrow:', type='primary', key=f"{K}|run") or \
            (res is not None and res['sig'] != sig and s.get(f"{K}|auto", False)):
        t0 = time.perf_counter()
        C, G, Bp = calcBoardMat(D)
        t = np.arange(0.0, tEnd + Ts * 1e-9, Ts)
        P = _profiles(lay, D, t, mission=mis)
        cpl = _cpl(D, Bp) if cpl_on else None
        bar = st.progress(0.0, text="Simulating the finite-volume model …")
        Tf, Tssf = simBoard(C, G, Bp, P, Ts, lay['Ta'], nSub, cpl=cpl, prog=_eta(bar, t0, "Finite-volume model"),
                            q=_q(D))
        bar.empty()
        dt = time.perf_counter() - t0
        s[f"{K}|rate"] = dt / max(D['N3'] * nStep, 1)
        s[f"{K}|res"] = res = {'sig': sig, 'lay': lay, 'D': D, 'D2': D2, 't': t, 'P': P, 'T': Tf, 'Tss': Tssf,
                               'C': C, 'G': G, 'Bp': Bp, 'nSub': nSub, 'time': dt, 'cpl': cpl, 'q': _q(D)}
    if res is None:
        st.info("Press *Simulate* to run the board model.", icon=':material/info:')
        return None
    if res['sig'] != sig:
        st.caption(":material/history: The results belong to the previous inputs — press *Simulate* to update.")
    return res


def _view_inputs(D, key):
    """View (top / cross-section) and the row of the top view; returns [view, row]."""
    c = st.columns([2, 2, 3])
    v = c[0].segmented_control("View", ['Top view (x-y)', 'Cross-section (x-z)'], default='Top view (x-y)',
                               key=f"{K}|view", help="Two views of the same 3D result.") or 'Top view (x-y)'
    view = 'xy' if v.startswith('Top') else 'xz'
    names = D['rowName']
    opts = list(range(len(names)))[::-1]
    lab = {r: (f"{names[r]} (chip)" if r >= D['nBoard'] else
               f"{names[r]} ({'below the board' if r < D['nBelow'] else 'board'})") for r in opts}
    row = c[1].selectbox("Layer of the top view", opts, index=opts.index(D['nBoard'] - 1), key=f"{K}|row",
                         format_func=lambda r: lab[r], disabled=view != 'xy',
                         help="Row of the stack shown in the top view: the dies (only under the chips), the top "
                              "copper, any inner layer, or the layers below the board (e.g. the heat sink).")
    c[2].caption(f"Cross-section at y = {fmt(D['yCut'] * 1e3)} mm (Layout › cut). z not to scale: every layer "
                 f"gets its own rows.")
    return [view, row]


def results(res):
    D, t, P, T, Tss, lay = res['D'], res['t'], res['P'], res['T'], res['Tss'], res['lay']
    Tj = _tj(D, T, P)
    Tjs = _tj(D, Tss[None, :], P[-1:])[0]
    names = [c['name'] for c in D['chips']]
    cm.section(None, "Result")
    cm.tiles([("T_max board (steady)", f"{fmt(Tss[D['idx'][D['nBoard'] - 1].ravel()].max())} °C",
               "Hottest node of the top board layer, steady state for the last losses."),
              ("T_j,max (transient)", f"{fmt(Tj.max())} °C", "Highest junction temperature of the run (hottest die "
                                                             "cell)."),
              ("Total loss (last)", f"{fmt(P[-1].sum() + (0.0 if res.get('q') is None else res['q'].sum()))} W",
               None if res.get('q') is None else f"Chips {fmt(P[-1].sum())} W + board {fmt(res['q'].sum())} W "
               "(constant)."),
              ("Simulation time", f"{fmt(res['time'])} s", f"{D['N3']:,} nodes × {len(t)} steps")])
    rows = []
    for i, c in enumerate(D['chips']):
        m = c['Tjmax'] - Tj[:, i].max()
        lvl = 'good' if m > 25 else ('warning' if m > 0 else 'critical')
        rows.append({'chip': c['name'], 'technology': c['tech'], 'P last (W)': P[-1, i],
                     'T_j steady (°C)': Tjs[i], 'T_j max run (°C)': Tj[:, i].max(), 'T_j,max (°C)': c['Tjmax'],
                     'margin (K)': m, 'status': {'good': 'OK', 'warning': 'tight', 'critical': 'too hot'}[lvl]})
    table(pd.DataFrame(rows), hide_index=True)
    st.caption("T_j: hottest cell of the resolved die. Margin = T_j,max − T_j max of the run; OK > 25 K, tight 0–25 K."
               + (" Electro-thermal coupling on: the losses rise with T_j (tab *Losses*)." if res.get('cpl') is not None
                  else ""))
    [view, row] = _view_inputs(D, 'res')
    xy = view == 'xy'
    [Fs, x, y, geo] = _disp(D, Tss, view, row)
    [F, _, _, _] = _disp(D, T, view, row)
    lim = [lay['Ta'], max(float(np.nanmax(T)), lay['Ta'] + 1e-6)]
    tabs = st.tabs(["Steady state", "Transient field", "Profiles & probes", "Junction temperatures", "Losses"])
    with tabs[0]:
        show(_heat(Fs[0], x, y, '°C', 'Steady-state temperature', equal=xy, scale=TEMP),
             height=430 if xy else 330, key=f"{K}|ss")
    with tabs[1]:
        i = st.slider("Time step", 0, len(t) - 1, len(t) - 1, key=f"{K}|tsl")
        st.caption(f"t = {fmt(t[i])} s")
        show(_heat(F[i], x, y, '°C', f"Temperature at t = {fmt(t[i])} s", equal=xy, zlim=lim),
             height=430 if xy else 330, key=f"{K}|tr")
    with tabs[2]:
        from gui.core.field_explore import profiles, probes
        i = st.session_state.get(f"{K}|tsl", len(t) - 1)
        lab = ('x (mm)', 'y (mm)') if xy else ('x (mm)', 'z (mm)')
        profiles([(f"t = {fmt(t[i])} s", F[i], False), ('steady state', Fs[0], True)], x, y, f"{K}|prof", vlabel=lab)
        st.caption("Time step from the slider of the tab *Transient field*.")
        probes(geo, [('T', F.reshape(len(t), -1), False)], t, f"{K}|probe", mark=t[i])
    with tabs[3]:
        fig = go.Figure([line(t, Tj[:, k], n, k) for k, n in enumerate(names)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='T_j (°C)')
        show(fig, key=f"{K}|tj")
    with tabs[4]:
        Pe = _peff(res)
        fig = go.Figure([line(t, P[:, k], n, k) for k, n in enumerate(names)] +
                        ([line(t, Pe[:, k], f"{n} with coupling", k, dash='dash') for k, n in enumerate(names)]
                         if res.get('cpl') is not None else []))
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)')
        show(fig, key=f"{K}|p", info="**Losses of the chips.** Solid: the nominal profile; dashed (electro-thermal "
             "coupling on): the losses at the simulated junction temperature.")
    return Tj


def _geo_sig(lay):
    """Signature of everything but the loads (the step responses do not depend on them)."""
    return json.dumps([{k: v for k, v in lay.items() if k != 'chips'},
                       [{k: c.get(k) for k in ('name', 'tech', 'x', 'y', 'w', 'l')} for c in lay['chips']]],
                      sort_keys=True, default=str)


def _stepNet(p, M, Nr, t):
    """Step responses Z(t) (nt x M x M) of the coupled Cauer network (sfNet matrices, modal solution)."""
    from src.py.model.SF.sfNet import sfNet
    [Gm, Cm, E] = sfNet(p, M, Nr)
    Si = 1 / np.sqrt(np.asarray(Cm, dtype=float).ravel()) if np.ndim(Cm) == 1 else 1 / np.sqrt(np.diag(Cm))
    Gm = np.asarray(Gm.toarray() if hasattr(Gm, 'toarray') else Gm, dtype=float)
    E = np.asarray(E.toarray() if hasattr(E, 'toarray') else E, dtype=float)
    lam, V = np.linalg.eigh(Si[:, None] * Gm * Si[None, :])
    B = V.T @ (Si[:, None] * E)                                         # modes x M
    O = E.T @ (Si[:, None] * V)                                         # M x modes (junction nodes)
    F = (1 - np.exp(-np.outer(t, lam))) / lam[None, :]                  # nt x modes
    return np.einsum('im,tm,mj->tij', O, F, B)


def _nodal(Zt, t, Rth, Nr):
    """
    Coupled Cauer network of the chips (SF mode, sfNet: Nr RC stages per chip from the junction to the cooling,
    couplings between the chips at every stage) fitted to the step responses of the board model (log-spaced times).
    """
    from scipy.optimize import least_squares
    M = Rth.shape[0]
    tau = np.geomspace(max(t[0], 1e-6) * 2, t[-1] / 3, Nr)
    Rl = np.outer(np.diag(Rth), np.ones(Nr)) / Nr
    nG = Nr * M * (M - 1) // 2
    p0 = np.concatenate([np.log(Rl).ravel(), np.log(tau[None, :] / Rl).ravel(),
                         np.full(nG, np.log(0.1 / np.mean(np.diag(Rth))))])
    sc = max(float(Zt.max()), 1e-12)
    sol = least_squares(lambda p: ((_stepNet(p, M, Nr, t) - Zt) / sc).ravel(), p0, method='trf',
                        bounds=(p0 - np.log(1e4), p0 + np.log(1e4)), max_nfev=300)
    p = sol.x
    return {'p': p, 'Nr': Nr, 'Zn': _stepNet(p, M, Nr, t), 'R': np.exp(p[0:M * Nr]).reshape(M, Nr),
            'C': np.exp(p[M * Nr:2 * M * Nr]).reshape(M, Nr), 'g': np.exp(p[2 * M * Nr:]).reshape(Nr, -1),
            'tau': np.array([t[int(np.argmax(Zt[:, i, i] >= 0.632 * Zt[-1, i, i]))] for i in range(M)])}


def _logSteps(C, G, Bp, j, tEnd, prog):
    """Response to a 1 W step in chip j on log-spaced times (segments with doubling time step, same model)."""
    tMin, n = tEnd * 1e-5, 30
    nSeg = int(np.ceil(np.log2(tEnd / (n * tMin) + 1)))
    M = Bp.shape[1]
    t, out, th, t0 = [0.0], [np.zeros(len(C))], None, 0.0
    for k in range(nSeg):
        Ts = tMin * 2 ** k
        P = np.zeros((n + 1, M))
        P[1:, j] = 1.0
        T, _ = simBoard(C, G, Bp, P, Ts, 0.0, 1, T0=th, steady=False)
        t += list(t0 + Ts * np.arange(1, n + 1))
        out.append(T[1:])
        th, t0 = T[-1], t0 + Ts * n
        prog((k + 1) / nSeg)
    return [np.array(t), np.vstack(out)]


@st.cache_data(show_spinner="Fitting the R/C network …", max_entries=8)
def _nodal_cached(Zt, t, Rth, Nr, sig):
    return _nodal(Zt, t, Rth, Nr)


def characterisation(res):
    cm.section(4, "Thermal characterisation", "Thermal resistance matrix between the chips and to the cooling (exact "
                                              "steady state of the layer-resolved model), step responses Z_th(t) of "
                                              "every chip and a nodal R/C network of the chips fitted to them.")
    D, lay, G, Bp, C = res['D'], res['lay'], res['G'], res['Bp'], res['C']
    names = [c['name'] for c in D['chips']]
    M = len(names)
    import scipy.sparse.linalg as spl
    B = np.asarray(D['B']) > 0

    # Rth matrix (exact steady state, 1 W per chip)
    luG = spl.splu(G.tocsc())
    TH = np.column_stack([luG.solve(Bp[:, j]) for j in range(M)])          # N x M, θ per W
    Rhot = np.array([[TH[B[:, i], j].max() for j in range(M)] for i in range(M)])
    Ravg = Bp.T @ TH                                                     # die average (Bp: weights, Σ = 1)
    tjdef = st.segmented_control("Junction temperature", ['Die average', 'Hottest cell'], default='Die average',
                                 key=f"{K}|tjdef", help="Die average (virtual junction temperature T_vj, as in "
                                 "datasheets): linear in the losses, so R_th matrix and Cauer network superpose "
                                 "exactly. Hottest cell: the hot spot of the die; it moves with the load distribution "
                                 "(max of a sum ≠ sum of the maxima), so superposition over-estimates it when "
                                 "several chips are loaded.") or 'Die average'
    Rth = Ravg if tjdef == 'Die average' else Rhot
    bot = np.array([float(D['gBot'] @ TH[:, j]) for j in range(M)])        # share of 1 W leaving through the bottom
    c = st.columns([3, 2])
    with c[0]:
        fig = go.Figure(go.Heatmap(z=Rth, x=names, y=names, colorscale='Reds', text=np.vectorize(fmt)(Rth),
                                   texttemplate='%{text}', colorbar=dict(title='K/W'),
                                   hovertemplate='ΔT_j of %{y} per W in %{x}: %{z:.3g} K/W<extra></extra>'))
        fig.update_layout(xaxis_title='heated chip j (1 W)', yaxis_title=f"junction of chip i ({tjdef.lower()})",
                          yaxis=dict(autorange='reversed'))
        show(fig, height=330, key=f"{K}|rth", info="**Thermal resistance matrix R_th,ij** = rise of the junction "
             "temperature of chip i per W in chip j (steady state, to the coolant/ambient). Diagonal: self-heating "
             "R_th,ja; off-diagonal: thermal coupling between the chips. T_j = T_a + Σ_j R_th,ij·P_j.")
    with c[1]:
        table(pd.DataFrame({'chip': names, 'R_th,ja (K/W)': np.diag(Rth),
                            'coupling to others (K/W)': Rth.sum(axis=1) - np.diag(Rth),
                            'max coupling factor': [max([Rth[i, j] / Rth[i, i] for j in range(M) if j != i] or [0])
                                                    for i in range(M)],
                            'heat to the coolant below (%)': 100 * bot,
                            'heat to the top (%)': 100 * (1 - bot)}), hide_index=True)
        st.caption("Coupling factor R_th,ij/R_th,ii: how much a W in a neighbour heats chip i compared to a W in "
                   "chip i itself. The heat split shows the two paths to the cooling (coolant h below, natural "
                   "convection h_top on top).")

    # Step responses
    from scipy.sparse.linalg import eigsh
    import scipy.sparse as sp
    gsig = _geo_sig(lay)
    stp = st.session_state.get(f"{K}|step")
    try:
        lam = float(eigsh(G.tocsc(), k=1, M=sp.diags(C).tocsc(), sigma=0, which='LM', return_eigenvectors=False)[0])
        tauMax = 1 / max(lam, 1e-12)
    except Exception:
        tauMax = float(res['t'][-1])
    tStep = 5 * tauMax
    nT = 30 * int(np.ceil(np.log2(tStep / (30 * tStep * 1e-5) + 1)))
    rate = st.session_state.get(f"{K}|rate")
    cc = st.columns([2, 1, 3], vertical_alignment='center')
    run = cc[0].button("Run step responses", icon=':material/stairs:', key=f"{K}|steprun",
                       help="One simulation per chip with a 1 W step (the others off) from 10⁻⁵·t_end to 5× the "
                            "slowest time constant on log-spaced times. The snapshots also train the POD model below.")
    Nr = int(cc[1].selectbox("Stages per chip", [1, 2, 3, 4], index=2, key=f"{K}|Nr", help="RC stages of the "
                             "network per chip (1: one node per chip = R_th/C_th matrix form)."))
    cc[2].caption(f"Slowest time constant τ_max = {fmt(tauMax)} s → {M} runs up to {fmt(tStep)} s"
                  + (f", expected ≈ {fmt(rate * D['N3'] * nT * M * 1.3)} s." if rate else "."))
    if run:
        t0 = time.perf_counter()
        bar = st.progress(0.0, text="Step responses …")
        Z, snap = None, []
        for j in range(M):
            prog = _eta(bar, t0, f"Step response {j + 1}/{M} ({names[j]})")
            ts, Tj_ = _logSteps(C, G, Bp, j, tStep, lambda x, j=j, p=prog: p((j + x) / M))
            Z = np.zeros((len(ts), M, M)) if Z is None else Z
            Za = np.zeros((len(ts), M, M)) if j == 0 else Za
            Z[:, :, j] = np.column_stack([Tj_[:, B[:, i]].max(axis=1) for i in range(M)])
            Za[:, :, j] = Tj_ @ Bp
            snap.append(Tj_[::6])
        bar.empty()
        st.session_state[f"{K}|step"] = stp = {'sig': gsig, 't': ts, 'Z': Z, 'Za': Za, 'snap': np.vstack(snap),
                                               'time': time.perf_counter() - t0}
        st.session_state[f"{K}|podsrc"] = 'Step responses'
    if stp is None or stp['sig'] != gsig:
        st.info("Press *Run step responses* for Z_th(t), the nodal R/C network of the chips and the POD training on "
                "step responses.", icon=':material/info:')
        return [None, None]
    if 'Za' not in stp:
        st.info("Run the step responses again (the stored ones are from an older version).", icon=':material/info:')
        return [None, None]
    ts, Z = stp['t'], (stp['Za'] if tjdef == 'Die average' else stp['Z'])
    try:
        net = _nodal_cached(Z[1:], ts[1:], Rth, Nr, gsig + tjdef)
        Zn = net['Zn']
    except Exception as e:
        st.warning(f"Network could not be fitted: {e}", icon=':material/warning:')
        net, Zn = None, None
    tabs = st.tabs(["Step responses Z_th(t)", "R/C network of the chips"])
    with tabs[0]:
        fig = go.Figure()
        for j in range(M):
            for i in range(M):
                fig.add_trace(go.Scatter(x=ts[1:], y=Z[1:, i, j], mode='lines', name=f"{names[i]} ← {names[j]}",
                                         line=dict(color=SERIES[j % len(SERIES)], dash=None if i == j else 'dot',
                                                   width=2.5 if i == j else 1.5)))
                if Zn is not None:
                    fig.add_trace(go.Scatter(x=ts[1:], y=Zn[:, i, j], mode='lines', showlegend=i == 0 and j == 0,
                                             name='R/C network', line=dict(color=INK2, dash='dash', width=1)))
        fig.update_layout(xaxis=dict(type='log', title='t (s)'), yaxis_title='Z_th (K/W)')
        show(fig, height=380, key=f"{K}|zth", info="**Thermal impedance** Z_th,ij(t): junction temperature rise of "
             "chip i after a 1 W step in chip j. Solid: self-heating (colour = heated chip), dotted: coupling; the "
             "coupling starts later because the heat has to spread to the neighbour first. Grey dashed: the fitted "
             "nodal network.")
    with tabs[1]:
        if net is not None:
            err = float(np.max(np.abs(Zn - Z[1:]))) / max(float(Z.max()), 1e-12)
            from gui.views.tools.sketch import coupled
            c = st.columns([3, 2])
            with c[0]:
                rows = []
                for i in range(M):
                    for k in range(Nr):
                        rows.append({'chip': names[i], 'stage': k + 1, 'R (K/W)': net['R'][i, k],
                                     'C (Ws/K)': net['C'][i, k], 'τ = RC (s)': net['R'][i, k] * net['C'][i, k]})
                table(pd.DataFrame(rows), hide_index=True)
                pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
                table(pd.DataFrame([{'coupling': f"{names[i]} ↔ {names[j]}",
                                     **{f"R stage {k + 1} (K/W)": 1 / net['g'][k, n] for k in range(Nr)}}
                                    for n, (i, j) in enumerate(pairs)]), hide_index=True)
            with c[1]:
                if M <= 4:
                    from gui.views.tools import sketch
                    sketch.show(coupled(M, min(Nr, 3), names), "Ladder per chip from the junction (top) to the "
                                "cooling; dashed: couplings between the chips at every stage.")
            st.caption(f"Coupled Cauer network of the SF mode: {Nr} RC stage(s) per chip from the junction to the "
                       f"cooling, the chips coupled at every stage. It reproduces all step responses of the board "
                       f"model within {100 * err:.1f} % of the largest Z_th (grey dashed in the first tab). τ63 "
                       f"(self-heating): {', '.join(f'{n}: {fmt(v)} s' for n, v in zip(names, net['tau']))}. Use it "
                       f"as a compact R/C model of the board in a circuit simulator.")
            _netMatrices(net, names, Rth)
    if net is not None:
        net['sig'] = gsig
        net['tjdef'] = tjdef
        net['Kavg'] = Ravg @ np.linalg.inv(Rth)                          # network rise → die average (coupling)
    return [stp, net]


def _netMatrices(net, names, Rth):
    """Conductance matrix G_th, capacitance matrix C_th and input matrix E of the fitted network (nodal form)."""
    from src.py.model.SF.sfNet import sfNet
    M, Nr = len(names), net['Nr']
    [Gm, Cm, E] = sfNet(net['p'], M, Nr)
    lab = [f"{n}·{k + 1}" if Nr > 1 else n for n in names for k in range(Nr)]
    rows = np.arange(0, M) * Nr
    Rn = np.linalg.inv(Gm)[rows][:, rows]
    st.markdown("**Matrix form of the network** — C_th·dT/dt = E·P − G_th·T (T: rise over the coolant/ambient at the "
                f"{M * Nr} nodes; node 'chip·k' is stage k of the chip, ·1 = junction)")
    c = st.columns(2)
    with c[0]:
        a = float(np.max(np.abs(Gm)))
        fig = go.Figure(go.Heatmap(z=Gm, x=lab, y=lab, colorscale='RdBu_r', zmin=-a, zmax=a,
                                   text=np.vectorize(fmt)(Gm), texttemplate='%{text}' if M * Nr <= 12 else None,
                                   colorbar=dict(title='W/K'),
                                   hovertemplate='G_th[%{y}, %{x}] = %{z:.4g} W/K<extra></extra>'))
        fig.update_layout(yaxis=dict(autorange='reversed'), title='G_th (W/K)')
        show(fig, height=360, key=f"{K}|Gth", info="**Thermal conductance matrix G_th** of the network: off-diagonal "
             "−1/R between two connected nodes (ladder stage to stage, coupling between the chips), diagonal: sum of "
             "all conductances at the node incl. the one to the cooling. Symmetric, positive definite.")
    with c[1]:
        gc = Gm.sum(axis=1)
        gc[np.abs(gc) < 1e-9 * np.max(np.abs(Gm))] = 0.0
        table(pd.DataFrame({'node': lab, 'C_th (Ws/K)': Cm, 'Σ_j G_th,ij to the cooling (W/K)': gc}),
              hide_index=True)
        st.caption("C_th is diagonal (one capacitance per node to the reference). The row sum of G_th is the "
                   "conductance of the node to the cooling: only the last stage of each ladder is connected to it.")
    c = st.columns([3, 2], vertical_alignment='bottom')
    with c[0]:
        table(pd.DataFrame(Rn, index=names, columns=names).reset_index(names='R_th of the network (K/W)'),
              hide_index=True)
        st.caption(f"Steady-state R_th matrix of the network, (G_th⁻¹) at the junction nodes; it deviates from the "
                   f"exact matrix of the board model above by {fmt(100 * np.max(np.abs(Rn - Rth)) / Rth.max())} % of "
                   f"the largest entry.")
    with c[1]:
        buf = io.StringIO()
        buf.write("# Coupled Cauer network of the board: C_th*dT/dt = E*P - G_th*T, T_j = T[junction nodes]\n")
        pd.DataFrame(Gm, index=lab, columns=lab).to_csv(buf, index_label='G_th (W/K)')
        buf.write("\n")
        pd.DataFrame({'C_th (Ws/K)': Cm}, index=lab).to_csv(buf, index_label='node')
        buf.write("\n")
        pd.DataFrame(E, index=lab, columns=names).to_csv(buf, index_label='E (-)')
        st.download_button("Download G_th, C_th, E (CSV)", buf.getvalue(), file_name='board_cauer_network.csv',
                           mime='text/csv', icon=':material/download:', key=f"{K}|netcsv", width='stretch')


def _simNet(net, M, P, Ts, Ta, nSub=4, cpl=None, dq=None):
    """
    Junction temperatures (Nt x M) (°C) of the fitted network for the losses P (Nt x M): exact modal solution per
    sub-step (zero-order hold); with electro-thermal coupling the junction temperature of the previous sub-step sets
    the losses (as in simBoard). The network gives the hottest die cell; the coupling uses the die average like the
    board model, mapped with the exact steady-state matrices (Kavg = R_avg·R_th⁻¹). dq: steady junction temperature
    rise of the constant board losses (hottest cell, die average), added as an offset (no transient).
    """
    from src.py.model.SF.sfNet import sfNet
    Nr = net['Nr']
    [Gm, Cm, E] = sfNet(net['p'], M, Nr)
    S = np.sqrt(Cm)
    lam, V = np.linalg.eigh(Gm / S[:, None] / S[None, :])
    h = Ts / max(int(nSub), 1)
    a = np.exp(-lam * h)
    gain = -np.expm1(-lam * h) / lam
    Bz = V.T @ (E / S[:, None])                                         # modes x M
    Oz = (V / S[:, None])[np.arange(0, M) * Nr]                          # M x modes (junction nodes)
    Oa = net.get('Kavg', np.eye(M)) @ Oz                                 # die-average rise
    z = np.zeros(len(lam))
    dh, da = (np.zeros(M), np.zeros(M)) if dq is None else dq
    Tj = np.zeros((P.shape[0], M))
    for k in range(1, P.shape[0]):
        for _ in range(0, max(int(nSub), 1)):
            p = P[k] if cpl is None else P[k] * np.maximum(1 + cpl['alpha'] * (Oa @ z + da + Ta - cpl['Tref']), 0)
            z = a * z + gain * (Bz @ p)
        Tj[k] = Oz @ z
    Tj[1:] += dh
    return Tj + Ta


def predict(res, rom, net):
    cm.section(6, "Prediction with the reduced models", "Run the POD model and/or the Cauer network on another load "
                                                        "(e.g. a different mission profile) without the finite-volume "
                                                        "model — there is no reference, so check their errors on the "
                                                        "loads of the simulation first (sections 4 and 5).")
    s = st.session_state
    D, lay = res['D'], res['lay']
    names = [c['name'] for c in D['chips']]
    M = len(names)
    os.makedirs(MIS_DIR, exist_ok=True)
    files = sorted(f[:-4] for f in os.listdir(MIS_DIR) if f.endswith('.csv'))
    c = st.columns([2, 2, 1, 1], vertical_alignment='bottom')
    name = c[0].selectbox("Mission profile", files, key=f"{K}|prfile", help="Sequence file of data/board/mission "
                          "(create or edit sequences under *Load › Mission profile* in section 2).")
    up = c[1].file_uploader("or upload a CSV", type=['csv'], key=f"{K}|prup")
    rept = cm.num(c[2], "Repeat", 1, 'board|prrep', format='%d', min_value=1, max_value=1000)
    Ts0 = float(res['t'][1] - res['t'][0]) if len(res['t']) > 1 else 1.0
    Ts = cm.num(c[3], "T_s (s)", Ts0, 'board|prTs', min_value=1e-3, max_value=100.0)
    opts = ['POD model'] + (['Cauer network'] if net is not None else [])
    c = st.columns([2, 3], vertical_alignment='center')
    mdl = c[0].segmented_control("Models", opts, default=opts, selection_mode='multi',
                                 key=f"{K}|prmdl|{len(opts)}") or []
    c[1].caption(f"POD: K = {rom['K']} modes (section 5) · Cauer: "
                 + (f"{net['Nr']} stage(s) per chip (section 4)" if net is not None else "run the step responses in "
                 "section 4 first") + " · electro-thermal coupling: " + ("on" if res.get('cpl') is not None else "off")
                 + " (switch in section 3).")
    if not files and up is None:
        st.info("No mission profile in data/board/mission.", icon=':material/info:')
        return
    try:
        seq = (pd.read_csv(up) if up is not None else pd.read_csv(os.path.join(MIS_DIR, f"{name}.csv"))
               ).to_dict('records')
        tEnd = genMission(seq, D['chips'], np.array([0.0, 1.0]), rept)[1]
        t = np.arange(0.0, tEnd + Ts * 1e-9, Ts)
        P = genMission(seq, D['chips'], t, rept)[0]
    except Exception as e:
        st.error(f"Mission profile could not be read: {e}", icon=':material/error:')
        return
    nT = len(t)
    st.caption(f"{fmt(tEnd)} s with T_s = {fmt(Ts)} s → {nT:,} steps. Chips without their own column in the sequence "
               f"use scale × P of the chip table.")
    tjdef = s.get(f"{K}|tjdef") or 'Die average'
    sig = json.dumps([seq, rept, Ts, rom['K'], rom['E'][-1], None if net is None else list(net['p']), mdl,
                      res['sig'], tjdef], default=str)
    out = s.get(f"{K}|pred")
    if st.button("Run prediction", icon=':material/play_arrow:', type='primary', key=f"{K}|prrun",
                 disabled=not mdl):
        out = {'sig': sig, 't': t, 'P': P, 'Tj': {}, 'time': {}, 'file': up.name[:-4] if up is not None else name}
        cpl = res.get('cpl')
        if 'POD model' in mdl:
            t0 = time.perf_counter()
            bar = st.progress(0.0, text="POD model …")
            Tr = simRom(rom, P, Ts, lay['Ta'], res['nSub'], cpl=cpl, prog=_eta(bar, t0, "POD model"),
                        q=res.get('q'))
            bar.empty()
            out['Tj']['POD model'] = (Tr - lay['Ta']) @ res['Bp'] + lay['Ta'] if tjdef == 'Die average' else \
                _tj(D, Tr, P)
            out['time']['POD model'] = time.perf_counter() - t0
        if 'Cauer network' in mdl and net is not None:
            t0 = time.perf_counter()
            dq = None
            if res.get('q') is not None:                                # board losses: steady offset of T_j
                import scipy.sparse.linalg as spl
                thq = spl.spsolve(res['G'].tocsc(), res['q'])
                dq = (_tj(D, thq[None, :] + lay['Ta'], np.zeros((1, M)))[0] - lay['Ta'], res['Bp'].T @ thq)
            out['Tj']['Cauer network'] = _simNet(net, M, P, Ts, lay['Ta'], res['nSub'], cpl=cpl, dq=dq)
            out['time']['Cauer network'] = time.perf_counter() - t0
        s[f"{K}|pred"] = out
    if out is None or not out['Tj']:
        st.info("Press *Run prediction*.", icon=':material/info:')
        return
    if out['sig'] != sig:
        st.caption(":material/history: The prediction belongs to the previous inputs — press *Run prediction*.")
    t, P = out['t'], out['P']
    rate = s.get(f"{K}|rate")
    rows = []
    for m, Tj in out['Tj'].items():
        for i in range(M):
            rows.append({'model': m, 'chip': names[i], 'T_j,max (°C)': Tj[:, i].max(),
                         't at T_j,max (s)': t[int(np.argmax(Tj[:, i]))], 'T_j,mean (°C)': Tj[:, i].mean(),
                         'ΔT_j swing (K)': Tj[:, i].max() - Tj[:, i].min(),
                         'margin to T_j,max (K)': D['chips'][i]['Tjmax'] - Tj[:, i].max()})
    c = st.columns([3, 2])
    with c[0]:
        fig = go.Figure()
        for n, (m, Tj) in enumerate(out['Tj'].items()):
            for i in range(M):
                fig.add_trace(go.Scatter(x=t, y=Tj[:, i], mode='lines', name=f"{names[i]} ({m})",
                                         line=dict(color=SERIES[i % len(SERIES)], dash=None if n == 0 else 'dash')))
        fig.update_layout(xaxis_title='t (s)', yaxis_title='T_j (°C)')
        show(fig, height=380, key=f"{K}|prfig", info="**Predicted junction temperatures** for the mission profile "
             "(solid: POD model, dashed: Cauer network; colour = chip), defined as in section 4 (die average or "
             "hottest cell).")
    with c[1]:
        fig = go.Figure([line(t, P[:, i], names[i], i) for i in range(M)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P₀ (W)')
        show(fig, height=380, key=f"{K}|prP", info="**Nominal losses** of the mission profile (with electro-thermal "
             "coupling the models raise them with T_j).")
    table(pd.DataFrame(rows), hide_index=True)
    tFV = rate * D['N3'] * len(t) * res['nSub'] if rate else None
    st.caption(f"Mission '{out['file']}': run time " + ", ".join(f"{m} {fmt(v)} s" for m, v in out['time'].items())
               + (f" — the finite-volume model would need about {fmt(tFV)} s." if tFV else ".")
               + (f" Largest difference between the two models: "
                  f"{fmt(float(np.max(np.abs(out['Tj']['POD model'] - out['Tj']['Cauer network']))))} K."
                  if len(out['Tj']) == 2 else ""))
    df = pd.DataFrame({'t (s)': t, **{f"P {n} (W)": P[:, i] for i, n in enumerate(names)},
                       **{f"T_j {n} {m} (°C)": Tj[:, i] for m, Tj in out['Tj'].items() for i, n in enumerate(names)}})
    st.download_button("Download the prediction (CSV)", df.to_csv(index=False), file_name=f"prediction_"
                       f"{out['file']}.csv", mime='text/csv', icon=':material/download:', key=f"{K}|prcsv")


def pod(res, stp=None):
    cm.section(5, "POD reduced model", "Proper orthogonal decomposition of training snapshots and Galerkin "
                                       "projection of the finite-volume model onto the first K modes; the reduced "
                                       "model is then run with the loads of the simulation (e.g. the mission "
                                       "profile) and compared to the full model.")
    D, t, T, lay = res['D'], res['t'], res['T'], res['lay']
    src = st.segmented_control("Training snapshots", ['Step responses', 'This run'],
                               default=None if f"{K}|podsrc" in st.session_state else
                               ('Step responses' if stp is not None else 'This run'), key=f"{K}|podsrc",
                               help="Step responses: train on the 1 W steps of every chip (section 4) and test on "
                                    "the loads of this run — a real test of the reduced model. This run: train and "
                                    "test on the same data.") or 'This run'
    if src == 'Step responses' and stp is None:
        st.info("Run the step responses in section 4 first; the run itself is used meanwhile.", icon=':material/info:')
        src = 'This run'
    snap = stp['snap'] + lay['Ta'] if src == 'Step responses' else T
    if src == 'Step responses' and res.get('q') is not None:          # field of the board losses as one snapshot
        import scipy.sparse.linalg as spl
        snap = np.vstack([snap, spl.spsolve(res['G'].tocsc(), res['q']) + lay['Ta']])
    Kmax = int(min(30, len(snap)))
    c = st.columns([1, 3])
    Kr = c[0].slider("Modes K", 1, Kmax, min(5, Kmax), key=f"{K}|K")
    rom = romBoard(res['C'], res['G'], res['Bp'], snap, lay['Ta'], Kr)
    t0 = time.perf_counter()
    Tr = simRom(rom, res['P'], t[1] - t[0] if len(t) > 1 else 1.0, lay['Ta'], res['nSub'], cpl=res.get('cpl'),
                q=res.get('q'))
    tr = time.perf_counter() - t0
    err = np.abs(Tr - T)
    rise = max(T.max() - lay['Ta'], 1e-9)
    c[0].caption(f"Trained on: **{src.lower()}** · tested on: the loads of this run"
                 + (" (mission profile)" if st.session_state.get(f"{K}|loadMode") == 'Mission profile' else ""))
    c[0].metric("Captured energy E_K", f"{100 * rom['E'][Kr - 1]:.4f} %")
    c[0].metric("Max. error", f"{fmt(err.max())} K", f"{100 * err.max() / rise:.2f} % of the rise", delta_color='off')
    c[0].metric("Unknowns", f"{Kr} instead of {D['N3']:,}", f"{fmt(res['time'] / max(tr, 1e-9))}× faster",
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
            view, row = ('xy' if (st.session_state.get(f"{K}|view") or 'Top').startswith('Top') else 'xz'), \
                st.session_state.get(f"{K}|row")
            [Fm, x, y, _] = _disp(D, rom['Phi'][:, k - 1], view, row)
            a = float(np.nanmax(np.abs(Fm))) or 1.0
            show(_heat(Fm[0], x, y, '–', f"POD mode {k} ({'top view' if view == 'xy' else 'cross-section'})",
                       equal=view == 'xy', zlim=[-a, a], scale='RdBu_r'), height=380, key=f"{K}|ph")
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
    cm.section(7, "Export to TherMOS", "Training data (random loss levels of all chips, independent excitation) and "
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
        D, lay = res['D2'], res['lay']
        C2, G2, B2 = calcBoardMat(D)
        Ts = float(res['t'][1] - res['t'][0]) if len(res['t']) > 1 else 1.0
        tTr = np.arange(0.0, st.session_state[f"{cm.KEY}|board|tTrain"] + Ts * 1e-9, Ts)
        Ptr = _profiles(lay, D, tTr, kind='random', seed=7)
        Ttr, _ = simBoard(C2, G2, B2, Ptr, Ts, lay['Ta'], res['nSub'])
        Tte, _ = simBoard(C2, G2, B2, res['P'], Ts, lay['Ta'], res['nSub'])
        folder = os.path.join(A.ROOT, 'data', 'board')
        saveBoard(D, tTr, Ptr, Ttr, os.path.join(folder, f"{name}_train.mat"))
        saveBoard(D, res['t'], res['P'], Tte, os.path.join(folder, f"{name}_test.mat"))
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
    D2 = res['D2']
    C2, G2, B2 = calcBoardMat(D2)
    _, Ts2 = simBoard(C2, G2, B2, res['P'][-1:], 1.0, res['lay']['Ta'], 1, q=_q(D2, len(C2)))
    Tj2 = _tj(D2, Ts2[None, :D2['geo'].shape[0]], res['P'][-1:])[0]
    Tj3 = _tj(res['D'], res['Tss'][None, :], res['P'][-1:])[0]
    cmp = ', '.join(f"{c['name']}: {fmt(a)} vs. {fmt(b)} °C" for c, a, b in zip(D2['chips'], Tj2, Tj3))
    st.caption(f"The PO mode needs one temperature per node of a plane, so the export uses the lumped top-view model "
               f"(board lumped over its thickness, layers below as an equivalent sink). Its steady junction "
               f"temperatures differ from the layer-resolved model by {fmt(float(np.max(np.abs(Tj2 - Tj3))))} K at "
               f"most (lumped vs. resolved: {cmp}). "
               "The data store the distributed heat sink h_v and the anisotropic conductivity k_y, so that the "
               "Galerkin projection of the PO mode includes the cooling (protocol 2/3, corrected version). Loading "
               "the config replaces the current GUI configuration.")


def pinn_struct(net, names):
    """Physics structure of the PI mode (setup['physPI']) from the fitted coupled Cauer network of the chips."""
    from src.py.model.SF.sfNet import sfNet
    M, Nr = len(names), net['Nr']
    [Gm, Cm, E] = sfNet(net['p'], M, Nr)
    return {'type': 'cauer', 'src': [f"x{i + 1}" for i in range(M)],
            'cauer': {'G': np.asarray(Gm).tolist(), 'C': np.asarray(Cm).ravel().tolist(), 'E': np.asarray(E).tolist(),
                      'out': [i * Nr + 1 for i in range(M)],
                      'labels': [f"{n}·{k + 1}" if Nr > 1 else n for n in names for k in range(Nr)]},
            'origin': f"board designer: coupled Cauer network ({Nr} stage(s) per chip)"}


def _tj_def(res, T, P):
    """Junction temperatures (die average or hottest cell, as selected in section 4)."""
    if (st.session_state.get(f"{K}|tjdef") or 'Die average') == 'Die average':
        return (T - res['lay']['Ta']) @ res['Bp'] + res['lay']['Ta']
    return _tj(res['D'], T, P)


def export_pinn(res, net):
    cm.section(8, "Export for the PI mode (multi-output PINN)",
               "Junction temperatures of all chips as 1D data (training: random load levels of every chip, several "
               "profiles; validation: another random profile; testing: the designed profile), the coupled Cauer "
               "network of section 4 as the physics of the PINN, and a ready-made PI config.")
    D, lay = res['D'], res['lay']
    names = [c['name'] for c in D['chips']]
    M = len(names)
    c = st.columns([2, 1, 1, 1], vertical_alignment='bottom')
    name = c[0].text_input("Data name", 'myBoard', key=f"{K}|pname", help="data/board/<name>_pinn.xlsx, "
                           "<name>_pinn.json (network) and config/gui/board_<name>_PI.json").strip().replace(' ', '_')
    cm.num(c[1], "Training profiles", 4, 'board|pN', format='%d', min_value=1, max_value=20,
           help="Number of random training profiles (sheets train1, train2, …).")
    cm.num(c[2], "Duration per profile (s)", float(res['t'][-1]), 'board|pT', min_value=1.0)
    go_ = c[3].button("Export PINN data", icon=':material/save:', key=f"{K}|pexp", width='stretch')
    if net is None:
        st.caption(":material/info: No R/C network fitted in section 4: the config uses the PINN without physics "
                   "(phys = 0). Fit the network first to train with the Cauer physics.")
    if go_ and name:
        from gui.core import config as conf
        from gui import adapter as A
        Ts = float(res['t'][1] - res['t'][0]) if len(res['t']) > 1 else 1.0
        nTr = int(st.session_state[f"{cm.KEY}|board|pN"])
        tTr = np.arange(0.0, st.session_state[f"{cm.KEY}|board|pT"] + Ts * 1e-9, Ts)
        bar, t0 = st.progress(0.0), time.perf_counter()
        sheets = {}

        def sim(P, k, n):
            T, _ = simBoard(res['C'], res['G'], res['Bp'], P, Ts, lay['Ta'], res['nSub'], cpl=res.get('cpl'),
                            q=res.get('q'), prog=lambda x: bar.progress(min((k + x) / n, 1.0),
                                                                     text=f"Simulating profile {k + 1}/{n} · "
                                                                          f"{fmt(time.perf_counter() - t0)} s"))
            Pe = _peff({**res, 'P': P, 'T': T}, T) if res.get('cpl') else P
            return _tj_def(res, T, Pe), Pe

        jobs = [(f"train{k + 1}", tTr, _profiles(lay, D, tTr, kind='random', seed=11 + 17 * k)) for k in range(nTr)]
        jobs += [('val', tTr, _profiles(lay, D, tTr, kind='random', seed=997)), ('test', res['t'], res['P'])]
        for k, (sh, t, P) in enumerate(jobs):
            Tj, Pe = sim(P, k, len(jobs))
            df = pd.DataFrame({'time': t, 'time_id': t, 'id': k + 1})
            for i, n in enumerate(names):
                df[f"P_{n}"] = P[:, i]
            df['Ta'] = lay['Ta']
            for i, n in enumerate(names):
                df[f"T_{n}"] = Tj[:, i]
            sheets[sh] = df
        bar.empty()
        folder = os.path.join(A.ROOT, 'data', 'board')
        os.makedirs(folder, exist_ok=True)
        with pd.ExcelWriter(os.path.join(folder, f"{name}_pinn.xlsx")) as xw:
            for sh, df in sheets.items():
                df.to_excel(xw, sheet_name=sh, index=False)
        struct = pinn_struct(net, names) if net is not None else None
        if struct is not None:
            with open(os.path.join(folder, f"{name}_pinn.json"), 'w', encoding='utf-8') as f:
                json.dump(struct, f, indent=1)
        tau = float(np.max(net['tau'])) if net is not None else float(res['t'][-1]) / 5
        W = int(min(max(3 * tau / Ts, 50), 2000, len(tTr) - 1))
        cfg = conf.read_config('tutorial_PI')
        cfg['Run'].update({'name': f"board_{name}_PI", 'config': '/gui/_gui_run', 'mode': 'PI', 'format': 'xlsx',
                           'datDim': 1, 'datSep': 2, 'trFile': [f"{name}_pinn"], 'vlFile': [f"{name}_pinn"],
                           'teFile': f"{name}_pinn", 'trSheet': [f"train{k + 1}" for k in range(nTr)],
                           'vlSheet': ['val'], 'teSheet': 'test', 'inp': [f"P_{n}" for n in names] + ['Ta'],
                           'out': [f"T_{n}" for n in names], 'ref': ['Ta'], 'selPI': 2, 'archPI': None,
                           'physPI': struct, 'train': 1, 'test': 1, 'featRoll': 0})
        cfg['Exp']['gen'].update({'Ts': Ts, 'init': 1})
        cfg['Mdl']['dl'].update({'W': W, 'stride': max(W // 10, 1), 'epoch': 60, 'batch': 32, 'initLr': 0.002,
                                 'valPat': 12, 'gradTh': 1, 'lrDropPr': 20, 'lrDropFa': 0.5, 'shu': 3})
        cfg['Mdl']['pinn'].update({'phys': 2 if struct else 0, 'alpha': 1, 'beta': 1, 'gamma': 0,
                                   'warm': max(W // 20, 1), 'x0': 1, 'col': 3, 'pred': 1, 'TrefSrc': M + 1,
                                   'node': 1})
        cfg['Mdl']['gen']['dev'] = 1
        conf.save_config(cfg, f"board_{name}_PI")
        st.session_state[f"{K}|pexported"] = (f"board_{name}_PI", name, W, Ts)
        st.toast(f"Exported PINN data '{name}'", icon=':material/check:')
    done = st.session_state.get(f"{K}|pexported")
    if done:
        cfgname, nm, W, Ts = done
        st.success(f"Written: data/board/{nm}_pinn.xlsx (sheets train…, val, test; columns P_<chip> in W, Ta, "
                   f"T_<chip> in °C), data/board/{nm}_pinn.json (Cauer network) and config/gui/{cfgname}.json (PI mode, LSTM, "
                   f"W = {W} samples = {fmt(W * Ts)} s, β = 1, physics = the network, T_ref = Ta).",
                   icon=':material/check_circle:')
        if st.button(f"Load config '{cfgname}' in the GUI", icon=':material/upload:', key=f"{K}|pload"):
            from gui.core import config as conf
            from gui.core import forms
            forms.set_config(conf.read_config(cfgname), cfgname)
            st.toast(f"Config '{cfgname}' loaded — open Modes › PI (views Physics, Network, Run)",
                     icon=':material/check:')
    st.caption("The PINN predicts the junction temperatures of all chips at once: the network outputs the node "
               "temperatures of the Cauer network (hidden stages included), the physics loss enforces C·dT/dt = E·P "
               "− G·T between them. The network of section 4 neglects the constant board losses and is fitted to the "
               "step responses only — the PINN corrects such deviations from the data, the physics keeps it plausible "
               "between the training load levels.")


def interpretation(res, Tj, rom, err):
    D, D2, lay = res['D'], res['D2'], res['lay']
    i = int(np.argmax(Tj.max(axis=0) - np.array([c['Tjmax'] for c in D['chips']])))
    c = D['chips'][i]
    m = c['Tjmax'] - Tj[:, i].max()
    Tss = res['Tss']
    p = [f"The hottest chip relative to its limit is **{c['name']}** ({c['tech']}): T_j reaches "
         f"**{fmt(Tj[:, i].max())} °C**, {'a margin of ' + fmt(m) + ' K' if m >= 0 else fmt(-m) + ' K above'} its "
         f"maximum junction temperature of {fmt(c['Tjmax'])} °C."]
    p.append(f"The board conducts heat sideways with k_eff = {fmt(D2['kEff'])} W/(m K) over its {fmt(D2['tb'] * 1e3)}"
             f" mm thickness; the stack-up resistance is {fmt(D2['Rz'] * 1e6)} K mm²/W through the board. The spreading "
             f"length √(k_eff t_b / h_eff) = {fmt(np.sqrt(D2['kEff'] * D2['tb'] / D2['hEff']) * 1e3)} mm tells how far "
             f"the heat of a chip spreads — chips closer than about twice this length heat each other.")
    Ttop = Tss[D['idx'][D['nBoard'] - 1].ravel()]
    if D['nBelow']:
        Tsink = Tss[D['idx'][D['nBelow'] - 1].ravel()]
        p.append(f"Below the board the heat passes the interface material and the carrier/heat sink: the top of the "
                 f"layers below reaches {fmt(Tsink.max())} °C at its hottest point, "
                 f"{fmt(Ttop.max() - Tsink.max())} K below the hottest point of the board (steady state). This drop "
                 f"is set by the board stack-up and the interface material — a thinner or better gap filler lowers "
                 f"it directly; select the layers in *Layer of the top view* to follow the heat downwards.")
    if D2['via'].any():
        p.append(f"The thermal vias lower the through-plane resistance of the board from {fmt(D2['Rz'] * 1e6)} to "
                 f"{fmt(D2['RzVia'] * 1e6)} K mm²/W in the via region — compare the junction temperatures with "
                 f"'No vias' to see the benefit.")
    p.append("Top view and cross-section show the same layer-resolved result: the cross-section makes the "
             "temperature drop across the die attach, the board and the layers below visible, the top view the "
             "spreading and the coupling of the chips.")
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
    [stp, net] = characterisation(res)
    rom, err = pod(res, stp)
    predict(res, rom, net)
    export(res)
    export_pinn(res, net)
    cm.equations([
        ("Layer-resolved model (simulation, both views)",
         [r"C_i\,\dot T_i = \sum_j G_{ij}(T_j - T_i) + g_{out,i}(T_a - T_i) + B_i P + q_i,\quad "
          r"G_{ij}^{z} = \frac{A_i}{\Delta z_i/2k_{z,i} + \Delta z_j/2k_{z,j}},\quad "
          r"G_{ij}^{x} = \frac{\Delta y\,\bar{\Delta z}}{\Delta x/2k_{x,i} + \Delta x/2k_{x,j}}"],
         "Nodes on the x-y grid in every row of the stack (layers below, board layers, die attach and die over the "
         "chip footprints); h on the bottom of the lowest row, h_top on the top of every column; q_i: constant "
         "additional losses of the board (P of a region spread over its control volumes in the selected layer)."),
        ("Lumped board (export to the PO mode)", [r"k_{eff} = \frac{\sum_l k_{xy,l} t_l}{t_b},\quad (\rho c)_{eff} = "
                                    r"\frac{\sum_l \rho_l c_l t_l}{t_b},\quad h_{eff} = \frac{1}{\sum_l t_l/k_{z,l} + "
                                    r"1/h} + h_{top}"], "Stack-up of thickness t_b lumped into one plane."),
        ("Heat equation", [r"(\rho c)\,\partial_t T = \nabla\cdot(k\nabla T) - h_v (T - T_a) + Q,\quad "
                           r"h_v = h_{eff}/t_b,\quad Q = \frac{P}{A_{chip} t_b}"], None),
        ("Layers below the board (lumped export model): heat sink plane T₂",
         [r"(\rho c t)_2\,\partial_t T_2 = \nabla\cdot((k t)_2\nabla T_2) + \frac{T - T_2}{R_{12}} - "
          r"\frac{T_2 - T_a}{R_{2a}}",
          r"R_{12} = \sum_{board}\frac{t_l}{k_{z,l}} + \sum_{i<last}\frac{t_i}{k_i} + \frac{t_{last}}{2k_{last}},\quad "
          r"R_{2a} = \frac{t_{last}}{2k_{last}} + \frac{1}{h}"],
         "With layers below, the board is cooled through the heat sink plane (and h_top on top); R in m²K/W per "
         "area."),
        ("Finite volumes", [r"C\,\dot\theta = -G\,\theta + B\,P,\quad C_i = (\rho c)_i V_i,\quad G_{ij} = "
                            r"-\frac{A_{ij}}{d/2k_i + d/2k_j}"], "Vertex-centred control volumes, implicit Euler."),
        ("Copper coverage and vias", [r"k_{xy} = c\,k_{Cu} + (1-c)\,k_{FR4,xy},\quad k_z = \left(\frac{c}{k_{Cu}} + "
                                      r"\frac{1-c}{k_{FR4,z}}\right)^{-1},\quad k_{z,via} = \varphi_{Cu} k_{Cu} + "
                                      r"\varphi_h k_{fill} + (1-\varphi_{Cu}-\varphi_h)\,k_{FR4,z}"],
         "Patterned Cu layer with coverage c; FR4 in the via region with the area fractions of the barrels φ_Cu = "
         "π(r² − (r − t_pl)²)/p² and of the holes φ_h."),
        ("Junction temperature", [r"T_j = \max_{die\ cells} T"], "The die and the die attach are resolved (the "
                                                                   "lumped export model uses T + P·R_jb)."),
    ], sym=[('k_xy, k_z', 'in-plane and through-plane conductivity of layer l', 'W/(m K)'),
            ('t_l, t_b', 'layer and board thickness', 'm'), ('h, h_top', 'heat transfer coefficient bottom, top',
                                                             'W/(m² K)'),
            ('h_v', 'distributed heat sink', 'W/(m³ K)'), ('Q', 'volumetric heat source', 'W/m³'),
            ('V_i, A_ij, d', 'control volume, face area, node distance', 'm², m, m (per m depth)'),
            ('R_jb', 'junction-to-board resistance of die and die attach', 'K/W'),
            ('q_i', 'additional loss of the board in node i (traces, shunts, passives)', 'W'),
            ('c', 'copper coverage of a layer', '–'), ('p, r, t_pl', 'via pitch, drill radius, plating thickness',
                                                       'm')], title="Model equations")
    interpretation(res, Tj, rom, err)
