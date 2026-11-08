#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         pinn_phys.py
# Date:         08.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
GUI of the physics of the PI mode (src/py/model/PINN/pinnPhys):
- equation_panel(cfg): the PINN equations with every parameter of the configuration, where it enters and what a larger
  value does (Parameters view and Theory)
- view(m): view 'Physics' — loss sources, Foster or Cauer structure (setup['physPI']), import from the other modules
  (RC mode: Foster fit, SF mode: Foster matrix or coupled Cauer network, trained models, board designer network),
  conversions Foster <-> Cauer, step responses Z_th(t), network diagram and a check of the physics alone against the
  training data
- lab(cfg): example 'variation of α, β, γ': a small PINN trained on synthetic data with the real pinnFit, with sweeps
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import copy
import glob
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core import forms
from gui.core.style import table, show, line, fmt
from gui.views.tools import _common as cm
from src.py.model.PINN import pinnPhys as pp

#######################################################################################################################
# Settings
#######################################################################################################################
K = 'piP'
COL_P = '#e07b00'
COL_D, COL_PH, COL_I, COL_N = '#d03b3b', '#3f8f4f', '#2a78d6', '#8a4fb3'
FOSTER_COLS = [('output', '–', "Output i (1 = first of Setup › outputs) whose temperature the term belongs to."),
               ('source', '–', "Loss source j (row of the table 'Loss sources') that heats the term."),
               ('R (K/W)', 'K/W', "Thermal resistance R_k of the term: steady contribution R_k·P_j. Coupling terms of "
                                  "a converted Cauer network can be negative."),
               ('τ (s)', 's', "Time constant τ_k = R_k·C_k: the term reaches 63 % of R_k·P_j after τ_k.")]
NODE_COLS = [('node', '–', "Name of the node (e.g. junction, case, heat sink)."),
             ('C (Ws/K)', 'Ws/K', "Thermal capacitance of the node to the reference."),
             ('output', '–', "Output i measured at this node (0 = hidden node, only in the physics loss).")]
EDGE_COLS = [('from', '–', "Node number (1 = first row of the node table)."),
             ('to', '–', "Node number, 0 = reference temperature T_ref (ambient / coolant)."),
             ('R (K/W)', 'K/W', "Thermal resistance between the two nodes (conductance 1/R in G).")]


#######################################################################################################################
# Helpers
#######################################################################################################################
def _outs(cfg):
    return list((cfg.get('Run') or {}).get('out') or []) or ['T']


def _inps(cfg):
    return list((cfg.get('Run') or {}).get('inp') or [])


def _ph(cfg, struct=None):
    """physDef of the configuration (optionally with another structure); (ph, error text)."""
    G = dict(cfg.get('Run') or {})
    if struct is not None:
        G['physPI'] = struct
        if struct.get('type') in ('foster', 'cauer'):                   # phys must match the structure type
            cfg = {**cfg, 'Mdl': {**cfg['Mdl'], 'pinn': {**(cfg['Mdl'].get('pinn') or {}),
                                                         'phys': 1 if struct['type'] == 'foster' else 2}}}
    try:
        return pp.physDef(cfg, G, len(_outs(cfg)), len(_inps(cfg)) or None), None
    except Exception as e:
        return None, str(e)


def _pin(cfg):
    return cfg['Mdl'].setdefault('pinn', {})


def _helpcols(cols):
    return {c: st.column_config.Column(c, help=f"{h} [{u}]") for c, u, h in cols}


#######################################################################################################################
# Equation panel (Parameters and Theory)
#######################################################################################################################
def equation_panel(cfg, expanded=True):
    ph, err = _ph(cfg)
    p = pp.physPara(cfg)
    dl = cfg['Mdl'].get('dl', {}) or {}
    with st.expander("How the parameters enter the PINN", icon=':material/function:', expanded=expanded):
        st.markdown("**1 Network** — predicts the states x of the physics network from a window of the inputs")
        st.latex(r"x(k) = \Delta T\cdot f_\theta\big(\,x_1 \dots x_F\ \text{of the window } [k-W+1,\,k]"
                 r",\ \color{" + COL_N + r"}{\Delta T_0}\,\big),\qquad \hat T = T_{ref} + O\,x")
        st.markdown("**2 Physics** — linear thermal network (Foster terms or Cauer nodes, view *Physics*)")
        st.latex(r"C\,\dot x = E\,P - G\,x,\qquad P_j = f_j(x_1, x_2, \dots, \hat T_1, \dots,\ "
                 r"\text{constants}),\qquad T_{ref}:\ \text{constant, feature or reference channel}")
        st.markdown("**3 Loss** — three dimensionless parts, the first $w$ samples of every window are skipped")
        st.latex(r"L = \color{" + COL_D + r"}{\alpha}\,\underbrace{\frac{1}{N}\sum_{k>\color{" + COL_N +
                 r"}{w}}\Big(\frac{\hat T - T}{\Delta T}\Big)^2}_{L_{data}} + \color{" + COL_PH +
                 r"}{\beta}\,\underbrace{\frac{1}{N}\sum_{k>w}\Big(\frac{C\dot x + G x - E P}{\mathrm{diag}(G)"
                 r"\,\Delta T}\Big)^2}_{L_{phys}} + \color{" + COL_I + r"}{\gamma}\,\underbrace{\frac{\sum_k e^{-t_k/"
                 r"\tau_{min}}\,(\hat T - T(0))^2}{\Delta T^2\sum_k e^{-t_k/\tau_{min}}}}_{L_{init}}")
        nS = ph['n'] if ph else '?'
        rows = [
            ('phys', '–', f"{pp.PHYS.get(p['phys'], p['phys'])}", "choice of C, G, E, O (view Physics)",
             "0 = no physics term; 1/2 = Foster / Cauer network of the view Physics"),
            ('alpha', 'α', fmt(p['alpha']), "weight of L_data", "pulls T̂ to the measured temperatures"),
            ('beta', 'β', fmt(p['beta']), "weight of L_phys",
             "pulls the states to the physics; too large: follows a wrong physics instead of the data"),
            ('gamma', 'γ', fmt(p['gamma']), "weight of L_init",
             "holds T̂ at T(0) during ~τ_min after a profile starts"),
            ('warm', 'w', str(p['warm']), "first w samples of a window not in L", "lets the recurrent state settle"),
            ('x0', 'ΔT₀', 'on' if p['x0'] else 'off', "extra input: measured rise at the window start",
             "gives the network the initial thermal state"),
            ('pred', '–', {1: 'whole profile', 2: 'windows'}.get(p['pred'], p['pred']), "testing",
             "1: one pass over every profile (state carried), 2: windows of W"),
            ('W, stride', 'W', f"{dl.get('W', '?')}, {dl.get('stride', '?')}", "window length and step of training",
             "W·Ts should cover the dominant time constant (τ_max)"),
            ('epoch, initLr, batch', 'θ', f"{dl.get('epoch')}, {fmt(dl.get('initLr', 0))}, {dl.get('batch')}",
             "Adam on L", "more epochs/larger lr: faster fit, risk of over-fitting / instability"),
            ('lrAdapt, lrDropPr, lrDropFa', 'η', f"{'adaptive' if int(dl.get('lrAdapt', 0) or 0) else 'fixed steps'}, "
             f"{dl.get('lrDropPr')}, {fmt(dl.get('lrDropFa', 0))}", "learning rate of Adam over the epochs",
             "adaptive: ×lrDropFa after lrDropPr epochs without better validation"),
            ('valPat', '–', str(dl.get('valPat')), "early stopping on the validation error",
             "validation = testing procedure (profiles from their first sample)"),
            ('gradTh', '–', fmt(dl.get('gradTh', 0)), "clipping of ‖∇L‖", "0 = off; 1 stabilises the LSTM"),
            ('TrefSrc, Tref', 'T_ref', ({0: f"{fmt(p['Tref'])} °C", -1: 'reference channels'}.get(
                p['TrefSrc'], f"x{p['TrefSrc']}")), "offset of T̂ = T_ref + O·x", "ambient / coolant temperature"),
            ('sources', 'P_j', ', '.join(ph['src']) if ph else '?', "E·P (heat input)",
             "losses that heat the network"),
            ('col', '–', {0: 'none', 1: 'validation inputs', 2: 'testing inputs', 3: 'both'}.get(p['col'], p['col']),
             "windows without temperatures, only in L_phys", "physics on unseen load profiles (little data)"),
            ('states', 'x', str(nS), "outputs of the network", "number of Foster terms / Cauer nodes (+ free)"),
        ]
        table(pd.DataFrame(rows, columns=['parameter (Mdl › pinn / dl)', 'symbol', 'current value', 'enters as',
                                          'larger value / effect']), hide_index=True)
        if err:
            st.warning(f"The physics of the current configuration is not valid: {err}", icon=':material/warning:')
        elif ph and ph['nP']:
            st.caption(f"Current physics: {ph['type']} ({ph['origin']}), {ph['nP']} physics states, time constants "
                       f"{', '.join(fmt(v) for v in ph['tau'][0:6])} s. ΔT is the 99th percentile of |T − T_ref| of "
                       f"the training data (printed in the log). The example *Variation of α, β, γ* in the Theory "
                       f"view "
                       f"shows the effect of the weights on a small case.")


#######################################################################################################################
# Data of the configuration
#######################################################################################################################
def load_data(cfg):
    """Training/validation data of the configuration (loader, offset; no normalisation), cached in the session."""
    from gui import adapter as A
    from gui.core.config import build_setup, ROOT
    keyd = json.dumps({k: cfg.get(k) for k in ('Exp', 'Dat', 'Par')} | {'Run': {
        k: v for k, v in (cfg.get('Run') or {}).items() if k not in ('archDL', 'archPI', 'physPI', 'mode')}},
        sort_keys=True, default=str)
    c = st.session_state.get(f"{K}|data")
    if c and c[0] == keyd:
        return c[1]
    from src.py.general.loadPara import loadPara
    from src.py.data.loadData import loadData
    from src.py.data.offRemove import offRemove
    schema = json.load(open(os.path.join(ROOT, 'gui', 'schema.json'), encoding='utf-8'))
    buf, cwd = io.StringIO(), os.getcwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(buf), A.session():
            setup = A.flat_setup(build_setup(copy.deepcopy(cfg), 'PI', schema), 'PI')
            para = loadPara(setup['config'], A.paths())
            data = loadData(para, setup, A.paths())
            for s in ('tr', 'te', 'vl'):
                data[s] = offRemove(data[s], para, int(setup.get('legacy', 0)))
    except SystemExit as e:
        raise RuntimeError(str(e) or buf.getvalue()[-500:])
    finally:
        os.chdir(cwd)
    out = {'data': data, 'para': para, 'setup': setup}
    st.session_state[f"{K}|data"] = (keyd, out)
    return out


def _decimate(X, y, r, n=60000):
    """Block mean of the inputs, sampled outputs: at most n samples in total (fits of RC/SF)."""
    N = sum(len(x) for x in X)
    q = max(1, int(np.ceil(N / n)))
    if q == 1:
        return X, y, r, 1
    bm = lambda a: np.asarray(a)[0:len(a) // q * q].reshape(-1, q, np.shape(a)[1]).mean(1)
    return [bm(x) for x in X], [np.asarray(v)[0:len(v) // q * q:q] for v in y], \
        [None if v is None else np.asarray(v)[0:len(v) // q * q:q] for v in r], q


def _profiles(D, ph, which='tr'):
    """Profiles of a split: raw inputs, outputs, references, loss sources P (profiles cut at the session IDs)."""
    d = D['data'][which]
    X2, y2, r2, _ = pp.split(d)
    init = int(D['para']['Exp']['gen'].get('init', 1) or 1)
    out = []
    for X, y, r in zip(X2, y2, r2):
        X, y = np.asarray(X, float), np.asarray(y, float)
        ref = np.asarray(pp.reference(X, None if r is None else np.asarray(r, float), ph, init), float)
        P = np.asarray(pp.sources(X, y, ph), float) if ph['S'] else np.zeros((len(X), 0))
        out.append((X, y, ref, P))
    return out, float(d['Ts'])


#######################################################################################################################
# Structure editors
#######################################################################################################################
def _nodes_edges(c):
    """Node and edge tables of a nodal network {'G', 'C', 'E', 'out', 'labels'}."""
    G, C = np.atleast_2d(np.asarray(c['G'], float)), np.asarray(c['C'], float).ravel()
    E = np.asarray(c['E'], float).reshape(len(C), -1)
    n = len(C)
    out = list(c.get('out') or [])
    node_out = [0] * n
    for i, o in enumerate(out):
        if o and 0 < int(o) <= n:
            node_out[int(o) - 1] = i + 1
    lab = list(c.get('labels') or [f"node {k + 1}" for k in range(n)])
    nodes = pd.DataFrame({'node': lab, 'C (Ws/K)': C, 'output': node_out})
    for j in range(E.shape[1]):
        nodes[f"P{j + 1}"] = E[:, j]
    edges, gref = [], G.sum(axis=1)
    for a in range(n):
        for b in range(a + 1, n):
            if G[a, b] < -1e-15:
                edges.append({'from': a + 1, 'to': b + 1, 'R (K/W)': -1 / G[a, b]})
        if gref[a] > 1e-12 * np.abs(G).max():
            edges.append({'from': a + 1, 'to': 0, 'R (K/W)': 1 / gref[a]})
    return nodes, pd.DataFrame(edges, columns=['from', 'to', 'R (K/W)'])


def _from_tables(nodes, edges, S, M):
    """Nodal network from the node and edge tables."""
    nodes = nodes.dropna(subset=['C (Ws/K)']).reset_index(drop=True)
    n = len(nodes)
    G = np.zeros((n, n))
    for _, e in edges.dropna().iterrows():
        a, b, R = int(e['from']), int(e['to']), float(e['R (K/W)'])
        if not 1 <= a <= n or not 0 <= b <= n or a == b or R <= 0:
            raise ValueError(f"edge {a} → {b} with R = {R}: nodes 1 … {n} (to = 0: reference) and R > 0 expected")
        g = 1 / R
        G[a - 1, a - 1] += g
        if b > 0:
            G[b - 1, b - 1] += g
            G[a - 1, b - 1] -= g
            G[b - 1, a - 1] -= g
    E = np.zeros((n, S))
    for j in range(S):
        if f"P{j + 1}" in nodes:
            E[:, j] = pd.to_numeric(nodes[f"P{j + 1}"], errors='coerce').fillna(0).to_numpy()
    out = [0] * M
    for k, o in enumerate(pd.to_numeric(nodes['output'], errors='coerce').fillna(0).astype(int)):
        if 0 < o <= M:
            out[o - 1] = k + 1
    return {'G': G.tolist(), 'C': nodes['C (Ws/K)'].astype(float).tolist(), 'E': E.tolist(), 'out': out,
            'labels': [str(v) for v in nodes['node']]}


def _foster_df(chains):
    rows = []
    for c in chains or []:
        for R, t in zip(np.ravel(c['R']), np.ravel(c['tau'])):
            rows.append({'output': int(c['out']), 'source': int(c['src']), 'R (K/W)': float(R), 'τ (s)': float(t)})
    return pd.DataFrame(rows, columns=[c[0] for c in FOSTER_COLS])


def _foster_chains(df):
    df = df.dropna(subset=['R (K/W)', 'τ (s)']).copy()
    df['output'] = pd.to_numeric(df['output'], errors='coerce').fillna(1).astype(int)
    df['source'] = pd.to_numeric(df['source'], errors='coerce').fillna(1).astype(int)
    out = []
    for (i, j), g in df.groupby(['output', 'source'], sort=True):
        out.append({'out': int(i), 'src': int(j), 'R': g['R (K/W)'].astype(float).tolist(),
                    'tau': g['τ (s)'].astype(float).tolist()})
    return out


def _sync(struct):
    """Working copy of the structure in the session (new version when the configuration changed elsewhere)."""
    s = st.session_state
    if s.get(f"{K}|last") != struct:
        s[f"{K}|base"] = copy.deepcopy(struct)
        s[f"{K}|last"] = copy.deepcopy(struct)
        s[f"{K}|ver"] = s.get(f"{K}|ver", 0) + 1
    return s[f"{K}|base"], s.get(f"{K}|ver", 0)


def _set_struct(cfg, struct, rerun=True):
    cfg['Run']['physPI'] = copy.deepcopy(struct)
    st.session_state[f"{K}|last"] = None                                # force a new editor version
    if struct.get('type') in ('foster', 'cauer'):
        _pin(cfg)['phys'] = 1 if struct['type'] == 'foster' else 2
    if rerun:
        st.rerun()


#######################################################################################################################
# Settings and constants
#######################################################################################################################
SETTINGS = [("Loss weights", ['alpha', 'beta', 'gamma', 'warm']),
            ("Initial state, prediction and collocation", ['x0', 'pred', 'col']),
            ("Reference temperature", ['TrefSrc', 'Tref', 'node'])]
CONST_COLS = [('name', '–', "Symbol used in the loss equations (letters, digits, _; not x1, x2, … or T, T1, …, which "
                            "are the features and outputs)."),
              ('value', 'unit of the constant', "Numerical value in the unit of the next column."),
              ('unit', '–', "Unit (documentation only, the value is used as it is)."),
              ('meaning', '–', "What the constant is (documentation only).")]
LEGACY = {'Rs': ('Ω', "stator resistance per phase (from the V1.x parameter Mdl › pinn › Rs)"),
          'Rth': ('K/W', "thermal resistance (from the V1.x parameter Mdl › pinn › Rth)"),
          'Cth': ('Ws/K', "thermal capacitance (from the V1.x parameter Mdl › pinn › Cth)")}
RESERVED = {'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'tanh', 'minimum', 'maximum', 'where', 'pi'}


def settings(cfg, phys):
    """Section 2: the scalar settings of the PINN (Mdl › pinn) with their help texts."""
    cm.section(2, "Settings of the PINN", "Loss weights, initial state, prediction, collocation and reference "
                                          "temperature (Mdl › pinn). The question marks explain every setting; the "
                                          "panel *How the parameters enter the PINN* above shows where they enter.")
    for title, names in SETTINGS:
        st.markdown(f"**{title}**")
        cols = st.columns(4)
        for i, v in enumerate(names):
            try:
                par = forms.find('Mdl', 'pinn', v)
            except KeyError:
                continue
            with cols[i % 4]:
                forms.param('Mdl', par, cfg)
    p = pp.physPara(cfg)
    dl = cfg['Mdl'].get('dl', {}) or {}
    W, stride = int(dl.get('W', 0) or 0), max(int(dl.get('stride', 1) or 1), 1)
    if p['x0']:
        st.caption(
            f"Initial state ΔT₀ (x0 = 1): **training** — every window (W = {W} samples, one every {stride} samples) "
            f"gets the measured temperature at its own first sample, so the network learns to continue from a known "
            f"state. **Validation and testing** — only the very first sample of every profile is measured; the "
            + ("network then runs over the whole profile in one pass (pred = 1)." if p['pred'] == 1 else
               f"windows follow each other and every window starts from the predicted temperature at the end of the "
               f"previous one (pred = 2).")
            + " The validation (early stopping, kept network) is computed exactly like the testing.")
    else:
        st.caption("Initial state off (x0 = 0): the network has to infer the thermal state of every window from the "
                   "inputs alone; the first samples of a window are uncertain (use a warm-up w).")


def constants(cfg, base):
    """Section 3: general list of constants of the loss equations (Mdl › pinn › const)."""
    import re
    p = _pin(cfg)
    cm.section(3, "Constants", "Named values for the loss equations, e.g. resistances, temperature coefficients, "
                               "iron-loss coefficients. Add, change or delete rows; unit and meaning are "
                               "documentation.")
    s = st.session_state
    txt = str(p.get('const') or '')
    if s.get(f"{K}|clast") != txt:
        s[f"{K}|clast"] = txt
        s[f"{K}|cver"] = s.get(f"{K}|cver", 0) + 1
    try:
        rows = pp.parseConstRows(txt)
    except ValueError as e:
        st.error(f"The constants could not be read ({e}); they are reset.", icon=':material/error:')
        rows = []
    used = ' '.join(str(v) for v in base.get('src') or [])
    for k, (u, mean) in LEGACY.items():                                 # V1.x parameters used in an equation
        if re.search(rf"\b{k}\b", used) and k not in {r['name'] for r in rows} and p.get(k) is not None:
            rows.append({'name': k, 'value': float(p[k]), 'unit': u, 'meaning': mean})
    df = pd.DataFrame(rows, columns=[c[0] for c in CONST_COLS])
    ed = st.data_editor(df, key=f"{K}|const|{s.get(f'{K}|cver', 0)}", num_rows='dynamic', hide_index=True,
                        width='stretch', column_config={
                            'name': st.column_config.TextColumn(help=CONST_COLS[0][2], required=True),
                            'value': st.column_config.NumberColumn(format='%.6g', help=CONST_COLS[1][2],
                                                                   required=True),
                            'unit': st.column_config.TextColumn(help=CONST_COLS[2][2]),
                            'meaning': st.column_config.TextColumn(help=CONST_COLS[3][2], width='large')})
    new, bad = [], []
    for _, r in ed.iterrows():
        nm = str(r['name'] or '').strip() if isinstance(r['name'], str) else ''
        if not nm or pd.isna(r['value']):
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*", nm) or re.fullmatch(r"x\d+|T\d*", nm) or nm in RESERVED:
            bad.append(nm)
            continue
        new.append({'name': nm, 'value': float(r['value']), 'unit': '' if pd.isna(r['unit']) else str(r['unit']),
                    'meaning': '' if pd.isna(r['meaning']) else str(r['meaning'])})
    if bad:
        st.warning(f"Not used (reserved or invalid names): {', '.join(bad)}. Names start with a letter and must not "
                   f"be x1, x2, … (features), T, T1, … (outputs) or a function name.", icon=':material/warning:')
    dup = sorted({r['name'] for r in new if [q['name'] for q in new].count(r['name']) > 1})
    if dup:
        st.warning(f"Defined twice (the last row wins): {', '.join(dup)}.", icon=':material/warning:')
    out = pp.constText(new)
    if out != txt:
        p['const'] = out
        s[f"{K}|clast"] = out
    unused = [r['name'] for r in new if not re.search(rf"\b{r['name']}\b", used)]
    st.caption(f"{len(new)} constant(s)" + (f"; not used in any loss equation yet: {', '.join(unused)}" if unused
                                            else "") + ". Stored in Mdl › pinn › const as 'name = value [unit] "
               "meaning; …'.")


#######################################################################################################################
# View 'Physics'
#######################################################################################################################
def view(m):
    cfg = forms.cfg()
    G = cfg['Run']
    p = _pin(cfg)
    outs, inps = _outs(cfg), _inps(cfg)
    M = len(outs)
    st.caption("All settings of the PINN: the physics (a linear thermal network whose states the neural network "
               "predicts), the loss weights, the initial state, the constants and the loss sources. The network "
               "architecture is in *Network*, the training settings (window W, epochs, learning rate) in "
               "*Parameters* › dl. The structure is stored with the configuration (setup['physPI']), the scalar "
               "settings in Mdl › pinn of the workbook.")
    equation_panel(cfg, expanded=False)

    # ==============================================================================
    # 1 Model
    # ==============================================================================
    cm.section(1, "Physics model")
    cur = int(p.get('phys', 1) if p.get('phys') is not None else 1)
    new = st.segmented_control("Physics of the residual", [0, 1, 2], default=cur if cur in (0, 1, 2) else 1,
                               format_func={0: 'None (data only)', 1: 'Foster terms', 2: 'Cauer / nodal network'}.get,
                               key=f"{K}|phys", help="Mdl › pinn › phys. Foster: Σ R_k(1 − e^(−t/τ_k)) per output "
                               "(and source); Cauer: nodes with capacitances, connected by resistances.")
    if new is not None and new != cur:
        p['phys'] = int(new)
        st.rerun()
    settings(cfg, cur)
    if cur == 0:
        st.info("No physics: every output is a free state of the network and β has no effect — the PINN is a DL "
                "network with the windows, the initial-state input and the whole-profile prediction of the PI mode.",
                icon=':material/info:')
        return
    struct = pp.structure(cfg, G, M)
    own = bool(G.get('physPI')) and G['physPI'].get('type') == struct['type'] and struct is not None and \
        struct.get('origin') == (G.get('physPI') or {}).get('origin')
    st.caption(f"Source of the structure: **{struct.get('origin') or 'view Physics'}**" +
               ("" if own else " — edit it below or import one (section 6) to replace the default term."))
    base, ver = _sync(struct)

    # ==============================================================================
    # 3 Constants, 4 Loss sources
    # ==============================================================================
    constants(cfg, base)
    cm.section(4, "Loss sources P_j", "Equations of the input features x1, x2, … (raw units, Setup › inputs), the "
                                       "predicted outputs T1, T2, … (T = output 'node') and the constants of "
                                       "section 3.")
    st.caption("Features: " + ', '.join(f"x{k + 1} = {n}" for k, n in enumerate(inps)) + " · outputs: " +
               ', '.join(f"T{k + 1} = {n}" for k, n in enumerate(outs)) + " · constants: " +
               (', '.join(f"{k} = {fmt(v)}" for k, v in pp.physPara(cfg)['const'].items()) or 'none') + ".")
    sdf = st.data_editor(pd.DataFrame({'source': [f"P{j + 1}" for j in range(len(base.get('src') or []))],
                                       'equation (W)': list(base.get('src') or [])}),
                         key=f"{K}|src|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                         disabled=['source'], column_config={
            'equation (W)': st.column_config.TextColumn(help="e.g. x4 (feature 4 is the loss), 3*Rs*x4**2 (copper "
                                                             "losses with the current in feature 4 and the constant "
                                                             "Rs of section 3), x1*x2, "
                                                             "3*Rs*(1+k*(T-20))*x4**2 (temperature dependent)")})
    src = [str(v).strip() for v in sdf['equation (W)'].tolist() if isinstance(v, str) and str(v).strip()]
    S = len(src)

    # ==============================================================================
    # 3 Structure
    # ==============================================================================
    cm.section(5, "Structure of the network")
    new_struct, err = None, None
    if struct['type'] == 'foster':
        st.markdown("One row per Foster term: Z_ij(t) = Σ_k R_k·(1 − e^(−t/τ_k)) between source j and output i. A "
                    "single output needs one chain (output 1, source 1); several outputs one chain each "
                    "(self-heating) and optional coupling chains (output i ≠ source j).")
        fdf = st.data_editor(_foster_df(base.get('foster')), key=f"{K}|fos|{ver}", num_rows='dynamic',
                             hide_index=True, width='stretch', column_config={
            'output': st.column_config.NumberColumn(min_value=1, max_value=max(M, 1), step=1, help=FOSTER_COLS[0][2]),
            'source': st.column_config.NumberColumn(min_value=1, max_value=max(S, 1), step=1, help=FOSTER_COLS[1][2]),
            'R (K/W)': st.column_config.NumberColumn(format='%.5g', help=FOSTER_COLS[2][2]),
            'τ (s)': st.column_config.NumberColumn(min_value=1e-9, format='%.5g', help=FOSTER_COLS[3][2])})
        new_struct = {'type': 'foster', 'src': src, 'foster': _foster_chains(fdf),
                      'origin': base.get('origin') or 'view Physics'}
    else:
        nodes, edges = _nodes_edges(base['cauer'])
        for j in range(S):
            if f"P{j + 1}" not in nodes:
                nodes[f"P{j + 1}"] = 0.0
        nodes = nodes[[c for c in nodes.columns if not c.startswith('P') or int(c[1:]) <= S]]
        c = st.columns([3, 2])
        with c[0]:
            st.markdown("**Nodes** (capacitance to the reference, measured output, share of every loss source)")
            ndf = st.data_editor(nodes, key=f"{K}|nod|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                                 column_config={**_helpcols(NODE_COLS[:1]),
                                                'C (Ws/K)': st.column_config.NumberColumn(min_value=1e-12,
                                                                                          format='%.5g',
                                                                                          help=NODE_COLS[1][2]),
                                                'output': st.column_config.NumberColumn(min_value=0, max_value=M,
                                                                                        step=1, help=NODE_COLS[2][2]),
                                                **{f"P{j + 1}": st.column_config.NumberColumn(
                                                    format='%.3g', help=f"Share of source P{j + 1} that enters this "
                                                    f"node (E matrix, usually 1 at the junction, 0 elsewhere).")
                                                   for j in range(S)}})
        with c[1]:
            st.markdown("**Resistances** (to = 0: reference temperature)")
            edf = st.data_editor(edges, key=f"{K}|edg|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                                 column_config={'from': st.column_config.NumberColumn(min_value=1, step=1,
                                                                                      help=EDGE_COLS[0][2]),
                                                'to': st.column_config.NumberColumn(min_value=0, step=1,
                                                                                    help=EDGE_COLS[1][2]),
                                                'R (K/W)': st.column_config.NumberColumn(min_value=1e-12,
                                                                                         format='%.5g',
                                                                                         help=EDGE_COLS[2][2])})
        try:
            new_struct = {'type': 'cauer', 'src': src, 'cauer': _from_tables(ndf, edf, S, M),
                          'origin': base.get('origin') or 'view Physics'}
        except Exception as e:
            err = str(e)
    ph = None
    if new_struct is not None and err is None:
        ph, err = _ph(cfg, new_struct)
    if err:
        st.error(f"The structure is not valid: {err}", icon=':material/error:')
    elif new_struct != struct:
        if own or new_struct != base:
            G['physPI'] = new_struct
            st.session_state[f"{K}|last"] = copy.deepcopy(new_struct)
    if ph is None:
        ph, _ = _ph(cfg, struct)

    # ==============================================================================
    # 4 Import and conversion
    # ==============================================================================
    cm.section(6, "Import from the other modules and conversion",
               "Fit the network on the training data with the RC or SF mode, load a trained model or the network of "
               "the board designer, or convert between Foster and Cauer.")
    imports(cfg, ph, struct)

    # ==============================================================================
    # 5 Properties
    # ==============================================================================
    if ph is not None and ph['nP']:
        properties(cfg, ph)
        check(cfg, ph)


#######################################################################################################################
# Import and conversion
#######################################################################################################################
def imports(cfg, ph, struct):
    outs = _outs(cfg)
    M = len(outs)
    t = st.tabs(["Fit on the training data", "Trained RC / SF model", "Board designer", "Convert Foster ↔ Cauer"])
    with t[0]:
        st.caption("Fits the network to the configured training data with the loss sources of section 4 and the "
                   "reference temperature (section 2 › T_ref source). RC mode: one Foster chain per output "
                   "(source j = output i, output-error fit, rcFit); SF mode: coupled Cauer network of all outputs "
                   "(sfFit, needs one loss source per output). Long data are block-averaged to ≤ 60 000 samples.")
        c = st.columns([1, 2, 2], vertical_alignment='bottom')
        Kr = int(c[0].number_input("Order K / stages", 1, 6, 2, key=f"{K}|fitK",
                                   help="Number of Foster terms per output (RC) or Cauer stages per source (SF)."))
        doF = c[1].button("Fit Foster (RC mode)", icon=':material/show_chart:', key=f"{K}|fitF", width='stretch')
        doC = c[2].button("Fit Cauer network (SF mode)", icon=':material/account_tree:', key=f"{K}|fitC",
                          width='stretch', disabled=ph is None or ph['S'] != M,
                          help=None if ph is not None and ph['S'] == M else
                          f"Needs {M} loss sources (one per output), there are {ph['S'] if ph else '?'}.")
        if (doF or doC) and ph is not None:
            try:
                with st.spinner("Loading the data and fitting …"):
                    new = fit_rc(cfg, ph, Kr) if doF else fit_sf(cfg, ph, Kr)
                st.session_state[f"{K}|fitmsg"] = new.pop('_msg', '')
                _set_struct(cfg, new)
            except Exception as e:
                st.error(f"The fit failed: {e}", icon=':material/error:')
        if st.session_state.get(f"{K}|fitmsg"):
            st.success(st.session_state[f"{K}|fitmsg"], icon=':material/check_circle:')
    with t[1]:
        from gui import adapter as A
        files = sorted(glob.glob(os.path.join(A.ROOT, 'mdl', 'mdl_rc_*.pkl')) +
                       glob.glob(os.path.join(A.ROOT, 'mdl', 'mdl_sf_*.pkl')))
        if not files:
            st.caption("No trained RC or SF model in mdl/ (run the RC or SF mode with these data first; the models "
                       "are saved as mdl/mdl_rc_<name>.pkl and mdl/mdl_sf_<name>.pkl).")
        else:
            c = st.columns([3, 1], vertical_alignment='bottom')
            f = c[0].selectbox("Model", files, format_func=os.path.basename, key=f"{K}|mdlf",
                               help="RC: Foster chain of output 1 (source 1). SF: nodal Foster model (self-heating "
                                    "chains of every output) or coupled Cauer network.")
            if c[1].button("Load", icon=':material/upload:', key=f"{K}|mdll", width='stretch'):
                try:
                    with open(f, 'rb') as fh:
                        mdl = pickle.load(fh)
                    if os.path.basename(f).startswith('mdl_rc_'):
                        new = {'type': 'foster', 'src': (ph['src'] if ph and ph['S'] else ['x1'])[0:1],
                               'foster': [pp.fromRC(mdl)], 'origin': f"RC mode ({os.path.basename(f)})"}
                    else:
                        new = pp.fromSF(mdl)
                        new['origin'] += f" ({os.path.basename(f)})"
                    _set_struct(cfg, new)
                except Exception as e:
                    st.error(f"The model could not be loaded: {e}", icon=':material/error:')
    with t[2]:
        board_import(cfg)
    with t[3]:
        if ph is None or not ph['nP']:
            st.caption("No valid network to convert.")
        elif struct['type'] == 'cauer':
            st.markdown("**Cauer → Foster** (exact modal decomposition: Z_ij(t) = Σ_k r_ijk(1 − e^(−t/τ_k)), "
                        "τ_k = 1/λ_k of C⁻¹G; same step responses, but the states are no longer node temperatures)")
            cpl = st.toggle("Keep the coupling chains (output i ≠ source j)", True, key=f"{K}|cpl",
                            help="Off: only the self-heating chains i = j — the cross-heating between the outputs is "
                                 "lost (compare the step responses).")
            if st.button("Convert to Foster", icon=':material/swap_horiz:', key=f"{K}|c2f"):
                _set_struct(cfg, {'type': 'foster', 'src': ph['src'], 'foster': pp.cauer2foster(ph, cpl),
                                  'origin': f"converted from the Cauer network ({struct.get('origin', '')})"})
        else:
            st.markdown("**Foster → Cauer** (continued-fraction expansion of every self-heating chain into a ladder "
                        "with the same Z_ii(t); node 1 of each ladder is the junction)")
            chains = struct.get('foster') or []
            cpl = [c for c in chains if c['out'] != c['src']]
            if cpl:
                st.warning(f"{len(cpl)} coupling chain(s) (output ≠ source) have no ladder equivalent and are "
                           f"dropped; "
                           f"fit a coupled Cauer network (SF mode) instead.", icon=':material/warning:')
            if st.button("Convert to Cauer", icon=':material/swap_horiz:', key=f"{K}|f2c"):
                try:
                    _set_struct(cfg, foster_to_cauer(chains, ph['S'], M, ph['src'], struct.get('origin', '')))
                except Exception as e:
                    st.error(f"Conversion failed: {e}", icon=':material/error:')


def foster_to_cauer(chains, S, M, src, origin=''):
    """Cauer ladders (one per self-heating chain) of Foster chains."""
    blocks, out, E, lab = [], [0] * M, [], []
    n0 = 0
    for c in chains:
        if c['out'] != c['src']:
            continue
        Rc, Cc = pp.foster2cauer(c['R'], c['tau'])
        G, Cn, _ = pp.ladder(Rc[None], Cc[None])
        blocks.append((G, Cn))
        e = np.zeros((len(Cn), S))
        e[0, int(c['src']) - 1] = 1.0
        E.append(e)
        out[int(c['out']) - 1] = n0 + 1
        lab += [f"{'junction' if k == 0 else 'stage'} T{c['out']}·{k + 1}" for k in range(len(Cn))]
        n0 += len(Cn)
    if not blocks:
        raise ValueError("no self-heating chain (output = source)")
    from scipy.linalg import block_diag
    G = block_diag(*[b[0] for b in blocks])
    return {'type': 'cauer', 'src': list(src), 'origin': f"converted from Foster ({origin})",
            'cauer': {'G': G.tolist(), 'C': np.concatenate([b[1] for b in blocks]).tolist(),
                      'E': np.vstack(E).tolist(), 'out': out, 'labels': lab}}


def fit_rc(cfg, ph, Kr):
    """Foster chains (one per output, source j = i or 1) fitted with the RC mode on the training data."""
    from src.py.model.RC.rcFit import rcFit
    D = load_data(cfg)
    para = copy.deepcopy(D['para'])
    para['Mdl']['rc']['K'] = Kr
    para['Mdl']['gen']['opt'] = 0
    prof, Ts = _profiles(D, ph, 'tr')
    vprof, _ = _profiles(D, ph, 'vl')
    M, chains, msg = ph['M'], [], []
    for i in range(M):
        j = i if ph['S'] > i else 0
        def mk(pr):
            return [np.asarray(a[3])[:, j:j + 1] for a in pr], [a[1][:, i:i + 1] - a[2][:, i:i + 1] for a in pr]
        P, y = mk(prof)
        Pv, yv = mk(vprof)
        P, y, _, q = _decimate(P, y, [None] * len(P))
        Pv, yv, _, _ = _decimate(Pv, yv, [None] * len(Pv))
        with contextlib.redirect_stdout(io.StringIO()):
            mdl = rcFit({'X2': P, 'y2': y, 'Ts': Ts * q, 'Nt': [len(v) for v in y]},
                        {'X2': Pv, 'y2': yv, 'Ts': Ts * q, 'Nt': [len(v) for v in yv]}, para, 0)
        chains.append(pp.fromRC(mdl, i + 1, j + 1))
        msg.append(f"T{i + 1}: R = {', '.join(fmt(v) for v in mdl['Rth'])} K/W, τ = "
                   f"{', '.join(fmt(v) for v in mdl['tau'])} s")
    return {'type': 'foster', 'src': ph['src'], 'foster': chains, 'origin': f'RC mode fit (K = {Kr})',
            '_msg': "Foster fit: " + '; '.join(msg)}


def fit_sf(cfg, ph, Nr):
    """Coupled Cauer network of the SF mode fitted on the training data (one source per output)."""
    from src.py.model.SF.sfFit import sfFit
    D = load_data(cfg)
    para = copy.deepcopy(D['para'])
    para['Mdl']['rc']['K'] = Nr
    para['Exp']['gen']['init'] = 1
    prof, Ts = _profiles(D, ph, 'tr')
    X, y, r = [a[3] for a in prof], [a[1] for a in prof], [a[2] for a in prof]
    X, y, r, q = _decimate(X, y, r)
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = sfFit({'X2': X, 'y2': y, 'r2': r, 't2': [np.arange(len(v)) * Ts * q for v in y], 'Ts': Ts * q,
                     'y': np.concatenate(y)}, None, para, 0, 2)
    new = pp.fromSF(mdl)
    new['src'] = ph['src']
    new['origin'] = f"SF mode fit (coupled Cauer network, {Nr} stage(s) per source)"
    new['_msg'] = f"Cauer fit: steady R_th matrix diagonal {', '.join(fmt(v) for v in np.diag(mdl['Rth']))} K/W"
    return new


def board_import(cfg):
    from gui import adapter as A
    M = len(_outs(cfg))
    folder = os.path.join(A.ROOT, 'data', 'board')
    files = sorted(glob.glob(os.path.join(folder, '*_pinn.json')))
    st.caption("The board designer (Theory & Tools › Board designer) fits a coupled Cauer network of the chips to "
               "the step responses of its finite-volume model. *Export to TherMOS* writes the PINN data "
               "(data/board/<name>_pinn.xlsx), this network (<name>_pinn.json) and a PI config. Or upload the CSV "
               "'Download G_th, C_th, E' of the board designer.")
    c = st.columns([3, 1], vertical_alignment='bottom')
    if files:
        f = c[0].selectbox("Exported network", files, format_func=os.path.basename, key=f"{K}|bf")
        if c[1].button("Load", icon=':material/upload:', key=f"{K}|bl", width='stretch'):
            with open(f, 'r', encoding='utf-8') as fh:
                new = json.load(fh)
            _set_struct(cfg, new)
    else:
        c[0].caption("No exported network in data/board yet.")
    up = st.file_uploader("Network CSV of the board designer (board_cauer_network.csv)", type=['csv'],
                          key=f"{K}|bup")
    if up is not None and st.button("Use the uploaded network", key=f"{K}|bupb"):
        try:
            new = parse_board_csv(up.getvalue().decode('utf-8'), M)
            _set_struct(cfg, new)
        except Exception as e:
            st.error(f"The CSV could not be read: {e}", icon=':material/error:')


def parse_board_csv(txt, M):
    """Nodal network from the CSV of the board designer (G_th, C_th, E tables separated by empty lines)."""
    blocks = [b for b in txt.replace('\r', '').split('\n\n') if b.strip()]
    tabs = []
    for b in blocks:
        lines = [ln for ln in b.split('\n') if ln.strip() and not ln.startswith('#')]
        if lines:
            tabs.append(pd.read_csv(io.StringIO('\n'.join(lines)), index_col=0))
    if len(tabs) < 3:
        raise ValueError("three tables (G_th, C_th, E) expected")
    G, C, E = tabs[0].to_numpy(float), tabs[1].iloc[:, 0].to_numpy(float), tabs[2].to_numpy(float)
    lab = [str(v) for v in tabs[0].index]
    S = E.shape[1]
    out = [int(np.argmax(E[:, i])) + 1 if i < S else 0 for i in range(M)]
    return {'type': 'cauer', 'src': [f"x{j + 1}" for j in range(S)], 'origin': 'board designer (CSV)',
            'cauer': {'G': G.tolist(), 'C': C.tolist(), 'E': E.tolist(), 'out': out, 'labels': lab}}


#######################################################################################################################
# Properties of the network
#######################################################################################################################
def properties(cfg, ph):
    outs = _outs(cfg)
    cm.section(7, "What the physics network does")
    Rss = ph['O'][:, 0:ph['nP']] @ np.linalg.solve(ph['G'], ph['E']) if ph['S'] else np.zeros((ph['M'], 0))
    cm.tiles([("Physics states", f"{ph['nP']}", "Foster terms or Cauer nodes predicted by the network."),
              ("Hidden states", f"{int(np.sum(~np.any(ph['O'][:, 0:ph['nP']] != 0, axis=0)))}",
               "States without a measured output: they only enter the physics loss."),
              ("τ_min … τ_max", f"{fmt(ph['tau'][0])} … {fmt(ph['tau'][-1])} s",
               "Time constants of the network (eigenvalues of C⁻¹G). W·Ts should cover τ_max."),
              ("Steady R_th (diagonal)", ', '.join(fmt(v) for v in np.diag(Rss)[0:4]) + ' K/W' if Rss.size else '–',
               "Temperature rise of output i per W of source i in steady state: O·G⁻¹·E.")])
    tabs = st.tabs(["Step responses Z_th(t)", "Network diagram", "Matrices"])
    with tabs[0]:
        t = np.geomspace(max(ph['tau'][0] / 20, 1e-6), ph['tau'][-1] * 6, 200)
        Z = pp.zth(ph, t)
        fig = go.Figure()
        k = 0
        for i in range(ph['M']):
            for j in range(ph['S']):
                if np.max(np.abs(Z[:, i, j])) > 0 and k < 16:
                    fig.add_trace(line(t, Z[:, i, j], f"Z {outs[i] if i < len(outs) else i + 1} ← P{j + 1}", k,
                                       dash=None if i == j else 'dot'))
                    k += 1
        fig.update_layout(xaxis=dict(type='log', title='t (s)'), yaxis_title='Z_th (K/W)')
        show(fig, key=f"{K}|zth", info="Response of output i to a 1 W step of source j (solid: i = j, self-heating; "
                                       "dotted: coupling). The final value is the steady R_th, the rise time the "
                                       "time constants.")
        if Rss.size:
            st.caption(f"Output 1 reaches {fmt(Rss[0, 0])} K per W of P1 in steady state; half of it after "
                       f"{fmt(float(t[np.argmax(Z[:, 0, 0] >= 0.5 * Z[-1, 0, 0])]))} s. A 1 W step of every source "
                       f"heats the outputs by {', '.join(fmt(v) for v in Rss.sum(axis=1)[0:6])} K.")
    with tabs[1]:
        try:
            st.graphviz_chart(diagram(ph, outs), width='content')
        except Exception as e:
            st.caption(f"Diagram not available: {e}")
    with tabs[2]:
        lab = ph['labels'][0:ph['nP']]
        c = st.columns(2)
        with c[0]:
            a = float(np.max(np.abs(ph['G'])))
            fig = go.Figure(go.Heatmap(z=ph['G'], x=lab, y=lab, colorscale='RdBu_r', zmin=-a, zmax=a,
                                       colorbar=dict(title='W/K' if ph['type'] == 'cauer' else '–')))
            fig.update_layout(yaxis=dict(autorange='reversed'), title='G')
            show(fig, height=340, key=f"{K}|G", info="Conductance matrix G of the residual (Foster: identity, "
                                                     "every term is its own state; Cauer: −1/R between connected "
                                                     "nodes, diagonal = sum of the conductances at the node).")
        with c[1]:
            df = pd.DataFrame({'state': lab, 'C': ph['C'], 'diag(G)': np.diag(ph['G'])})
            for j in range(ph['S']):
                df[f"E·P{j + 1}"] = ph['E'][:, j]
            for i in range(ph['M']):
                df[f"O→{outs[i] if i < len(outs) else i + 1}"] = ph['O'][i, 0:ph['nP']]
            table(df, hide_index=True)
            st.caption("C: capacitance (Cauer, Ws/K) or time constant (Foster, s); E: input of every source "
                       "(Cauer: share, Foster: R_k in K/W); O: which states sum up to an output.")


def diagram(ph, outs):
    L = ['digraph G {', 'rankdir=LR; nodesep=0.3; ranksep=0.5;',
         'node [fontname="Helvetica", fontsize=10, style="rounded,filled"];',
         'edge [fontname="Helvetica", fontsize=9];',
         'ref [label="T_ref", shape=box, fillcolor="#eeeeee"];']
    nP = ph['nP']
    for j, e in enumerate(ph['src']):
        L.append(f'P{j} [label="P{j + 1} = {e}", shape=ellipse, fillcolor="#fff4e5", color="{COL_P}"];')
    if ph['type'] == 'cauer':
        for k in range(nP):
            meas = [outs[i] if i < len(outs) else f"T{i + 1}" for i in range(ph['M']) if ph['O'][i, k]]
            L.append(f'n{k} [label="{ph["labels"][k]}\\nC = {fmt(ph["C"][k])} Ws/K' +
                     (f'\\nmeasured: {", ".join(meas)}' if meas else '\\n(hidden)') + '", shape=box, '
                     f'fillcolor="{"#fde8e8" if meas else "#f3e8f7"}"];')
        g = ph['G']
        for a in range(nP):
            for b in range(a + 1, nP):
                if g[a, b] < 0:
                    L.append(f'n{a} -> n{b} [dir=none, label="{fmt(-1 / g[a, b])} K/W"];')
            gr = g[a].sum()
            if gr > 1e-12 * np.abs(g).max():
                L.append(f'n{a} -> ref [dir=none, style=dashed, label="{fmt(1 / gr)} K/W"];')
        for j in range(ph['S']):
            for k in range(nP):
                if ph['E'][k, j]:
                    L.append(f'P{j} -> n{k} [color="{COL_P}"];')
    else:
        for k in range(nP):
            i = int(np.argmax(ph['O'][:, k]))
            L.append(f'n{k} [label="{ph["labels"][k]}\\nR = {fmt(ph["E"][k].sum())} K/W, τ = {fmt(ph["C"][k])} s", '
                     f'shape=box, fillcolor="#f3e8f7"];')
            j = int(np.argmax(np.abs(ph['E'][k])))
            L.append(f'P{j} -> n{k} [color="{COL_P}"];')
            L.append(f'n{k} -> o{i};')
        for i in range(ph['M']):
            L.append(f'o{i} [label="{outs[i] if i < len(outs) else i + 1} = T_ref + Σθ", shape=box, '
                     f'fillcolor="#fde8e8", color="{COL_D}"];')
        L.append('ref -> ' + ' ; ref -> '.join(f'o{i}' for i in range(ph['M'])) + ' [style=dashed];')
    L.append('}')
    return '\n'.join(L)


#######################################################################################################################
# Check against the data
#######################################################################################################################
def check(cfg, ph):
    outs = _outs(cfg)
    cm.section(8, "How good is the physics alone?", "Simulates the physics network with the loss sources of the "
                                                    "training data (exact, from the measured initial temperatures) "
                                                    "and compares it with the measured outputs.")
    if int((cfg.get('Run') or {}).get('datDim', 1) or 1) != 1:
        st.caption("Only for 1D data.")
        return
    if not st.toggle("Simulate the physics on the training data", False, key=f"{K}|chk",
                     help="Loads the training data of the configuration (cached in the session)."):
        return
    try:
        with st.spinner("Loading the data and simulating …"):
            D = load_data(cfg)
            prof, Ts = _profiles(D, ph, 'tr')
    except Exception as e:
        st.error(f"The data could not be loaded: {e}", icon=':material/error:')
        return
    err, show_k, sim = [], int(np.argmax([len(a[0]) for a in prof])), None
    for k, (X, y, ref, P) in enumerate(prof):
        x0 = None
        meas = np.any(ph['O'][:, 0:ph['nP']] != 0, axis=0)
        if ph['type'] == 'cauer':                                       # steady state of the initial rises
            x0 = np.zeros(ph['nP'])
            Om = ph['O'][:, 0:ph['nP']]
            d0 = (y[0] - ref[0])
            x0 = np.linalg.lstsq(Om, d0, rcond=None)[0] if np.any(meas) else x0
        else:
            x0 = np.zeros(ph['nP'])
            for i in range(ph['M']):
                idx = np.flatnonzero(ph['O'][i, 0:ph['nP']])
                if len(idx):
                    x0[idx] = (y[0, i] - ref[0, i]) * np.abs(ph['E'][idx].sum(1)) / max(
                        np.abs(ph['E'][idx].sum(1)).sum(), 1e-12)
        _, Tp = pp.simPhys(ph, P, Ts, x0)
        T = Tp + ref
        cov = [i for i in range(ph['M']) if np.any(ph['O'][i, 0:ph['nP']])]
        err.append((T - y)[:, cov])
        if k == show_k:
            sim = (np.arange(len(y)) * Ts, y, T, cov)
    e = np.concatenate(err)
    rmse = np.sqrt(np.mean(e ** 2, axis=0))
    dT = max(float(np.percentile(np.abs(np.concatenate([a[1] - a[2] for a in prof])), 99)), 1.0)
    cm.tiles([("Profiles", f"{len(prof)}", "Training profiles (cut at every session ID)."),
              ("RMSE physics alone", ', '.join(fmt(v) for v in rmse[0:4]) + ' K', "Per measured output."),
              ("Bias", ', '.join(fmt(v) for v in e.mean(0)[0:4]) + ' K', "Mean of physics − measurement."),
              ("ΔT scale", f"{fmt(dT)} K", "99th percentile of |T − T_ref|; RMSE/ΔT is the relative physics error.")])
    t, y, T, cov = sim
    n = max(1, len(t) // 4000)
    fig = go.Figure()
    for q, i in enumerate(cov[0:4]):
        fig.add_trace(line(t[::n], y[::n, i], f"measured {outs[i] if i < len(outs) else i + 1}", q))
        fig.add_trace(line(t[::n], T[::n, i], f"physics {outs[i] if i < len(outs) else i + 1}", q, dash='dot'))
    fig.update_layout(xaxis_title='t (s)', yaxis_title='T (°C)')
    show(fig, key=f"{K}|chkfig", info="Solid: measured, dotted: physics network alone (no neural network) on the "
                                      "longest training profile.")
    rel = float(np.max(rmse)) / dT
    cm.interpret([
        f"The physics alone deviates by {fmt(float(np.max(rmse)))} K RMSE ({fmt(100 * rel)} % of ΔT = {fmt(dT)} K) "
        f"from the measurement, with a bias of {fmt(float(e.mean()))} K. "
        + ("This is a good physics prior: β = 0.1 … 1 lets it regularise the network strongly."
           if rel < 0.05 else "This is a usable prior: start with β = 0.05 … 0.2." if rel < 0.15 else
           "The physics is a poor description of the data: keep β small (≤ 0.05), or improve it — fit it on the "
           "data (section 6, RC/SF mode), add loss sources (e.g. iron losses with the speed) or nodes."),
        "The physics loss does not need the physics to be exact: the network predicts the states and the residual "
        "only penalises how far they violate C·dx/dt = E·P − G·x. A wrong R or τ biases the prediction by roughly β "
        "times the physics error — the β sweep in the Theory view shows the trade-off."])


#######################################################################################################################
# Example: variation of α, β, γ
#######################################################################################################################
LAB_R, LAB_TAU, LAB_TS = [0.1, 0.3], [20.0, 300.0], 2.0


def _lab_profile(n, seed, Pmax=100.0):
    rng = np.random.default_rng(seed)
    P, k = np.zeros(n), 0
    while k < n:
        L = int(rng.integers(30, 150))
        P[k:k + L] = rng.uniform(0, Pmax) * (rng.random() > 0.2)
        k += L
    Ta = 25 + 3 * np.sin(np.arange(n) * LAB_TS / 1800 + seed)
    return P, Ta


@st.cache_data(show_spinner=False, max_entries=48)
def lab_run(alpha, beta, gamma, warm, x0, noise, mismatch, nTr, epochs, dev, nCol=6, seed=0):
    """Small PINN (LSTM) trained with pinnFit on synthetic data of a 2-term Foster network; test on a new profile."""
    from src.py.model.PINN.pinnFit import pinnFit
    from src.py.model.PINN.pinnSol import pinnSol
    true = {'type': 'foster', 'src': ['x1'], 'foster': [{'out': 1, 'src': 1, 'R': LAB_R, 'tau': LAB_TAU}]}
    model = {'type': 'foster', 'src': ['x1'], 'origin': 'example',
             'foster': [{'out': 1, 'src': 1, 'R': [r * (1 + mismatch) for r in LAB_R], 'tau': LAB_TAU}]}
    para = {'Mdl': {'dl': {'W': 150, 'stride': 10, 'epoch': int(epochs), 'batch': 16, 'initLr': 3e-3,
                           'valPat': int(epochs), 'gradTh': 1, 'lrDropPr': 10 ** 6, 'lrDropFa': 0.5, 'shu': 3},
                    'pinn': {'phys': 1, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'warm': int(warm),
                             'x0': int(x0), 'TrefSrc': 2, 'pred': 2},
                    'gen': {'dev': dev, 'err': 1e-3, 'nSub': 5}},
            'Exp': {'gen': {'init': 1}}, 'Par': {'gen': {'normX': 1, 'normY': 1}, 'loss': {'Rohm': 0}}}
    phT = pp.physDef(para, {'physPI': true}, 1, 2)
    rng = np.random.default_rng(seed + 100)

    def prof(n, s, noisy):
        P, Ta = _lab_profile(n, s)
        _, T = pp.simPhys(phT, P[:, None], LAB_TS)
        T = T[:, 0] + Ta
        Tm = T + (rng.normal(0, noise, n) if noisy else 0)
        return np.c_[P, Ta], Tm[:, None], Ta[:, None], T
    tr = [prof(600, s, True) for s in range(int(nTr))]
    col = [(prof(600, 200 + s, False)[0], None) for s in range(int(nCol))]       # inputs only
    vl = [prof(600, 50, True)]
    te = [prof(2400, 99, False)]
    mk = lambda L: {'X2': [a[0] for a in L], 'y2': [a[1] for a in L], 'r2': [a[2] for a in L], 'Ts': LAB_TS}
    torch_seed = __import__('torch').manual_seed
    torch_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = pinnFit({**mk(tr), 'col': col}, mk(vl), para, {'selPI': 2, 'physPI': model})
        out = pinnSol(mdl, mk(te), para, {'selPI': 2, 'physPI': model})
    X, Ttrue = te[0][0], te[0][3]
    Tp = out['y'][:, 0]
    phM = pp.physDef(para, {'physPI': model}, 1, 2)
    _, Tphys = pp.simPhys(phM, X[:, 0:1], LAB_TS)
    Tphys = Tphys[:, 0] + X[:, 1]
    x = out['xPhys'][0]
    dx = np.gradient(x, LAB_TS, axis=0)
    r = (phM['C'] * dx + x @ phM['G'].T - X[:, 0:1] @ phM['E'].T) / mdl['info']['dT']
    return {'t': np.arange(len(Tp)) * LAB_TS, 'P': X[:, 0], 'Ttrue': Ttrue, 'Tpred': Tp, 'Tphys': Tphys,
            'rmse': float(np.sqrt(np.mean((Tp - Ttrue) ** 2))), 'max': float(np.max(np.abs(Tp - Ttrue))),
            'rmsePhys': float(np.sqrt(np.mean((Tphys - Ttrue) ** 2))), 'res': float(np.sqrt(np.mean(r ** 2))),
            'info': {k: v for k, v in mdl['info'].items() if isinstance(v, (list, float, int))},
            'nWin': int(mdl['info']['nWin']), 'train': [a[1][:, 0] for a in tr]}


def lab(cfg):
    from src.py.general.getDevice import getDevice
    cm.section(None, "Example: variation of α, β, γ", "A small PINN (LSTM, W = 150 samples) is trained with the "
               "real pinnFit on synthetic data of a two-term Foster network (R = 0.1, 0.3 K/W, τ = 20, 300 s, "
               "Ts = 2 s, random load steps 0 … 100 W) and tested on a new 80-minute profile. The physics given to "
               "the PINN can be wrong (mismatch), the measurements noisy, and unlabelled load profiles (no "
               "temperatures) can be added for the physics loss.")
    c = st.columns(5)
    alpha = cm.num(c[0], "α (data)", 1.0, f"{K}|la", min_value=0.0, help="Weight of the data loss.")
    beta = cm.num(c[1], "β (physics)", 0.1, f"{K}|lb", min_value=0.0, help="Weight of the physics loss.")
    gamma = cm.num(c[2], "γ (initial)", 0.0, f"{K}|lg", min_value=0.0, help="Weight of the initial condition.")
    warm = cm.num(c[3], "warm-up w", 10, f"{K}|lw", format='%d', min_value=0, max_value=100)
    x0 = c[4].toggle("Initial state input x0", True, key=f"{K}|lx0")
    c = st.columns(5)
    noise = cm.num(c[0], "Noise σ (K)", 1.0, f"{K}|ln", min_value=0.0,
                   help="Measurement noise of the training temperatures (the test truth is clean).")
    mism = cm.num(c[1], "Physics mismatch (%)", 0.0, f"{K}|lm", min_value=-80.0, max_value=300.0,
                  help="The physics of the PINN uses R·(1 + mismatch): 0 % = exact physics.") / 100
    nTr = int(cm.num(c[2], "Training profiles (20 min each)", 1, f"{K}|lt", format='%d', min_value=1,
                     max_value=8, help="Little data: the physics helps most."))
    ep = int(cm.num(c[3], "Epochs", 40, f"{K}|le", format='%d', min_value=5, max_value=300))
    nCol = int(cm.num(c[4], "Unlabelled profiles", 6, f"{K}|lc", format='%d', min_value=0, max_value=20,
                      help="Collocation profiles: 20-min load profiles without temperatures. They only enter "
                           "the physics loss (Mdl › pinn › col uses the inputs of the validation/testing data)."))
    dev = 1 if getDevice(cfg, 0).type == 'cuda' else 0
    args = (alpha, beta, gamma, warm, int(x0), noise, mism, nTr, ep, dev, nCol)
    c = st.columns([1, 1, 3])
    if c[0].button("Train", icon=':material/play_arrow:', key=f"{K}|lrun", type='primary'):
        with st.spinner("Training the example PINN …"):
            st.session_state[f"{K}|lres"] = (args, lab_run(*args))
    if c[1].button("β sweep", icon=':material/stacked_line_chart:', key=f"{K}|lsw",
                   help="Trains the example for β = 0, 0.01, 0.1, 1, 10 with the other settings."):
        prog = st.progress(0.0)
        res = []
        for k, b in enumerate([0.0, 0.01, 0.1, 1.0, 10.0]):
            res.append((b, lab_run(alpha, b, gamma, warm, int(x0), noise, mism, nTr, ep, dev, nCol)))
            prog.progress((k + 1) / 5)
        prog.empty()
        st.session_state[f"{K}|lsweep"] = (args, res)
    c[2].caption("Training takes a few seconds on a GPU, up to a minute on the CPU; results are cached.")
    R = st.session_state.get(f"{K}|lres")
    if R:
        a, r = R
        if a != args:
            st.caption(":material/info: The settings changed — press *Train* to update the result.")
        cm.tiles([("Test RMSE PINN", f"{fmt(r['rmse'])} K", "Root mean square error on the clean test profile."),
                  ("Physics alone", f"{fmt(r['rmsePhys'])} K", "RC network of the PINN simulated without network."),
                  ("Physics residual", f"{fmt(r['res'])}", "RMS of r = (C·dx/dt + G·x − E·P)/ΔT of the prediction."),
                  ("Training windows", f"{r['nWin']}", "Windows of W = 150 samples, stride 10.")])
        t = st.tabs(["Test prediction", "Losses over the epochs"])
        with t[0]:
            fig = go.Figure([line(r['t'], r['Ttrue'], 'truth', 0, width=3), line(r['t'], r['Tpred'], 'PINN', 1),
                             line(r['t'], r['Tphys'], 'physics alone', 2, dash='dot')])
            fig.update_layout(xaxis_title='t (s)', yaxis_title='T (°C)')
            show(fig, key=f"{K}|lfig", info="Truth: network of the example; PINN: prediction over the whole test "
                                            "profile from its initial temperature; physics alone: the (possibly "
                                            "wrong) RC network of the PINN.")
        with t[1]:
            i = r['info']
            ep_ = np.arange(1, len(i['loss']) + 1)
            fig = go.Figure([line(ep_, i['lossData'], 'L_data (training)', 0), line(ep_, i['lossPhys'],
                                                                                   'L_phys (training)', 2),
                             line(ep_, np.asarray(i['valLoss']) / i['dT'] ** 2, 'L_data (validation)', 0, dash='dot')])
            fig.update_layout(xaxis_title='epoch', yaxis=dict(type='log', title='loss (dimensionless)'))
            show(fig, key=f"{K}|lloss", info="Both losses are relative to ΔT²; their weighted sum α·L_data + "
                                             "β·L_phys is minimised.")
    SW = st.session_state.get(f"{K}|lsweep")
    if SW:
        a, res = SW
        b = [max(v[0], 1e-3) for v in res]
        fig = go.Figure([line(b, [v[1]['rmse'] for v in res], 'test RMSE PINN (K)', 1),
                         line(b, [v[1]['rmsePhys'] for v in res], 'physics alone (K)', 2, dash='dot'),
                         line(b, [v[1]['res'] * v[1]['info']['dT'] for v in res], 'physics residual · ΔT (K)', 3,
                              dash='dash')])
        fig.update_layout(xaxis=dict(type='log', title='β (β = 0 drawn at 0.001)'), yaxis_title='K')
        show(fig, key=f"{K}|lsweepfig", info="Test error of the PINN and the remaining physics residual over β, "
                                             "the other settings fixed (noise, mismatch, data as set above).")
        best = min(res, key=lambda v: v[1]['rmse'])
        r0 = res[0][1]['rmse']
        cm.interpret([
            f"With noise σ = {fmt(a[5])} K, {a[7]} training profile(s), {a[10]} unlabelled profile(s) and a physics "
            f"mismatch of {fmt(100 * a[6])} %, "
            f"β = 0 (data only) gives {fmt(r0)} K test RMSE; the best β = {fmt(best[0])} gives "
            f"{fmt(best[1]['rmse'])} K ({fmt(100 * (1 - best[1]['rmse'] / max(r0, 1e-12)))} % "
            f"{'better' if best[1]['rmse'] <= r0 else 'worse'}).",
            "Why: with one labelled profile the network memorises its windows (training error ≈ 1 K, test ≈ 12 K). "
            "The physics loss forces the predicted states to obey C·dx/dt = E·P − G·x — and it needs no "
            "temperatures, so it is also evaluated on the unlabelled profiles: the network learns the response to "
            "load situations it has never seen measured. Without unlabelled profiles (0) β helps little, because "
            "the physics then only acts on the windows the data already fit. With a wrong physics (mismatch) a "
            "large β pulls the prediction towards the wrong network.",
            "Try: unlabelled profiles 0 vs 6 at β = 1; mismatch 50 % → the optimum moves to a small β; 8 training "
            "profiles → the data alone are enough and β matters less. α only sets the scale against β (the ratio "
            "β/α counts); γ helps when the profiles start in equilibrium. In a real configuration Mdl › pinn › col "
            "adds the inputs of the validation and/or testing data as unlabelled profiles."])


#######################################################################################################################
# References
#######################################################################################################################
