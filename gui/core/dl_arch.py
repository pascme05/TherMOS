#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         dl_arch.py
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
View 'Network' of the DL and PI modes: shows the network that will be trained (layers, tensor shapes, parameters,
diagram) and lets the user change it. The built-in networks (setup selDL: CNN, LSTM, CNN + LSTM) can be replaced by a
user-defined architecture (setup archDL / archPI, see src/py/model/DL/dlNet): a list of layers (convolution, LSTM, GRU,
transformer encoder, dense), the pooling of seq2point and the hidden dense layers of the head. Presets (dlNet.ARCHS)
are starting points; own architectures are stored in config/gui/dl_arch_lib.json.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import copy
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from gui.core import forms
from gui.core.style import table, fmt
from gui.core.theory_dl import arch_dot, summary, _net, _feat_names
from gui.core.theory_lumped import _get
from gui.views.tools import _common as cm
from src.py.model.DL.dlNet import ARCHS, BUILTIN, LAYER_TYPES, ACTS, POOLS

#######################################################################################################################
# Settings
#######################################################################################################################
K = 'dlA'
COLS = ['type', 'units', 'kernel', 'dilation', 'heads', 'bidir', 'norm', 'act', 'drop']
STAR = '★ '
W_SHOW = 1000                                                                 # longest window of the summary pass
TYPE_HELP = {'conv': "1D convolution over time: 'units' filters of length 'kernel' with the 'dilation' (gaps between "
                     "the taps); local patterns such as fast transients.",
             'lstm': "Long short-term memory: a state of 'units' values is carried through the window (slow "
                     "heating); 'bidir' reads the window in both directions.",
             'gru': "Gated recurrent unit: like the LSTM with one state vector, about 25 % fewer parameters.",
             'transformer': "Self-attention encoder layer (pre-norm, feed-forward 2 × units): every sample attends "
                            "to every other one; 'units' must be a multiple of 'heads'. A sinusoidal position code is "
                            "added; cost grows with W².",
             'dense': "Linear layer applied to every time step (e.g. an embedding before a transformer)."}


#######################################################################################################################
# Library
#######################################################################################################################
def _lib_file():
    from gui import adapter as A
    return os.path.join(A.ROOT, 'config', 'gui', 'dl_arch_lib.json')


def lib_load():
    try:
        with open(_lib_file(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def lib_save(name, arch):
    d = lib_load()
    d[name] = arch
    os.makedirs(os.path.dirname(_lib_file()), exist_ok=True)
    with open(_lib_file(), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def _presets():
    return {**ARCHS, **{STAR + n: v for n, v in lib_load().items()}}


#######################################################################################################################
# Functions
#######################################################################################################################
def _df(arch):
    return pd.DataFrame([{c: l.get(c) for c in COLS} for l in arch.get('body') or []], columns=COLS)


def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _arch(df, head, pool):
    """Architecture dictionary of the editor (rows without type or units are skipped)."""
    body = []
    for _, r in df.iterrows():
        if _nan(r['type']) or r['type'] not in LAYER_TYPES or _nan(r['units']):
            continue
        body.append({'type': r['type'], 'units': int(r['units']),
                     'kernel': 3 if _nan(r['kernel']) else int(r['kernel']),
                     'dilation': 1 if _nan(r['dilation']) else int(r['dilation']),
                     'heads': 4 if _nan(r['heads']) else int(r['heads']),
                     'bidir': bool(r['bidir']) if not _nan(r['bidir']) else False,
                     'norm': bool(r['norm']) if not _nan(r['norm']) else False,
                     'act': 'relu' if _nan(r['act']) else str(r['act']),
                     'drop': 0.0 if _nan(r['drop']) else float(r['drop'])})
    return {'body': body, 'head': head, 'pool': pool}


def _receptive(arch, W):
    """Samples of the window that reach the last output (convolutions only; recurrent/attention: the window)."""
    rf, full = 1, False
    for l in arch['body']:
        if l['type'] == 'conv':
            rf += (int(l['kernel']) - 1) * int(l['dilation'])
        elif l['type'] in ('lstm', 'gru', 'transformer'):
            full = True
    return W if full else min(rf, W), full


def _set(arch, ver_bump=True):
    s = st.session_state
    s[f"{K}|base"] = copy.deepcopy(arch)
    s[f"{K}|last"] = copy.deepcopy(arch)
    if ver_bump:
        s[f"{K}|ver"] = s.get(f"{K}|ver", 0) + 1


#######################################################################################################################
# View
#######################################################################################################################
MODE_KEYS = {'DL': ('selDL', 'archDL', 'dlA'), 'PI': ('selPI', 'archPI', 'piA')}


def view(m):
    global K
    s = st.session_state
    cfg = forms.cfg()
    G = cfg['Run']
    vsel, varch, K = MODE_KEYS.get(m, MODE_KEYS['DL'])
    dl = _get(cfg, 'Mdl', 'dl', default={}) or {}
    W, seq = int(dl.get('W', 600) or 600), int(dl.get('seq', 0) or 0)
    seq = min(seq, W)
    names = _feat_names(cfg)
    outs = list(_get(cfg, 'Run', 'out', default=[]) or []) or ['T']
    F, M = len(names), len(outs)
    sel = int(G.get(vsel) or 1)
    cur = G.get(varch) or None
    if m == 'PI':
        seq = 0
        try:
            from src.py.model.PINN.pinnPhys import physDef
            ph = physDef(cfg, G, M)
            nS, x0 = ph['n'], int(ph.get('x0', 1))
        except Exception as e:
            st.warning(f"The physics of the configuration is not valid ({e}); the network is shown with one state per "
                       f"output.", icon=':material/warning:')
            nS, x0 = M, 1
        names = names + ([f"ΔT0({o})" for o in outs] if x0 else [])
        st.caption(f"The network of the PINN: {len(names)} inputs ({', '.join(map(str, names[0:8]))}"
                   f"{' …' if len(names) > 8 else ''}) in windows of W = {W} samples → the {nS} states x of the "
                   f"physics network (view *Physics*), seq2seq; the outputs are T̂ = T_ref + O·x"
                   + (". ΔT0: measured temperature rise at the start of the window (Physics › settings › x0)."
                      if x0 else ".") + " W: Parameters › dl.")
        F, Mo = len(names), nS
    else:
        Mo = M
        st.caption(f"The network that the DL mode trains: {F} input features ({', '.join(map(str, names[0:8]))}"
                   f"{' …' if F > 8 else ''}) in windows of W = {W} samples → {M} outputs "
                   f"({', '.join(map(str, outs[0:8]))}{' …' if M > 8 else ''}), "
                   + ("seq2point (target sample s = %d)" % seq if seq > 0 else "seq2seq (one output per sample)")
                   + ". W and s: Parameters › dl.")
    c = st.columns([2, 3], vertical_alignment='center')
    on = c[0].toggle("User-defined architecture", value=cur is not None, key=f"{K}|on",
                     help=f"Off: the built-in network of the model variant (Parameters › Model variant). On: the "
                          f"layers below replace it (setup['{varch}']); it is saved with the configuration.")
    if not on:
        if cur is not None:
            G[varch] = None
        c[1].caption(f"Built-in network: **{BUILTIN.get(sel, 'CNN-LSTM')}** (model variant). Switch on to edit a "
                     f"copy of it or to start from another preset.")
        arch = None
    else:
        if cur is None:                                                     # start from the built-in network
            cur = {k: v for k, v in copy.deepcopy(ARCHS[BUILTIN.get(sel, 'CNN-LSTM')]).items() if k != 'note'}
            G[varch] = cur
        if s.get(f"{K}|last") != cur:                                       # config loaded / changed elsewhere
            _set(cur)
        c[1].caption("The model variant (CNN / LSTM / CNN + LSTM) is ignored while this is on.")
        arch = editor(F, Mo, seq, W, varch)
        if arch is not None:
            G[varch] = arch
            s[f"{K}|last"] = copy.deepcopy(arch)
    show_net(sel, arch if on else None, F, Mo, seq, W, names,
             outs if m != 'PI' else [f"x{k + 1}" for k in range(Mo)], pinn=m == 'PI')


def editor(F, M, seq, W, varch='archDL'):
    """Preset loader, layer table, head and pooling; returns the architecture or None if it is invalid."""
    s = st.session_state
    pre = _presets()
    with st.container(border=True):
        c = st.columns([3, 1, 4], vertical_alignment='bottom')
        p = c[0].selectbox("Start from a preset", list(pre), key=f"{K}|preset",
                           help="Standard architectures for thermal time series; ★: your library.")
        if c[1].button("Load", icon=':material/download:', key=f"{K}|load", width='stretch'):
            new = {k: v for k, v in copy.deepcopy(pre[p]).items() if k != 'note'}
            forms.cfg()['Run'][varch] = copy.deepcopy(new)
            _set(new)
            st.rerun()
        c[2].caption(pre[p].get('note', 'From your library (config/gui/dl_arch_lib.json).'))
        base = s.get(f"{K}|base") or {}
        ver = s.get(f"{K}|ver", 0)
        st.markdown("**Layers** (input → output; add rows at the bottom, delete with the row menu)")
        df = st.data_editor(_df(base), key=f"{K}|ed|{ver}", num_rows='dynamic', hide_index=True, width='stretch',
                            column_config={
            'type': st.column_config.SelectboxColumn('type', options=LAYER_TYPES, required=True,
                                                     help=' · '.join(f"{k}: {v}" for k, v in TYPE_HELP.items())),
            'units': st.column_config.NumberColumn('units', min_value=1, max_value=4096, step=1, required=True,
                                                   help="Channels (conv), state size (lstm, gru), model width "
                                                        "(transformer) or neurons (dense)."),
            'kernel': st.column_config.NumberColumn('kernel', min_value=1, max_value=512, step=1,
                                                    help="conv: filter length in samples."),
            'dilation': st.column_config.NumberColumn('dilation', min_value=1, max_value=1024, step=1,
                                                      help="conv: spacing of the filter taps (TCN: 1, 2, 4, 8 …)."),
            'heads': st.column_config.NumberColumn('heads', min_value=1, max_value=64, step=1,
                                                   help="transformer: attention heads (divides units)."),
            'bidir': st.column_config.CheckboxColumn('bidir', help="lstm/gru: read the window in both directions "
                                                                   "(doubles the output size)."),
            'norm': st.column_config.CheckboxColumn('norm', help="Normalisation after the layer (BatchNorm for "
                                                                 "conv, LayerNorm otherwise)."),
            'act': st.column_config.SelectboxColumn('act', options=ACTS, help="Activation after the layer "
                                                                              "(not used by transformer)."),
            'drop': st.column_config.NumberColumn('drop', min_value=0.0, max_value=0.9, step=0.05, format='%.2f',
                                                  help="Dropout rate during training.")})
        c = st.columns([2, 2, 2], vertical_alignment='bottom')
        htxt = c[0].text_input("Head: hidden dense units", ', '.join(str(u) for u in base.get('head') or []),
                               key=f"{K}|head|{ver}", help="Comma-separated neurons of the dense layers after the "
                               "body (ReLU); a linear layer to the outputs is always added. Empty: linear output.")
        pool = c[1].selectbox("Pooling (seq2point)", POOLS, index=POOLS.index(base.get('pool') or 'auto'),
                              key=f"{K}|pool|{ver}", disabled=seq == 0,
                              help="Reduction of the window to one vector: auto = last step after a recurrent "
                                   "layer, mean otherwise. seq2seq (s = 0) keeps every time step.")
        try:
            head = [int(float(v)) for v in htxt.replace(';', ',').split(',') if v.strip()]
        except ValueError:
            st.error("Head: comma-separated integers expected.", icon=':material/error:')
            return None
        arch = _arch(df, head, pool)
        if not arch['body']:
            st.warning("No layer: the network is a linear map per time step (head only).", icon=':material/warning:')
        with c[2].popover("Save to library", icon=':material/bookmark_add:', width='stretch'):
            nm = st.text_input("Name", "my network", key=f"{K}|libname")
            if st.button("Save", icon=':material/save:', key=f"{K}|libsave") and nm.strip():
                lib_save(nm.strip(), arch)
                st.toast(f"Architecture '{nm.strip()}' saved to config/gui/dl_arch_lib.json", icon=':material/check:')
    try:
        _net(1, F, M, seq, arch)
    except Exception as e:
        st.error(f"The architecture is not valid: {e}", icon=':material/error:')
        return None
    return arch


def show_net(sel, arch, F, M, seq, W, names, outs, pinn=False):
    """Summary, diagram and interpretation of the network."""
    Wd = min(W, W_SHOW)
    try:
        net = _net(sel, F, M, seq, arch)
        rows, yshape = summary(net, Wd, F)
    except Exception as e:
        st.warning(f"The network could not be built: {e}")
        return
    nP = sum(p.numel() for p in net.parameters())
    a = arch or {'body': [{'type': t, 'kernel': k, 'dilation': 1, 'units': 0} for t, k in
                          {1: [('conv', 5), ('conv', 3)], 2: [('lstm', 1)], 3: [('conv', 3), ('lstm', 1)]}[sel]]}
    rf, full = _receptive(a, W)
    cm.section(None, "Network")
    cm.tiles([("Parameters", f"{nP:,}", "Weights and biases that are trained."),
              ("Layers", f"{len(a['body'])} + head", "Body layers (rows of the table) and the dense head."),
              ("Receptive field", f"{rf:,} samples" + (" (window)" if full else ""),
               "Samples of the window that influence one output: Σ (k − 1)·d + 1 for convolutions, the whole window "
               "with a recurrent or attention layer."),
              ("Output", yshape, "B = batch; seq2point: one value per output, seq2seq: one per sample.")])
    c = st.columns([3, 2])
    with c[0]:
        try:
            st.graphviz_chart(arch_dot(net, Wd, F, names if F <= 6 else None, outs), width='content')
        except Exception as e:
            st.caption(f"Diagram not available: {e}")
    with c[1]:
        table(pd.DataFrame(rows), hide_index=True)
        if Wd < W:
            st.caption(f"Shapes for a window of {Wd} samples (W = {W} in training).")
    p = [f"The network has {nP:,} parameters. Compare this with the number of training windows (about the number of "
         f"training samples / stride): with far more parameters than windows the network can memorise the training "
         f"data — reduce the units or add dropout."]
    if not full and rf < W:
        p.append(f"Only the last {rf} of the {W} samples of a window influence an output (convolutions only). The "
                 f"network cannot see thermal history older than that: increase the kernels or dilations (TCN) or "
                 f"add a recurrent layer, or use rolled features (featRoll).")
    if any(l['type'] == 'transformer' for l in a['body']) and W > 1000:
        p.append(f"Self-attention compares all {W}² sample pairs of a window: training is slow and memory-hungry "
                 f"for W = {W}. Shorten the window or put strided convolutions in front.")
    if any(l.get('bidir') for l in a['body']) and seq > 0 and seq < W:
        p.append("A bidirectional layer uses the samples after the target sample s of the window — the model is "
                 "then not causal (it cannot run on-line).")
    if pinn:
        p.append("PINN: the last layer gives the states x of the physics network, scaled with the temperature scale "
                 "ΔT of the training data. For the whole-profile prediction (Physics › settings › pred = 1) a causal "
                 "network is best: LSTM/GRU (no bidirectional layers) carry the thermal state over the full profile; "
                 "convolutions only see their receptive field, attention layers fall back to window-wise "
                 "prediction.")
    cm.interpret(p + ["The presets are common choices for thermal sequences: CNN and TCN learn fast transients from "
                      "local filters, LSTM and GRU carry a state for the slow heating, the transformer attends to the "
                      "whole window. Train two or three of them with the same data and compare the test error in "
                      "*Results*; the Study page can tune their hyperparameters (Optuna)."])
