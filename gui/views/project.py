#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         project.py
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
Setup page: run settings (formerly start.py, adapter.RUN_FIELDS), input files and the parameters of the run section
shared by all modes.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import config as conf
from gui.core import dataset, files, forms

import os

import streamlit as st


#######################################################################################################################
# Widgets of the run settings
#######################################################################################################################
def _file(cfg, var, kind, label, help=None):
    opts = files.list_files(kind)
    cur = cfg[A.RUN_SECTION].get(var)
    if cur not in opts:
        opts = [cur] + opts
    cfg[A.RUN_SECTION][var] = st.selectbox(label, opts, index=opts.index(cur), key=f"g|{var}",
                                           help=help or f"Workbooks in the folder '{A.FOLDERS[kind]}/'.")


def run_field(cfg, var, kind, label, help=None):
    """Widget of one run setting, the value is written into the run section."""
    G = cfg[A.RUN_SECTION]
    if kind.startswith('expert:'):
        if not forms.expert():
            return
        kind = kind[7:]
    if kind == 'text':
        G[var] = st.text_input(label, value=str(G.get(var, '')), key=f"g|{var}", help=help)
    elif kind == 'toggle':
        v = st.toggle(label, value=bool(G.get(var)), key=f"g|{var}", help=help)
        G[var] = type(G.get(var, False))(v) if isinstance(G.get(var), (bool, int)) else v
    elif kind == 'int':
        G[var] = int(st.number_input(label, value=int(G.get(var, 0) or 0), step=1, key=f"g|{var}", help=help))
    elif kind.startswith('file:'):
        _file(cfg, var, kind[5:], label, help)
    elif kind.startswith('select:'):
        opts = kind[7:].split('|')
        cast = type(G.get(var)) if isinstance(G.get(var), (int, float)) else str
        opts = [cast(o) for o in opts]
        cur = G.get(var, opts[0])
        opts = opts if cur in opts else [cur] + opts
        G[var] = st.selectbox(label, opts, index=opts.index(cur), key=f"g|{var}", help=help)


def file_groups(cfg):
    """
    Input files grouped as in adapter.FILE_GROUPS: [(group dict, [RUN_FIELDS entries], active)]; one untitled group
    with all file settings if the adapter defines no groups.
    """
    fields = {f[0]: f for f in A.RUN_FIELDS if f[1].startswith('file:')}
    groups = getattr(A, 'FILE_GROUPS', None)
    if not groups:
        return [({'title': None}, list(fields.values()), True)] if fields else []
    out = []
    for g in groups:
        try:
            active = bool(g['active'](cfg))
        except Exception:
            active = True
        out.append((g, [fields[v] for v in g['fields'] if v in fields], active))
    return out


def active_files(cfg):
    """File settings used with the current configuration (all of them without FILE_GROUPS)."""
    return [f for _, fs, active in file_groups(cfg) if active for f in fs]


def _upload(cfg, fields, key):
    """Upload of a workbook into the folder of one file setting, which then uses it."""
    c1, c2 = st.columns([2, 3])
    with c1:
        var = st.selectbox("For", [f[0] for f in fields], key=f"up|for|{key}",
                           format_func=lambda v: next(f[2] for f in fields if f[0] == v))
    kind = next(f[1][5:] for f in fields if f[0] == var)
    with c2:
        up = st.file_uploader("Workbook", type=[e.lstrip('.') for e in A.FILE_EXT], key=f"up|{key}",
                              help=f"Stored in {A.FOLDERS[kind]}/ and selected for this setting.")
    if up is not None and st.session_state.get(f"upd|{key}") != up.file_id:
        st.session_state[f"upd|{key}"] = up.file_id
        cfg[A.RUN_SECTION][var] = files.save_upload(kind, up)
        st.session_state.pop(f"g|{var}", None)
        st.toast(f"{up.name} stored in {A.FOLDERS[kind]}/", icon=':material/upload:')
        st.rerun()


def _selector():
    """Parameter that selects the active file group (adapter.GROUP_SELECTOR), or None."""
    sel = getattr(A, 'GROUP_SELECTOR', None)
    try:
        return (sel[0], forms.find(*sel)) if sel else None
    except KeyError:
        return None


def files_form(cfg, upload=True):
    """Data form of TherMOS (format, dimension, split, files/sheets/IDs/ratios, channels)."""
    data_form(cfg)


def _lst(v):
    return [] if v is None else (list(v) if isinstance(v, (list, tuple)) else [v])


def _multi(G, var, label, opts, help):
    cur = [c for c in _lst(G.get(var)) if c not in opts]
    G[var] = st.multiselect(label, cur + opts, default=_lst(G.get(var)), key=f"g|{var}", help=help,
                            accept_new_options=True)


def _single(G, var, label, opts, help):
    cur = G.get(var)
    opts = opts if cur in opts else [cur] + opts
    G[var] = st.selectbox(label, opts, index=opts.index(cur), key=f"g|{var}", help=help, accept_new_options=True)


def _ints(G, var, label, help):
    txt = st.text_input(label, value=', '.join(str(int(v)) for v in _lst(G.get(var))), key=f"g|{var}", help=help)
    try:
        G[var] = [int(float(v)) for v in txt.replace(';', ',').split(',') if v.strip()]
    except ValueError:
        st.warning(f"{label}: comma separated integers expected")


def data_form(cfg):
    """Data settings of setup (formerly the 'Data' block of start.py)."""
    G = cfg[A.RUN_SECTION]
    c = st.columns(3)
    with c[0]:
        G['format'] = st.segmented_control("Format", ['mat', 'xlsx'], default=G.get('format', 'mat'), key="g|format",
                                           help="mat: MATLAB files (1D struct 'data' or 2D/3D fields), xlsx: one "
                                                "table per sheet.") or G.get('format', 'mat')
    with c[1]:
        dims = {1: '1D signals', 2: '2D field', 3: '3D field'}
        G['datDim'] = st.segmented_control("Dimension", list(dims), default=int(G.get('datDim', 1)), key="g|datDim",
                                           format_func=dims.get, help="1D: time series of losses and "
                                           "temperatures; 2D/3D: temperature fields on a grid (PO, PS).") or 1
    with c[2]:
        seps = {1: 'File', 2: 'Sheet', 3: 'ID', 4: 'Ratio'}
        G['datSep'] = st.segmented_control("Split by", list(seps), default=int(G.get('datSep', 1)), key="g|datSep",
                                           format_func=seps.get, help="How training, validation and testing data "
                                           "are separated: separate files, sheets of one workbook, profile IDs "
                                           "(column 'id') or a ratio of one file.") or 1
    fmt, sep = G['format'], int(G['datSep'])
    names = dataset.files(fmt)
    hf = f"Data sets in data/ (.{fmt}), searched recursively."

    # Files
    st.markdown("**Files**")
    c = st.columns(3)
    with c[0]:
        _multi(G, 'trFile', "Training", names, hf)
    with c[1]:
        _multi(G, 'vlFile', "Validation", names, hf)
    with c[2]:
        _single(G, 'teFile', "Testing", names, hf)
    inf = dataset.info(G['teFile'], fmt)
    if inf['kind'] in ('missing', 'error'):
        st.warning(f"Testing data set '{G['teFile']}.{fmt}' not found under data/ — the run will stop.")
    elif inf['kind'] != '1D' and int(G['datDim']) == 1:
        st.info(f"'{G['teFile']}' is a {inf['kind']} field: set Dimension to {inf['kind']} for PO/PS.")
    elif inf['kind'] == '1D' and int(G['datDim']) > 1:
        st.warning(f"'{G['teFile']}' contains 1D signals, but Dimension is {G['datDim']}D.")

    # Sheets, IDs, ratio
    if sep == 2:
        st.markdown("**Sheets**")
        sh = inf.get('sheets', [])
        c = st.columns(3)
        with c[0]:
            _multi(G, 'trSheet', "Training", sh, "Sheets of the training files.")
        with c[1]:
            _multi(G, 'vlSheet', "Validation", sh, "Sheets of the validation files.")
        with c[2]:
            _single(G, 'teSheet', "Testing", sh, "Sheet of the testing file.")
    elif sep == 3:
        st.markdown("**Profile IDs**")
        ids = inf.get('ids', [])
        if ids:
            st.caption(f"IDs in '{G['teFile']}': {', '.join(map(str, ids[:60]))}{' …' if len(ids) > 60 else ''}")
        c = st.columns(3)
        with c[0]:
            _ints(G, 'trID', "Training", "Comma separated; empty = all IDs not used for validation/testing.")
        with c[1]:
            _ints(G, 'vlID', "Validation", "Comma separated profile IDs.")
        with c[2]:
            _ints(G, 'teID', "Testing", "Comma separated profile IDs.")
    elif sep == 4:
        c = st.columns(2)
        with c[0]:
            G['rTr'] = st.slider("Training share", 0.1, 0.95, float(G.get('rTr', 0.7)), 0.05, key="g|rTr",
                                 help="Share of the samples used for training, the rest is testing.")
        with c[1]:
            G['rVl'] = st.slider("Validation share (of training)", 0.0, 0.5, float(G.get('rVl', 0.2)), 0.05,
                                 key="g|rVl", help="Taken from the training share.")

    # Channels
    if int(G['datDim']) == 1:
        st.markdown("**Channels**")
        cols = [x for x in inf.get('cols', []) if x not in ('time', 'time_id', 'id')]
        c = st.columns(3)
        with c[0]:
            _multi(G, 'inp', "Inputs (losses)", cols, "Input channels, usually power losses (W).")
        with c[1]:
            _multi(G, 'out', "Outputs (temperatures)", cols, "Temperatures to be predicted (°C).")
        with c[2]:
            _multi(G, 'ref', "References", cols, "Reference temperatures (e.g. coolant), one per output, used by "
                                                 "'Initialise Zero = 3'.")
        miss = [x for x in _lst(G['inp']) + _lst(G['out']) if cols and x not in cols]
        if miss:
            st.warning(f"Not in '{G['teFile']}': {', '.join(map(str, miss))}")


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cfg = forms.cfg()
    if A.LOGO and os.path.isfile(os.path.join(A.ROOT, A.LOGO)):
        st.image(os.path.join(A.ROOT, A.LOGO), width=420)
    else:
        st.title(A.NAME)
    st.markdown(f"{A.TITLE} — {A.TAGLINE}")

    with st.container(border=True):
        st.markdown("#### :material/rocket_launch: Start from a ready-made setup")
        c1, c2 = st.columns([3, 1], vertical_alignment='bottom')
        names = [n for n in conf.list_configs() if n != 'default']
        with c1:
            pick = st.selectbox("Setup script", names, key='sb|script', help="Configs generated from start*.py and "
                                "setup/**/*.py (run settings + config workbook). Loading replaces the current config.")
        with c2:
            if st.button("Load", icon=':material/download:', width='stretch', key='sb|scriptload'):
                forms.set_config(conf.read_config(pick), pick)
                st.toast(f"Setup '{pick}' loaded", icon=':material/check:')
                st.rerun()

    with st.container(border=True):
        st.markdown("#### :material/badge: Project")
        other = [f for f in A.RUN_FIELDS if not f[1].startswith('file:')]
        cols = st.columns(4)
        for i, (var, kind, label, hlp) in enumerate(other):
            with cols[i % 4]:
                run_field(cfg, var, kind, label, hlp)

    with st.container(border=True):
        st.markdown("#### :material/database: Data")
        data_form(cfg)

    if any(p['var'] not in A.RUN for p in forms.schema().get(A.RUN_SECTION, [])):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"#### :material/functions: {A.RUN_SECTION} parameters")
            with c2:
                forms.reset_section(A.RUN_SECTION)
            forms.section(A.RUN_SECTION, cols=4)

    st.markdown("#### Operating modes")
    cols = st.columns(min(len(A.MODES), 3))
    for i, (m, d) in enumerate(A.MODES.items()):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{d['icon']} {d['title']}**")
                st.caption(d['text'])
