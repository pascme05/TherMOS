#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         mode.py
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
Page of one operating mode with the views workflow (if defined), parameters, inputs, run and results.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import files, forms, theory, workflow
from gui.core import viz_inputs as vi
from gui.core import viz_results as vr
from gui.views.project import files_form, file_groups, active_files

import os

import streamlit as st

VIEW_ICONS = {'Workflow': ':material/account_tree:', 'Parameters': ':material/tune:',
              'Inputs': ':material/table_chart:',
              'Theory': ':material/function:', 'Network': ':material/hub:', 'Physics': ':material/device_thermostat:',
              'Run': ':material/play_arrow:', 'Results': ':material/monitoring:'}


def views(m):
    return (['Workflow'] if m in workflow.WORKFLOWS else []) + ['Parameters', 'Inputs'] + \
        (['Theory'] if m in theory.THEORY else []) + (['Network'] if m in ('DL', 'PI') else []) + \
        (['Physics'] if m == 'PI' else []) + ['Run', 'Results']


#######################################################################################################################
# Theory
#######################################################################################################################
def theory_view(m):
    theory.THEORY[m](forms.cfg())


def physics_view(m):
    from gui.core import pinn_phys
    pinn_phys.view(m)


def network_view(m):
    from gui.core import dl_arch
    dl_arch.view(m)


#######################################################################################################################
# Parameters
#######################################################################################################################
def variant(m):
    """Model variant of the mode (run setting sel<XX>), e.g. the ML regressor or the DL network."""
    d, G = A.MODES[m], forms.cfg()[A.RUN_SECTION]
    if not d.get('pick'):
        return
    var, opts = d['pick']
    cur = int(G.get(var) or list(opts)[0])
    cur = cur if cur in opts else list(opts)[0]
    G[var] = st.segmented_control("Model variant", list(opts), default=cur, format_func=opts.get, key=f"g|{var}",
                                  help=f"setup['{var}'] of the start script.") or cur


def load_example(m):
    """Button loading the tutorial / default setup script of the mode."""
    script = A.MODES[m].get('script')
    name = os.path.splitext(os.path.basename(script or ''))[0]
    from gui.core import config as conf
    if name not in conf.list_configs():
        return
    if st.button(f"Load example ({name})", icon=':material/school:', key=f"ex|{m}", type='tertiary',
                 help=f"Replaces the configuration with the settings of {script} (data, run settings, parameters)."):
        forms.set_config(conf.read_config(name), name)
        st.toast(f"Example '{name}' loaded", icon=':material/check:')
        st.rerun()


def parameters(m):
    d = A.MODES[m]
    c1, c2, c3 = st.columns([4, 2, 1], vertical_alignment='bottom')
    with c1:
        variant(m)
    with c2:
        load_example(m)
    with c3:
        forms.reset_section(d['section'])
    if m == 'PI':
        st.info("The PINN settings — physics, loss weights α, β, γ, warm-up, initial state, prediction, collocation, "
                "reference temperature, constants and loss sources — are in the view **Physics**. Here: the "
                "general settings and the training of the network (dl: window W, stride, epochs, learning rate and "
                "its schedule, batch, early stopping).", icon=':material/device_thermostat:')
    forms.section(d['section'], cats=d.get('cats'))
    for sec in d.get('extra', []):
        with st.expander(f"**{sec}** · {SECTION_TITLES.get(sec, '')}", expanded=False):
            forms.section(sec)


SECTION_TITLES = {'Exp': 'experiment: resampling, initial values, padding',
                  'Dat': 'data: dimension reduction, probe positions',
                  'Par': 'pre-processing: interpolation, normalisation, loss correction',
                  'Mdl': 'model'}


#######################################################################################################################
# Inputs
#######################################################################################################################
def inputs(m):
    cfg = forms.cfg()
    with st.expander("Input files", expanded=False):
        files_form(cfg, upload=False)
    if m in vi.INPUTS:
        vi.INPUTS[m](cfg)
        return
    fields = active_files(cfg)
    if not fields:
        st.info("This mode reads no input workbooks.")
        return
    title = next((g['title'] for g, _, a in file_groups(cfg) if a and g['title']), None)
    if title:
        st.caption(f"Input files of the **{title}** (selected by the topology).")
    for tab, (var, kind, label, _) in zip(st.tabs([f"{f[2]} · {cfg[A.RUN_SECTION].get(f[0])}" for f in fields]),
                                          fields):
        with tab:
            vi.workbook(kind[5:], cfg[A.RUN_SECTION].get(var), key=f"m{m}|{var}")


#######################################################################################################################
# Run
#######################################################################################################################
def _summary(m):
    cfg = forms.cfg()
    G = cfg[A.RUN_SECTION]
    d = A.MODES[m]
    seps = {1: 'files', 2: 'sheets', 3: 'IDs', 4: 'ratio'}
    rows = [('Simulation', f"{G.get('name', '')}  →  results/{G.get('name', '')}/")]
    if d.get('pick'):
        var, opts = d['pick']
        rows.append(('Model variant', opts.get(int(G.get(var) or 1), str(G.get(var)))))
    rows.append(('Data', f"{G.get('datDim')}D · .{G.get('format')} · split by {seps.get(int(G.get('datSep', 1)))}"))
    rows.append(('Training', ', '.join(map(str, G.get('trFile') or []))))
    rows.append(('Testing', str(G.get('teFile'))))
    if int(G.get('datDim', 1)) == 1:
        rows.append(('Inputs → outputs', f"{', '.join(map(str, G.get('inp') or []))}  →  "
                                         f"{', '.join(map(str, G.get('out') or []))}"))
    rows.append(('Train / test', f"{'train' if G.get('train') else 'load model'} · "
                                 f"{'test' if G.get('test') else 'no test'} · "
                                 f"{'MATLAB legacy' if int(G.get('legacy', 0)) else 'corrected numerics'}"))
    return rows


def _checks(m):
    """Warnings before a run: data dimension vs. model, missing model file, long runs."""
    from gui.core import dataset
    G = forms.cfg()[A.RUN_SECTION]
    out = []
    dim = int(G.get('datDim', 1))
    dOut = int(forms.value('Dat', 'gen', 'dOut') or 1)
    if not (dOut == dim or (dim == 3 and dOut == 2) or (dim > 1 and dOut == 1)):
        out.append(('error', f"Data reduction {dim}D → {dOut}D is not possible: set Parameters › Dat › Data "
                             f"Reduction to {dim}D (or 1D for a probe of a field)."))
    if m in ('PO', 'PS') and (dim == 1 or dOut == 1):
        out.append(('error', "PO/PS need 2D or 3D field data (Setup › Data › Dimension) and Data Reduction ≥ 2D."))
    if m not in ('PO', 'PS') and dim > 1 and dOut == 1:
        out.append(('info', f"The {dim}D field is reduced to the probe (inpX, inpY) → (outX, outY) of the Dat "
                            f"parameters for this 1D model."))
    if m not in ('PO', 'PS') and dOut > 1:
        out.append(('error', f"The {A.MODES[m]['title']} model works on 1D signals; set Data Reduction to 1D."))
    if m == 'PS' and dim == 3 and int(G.get('selPS') or 1) != 2:
        out.append(('warning', "3D data with the 2D variant: choose the 3D variant under Parameters."))
    names = [G.get('teFile')] + list(G.get('trFile') or [])
    miss = [n for n in names if dataset.info(n, G.get('format', 'mat'))['kind'] in ('missing', 'error')]
    if miss:
        out.append(('error', f"Data set(s) not found under data/: {', '.join(map(str, miss))}"))
    if not G.get('train'):
        tag = m.lower()
        f = [os.path.join(A.ROOT, 'mdl', f"mdl_{tag}_{G.get('name')}{e}") for e in ('.pkl', '.pt')]
        if not any(os.path.isfile(x) for x in f):
            out.append(('warning', f"Training is off, but no stored model mdl/mdl_{tag}_{G.get('name')}.pkl/.pt "
                                   f"was found; train the model first."))
    if m in ('DL', 'PI') and G.get('train'):
        out.append(('info', "Deep learning on the CPU: a training run can take from minutes to hours "
                            "(epochs, window length, data size)."))
    return out


def run_view(m):
    cfg = forms.cfg()
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.markdown("**Run settings**")
            for k, v in _summary(m):
                st.markdown(f"<span style='color:#52514e'>{k}</span> &nbsp; {v}", unsafe_allow_html=True)
    with c2:
        checks = _checks(m)
        for kind, txt in checks:
            getattr(st, kind)(txt)
        start = st.button(f"Run {A.MODES[m]['title']}", type='primary', icon=':material/play_arrow:',
                          width='stretch', key=f"run|{m}", disabled=any(k == 'error' for k, _ in checks))
        st.caption("The simulation runs locally in this process; the console output is streamed below.")

    if start:
        from gui.core.runner import run_mode
        with st.status(f"Running {A.MODES[m]['title']} …", expanded=True) as status:
            box = st.empty()
            run = run_mode(cfg, m, forms.schema(), stream=lambda txt: box.code(txt[-8000:], language=None,
                                                                                height=380))
            if run['error']:
                status.update(label=f"Stopped with an error after {run['duration']:.1f} s", state='error')
            else:
                status.update(label=f"Finished in {run['duration']:.1f} s", state='complete', expanded=False)
        st.session_state.runs[m] = run
        try:
            files.save_run(run)
        except Exception as e:
            st.warning(f"The run could not be stored for the results browser: {e}")
        if not run['error']:
            st.session_state[f"goto|{m}"] = 'Results'
            st.rerun()
        else:
            st.error(run['error'])
            with st.expander("Console log"):
                st.code(run['log'][-20000:], language=None)


#######################################################################################################################
# Results
#######################################################################################################################
def results_view(m):
    cfg = forms.cfg()
    root = A.results_root(cfg)
    past = []
    if os.path.isdir(root):
        for f in files.list_runs('', root=root):
            meta = files.run_meta(f)
            if meta and meta.get('mode') == m:
                past.append((f, meta))
    opts = (['session'] if m in st.session_state.runs else []) + [f for f, _ in past]
    if not opts:
        vr.run_view(None)
        return
    lab = {f: f"{os.path.basename(f)} · {meta.get('time', '')}" for f, meta in past}
    lab['session'] = "Latest run of this session"
    sel = st.selectbox("Results", opts, format_func=lambda o: lab[o], key=f"res|pick|{m}",
                       help=f"GUI runs of this mode in {os.path.relpath(root, A.ROOT)}/")
    run = st.session_state.runs[m] if sel == 'session' else files.load_run(sel)
    vr.run_view(run, key=f"res|{m}")


#######################################################################################################################
# Page
#######################################################################################################################
def make_page(m):
    def page():
        d = A.MODES[m]
        if A.MODE_VAR:                                                  # activity rules and saved configs see the mode
            forms.cfg()[A.RUN_SECTION][A.MODE_VAR] = m
        st.title(f"{m} · {d['title']}" if len(A.MODES) > 1 else d['title'])
        st.caption(d['text'])
        vs = views(m)
        key = f"view|{m}"
        if f"goto|{m}" in st.session_state:
            st.session_state[key] = st.session_state.pop(f"goto|{m}")
        if st.session_state.get(key) not in vs:
            st.session_state[key] = vs[0]
        view = st.segmented_control("View", vs, key=key, label_visibility='collapsed',
                                    format_func=lambda v: f"{VIEW_ICONS[v]} {v}") or vs[0]
        st.divider()
        {'Workflow': workflow.view, 'Parameters': parameters, 'Inputs': inputs, 'Theory': theory_view,
         'Network': network_view, 'Physics': physics_view, 'Run': run_view, 'Results': results_view}[view](m)

    page.__name__ = f"mode{m}"
    return page
