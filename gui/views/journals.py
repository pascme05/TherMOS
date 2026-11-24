#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         journals.py
# Date:         22.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Journal pages (Journals › TherMOS journal, Journals › PINN journal): every result of the papers can be loaded and
reproduced from here (gui/core/journals.py).
TherMOS journal: one card per setup of setup/journal; 'Load' sets its GUI config and opens the operating mode.
PINN journal: overview of all setups (results of the paper, runtime, results present), then per setup its
description and results, its settings (editable, CHANGE marks the lines to adapt), the run in the background with
its log, and the method configs config/journal_pinn/*.json.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import config as conf
from gui.core import forms
from gui.core import journals as J
from gui.core.style import table
from gui.views.registry import PAGES

import os
import pandas as pd
import streamlit as st


#######################################################################################################################
# TherMOS journal
#######################################################################################################################
def _load_thermos(t):
    forms.set_config(conf.read_config(t['id']), t['id'])
    st.session_state[f"goto|{t['mode']}"] = 'Parameters'
    st.session_state['journal'] = t['id']


def thermos_page():
    st.title("TherMOS journal")
    st.caption("Setups of the TherMOS journal paper (setup/journal). *Load* sets the configuration of the setup "
               "(config/gui/<setup>.json) and opens its operating mode; press *Run* there to reproduce the result. "
               "Your current configuration is replaced, save it first in the sidebar if needed. On the command line: "
               "python setup/journal/<setup>.py.")
    lst = [t for t in J.THERMOS if os.path.isfile(os.path.join(conf.CONFIG_DIR, t['id'] + '.json'))]
    if not lst:
        st.info("No configs of the journal setups found in config/gui.")
        return
    table(pd.DataFrame([{'section': t['sec'], 'setup': t['id'], 'case': t['title'],
                         'mode': f"{t['mode']} · {A.MODES[t['mode']]['title']}"} for t in lst]),
          hide_index=True, width='stretch')
    c1, c2 = st.columns([2, 5])
    with c1:
        ids = [t['id'] for t in lst]
        cur = st.session_state.get('journal')
        sel = st.radio("Setup", ids, index=ids.index(cur) if cur in ids else 0, key='jt|sel',
                       format_func=lambda i: next(f"{t['sec']} · {t['title']}" for t in lst if t['id'] == i),
                       label_visibility='collapsed')
    t = next(x for x in lst if x['id'] == sel)
    with c2:
        with st.container(border=True):
            st.markdown(f"### {t['sec']} · {t['title']}")
            st.markdown(f":gray-badge[{t['mode']} · {A.MODES[t['mode']]['title']}] :gray-badge[setup/journal/"
                        f"{t['id']}.py]")
            st.markdown(t['text'])
            b1, b2, _ = st.columns([2, 2, 3])
            with b1:
                if st.button("Load setup", type='primary', icon=':material/download:', key='jt|load',
                             width='stretch'):
                    _load_thermos(t)
                    st.toast(f"Setup '{t['id']}' loaded", icon=':material/check:')
                    if t['mode'] in PAGES:
                        st.switch_page(PAGES[t['mode']])
            with b2:
                if t['mode'] in PAGES:
                    st.page_link(PAGES[t['mode']], label="Open the mode", icon=':material/arrow_forward:')
            if st.session_state.get('cfg_name') == t['id']:
                st.caption(":material/check_circle: This setup is the active configuration.")
        R = conf.read_config(t['id'])[A.RUN_SECTION]
        keys = ['name', 'config', 'datDim', 'trFile', 'teFile', 'vlFile', 'inp', 'out', 'legacy']
        st.markdown("**Settings of the setup**")
        table(pd.DataFrame([{'setting': k, 'value': str(R.get(k))} for k in keys if k in R]), hide_index=True,
              width='stretch')


#######################################################################################################################
# PINN journal: helpers
#######################################################################################################################
def _txt(v):
    return repr(v)


def _rows(sid):
    return J.parse_setup(J.script(sid))[0]


def _changes(sid):
    """Changed setup values of a setup (session state, relative to the script)."""
    return st.session_state.setdefault('jp|chg', {}).setdefault(sid, {})


def _setup(sid):
    s = {r['key']: r['value'] for r in _rows(sid)}
    s.update(_changes(sid))
    return s


def _overview():
    rows = []
    for i, p in enumerate(J.PINN, start=1):
        if not os.path.isfile(J.script(p['id'])):
            continue
        s = _setup(p['id'])
        stt = J.status(s)
        _, state = J.read_log(J.log_file(s), 1)
        rows.append({'#': i, 'setup': p['id'], 'results of the paper': p['res'], 'content': p['what'],
                     'runtime': p['time'], 'results present': f"{sum(e for _, e in stt)} / {len(stt)}" if stt else '–',
                     'last run': {'none': '–', 'running': 'running', 'done': 'done', 'error': 'error'}[state]})
    table(pd.DataFrame(rows), hide_index=True, width='stretch')


def _results(sid, s):
    stt = J.status(s)
    if not stt:
        st.caption("This setup writes no run results (figure data and PDFs only).")
    else:
        rows = []
        for f, ok in stt:
            r = {'file': f, 'present': '✓' if ok else 'missing'}
            if ok and f.endswith('.json'):
                r.update({k: round(v, 3) if k != 'train (s)' else round(v) for k, v in
                          J.summary(f, s.get('out', 'results/pfhb_v2')).items()})
            rows.append(r)
        df = pd.DataFrame(rows)
        table(df, hide_index=True, width='stretch')
        st.caption(f"Folder: {s.get('out')} (data: data/board/pfhb). Columns N1 … T5: hot-spot RMSE (K) on the "
                   "testing sets (Table V of the paper); train: training time (s). Missing runs are computed by "
                   "*Run*, present ones are skipped (setup['skip'] = 1).")
    if s.get('figs') or s.get('build'):
        d = os.path.join(J.ROOT, 'docu', 'paper', 'figures')
        pdf = sorted(f for f in os.listdir(d) if f.endswith('.pdf')) if os.path.isdir(d) else []
        if not pdf:
            st.info("No figure PDFs in docu/paper/figures (the paper folder is not part of the repository).")
        else:
            st.markdown("**Figure PDFs** (docu/paper/figures)")
            c = st.columns(4)
            for i, f in enumerate(pdf):
                with open(os.path.join(d, f), 'rb') as fh:
                    c[i % 4].download_button(f, fh.read(), file_name=f, mime='application/pdf', key=f"jp|pdf|{f}",
                                             icon=':material/picture_as_pdf:', width='stretch')


def _settings(sid):
    rows = _rows(sid)
    chg = _changes(sid)
    df = pd.DataFrame([{'group': r['group'], 'parameter': r['key'],
                        'value': _txt(chg.get(r['key'], r['value'])), 'CHANGE': r['change'],
                        'description': r['comment'].replace('CHANGE: ', '')} for r in rows])
    st.caption("Values as Python literals (text in quotes, lists in brackets). *CHANGE* marks the lines to adapt: "
               "name and results folder of a test, epoch (e.g. 2 for a quick check of the whole chain), meshes. "
               "The changes are used by *Run* in this session; *Save to setup file* writes them into "
               f"{J.PINN_DIR}/{sid}.py (then also used on the command line).")
    ed = st.data_editor(df, hide_index=True, width='stretch', key=f"jp|ed|{sid}", disabled=['group', 'parameter',
                        'CHANGE', 'description'],
                        column_config={'CHANGE': st.column_config.CheckboxColumn(width='small'),
                                       'value': st.column_config.TextColumn(width='medium')})
    new, err = {}, []
    for r, v in zip(rows, ed['value']):
        try:
            x = J.parse_value(v, r['value'])
        except ValueError as e:
            err.append(f"{r['key']}: {e}")
            continue
        if x != r['value']:
            new[r['key']] = x
    if err:
        st.error("; ".join(err))
    st.session_state['jp|chg'][sid] = new
    c1, c2, _ = st.columns([2, 2, 3])
    if new:
        st.caption("Changed: " + ", ".join(f"{k} = {v!r}" for k, v in new.items()))
    if c1.button("Save to setup file", icon=':material/save:', key=f"jp|save|{sid}", disabled=not new or bool(err),
                 width='stretch'):
        J.write_setup(J.script(sid), new)
        st.session_state['jp|chg'][sid] = {}
        st.session_state.pop(f"jp|ed|{sid}", None)
        st.toast(f"Saved to {J.PINN_DIR}/{sid}.py", icon=':material/check:')
        st.rerun()
    if c2.button("Discard changes", icon=':material/undo:', key=f"jp|undo|{sid}", disabled=not new,
                 width='stretch'):
        st.session_state['jp|chg'][sid] = {}
        st.session_state.pop(f"jp|ed|{sid}", None)
        st.rerun()


def _run(sid, s):
    path = J.script(sid)
    chg = _changes(sid)
    procs = st.session_state.setdefault('jp|proc', {})
    p = procs.get(sid)
    alive = p is not None and p.poll() is None
    st.markdown("**Command line** (in the TherMOS folder; the values of the setup file)")
    st.code(f"python {J.PINN_DIR}/{sid}.py", language='bash')
    if chg:
        st.caption("With the changes of *Settings* (not saved to the file): "
                   f"python -m gui.core.journals {J.PINN_DIR}/{sid}.py '{__import__('json').dumps(chg)}'")
    st.warning(f"Runtime of this setup: {next(x['time'] for x in J.PINN if x['id'] == sid)} (RTX 3070). The run "
               "continues in the background when the page is left; existing results are skipped, so a stopped run "
               "restarts where it ended.", icon=':material/schedule:')
    c1, c2, c3, _ = st.columns([2, 2, 2, 2])
    if c1.button("Run in background", type='primary', icon=':material/play_arrow:', key=f"jp|run|{sid}",
                 disabled=alive, width='stretch'):
        procs[sid], _ = J.start(path, chg)
        st.toast(f"Started {sid}", icon=':material/play_arrow:')
        st.rerun()
    if c2.button("Stop", icon=':material/stop:', key=f"jp|stop|{sid}", disabled=not alive, width='stretch'):
        p.terminate()
        st.toast(f"Stopped {sid}", icon=':material/stop:')
    c3.button("Refresh log", icon=':material/refresh:', key=f"jp|ref|{sid}", width='stretch')
    log = J.log_file(s)
    txt, state = J.read_log(log)
    if alive:
        state = 'running'
    elif p is not None and state == 'running':
        state = 'error' if p.returncode else 'stopped'
    lab = {'none': ('gray', 'no run yet'), 'running': ('blue', 'running'), 'done': ('green', 'done'),
           'error': ('red', 'error'), 'stopped': ('orange', 'stopped')}[state]
    st.markdown(f"**Log** :{lab[0]}-badge[{lab[1]}] · {os.path.relpath(log, J.ROOT)}")
    if txt:
        st.code(txt, language=None, height=360)


def _configs():
    names = J.config_names()
    st.caption("Parameters of the data sets and methods, used by all setups (config/journal_pinn/<name>.json, "
               "sections as in config/gui/start.json). *CHANGE* marks the parameters that depend on the computer "
               "(GPU memory, runtime); all other values are those of the paper. Mesh.<mesh>.x replaces x on one mesh.")
    name = st.segmented_control("Config", names, default='pinn' if 'pinn' in names else names[0], key='jp|cfg')
    if not name:
        return
    cfg = J.read_config(name)
    info = cfg.get('Info', {})
    st.markdown(f"**{info.get('title', name)}** — {info.get('desc', '')}")
    rows = J.config_rows(cfg)
    df = pd.DataFrame([{'parameter': r['parameter'], 'value': _txt(r['value']), 'CHANGE': r['change'],
                        'description': r['comment']} for r in rows])
    ed = st.data_editor(df, hide_index=True, width='stretch', key=f"jp|ced|{name}",
                        disabled=['parameter', 'CHANGE', 'description'],
                        column_config={'CHANGE': st.column_config.CheckboxColumn(width='small'),
                                       'value': st.column_config.TextColumn(width='small')})
    new, err = {}, []
    for r, v in zip(rows, ed['value']):
        try:
            x = J.parse_value(v, r['value'])
        except ValueError as e:
            err.append(f"{r['parameter']}: {e}")
            continue
        if x != r['value']:
            new[r['parameter']] = x
    if err:
        st.error("; ".join(err))
    if new:
        st.caption("Changed: " + ", ".join(f"{k} = {v!r}" for k, v in new.items()))
    if st.button(f"Save {name}.json", icon=':material/save:', key=f"jp|csave|{name}", disabled=not new or bool(err)):
        J.write_config(name, J.set_values(cfg, new))
        st.session_state.pop(f"jp|ced|{name}", None)
        st.toast(f"Saved to {J.PINN_CFG}/{name}.json", icon=':material/check:')
        st.rerun()


#######################################################################################################################
# PINN journal
#######################################################################################################################
def pinn_page():
    st.title("PINN journal")
    st.caption("Setups of the PINN journal paper (half-bridge module, setup/journal_pinn), one per group of figures "
               "and tables, in run order. Select a setup: *Description & results* shows what it reproduces and which "
               "results exist, *Settings* the values to adapt (CHANGE), *Run* starts it in the background, *Method "
               "configs* the parameters of the data sets and methods.")
    lst = [p for p in J.PINN if os.path.isfile(J.script(p['id']))]
    if not lst:
        st.info(f"No setups found in {J.PINN_DIR}.")
        return
    _overview()
    ids = [p['id'] for p in lst]
    sid = st.selectbox("Setup", ids, key='jp|sel',
                       format_func=lambda i: next(f"{ids.index(i) + 1} · {p['res']} — {p['what']}" for p in lst
                                                  if p['id'] == i))
    s = _setup(sid)
    _, doc = J.parse_setup(J.script(sid))
    t1, t2, t3, t4 = st.tabs(["Description & results", "Settings", "Run", "Method configs"])
    with t1:
        st.markdown(doc.replace('\n', ' '))
        _results(sid, s)
    with t2:
        _settings(sid)
    with t3:
        _run(sid, _setup(sid))
    with t4:
        _configs()
