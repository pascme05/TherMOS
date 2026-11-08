#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         tutorials.py
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
Tutorials page: guided scenarios for the operating modes (gui/core/tutorials.py). A tutorial loads its own config,
then the steps lead through the views of the mode page.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms
from gui.core import tutorials as tut
from gui.core.style import table
from gui.views.registry import PAGES

import pandas as pd
import streamlit as st

LEVEL_COL = {'Beginner': 'green', 'Intermediate': 'blue', 'Advanced': 'orange'}


#######################################################################################################################
# Helper
#######################################################################################################################
def _mode(m):
    return f"{m} · {A.MODES[m]['title']}" if m in A.MODES else str(m)


RUN_LABELS = {'physPI': 'physics structure (view Physics)', 'selPI': 'network variant (1 CNN, 2 LSTM, 3 CNN + LSTM)'}


def _value(k, v):
    """Short text of a setting (the physics structure of the PI mode as a summary)."""
    if isinstance(v, dict) and v.get('type') == 'foster':
        return "Foster: " + "; ".join(f"T{c['out']} ← P{c['src']}: R = {', '.join(f'{r:.4g}' for r in c['R'])} K/W, "
                                      f"τ = {', '.join(f'{x:.4g}' for x in c['tau'])} s" for c in v['foster']) + \
            f" · sources {', '.join(v.get('src') or [])}"
    if isinstance(v, dict) and v.get('type') == 'cauer':
        return f"Cauer network, {len(v['cauer']['C'])} nodes · sources {', '.join(v.get('src') or [])}"
    if k.endswith('physPI') and v is None:
        return "none (default single Foster term)"
    return v


def _changes(t, changes=None, head=True):
    """Readable table of the base config and the changes of a tutorial (or of one of its runs)."""
    rows = [{'setting': 'base configuration', 'value': f"config/{t.get('base', 'default')}.json"}] if head else []
    runs = {var: label for var, _, label, _ in A.RUN_FIELDS}
    for k, v in (t['changes'] if changes is None else changes).items():
        v = _value(k, v)
        label, (sec, *rest) = k, k.split('.')
        if sec == A.RUN_SECTION and len(rest) == 1:
            label = f"Run settings › {runs.get(rest[0], RUN_LABELS.get(rest[0], rest[0]))}"
        elif len(rest) == 2:
            try:
                p = forms.find(sec, *rest)
                label = f"{sec} › {p['cat']} › {p['name']}"
                if p.get('options') and v in p['options']:
                    v = f"{v}: {p['options'][v]}"
                elif p.get('unit'):
                    v = f"{v} {p['unit']}"
            except KeyError:
                pass
        rows.append({'setting': label, 'value': '(empty)' if v is None else str(v)})
    return pd.DataFrame(rows, columns=['setting', 'value'])


def runs(t):
    """Runs of the table 'What you should see': their changes and a button that loads each one."""
    st.markdown("**Runs of the table *What you should see*** — changes relative to the tutorial configuration; "
                "load a run and press *Run* to reproduce its row.")
    rows = []
    for r in t['runs']:
        ch = _changes(t, r['changes'], head=False)
        txt = '; '.join(f"{a} = {b}" for a, b in zip(ch['setting'], ch['value']))
        rows.append({'run': r['label'], 'changes': txt or 'none (the tutorial configuration)'})
    table(pd.DataFrame(rows), hide_index=True, width='stretch')
    c = st.columns([3, 1], vertical_alignment='bottom')
    k = c[0].selectbox("Run", range(len(t['runs'])), format_func=lambda i: t['runs'][i]['label'],
                       key=f"tut|run|{t['id']}")
    if c[1].button("Load this run", icon=':material/download:', key=f"tut|runl|{t['id']}", width='stretch'):
        forms.set_config(tut.run_config(t['id'], k), f"tutorials/{t['id']}")
        if t['mode'] in A.MODES:
            st.session_state[f"goto|{t['mode']}"] = 'Run'
        st.session_state['tutorial'] = t['id']
        st.toast(f"Run '{t['runs'][k]['label']}' loaded", icon=':material/check:')
        if _target(t) in PAGES:
            st.switch_page(PAGES[_target(t)])


def _target(t):
    """Page the tutorial opens: its own page ('study', a tool) or the mode page."""
    return t.get('page', t['mode'])


def _title(t):
    p = _target(t)
    if p == 'study':
        return 'Study'
    if p in A.MODES:
        return A.MODES[p]['title']
    return getattr(A, 'TOOLS', {}).get(p, (None, p))[1]


def _load(t):
    forms.set_config(tut.load(t['id']), f"tutorials/{t['id']}")
    if t['mode'] in A.MODES:
        st.session_state[f"goto|{t['mode']}"] = 'Theory' if 'Theory' in _views(t['mode']) else 'Parameters'
    for k, v in (t.get('preset') or {}).items():                    # widget values of the study / tool page
        st.session_state[k] = v
    st.session_state['tutorial'] = t['id']


def _views(m):
    from gui.views.mode import views
    return views(m)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Tutorials")
    if not tut.TUTORIALS:
        st.info("No tutorials defined yet.")
        return
    st.caption("Guided scenarios for the operating modes. *Load* sets the configuration of the tutorial — your current "
               "configuration is replaced, save it first in the sidebar if needed. The values under *What you should "
               "see* come from runs of the tutorial configurations.")
    modes = list(dict.fromkeys([m for m in A.MODES if any(t['mode'] == m for t in tut.TUTORIALS)] +
                               [t['mode'] for t in tut.TUTORIALS]))
    m = st.segmented_control("Mode", ['all'] + modes, default='all', key='tut|mode',
                             format_func=lambda x: 'All modes' if x == 'all' else _mode(x))
    lst = [t for t in tut.TUTORIALS if m in (None, 'all') or t['mode'] == m]
    c1, c2 = st.columns([2, 5])
    with c1:
        ids = [t['id'] for t in lst]
        cur = st.session_state.get('tutorial')
        sel = st.radio("Tutorial", ids, index=ids.index(cur) if cur in ids else 0, key='tut|sel',
                       format_func=lambda i: next(f"{t['mode']} · {t['title']}" for t in lst if t['id'] == i),
                       label_visibility='collapsed')
    t = next(x for x in lst if x['id'] == sel)
    with c2:
        with st.container(border=True):
            st.markdown(f"### {t['title']}")
            st.markdown(f":{LEVEL_COL[t['level']]}-badge[{t['level']}] :gray-badge[:material/schedule: about "
                        f"{t['minutes']} min] :gray-badge[{_mode(t['mode'])}]")
            st.markdown(f"**Scenario.** {t['story']}")
            st.markdown("**You will learn to** " + "; ".join(t['goals']) + ".")
            b1, b2, _ = st.columns([2, 2, 3])
            with b1:
                if st.button("Load tutorial", type='primary', icon=':material/download:', key='tut|load',
                             width='stretch'):
                    _load(t)
                    st.toast(f"Tutorial '{t['title']}' loaded", icon=':material/check:')
                    if _target(t) in PAGES:
                        st.switch_page(PAGES[_target(t)])
            with b2:
                if _target(t) in PAGES:
                    st.page_link(PAGES[_target(t)], label=f"Open {_title(t)}", icon=':material/arrow_forward:')
            if st.session_state.get('cfg_name') == f"tutorials/{t['id']}":
                st.caption(":material/check_circle: This tutorial is the active configuration.")
        t1, t2, t3, t4 = st.tabs(["Steps", "What you should see", "Exercises", "Settings of the tutorial"])
        with t1:
            for i, s in enumerate(t['steps'], start=1):
                st.markdown(f"{i}. {s}")
        with t2:
            st.markdown(t['look'] or "–")
        with t3:
            for s in t['exercise']:
                st.markdown(f"- {s}")
        with t4:
            table(_changes(t), hide_index=True, width='stretch')
            st.caption(f"Config file: {A.CONFIG_DIR}/tutorials/{t['id']}.json — all other parameters are those of the "
                       f"base configuration. Load it in the sidebar or import it as JSON.")
            if t.get('runs'):
                runs(t)
