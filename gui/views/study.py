#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         study.py
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
Study page: grid search, optimisation (built-in random + local search, or Optuna TPE) and forward feature selection
for one operating mode, on top of the current configuration. Every evaluation is a full train + test run of the mode
(nothing saved, no plots); the objective is an error metric on the validation data (default) or the testing data.
The best setting can be loaded into the configuration. Engine: gui/core/study.py.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import json
import math
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui import adapter as A
from gui.core import forms, study as S
from gui.core.style import table, show, line, fmt
from gui.views.tools import _common as cm

#######################################################################################################################
# Settings
#######################################################################################################################
K = 'st'
KINDS = {'grid': 'Grid search', 'opti': 'Optimisation', 'feat': 'Feature selection'}


#######################################################################################################################
# Helpers
#######################################################################################################################
def _label(p):
    return f"{p['name'] or p['var']} ({p['cat']}.{p['var']})"


def _rows_df(rows, metric):
    df = pd.DataFrame([{'#': i + 1, **r['x'], metric: r['obj'], **{k: v for k, v in r['metrics'].items()
                                                                     if k != metric},
                        'time (s)': r['time'], 'error': r.get('error') or ''} for i, r in enumerate(rows)])
    return df


def _save(st_):
    G = forms.cfg()[A.RUN_SECTION]
    folder = os.path.join(A.ROOT, 'results', f"study_{st_['mode']}_{G.get('name', '')}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'study.json'), 'w', encoding='utf-8') as f:
        json.dump(st_, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    return folder


def _best(rows, metric):
    b = None
    for r in rows:
        if S.better(r['obj'], None if b is None else b['obj'], metric):
            b = r
    return b


#######################################################################################################################
# Runner
#######################################################################################################################
def _run(state, plan, schema):
    """Evaluates the change dicts of plan (list or generator) and appends the rows to state (live table)."""
    bar = st.progress(0.0, text="Starting …")
    box = st.empty()
    n = state['n']
    for i, x in enumerate(plan):
        bar.progress(min(i / max(n, 1), 1.0), text=f"Run {i + 1} of {n}: " +
                     ', '.join(f"{k.split('.')[-1]} = {v}" for k, v in x.items()))
        cfg = S.set_values(state['base'], x, schema)
        r = S.evaluate(cfg, state['mode'], schema, state['metric'])
        r['x'] = {k: (', '.join(v) if isinstance(v, list) else v) for k, v in x.items()}
        r['raw'] = x
        state['rows'].append(r)
        if state.get('_opt') is not None:
            state['_opt'].tell(x, r['obj'])
        box.dataframe(_rows_df(state['rows'], state['metric']), hide_index=True, height=240)
    bar.progress(1.0, text="Done")
    state['done'] = True


def _opt_plan(opt, budget):
    for _ in range(budget):
        yield opt.ask()


def _feat_plan(state, cand, budget):
    """Greedy forward selection: per round every remaining channel is added to the best set of the last round."""
    chosen, best, used = [], None, 0
    while used < budget and len(chosen) < len(cand):
        start = len(state['rows'])
        for s in S.forward_next(cand, chosen):
            if used >= budget:
                break
            used += 1
            yield {f"{A.RUN_SECTION}.inp": s}
        rnd = state['rows'][start:]
        b = _best(rnd, state['metric'])
        if b is None or (best is not None and not S.better(b['obj'], best, state['metric'])):
            state['stop'] = "no further improvement"
            return
        best, chosen = b['obj'], list(b['raw'][f"{A.RUN_SECTION}.inp"])


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cfg = forms.cfg()
    schema = forms.schema()
    st.title("Study")
    st.caption("Parameter studies on top of the current configuration: every evaluation trains and tests the model "
               "once (nothing is saved, no figures). TherMOS itself has no optimiser — this page runs main() "
               "repeatedly with changed parameters.")
    cm.question("Which model order, window length or learning rate gives the smallest error, and which inputs are "
                "needed?")

    # ------------------------------------------
    # Setup
    # ------------------------------------------
    c = st.columns([2, 3, 1.2, 2])
    modes = list(A.MODES)
    mode = c[0].selectbox("Operating mode", modes, format_func=lambda m: f"{m} · {A.MODES[m]['title']}",
                          key=f"{K}|mode")
    kind = c[1].segmented_control("Study", list(KINDS), format_func=KINDS.get, default='grid', key=f"{K}|kind") \
        or 'grid'
    metric = c[2].selectbox("Objective", list(S.METRICS), key=f"{K}|metric",
                            help="MAE, RMSE, MAX are minimised (K), FIT is maximised (%).")
    useVal = c[3].toggle("Objective on validation data", True, key=f"{K}|val",
                         help="Recommended: the testing data stay unseen; check the best setting with a normal run "
                              "of the mode afterwards.")
    base, note = S.study_cfg(cfg, mode, useVal)
    G = cfg[A.RUN_SECTION]
    st.caption(f"Config '{st.session_state.get('cfg_name', '')}' · data {G.get('datDim')}D · objective {metric} on "
               f"{note}. Mode-specific settings (model variant, data) are taken from the current configuration.")
    last = st.session_state.get('runs', {}).get(mode)
    tRun = last.get('duration') if last else None

    tun = S.tunable(schema, mode)
    plan, space, n = None, None, 0
    with st.container(border=True):
        if kind == 'grid':
            st.markdown("**Grid search** — every combination of the values is evaluated.")
            keys = st.multiselect("Parameters (1–3)", [k for k, _ in tun], max_selections=3,
                                  format_func=lambda k: _label(dict(tun)[k]), key=f"{K}|gk",
                                  default=[k for k, _ in tun if k.endswith('.K')][0:1])
            specs = {}
            for k in keys:
                p = dict(tun)[k]
                cur = S.get_value(cfg, k, schema)
                dflt = (f"{max(int(cur) - 2, 1)}:{int(cur) + 3}" if p['type'] == 'int' else
                        f"{cur / 4:g}, {cur / 2:g}, {cur:g}, {2 * cur:g}" if cur else "0, 0.5, 1")
                txt = st.text_input(f"Values of {_label(p)}", dflt, key=f"{K}|gv|{k}",
                                    help="Comma list (1, 2, 4), integer range (1:8) or a:b:n (n values, log spaced "
                                         "if b/a ≥ 10).")
                try:
                    specs[k] = S.parse_values(txt, p['type'])
                except ValueError:
                    st.warning(f"Could not read the values of {_label(p)}.")
            if specs:
                plan = S.grid(specs)
                n = len(plan)
        elif kind == 'opti':
            avail = S.optuna_available()
            c = st.columns([2, 1, 1])
            meth = c[0].segmented_control("Method", ['built-in', 'Optuna (TPE)'], default='built-in',
                                          key=f"{K}|meth", help="Built-in: random search over half the budget, then "
                                          "coordinate search around the best point. Optuna: tree-structured Parzen "
                                          "estimator (pip install optuna).") or 'built-in'
            if meth.startswith('Optuna') and not avail:
                c[0].warning("Optuna is not installed: pip install optuna", icon=':material/warning:')
            budget = int(c[1].number_input("Budget (runs)", 2, 500, 12, key=f"{K}|budget"))
            keys = st.multiselect("Parameters (1–4)", [k for k, _ in tun], max_selections=4,
                                  format_func=lambda k: _label(dict(tun)[k]), key=f"{K}|ok",
                                  default=[k for k, _ in tun if k.endswith('.K')][0:1])
            space = {}
            for k in keys:
                p = dict(tun)[k]
                cur = float(S.get_value(cfg, k, schema) or 1)
                cc = st.columns([3, 1, 1, 1])
                cc[0].markdown(f"{_label(p)} · current {fmt(cur)}")
                lo = cc[1].number_input("min", value=float(max(cur / 4, 1) if p['type'] == 'int' else cur / 10),
                                        key=f"{K}|lo|{k}", format='%g', label_visibility='collapsed',
                                        help="lower bound")
                hi = cc[2].number_input("max", value=float(max(cur * 3, lo + 1) if p['type'] == 'int' else cur * 10),
                                        key=f"{K}|hi|{k}", format='%g', label_visibility='collapsed',
                                        help="upper bound")
                log = cc[3].checkbox("log", value=p['type'] == 'float' and lo > 0, key=f"{K}|log|{k}",
                                     help="Search on a logarithmic scale (learning rates, regularisation).")
                if hi <= lo:
                    st.warning(f"{_label(p)}: max must be larger than min.")
                else:
                    space[k] = (lo, hi, p['type'], log and lo > 0)
            n = budget if space else 0
        else:
            cand = list(G.get('inp') or [])
            st.markdown("**Forward feature selection** — start with no input, add the channel that improves the "
                        "objective most, repeat until no channel helps (as in setup/opti/opti_DL_Feat).")
            cand = st.multiselect("Candidate input channels", cand, default=cand, key=f"{K}|cand")
            budget = int(st.number_input("Budget (runs)", 1, 500, int(len(cand) * (len(cand) + 1) / 2) or 1,
                                         key=f"{K}|fbudget",
                                         help=f"A complete selection needs up to {len(cand) * (len(cand) + 1) // 2} "
                                              f"runs."))
            if mode == 'PI':
                st.info("The physics loss of the PI mode reads T_amb from feature 2 and I_s from feature 4; feature "
                        "selection changes these positions — use β = 0 or keep the order.", icon=':material/info:')
            if mode in ('RC', 'SF', 'PO', 'PS'):
                st.info("This mode does not use a list of input features in the same way; feature selection is "
                        "meant for SS, ML, DL and PI.", icon=':material/info:')
            n = budget if cand else 0
        c = st.columns([1, 3])
        est = f" · ≈ {fmt(n * tRun / 60)} min (last run of {mode}: {fmt(tRun)} s)" if tRun else \
            " · run the mode once to get a time estimate"
        c[1].caption(f"{n} runs{est}. Changing any widget stops a running study; the finished runs are kept.")
        go_ = c[0].button("Start study", type='primary', icon=':material/play_arrow:', key=f"{K}|go",
                          disabled=n == 0, width='stretch')

    if go_:
        state = {'mode': mode, 'kind': kind, 'metric': metric, 'note': note, 'base': base, 'rows': [], 'n': n,
                 'done': False, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        st.session_state['study'] = state
        if kind == 'grid':
            _run(state, plan, schema)
        elif kind == 'opti':
            opt = S.OptunaOpt(space, metric, n) if (st.session_state.get(f"{K}|meth", '').startswith('Optuna')
                                                    and S.optuna_available()) else S.RandomLocal(space, metric, n)
            state['_opt'] = opt
            state['method'] = type(opt).__name__
            _run(state, _opt_plan(opt, n), schema)
            state.pop('_opt', None)
        else:
            _run(state, _feat_plan(state, cand, n), schema)
        try:
            state['folder'] = _save({k: v for k, v in state.items() if k not in ('base', '_opt')})
        except Exception as e:
            st.warning(f"study.json could not be written: {e}")
    results(st.session_state.get('study'))


#######################################################################################################################
# Results
#######################################################################################################################
def results(state):
    if not state or not state['rows']:
        return
    metric, rows = state['metric'], state['rows']
    cm.section(None, f"Result: {KINDS[state['kind']]} of {state['mode']} ({len(rows)} runs, {state['time']})")
    ok = [r for r in rows if not (isinstance(r['obj'], float) and math.isnan(r['obj']))]
    b = _best(rows, metric)
    if b is None:
        st.error("No run produced a result. First error: " + str(rows[0].get('error')))
        return
    cm.tiles([("Best " + metric, f"{fmt(b['obj'])} {'%' if metric == 'FIT' else 'K'}", f"on {state['note']}"),
              ("Best setting", ', '.join(f"{k.split('.')[-1]} = {v}" for k, v in b['x'].items()), None),
              ("Runs", f"{len(ok)} ok / {len(rows)}", None),
              ("Total time", f"{fmt(sum(r['time'] for r in rows) / 60)} min", None)])
    df = _rows_df(rows, metric)
    keys = list(rows[0]['x'])
    if state['kind'] == 'grid' and len(keys) == 1:
        d = df.sort_values(keys[0])
        fig = go.Figure(line(d[keys[0]], d[metric], metric, 0, mode='lines+markers'))
        fig.update_layout(xaxis_title=keys[0].split('.')[-1], yaxis_title=metric)
        show(fig, key=f"{K}|g1")
    elif state['kind'] == 'grid' and len(keys) >= 2:
        pv = df.pivot_table(index=keys[1], columns=keys[0], values=metric, aggfunc='min')
        fig = go.Figure(go.Heatmap(z=pv.values, x=[str(v) for v in pv.columns], y=[str(v) for v in pv.index],
                                   colorscale='Viridis_r' if S.METRICS[metric] == 'min' else 'Viridis',
                                   colorbar=dict(title=metric), texttemplate='%{z:.3g}'))
        fig.update_layout(xaxis_title=keys[0].split('.')[-1], yaxis_title=keys[1].split('.')[-1])
        show(fig, key=f"{K}|g2")
        if len(keys) > 2:
            st.caption(f"Minimum over {keys[2].split('.')[-1]} per cell.")
    else:
        obj = np.array([r['obj'] for r in rows], dtype=float)
        acc = np.fmin.accumulate(obj) if S.METRICS[metric] == 'min' else np.fmax.accumulate(obj)
        fig = go.Figure([go.Scatter(x=np.arange(1, len(obj) + 1), y=obj, mode='markers', name='run'),
                         line(np.arange(1, len(obj) + 1), acc, 'best so far', 1)])
        fig.update_layout(xaxis_title='run', yaxis_title=metric)
        show(fig, key=f"{K}|op")
    table(df, hide_index=True)
    if state.get('stop'):
        st.caption(f"Stopped: {state['stop']}.")
    c = st.columns([1, 3])
    if c[0].button("Load best into the config", icon=':material/upload:', key=f"{K}|load"):
        cfg = S.set_values(forms.cfg(), b['raw'], forms.schema())
        forms.set_config(cfg, st.session_state.get('cfg_name', 'default'))
        st.toast("Best setting loaded — run the mode on the testing data to confirm it", icon=':material/check:')
    if state.get('folder'):
        c[1].caption(f"Stored in {os.path.relpath(state['folder'], A.ROOT)}/study.json")
    best = ', '.join(f"{k.split('.')[-1]} = {v}" for k, v in b['x'].items())
    cm.interpret([
        f"The best of {len(ok)} runs is {best} with {metric} = {fmt(b['obj'])} on {state['note']}.",
        "The objective is measured on the validation data, so the best setting is selected without looking at the "
        "test data; confirm it with a normal run of the mode (Load best → Run). Differences smaller than the "
        "run-to-run scatter (DL/PI: random initialisation) are not significant."
        if 'validation' in state['note'] else
        "The objective was measured on the testing data: the best setting is optimistic for unseen data. Use a "
        "validation set (Setup › Data) for an unbiased selection.",
        "A minimum inside the searched range is reliable; a best value at the border of the range means the "
        "optimum may lie outside — extend the range."])
