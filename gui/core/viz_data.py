#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         viz_data.py
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
Input data viewer of TherMOS (Input data page and the Inputs view of every mode). It shows the data sets selected on
the Setup page as they are stored in data/ (before resampling and normalisation):
    1D  losses (inputs, W) and temperatures (outputs, °C) over time per profile (file / sheet / ID), statistics
    2D/3D  material maps (k, ρ, c_p, h), heat input and temperature field at a time slider, total heat and the
           temperature range over time
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import dataset
from gui.core.style import show, line, fmt, table, color, SEQ, TEMP
from gui.core.viz_thermos import _grid, heat

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


#######################################################################################################################
# 1D
#######################################################################################################################
def _profile(df, G, key):
    """Rows of one profile (ID) of a 1D table."""
    if 'id' not in df.columns:
        return df, None
    ids = sorted(pd.unique(df['id'].dropna()).astype(int).tolist())
    used = {'train': G.get('trID') or [], 'validation': G.get('vlID') or [], 'test': G.get('teID') or []}
    lab = {i: f"{i}" + ''.join(f" · {k}" for k, v in used.items() if i in v) for i in ids}
    i = st.selectbox("Profile ID", ids, format_func=lab.get, key=f"{key}|id",
                     index=ids.index((G.get('teID') or ids)[0]) if (G.get('teID') or ids)[0] in ids else 0)
    return df[df['id'] == i], i


def view1d(cfg, key):
    G = cfg[A.RUN_SECTION]
    fmt_, sep = G.get('format', 'mat'), int(G.get('datSep', 1))
    files = list(dict.fromkeys([G.get('teFile')] + list(G.get('trFile') or []) + list(G.get('vlFile') or [])))
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("Data set", files, key=f"{key}|file",
                            format_func=lambda n: n + (' · test' if n == G.get('teFile') else ''))
    inf = dataset.info(name, fmt_)
    sheet = None
    with c2:
        if inf.get('sheets'):
            sheets = inf['sheets']
            sheet = st.selectbox("Sheet", sheets, key=f"{key}|sheet",
                                 index=sheets.index(G['teSheet']) if G.get('teSheet') in sheets else 0)
    df = dataset.table(name, fmt_, sheet)
    if sep == 3 or 'id' in df.columns:
        df, _ = _profile(df, G, key)
    t = df['time'].to_numpy(dtype=float) if 'time' in df.columns else np.arange(len(df), dtype=float)
    t = t - t[0] if len(t) else t
    inp = [c for c in (G.get('inp') or []) if c in df.columns]
    out = [c for c in (G.get('out') or []) if c in df.columns]
    ref = [c for c in (G.get('ref') or []) if c in df.columns]
    if not inp and not out:
        st.warning("None of the selected channels is in this data set (Setup › Data › Channels).")
        return
    s = max(1, len(t) // 5000)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Inputs", "Outputs and references"))
    for i, c in enumerate(inp):
        fig.add_trace(line(t[::s], df[c].to_numpy(dtype=float)[::s], c, i), 1, 1)
    for i, c in enumerate(out):
        fig.add_trace(line(t[::s], df[c].to_numpy(dtype=float)[::s], c, i + len(inp)), 2, 1)
    for i, c in enumerate(ref):
        fig.add_trace(line(t[::s], df[c].to_numpy(dtype=float)[::s], c, i + len(inp) + len(out), dash='dot'), 2, 1)
    fig.update_yaxes(title_text="Input (W, A, V, …)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    show(fig, height=560, key=f"{key}|ts", info="**Raw input data of the selected profile.** Top: model inputs "
         "(usually losses in W), bottom: temperatures to be predicted (solid) and references (dotted). The model "
         "can only learn the dynamics that are excited here: a single step identifies the time constants, a "
         "random load profile also the coupling between the inputs.")
    dt = np.diff(t)
    st.caption(f"{len(t)} samples · {fmt(t[-1] if len(t) else 0)} s · sampling {fmt(np.median(dt) if len(dt) else 0)} s "
               f"(resampled to Ts = {fmt(cfg['Exp']['gen']['Ts'])} s for the models)")
    with st.expander("Statistics"):
        table(df[inp + out + ref].describe().T, height=40 + 35 * len(inp + out + ref))


#######################################################################################################################
# 2D / 3D
#######################################################################################################################
def view_field(cfg, key):
    G = cfg[A.RUN_SECTION]
    files = list(dict.fromkeys([G.get('teFile')] + list(G.get('trFile') or []) + list(G.get('vlFile') or [])))
    name = st.selectbox("Data set", files, key=f"{key}|ffile",
                        format_func=lambda n: n + (' · test' if n == G.get('teFile') else ''))
    D = dataset.field(name)
    g = np.asarray(D['geo'], dtype=float)
    X, y = np.asarray(D['X'], dtype=float), np.asarray(D['y'], dtype=float)
    Ts = float(D.get('Ts', 1))
    t = np.asarray(D.get('t', np.arange(len(y)) * Ts), dtype=float).ravel()
    z0 = float(cfg['Dat']['gen'].get('inpZ', 0) or 0)
    d3 = g.shape[1] == 3
    ext = np.ptp(g, axis=0) * 1e3
    c = st.columns(4)
    c[0].metric("Nodes", f"{len(g):,}", help="Grid nodes of the field (one temperature per node).")
    c[1].metric("Domain (mm)", ' × '.join(f"{v:.3g}" for v in ext), help="Extent of the geometry.")
    c[2].metric("Time steps", f"{len(y)}", help=f"Snapshots, Ts = {fmt(Ts)} s.")
    c[3].metric("T range (°C)", f"{np.nanmin(y):.1f} … {np.nanmax(y):.1f}", help="Minimum and maximum temperature.")
    if d3:
        from gui.core.field_explore import zslice
        z0 = zslice(g, key, z0)
    if np.all(np.asarray(D.get('hc', 0)) == 0):
        st.info("hc = 0 everywhere: the data carry no convection boundary. The pure Galerkin projection (POD "
                "protocol 2/3) then has no heat sink — use the calibrated protocol 5 or 6 (F-P16).")

    tab = st.tabs(["Materials", "Heat input & temperature", "Over time", "Profiles & probes"])
    with tab[0]:
        cols = st.columns(2)
        for i, (k, ttl, u) in enumerate((('k', 'Thermal conductivity k', 'W/mK'), ('rho', 'Density ρ', 'kg/m³'),
                                         ('Cp', 'Specific heat c_p', 'J/kgK'), ('hc', 'Heat transfer coeff. h',
                                                                                 'W/m²K'))):
            if k not in D:
                continue
            [F, x, yy] = _grid(np.asarray(D[k], dtype=float).ravel(), D, z0)
            with cols[i % 2]:
                show(heat(F[0] if F.ndim == 3 else F, x, yy, ttl, u, SEQ), height=320, key=f"{key}|m{k}",
                     info="**Material map.** Each colour region is one material; the POD Galerkin projection "
                          "(protocol 2/3) uses k and ρ·c_p from these maps.")
    with tab[1]:
        it = st.slider("Time step", 0, len(y) - 1, len(y) - 1, key=f"{key}|fit")
        [F, x, yy] = _grid(np.vstack([X[it], y[it]]), D, z0)
        cols = st.columns(2)
        with cols[0]:
            show(heat(F[0], x, yy, f"Heat input q · t = {t[it] if it < len(t) else it * Ts:.0f} s", 'W/m³', SEQ),
                 height=340, key=f"{key}|q", info="**Volumetric heat generation** (losses of the devices).")
        with cols[1]:
            show(heat(F[1], x, yy, f"Temperature · t = {t[it] if it < len(t) else it * Ts:.0f} s", '°C', TEMP),
                 height=340, key=f"{key}|T", info="**Temperature field** of the data set at this time step.")
    with tab[2]:
        dV = float(D.get('dx', 1)) * float(D.get('dy', 1)) * (float(D.get('dz', 1)) if d3 else 1.0)
        tt = t if len(t) == len(y) else np.arange(len(y)) * Ts
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(line(tt, X.sum(axis=1) * dV, 'Σ q·dV', 0), secondary_y=False)
        fig.add_trace(line(tt, y.max(axis=1), 'T max', 1), secondary_y=True)
        fig.add_trace(line(tt, y.mean(axis=1), 'T mean', 2), secondary_y=True)
        fig.update_yaxes(title_text='Heat (W, ×depth for 2D)', secondary_y=False)
        fig.update_yaxes(title_text='Temperature (°C)', secondary_y=True)
        fig.update_xaxes(title_text='Time (s)')
        show(fig, height=380, key=f"{key}|ot", info="**Total heat and temperature over time.** Σ q·dV is the heat "
             "input (per metre depth for 2D data); T max is the hot spot, T mean the average of all nodes.")
    with tab[3]:
        from gui.core.field_explore import profiles, probes
        it2 = st.slider("Time step of the profiles", 0, len(y) - 1, len(y) - 1, key=f"{key}|pit")
        [F, x, yy] = _grid(y[it2], D, z0)
        profiles([(f"T at t = {it2 * Ts:.0f} s", F[0] if F.ndim == 3 else F, False)], x, yy, f"{key}|prof")
        tt = t if len(t) == len(y) else np.arange(len(y)) * Ts
        probes(g, [('T', y, False)], tt, f"{key}|probe", z=z0, mark=tt[it2])


#######################################################################################################################
# Entry
#######################################################################################################################
def view(cfg, key='inp'):
    G = cfg[A.RUN_SECTION]
    if not G.get('teFile'):
        st.info("Select the data sets on the Setup page.")
        return
    inf = dataset.info(G['teFile'], G.get('format', 'mat'))
    if inf['kind'] in ('missing', 'error'):
        st.warning(f"Data set '{G['teFile']}' not found under data/ ({inf.get('error', 'missing')}).")
        return
    if inf['kind'] == '1D':
        view1d(cfg, key)
    else:
        view_field(cfg, key)
