#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Thermal Modelling, Model Order Reduction
# File:         field_explore
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
Interactive insight into 2D/3D temperature fields, shared by the PO/PS results, the data viewer and the board designer:
- profiles():  line profiles T(x) at y = y₀ and T(y) at x = x₀ through a chosen point (several fields overlaid, e.g.
               measured and predicted), with the position marked in a small heatmap
- probes():    temperature over time at probe points (hot spot, coolest node, mean, any x/y/z point) for several fields
               (measured solid, predicted dashed) and the error at the probes
- zslice():    z position of the plane shown for 3D data
All coordinates are in metres in the data and in mm on screen.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import show, line, fmt, color, INK2, TEMP


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _near(v, v0):
    return int(np.argmin(np.abs(np.asarray(v) - v0)))


#######################################################################################################################
# z slice
#######################################################################################################################
def zslice(geo, key, z0=0.0, label="z plane (mm)"):
    """z of the plane for 3D node coordinates (geo N×3, m); None for 2D."""
    g = np.asarray(geo, dtype=float)
    if g.shape[1] < 3:
        return None
    z = np.unique(np.round(g[:, 2] - g[:, 2].min(), 12))
    if len(z) < 2:
        return float(z[0]) if len(z) else 0.0
    i = st.select_slider(label, options=list(range(len(z))), value=_near(z, z0), key=f"{key}|z",
                         format_func=lambda k: f"{z[k] * 1e3:.3g}", help="Position of the x-y plane shown in the "
                         "fields and profiles (3D data). Move it through the stack to see the heat flowing down.")
    return float(z[i])


#######################################################################################################################
# Line profiles
#######################################################################################################################
def profiles(fields, x, y, key, unit='°C', vlabel=('x (mm)', 'y (mm)'), info=None):
    """
    Line profiles through a point of 2D fields on a regular grid.
    fields: list of (label, F (ny×nx), dashed); x, y: grid axes (m)
    """
    if not fields:
        return
    F0 = np.asarray(fields[0][1], dtype=float)
    iy0, ix0 = np.unravel_index(int(np.nanargmax(F0)), F0.shape)
    c = st.columns([2, 2, 1])
    ix = c[0].select_slider(f"x₀ for T({vlabel[1].split()[0]}) ({vlabel[0].split()[1]})", list(range(len(x))),
                            value=int(ix0), key=f"{key}|px", format_func=lambda k: f"{x[k] * 1e3:.3g}")
    iy = c[1].select_slider(f"{vlabel[1].split()[0]}₀ for T(x) ({vlabel[1].split()[1]})", list(range(len(y))),
                            value=int(iy0), key=f"{key}|py", format_func=lambda k: f"{y[k] * 1e3:.3g}")
    if c[2].button("Hot spot", icon=':material/my_location:', key=f"{key}|phot", width='stretch',
                   help="Place the cross at the hottest point of the first field."):
        st.session_state[f"{key}|px"], st.session_state[f"{key}|py"] = int(ix0), int(iy0)
        st.rerun()
    cols = st.columns([1, 2, 2])
    with cols[0]:
        fig = go.Figure(go.Heatmap(z=F0, x=x * 1e3, y=y * 1e3, colorscale=TEMP if unit == '°C' else 'RdBu_r',
                                   showscale=False, hovertemplate='%{x:.1f}, %{y:.1f} mm: %{z:.3g}<extra></extra>'))
        fig.add_hline(y=y[iy] * 1e3, line=dict(color=INK2, width=1.5, dash='dot'))
        fig.add_vline(x=x[ix] * 1e3, line=dict(color=INK2, width=1.5, dash='dot'))
        fig.update_layout(title=fields[0][0], xaxis_title=vlabel[0], yaxis_title=vlabel[1], margin=dict(l=40, r=10))
        show(fig, height=300, key=f"{key}|pmap", info="**Position of the profiles.** Dotted lines: the cuts shown "
             "on the right (horizontal → T(x), vertical → T(along the second axis)).")
    for col, (ax, cut, sl, ttl) in zip(cols[1:], ((x, lambda F: F[iy, :], vlabel[0],
                                                     f"T(x) at {vlabel[1].split()[0]} = {y[iy] * 1e3:.3g} mm"),
                                                    (y, lambda F: F[:, ix], vlabel[1],
                                                     f"T({vlabel[1].split()[0]}) at x = {x[ix] * 1e3:.3g} mm"))):
        with col:
            fig = go.Figure([line(ax * 1e3, cut(np.asarray(F, dtype=float)), lab, i, dash='dash' if d else None)
                             for i, (lab, F, d) in enumerate(fields)])
            fig.update_layout(title=ttl, xaxis_title=sl, yaxis_title=f"T ({unit})" if unit == '°C' else unit)
            show(fig, height=300, key=f"{key}|line{sl[0]}", info=info or "**Line profile.** Steep slopes = high heat "
                 "flux (q = −k·dT/dx); a flat profile = good heat spreading. Where measured and predicted curves "
                 "separate, the reduced model misses a local detail (too few modes, missing boundary).")


#######################################################################################################################
# Probes over time
#######################################################################################################################
def probes(geo, series, t, key, z=None, mark=None, unit='°C'):
    """
    Temperature over time at probe points.
    geo: node coordinates (N×2 or N×3, m); series: list of (label, Y (nt×N), dashed); t: time (s)
    z: z of the plane for 3D custom points; mark: time to mark (s)
    """
    g = np.asarray(geo, dtype=float)
    g = g - g.min(axis=0)
    Y0 = np.asarray(series[0][1], dtype=float)
    opts = ["Hot spot (max over the run)", "Coolest node", "Mean of all nodes", "Point x/y"]
    c = st.columns([3, 1, 1])
    sel = c[0].multiselect("Probes", opts, default=opts[:1] + opts[2:3], key=f"{key}|pr",
                           help="Hot spot / coolest node: node with the highest / lowest temperature of the first "
                                "series over the whole run. Point: the node next to the given position.")
    px = c[1].number_input("x (mm)", 0.0, float(g[:, 0].max() * 1e3), float(g[np.argmax(Y0.max(axis=0)), 0] * 1e3),
                           key=f"{key}|prx", disabled="Point x/y" not in sel)
    py = c[2].number_input("y (mm)", 0.0, float(g[:, 1].max() * 1e3), float(g[np.argmax(Y0.max(axis=0)), 1] * 1e3),
                           key=f"{key}|pry", disabled="Point x/y" not in sel)
    idx = []
    for s in sel:
        if s.startswith("Hot"):
            idx.append((s.split(' (')[0], int(np.argmax(Y0.max(axis=0)))))
        elif s.startswith("Cool"):
            idx.append((s, int(np.argmin(Y0.min(axis=0)))))
        elif s.startswith("Mean"):
            idx.append((s, None))
        else:
            p = np.array([px, py] + ([z * 1e3] if g.shape[1] == 3 and z is not None else
                                     [0.0] * (g.shape[1] - 2))) * 1e-3
            idx.append((f"x={px:.3g}, y={py:.3g} mm", int(np.argmin(np.sum((g - p) ** 2, axis=1)))))
    if not idx:
        st.caption("Select at least one probe.")
        return
    fig = go.Figure()
    errs = []
    for k, (nm, n) in enumerate(idx):
        for j, (lab, Y, d) in enumerate(series):
            Y = np.asarray(Y, dtype=float)
            v = Y.mean(axis=1) if n is None else Y[:, n]
            fig.add_trace(go.Scatter(x=t[:len(v)], y=v, mode='lines', name=f"{nm} · {lab}", legendgroup=nm,
                                     line=dict(color=color(k), dash='dash' if d else None, width=2)))
        if len(series) > 1:
            a = np.asarray(series[0][1], dtype=float)
            b = np.asarray(series[1][1], dtype=float)
            e = (a.mean(axis=1) - b.mean(axis=1)) if n is None else (a[:, n] - b[:, n])
            errs.append({'probe': nm, 'node': '–' if n is None else n,
                         'MAE (K)': float(np.mean(np.abs(e))), 'max |e| (K)': float(np.max(np.abs(e)))})
    if mark is not None:
        fig.add_vline(x=mark, line=dict(color=INK2, dash='dot'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title=f"T ({unit})")
    show(fig, height=360, key=f"{key}|prf", info="**Probes over time.** Colour = probe, solid = first series "
         "(measured / full model), dashed = second (prediction / reduced model). The dotted line marks the time "
         "step of the field above. Thermal time constants show as the delay between a load step and the rise.")
    if errs:
        import pandas as pd
        from gui.core.style import table
        table(pd.DataFrame(errs), hide_index=True)
