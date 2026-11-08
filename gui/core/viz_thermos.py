#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         viz_thermos.py
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
Interactive results of the TherMOS modes (registered in viz_results.RENDER). Every view starts with the key figures
(error metrics of the testing data), followed by the prediction over time and the model-specific charts:
    RC  Foster pairs (Rth, τ) and the thermal impedance Zth(t)
    SS  poles, time constants and the steady-state thermal resistance matrix of the identified model
    SF  thermal resistance matrix and the Foster pairs per node
    PO/PS  POD energy, mode shapes, temperature field measured / predicted / error with a time slider
    ML  prediction vs. measurement
    DL/PINN  training and validation loss
The data come from adapter.compact (metrics, thinned predictions, model data without fitted estimators).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core.style import show, line, fmt, table, color, SEQ, DIV, TEMP, INK2, MUTED

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

METRICS = [('MAE', 'MAE (K)', "Mean absolute error |T − T̂| over all test samples."),
           ('RMSE', 'RMSE (K)', "Root mean square error, weights large errors more than the MAE."),
           ('MAX', 'MAX (K)', "Largest absolute error of a single sample — the worst moment of the prediction."),
           ('FIT', 'FIT (%)', "100·(1 − ‖T − T̂‖/‖T − mean(T)‖): share of the temperature variation explained by the "
                              "model (100 % = perfect, 0 % = as good as a constant). MATLAB calls this NRMSE."),
           ('NRMSE', 'NRMSE (%)', "RMSE normalised by the temperature range of the channel.")]


#######################################################################################################################
# Helper
#######################################################################################################################
def _names(setup, n, key='out', pre='T'):
    lst = list((setup or {}).get('Run', {}).get(key) or [])
    return [str(lst[i]) if i < len(lst) else f"{pre}{i + 1}" for i in range(n)]


def _tag(data):
    return next(iter(data['results']))


def _is_field(grt):
    D = grt.get('Data')
    return isinstance(D, dict) and 'geo' in D and np.asarray(grt['y']).shape[1] > 20


def tiles(data, setup, key):
    """Key figures: average metrics over all outputs, table per output (all / transient / steady-state samples)."""
    tag = _tag(data)
    r = data['results'][tag]
    err = r['err']
    n = len(np.atleast_1d(err['tot']['MAE']))
    cols = st.columns(len(METRICS) + 1)
    for c, (k, lab, hlp) in zip(cols, METRICS):
        c.metric(lab, fmt(float(np.nanmean(err['tot'][k]))), help=hlp + (" Average over all outputs." if n > 1
                                                                         else ""))
    t = r.get('time', {})
    cols[-1].metric("Train / test (s)", f"{fmt(t.get('train', 0))} / {fmt(t.get('test', 0))}",
                    help="Training time of the model and time of the prediction of the testing data.")
    if n <= 50:
        rows = []
        names = _names(setup, n) if n <= 10 else [f"node {i}" for i in range(n)]
        for i in range(n):
            row = {'output': names[i]}
            for s, sl in (('tot', 'all'), ('tr', 'transient'), ('ss', 'steady')):
                for k in ('MAE', 'MAX'):
                    row[f"{k} {sl} (K)"] = float(np.atleast_1d(err[s][k])[i])
            row['FIT (%)'] = float(np.atleast_1d(err['tot']['FIT'])[i])
            rows.append(row)
        with st.expander("Errors per output", expanded=n > 1):
            st.caption("Transient samples: |dT| > th·max|dT| (Par › gen › th), the rest is steady-state. A model "
                       "with a small steady-state but large transient error has the right resistances but wrong "
                       "time constants.")
            table(pd.DataFrame(rows), hide_index=True)


def response(data, setup, key):
    """Measured and predicted temperature over time with the error (1D outputs or selected nodes of a field)."""
    tag = _tag(data)
    p, g = data['pred'][tag], data['grt'][tag]
    yT, yP = np.asarray(g['y'], dtype=float), np.asarray(p['y'], dtype=float)
    yT, yP = yT.reshape(len(yT), -1), yP.reshape(len(yP), -1)
    t = np.arange(len(yT)) * float(g.get('Ts', 1))
    field = _is_field(g)
    if field:
        idx = [int(np.argmax(np.max(yT, axis=0)))]
        st.caption(f"Field data: node {idx[0]} (hottest node of the test data) is shown; the field view below "
                   f"shows all nodes.")
        names = [f"node {i}" for i in idx]
    else:
        names = _names(setup, yT.shape[1])
        idx = list(range(yT.shape[1]))
        if len(idx) > 4:
            pick = st.multiselect("Outputs", names, default=names[:4], key=f"{key}|outs")
            idx = [names.index(x) for x in pick]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    for j, i in enumerate(idx):
        nm = names[j] if field else names[i]
        grp = dict(legendgroup=nm, legendgrouptitle_text=nm)              # one group per output: T, T̂ and error
        fig.add_trace(line(t, yT[:, i], "measured", j, **grp), 1, 1)
        fig.add_trace(line(t, yP[:, i], "predicted", j, dash='dash', legendgroup=nm), 1, 1)
        fig.add_trace(line(t, yT[:, i] - yP[:, i], "error T − T̂", j, width=1.3, dash='dot', legendgroup=nm), 2, 1)
    fig.update_layout(legend=dict(groupclick='togglegroup'))
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="T − T̂ (K)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    show(fig, height=520, key=f"{key}|resp",
         info="**Temperature over time (testing data).** Solid: measured, dashed: predicted, bottom: error "
              "T − T̂. The legend is grouped per output: a click on an entry hides or shows the measured and "
              "predicted temperature and the error of that output together (double click: only this output). An "
              "error that drifts with time points to a wrong steady-state resistance; an error that peaks at load "
              "steps and decays points to wrong time constants.")


def scatter(data, setup, key):
    """Prediction vs. measurement and the residual distribution."""
    tag = _tag(data)
    yT = np.asarray(data['grt'][tag]['y'], dtype=float)
    yP = np.asarray(data['pred'][tag]['y'], dtype=float)
    yT, yP = yT.reshape(len(yT), -1), yP.reshape(len(yP), -1)
    step = max(1, yT.size // 20000)
    a, b = yT.ravel()[::step], yP.ravel()[::step]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scattergl(x=a, y=b, mode='markers', marker=dict(size=4, color=color(0), opacity=0.5),
                                     name='samples'))
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(color=INK2, dash='dash'),
                                 name='T̂ = T'))
        fig.update_layout(xaxis_title="Measured (°C)", yaxis_title="Predicted (°C)")
        show(fig, height=380, key=f"{key}|sc", info="**Prediction vs. measurement.** Every point is one sample. "
             "Points on the dashed line are exact; a fan opening at high temperatures means the model "
             "under- or overestimates the thermal resistance.")
    with c2:
        fig = go.Figure(go.Histogram(x=(yT - yP).ravel()[::step], nbinsx=60, marker_color=color(1),
                                     histnorm='probability density', name='error'))
        fig.update_layout(xaxis_title="Error T − T̂ (K)", yaxis_title="Density (1/K)", showlegend=False)
        show(fig, height=380, key=f"{key}|hist", info="**Residual distribution.** Centred at 0 = no bias. A "
             "shifted peak is a systematic offset (e.g. wrong initial value or reference); long tails are "
             "transients the model does not follow.")


#######################################################################################################################
# Model views
#######################################################################################################################
def rc_view(data, setup, key):
    m = data['mdls'].get('rc', {})
    R, tau = np.atleast_1d(m.get('Rth', [])).astype(float), np.atleast_1d(m.get('tau', [])).astype(float)
    if not len(R):
        return
    o = np.argsort(tau)
    R, tau = R[o], tau[o]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Foster network**")
        table(pd.DataFrame({'pair': np.arange(1, len(R) + 1), 'Rth (K/W)': R, 'τ (s)': tau, 'Cth (Ws/K)': tau / R}),
              hide_index=True)
        st.caption(f"Total Rth = {fmt(R.sum())} K/W: the steady-state temperature rise per watt. Pairs with small τ "
                   f"describe the chip and solder, large τ the heat sink.")
    with c2:
        t = np.logspace(np.log10(max(tau.min() / 20, 1e-4)), np.log10(tau.max() * 20), 300)
        fig = go.Figure(line(t, np.sum(R[:, None] * (1 - np.exp(-t[None, :] / tau[:, None])), axis=0), 'Zth(t)', 0))
        for i in range(len(R)):
            fig.add_vline(x=tau[i], line=dict(color=MUTED, dash='dot', width=1))
        fig.update_layout(xaxis=dict(type='log', title='Time (s)'), yaxis_title='Zth (K/W)', showlegend=False)
        show(fig, height=340, key=f"{key}|zth", info="**Thermal impedance Zth(t) = Σ Rᵢ(1 − e^(−t/τᵢ)).** "
             "Temperature rise per watt after a loss step, on a log time axis. Each dotted line is one time "
             "constant τᵢ; the curve rises by Rᵢ around τᵢ.")


def ss_view(data, setup, key):
    sysd = data['mdls'].get('ss', {}).get('sys') or data['mdls'].get('ps', {}).get('sys')
    if not isinstance(sysd, dict) or 'A' not in sysd:
        return
    A, B, C = (np.atleast_2d(np.asarray(sysd[k], dtype=float)) for k in ('A', 'B', 'C'))
    Ts = float(sysd.get('Ts') or (setup or {}).get('Exp', {}).get('gen', {}).get('Ts', 1) or 1)
    lam = np.linalg.eigvals(A)
    modal = 'ss' not in data['mdls']                                     # PS: states drive the POD coordinates
    c1, c2 = st.columns(2)
    with c1:
        th = np.linspace(0, 2 * np.pi, 200)
        fig = go.Figure(go.Scatter(x=np.cos(th), y=np.sin(th), mode='lines', line=dict(color=MUTED, dash='dot'),
                                   name='unit circle'))
        fig.add_trace(go.Scatter(x=lam.real, y=lam.imag, mode='markers', marker=dict(symbol='x', size=10,
                                 color=color(1)), name='poles'))
        fig.update_layout(xaxis_title='Re λ', yaxis_title='Im λ', yaxis=dict(scaleanchor='x'))
        show(fig, height=360, key=f"{key}|poles", info="**Poles of the discrete model.** A thermal system is "
             "passive: all poles real, between 0 and 1. Poles close to 1 are slow time constants; complex or "
             "negative poles are artefacts of the identification (oscillations that do not exist physically).")
    with c2:
        mag = np.clip(np.abs(lam), 1e-12, 1 - 1e-12)
        tau = -Ts / np.log(mag)
        table(pd.DataFrame({'pole λ': [f"{v.real:.5f}{v.imag:+.5f}j" if abs(v.imag) > 1e-9 else f"{v.real:.5f}"
                                       for v in lam], 'τ = −Ts/ln|λ| (s)': tau}).sort_values('τ = −Ts/ln|λ| (s)'),
              hide_index=True)
        if modal:
            st.caption("PS: the outputs of this state-space model are the POD coordinates aₖ(t) of the field.")
            return
        try:
            G = C @ np.linalg.solve(np.eye(len(A)) - A, B)
            st.markdown("**Steady-state gain (K/W)**")
            st.caption("Temperature rise of each output (row) per watt of each input (column) after all transients: "
                       "the thermal resistance matrix incl. coupling. Off-diagonal values are the heating of one "
                       "device by its neighbours.")
            table(pd.DataFrame(G, index=_names(setup, G.shape[0]), columns=_names(setup, G.shape[1], 'inp', 'P')))
        except np.linalg.LinAlgError:
            pass


def sf_view(data, setup, key):
    m = data['mdls'].get('sf', {})
    R = m.get('Rth')
    if R is None:
        return
    R = np.atleast_2d(np.asarray(R, dtype=float))
    names = _names(setup, R.shape[0])
    fig = go.Figure(go.Heatmap(z=R, x=_names(setup, R.shape[1], 'inp', 'P'), y=names, colorscale=SEQ,
                               colorbar=dict(title='K/W'), texttemplate='%{z:.3g}'))
    fig.update_layout(yaxis=dict(autorange='reversed'))
    show(fig, height=360, key=f"{key}|rth", info="**Steady-state thermal resistance matrix Rth.** Rth[i, k] = "
         "temperature rise of node i per watt in node k. The diagonal is the self-heating, off-diagonal values are "
         "the thermal coupling.")
    if m.get('type') == 'net':
        Rl, Cl, Rc = (np.asarray(m[k], dtype=float) for k in ('Rlad', 'Clad', 'Rcpl'))
        Nr = Rl.shape[1]
        if len(Rl) <= 4:
            from gui.views.tools import sketch
            sketch.show(sketch.coupled(len(Rl), min(Nr, 3), names[:len(Rl)]),
                        "Fitted coupled Cauer network: one ladder per source, couplings at every stage.")
        st.markdown("**Ladders** (stage 1 = junction, the last stage connects to the reference)")
        table(pd.DataFrame(np.column_stack([Rl, Cl, np.cumsum(Rl, axis=1)[:, -1]]), index=names[:len(Rl)],
                           columns=[f"R'{k + 1} (K/W)" for k in range(Nr)] + [f"C'{k + 1} (Ws/K)" for k in range(Nr)] +
                                   ["Σ R' (K/W)"]))
        rows = []
        for k in range(Nr):
            for i in range(len(Rl)):
                for j in range(i + 1, len(Rl)):
                    rows.append({'stage': k + 1, 'between': f"{names[i]} – {names[j]}", 'R_ij (K/W)': Rc[k, i, j],
                                 'conductance (W/K)': 1 / Rc[k, i, j]})
        st.markdown("**Coupling resistances** (large R = negligible coupling)")
        table(pd.DataFrame(rows), hide_index=True)
        if m.get('nAct'):
            st.warning(f"{m['nAct']} resistances/capacitances ended at their bounds (factor {fmt(m.get('bnd'))}): "
                       "increase Mdl › sf › bnd or change the number of stages.")
    if m.get('Rth_rc') is not None:
        Rf, tf = np.asarray(m['Rth_rc'], dtype=float), np.asarray(m['tau_rc'], dtype=float)
        st.markdown("**Foster pairs per node**")
        table(pd.DataFrame(np.column_stack([Rf.reshape(len(Rf), -1), tf.reshape(len(tf), -1)]), index=names[:len(Rf)],
                           columns=[f"R{i + 1} (K/W)" for i in range(Rf.reshape(len(Rf), -1).shape[1])] +
                                   [f"τ{i + 1} (s)" for i in range(tf.reshape(len(tf), -1).shape[1])]))


def _grid(v, D, z0=0.0):
    """Field values at the nodes on a regular grid with the toolkit's map2D/map3D (as plotField)."""
    from src.py.general.map2D import map2D
    from src.py.general.map3D import map3D
    g = np.asarray(D['geo'], dtype=float)
    g = g - g.min(axis=0)
    x = np.arange(0, float(D['Lx']) + float(D['dx']) * 1e-6, float(D['dx']))
    y = np.arange(0, float(D['Ly']) + float(D['dy']) * 1e-6, float(D['dy']))
    if g.shape[1] == 2:
        return [map2D(v, g[:, 0], g[:, 1], x, y, 2), x, y]
    z = np.arange(0, float(D['Lz']) + float(D['dz']) * 1e-6, float(D['dz']))
    iz = int(np.argmin(np.abs(z - z0)))
    return [map3D(v, g[:, 0], g[:, 1], g[:, 2], x, y, z, 2)[..., iz], x, y]


def heat(F, x, y, title, unit, scale=SEQ, lim=None, mid=None):
    """Heatmap of a field (mm axes, equal aspect for compact domains)."""
    hm = go.Heatmap(z=F, x=x * 1e3, y=y * 1e3, colorscale=scale, colorbar=dict(title=unit, len=0.9),
                    zmin=None if lim is None else lim[0], zmax=None if lim is None else lim[1], zmid=mid,
                    hovertemplate='x %{x:.1f} mm<br>y %{y:.1f} mm<br>%{z:.3g} ' + unit + '<extra></extra>')
    fig = go.Figure(hm)
    r = (y[-1] - y[0]) / max(x[-1] - x[0], 1e-12)
    fig.update_layout(title=title, xaxis_title='x (mm)', yaxis_title='y (mm)',
                      yaxis=dict(scaleanchor='x') if 0.2 < r < 5 else {})
    return fig


def pod_view(data, setup, key):
    tag = _tag(data)
    m = data['mdls'].get(tag, {})
    g, p = data['grt'][tag], data['pred'][tag]
    D = g.get('Data') or {}
    z0 = float((setup or {}).get('Dat', {}).get('gen', {}).get('inpZ', 0) or 0)
    lam = np.asarray(m.get('lam', []), dtype=float)
    K = int(m.get('K', 0) or 0)
    if len(lam):
        E = np.cumsum(lam ** 2) / np.sum(lam ** 2)
        n = min(len(lam), max(20, 2 * K))
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = make_subplots(specs=[[{'secondary_y': True}]])
            fig.add_trace(go.Bar(x=np.arange(1, n + 1), y=lam[:n] ** 2 / np.sum(lam ** 2), name='energy per mode',
                                 marker_color=[color(0) if i < K else MUTED for i in range(n)]), secondary_y=False)
            fig.add_trace(line(np.arange(1, n + 1), E[:n] * 100, 'cumulative energy', 1, mode='lines+markers'),
                          secondary_y=True)
            fig.update_yaxes(type='log', title_text='σₖ² / Σσ²', secondary_y=False)
            fig.update_yaxes(title_text='cumulative (%)', secondary_y=True)
            fig.update_xaxes(title_text='mode k')
            show(fig, height=340, key=f"{key}|energy", info="**POD energy.** Bars: share of the snapshot variance "
                 "captured by mode k (log scale, coloured = used by the model). Line: cumulative share. A fast drop "
                 "of the bars means few modes describe the field — the reason POD works for heat conduction.")
        with c2:
            st.metric("Modes K", K, help="Number of POD modes of the reduced model.")
            st.metric("Energy captured", f"{100 * E[K - 1]:.4f} %" if K else '–',
                      help="Σσ²(1…K)/Σσ² of the training snapshots.")
            st.metric("Reduction", f"{len(np.asarray(g['y'])[0])} → {K}",
                      help="Nodes of the field → states of the reduced model.")
            if m.get('protocol') is not None:
                st.caption(f"Protocol {m['protocol']}" + (f", cond(C)+cond(G) = {fmt(m['matCond'])}"
                                                          if m.get('matCond') is not None else ''))
    if not D or 'geo' not in D:
        return
    from gui.core.field_explore import zslice, profiles, probes
    zs = zslice(D['geo'], key, z0)
    z0 = z0 if zs is None else zs

    # Mode shapes
    Phi = m.get('rPhi')
    if Phi is not None:
        Phi = np.asarray(Phi, dtype=float)
        k = st.slider("Mode shape φₖ", 1, Phi.shape[1], 1, key=f"{key}|k")
        [F, x, y] = _grid(Phi[:, k - 1], D, z0)
        a = float(np.nanmax(np.abs(F))) or 1.0
        show(heat(F[0] if F.ndim == 3 else F, x, y, f"Mode {k}", '-', DIV, [-a, a], 0), height=360,
             key=f"{key}|phi", info="**Spatial POD mode φₖ.** Mode 1 is the dominant heating pattern; higher modes "
             "describe how the heat spreads in time (sign changes = heat moving from one region to another). The "
             "field is T(x, t) ≈ T̄(x) + Σ aₖ(t)·φₖ(x).")

    # Fields
    yT, yP = np.asarray(g['y'], dtype=float), np.asarray(p['y'], dtype=float)
    Ts = float(g.get('Ts', 1))
    it = st.slider("Time step", 0, len(yT) - 1, len(yT) - 1, key=f"{key}|t",
                   help="Time of the field below (testing data).")
    [F, x, y] = _grid(np.vstack([yT[it], yP[it]]), D, z0)
    lim = [float(np.nanmin(F)), float(np.nanmax(F))]
    E = F[0] - F[1]
    e = float(np.nanmax(np.abs(E))) or 1e-9
    c = st.columns(3)
    for col, (Fi, ttl, u, sc, l, md) in zip(c, ((F[0], 'Measured', '°C', TEMP, lim, None),
                                                (F[1], 'Predicted', '°C', TEMP, lim, None),
                                                (E, 'Error T − T̂', 'K', DIV, [-e, e], 0))):
        with col:
            show(heat(Fi, x, y, f"{ttl} · t = {it * Ts:.0f} s", u, sc, l, md), height=340, key=f"{key}|f{ttl[0]}",
                 info="**Temperature field.** Measured and predicted share one colour scale; the error uses a "
                      "symmetric scale (red: model too cold, blue: too warm). Move the time slider to follow "
                      "the transient.")
    tabs = st.tabs(["Line profiles", "Probes over time", "Error over time"])
    with tabs[0]:
        profiles([('Measured', F[0], False), ('Predicted', F[1], True)], x, y, f"{key}|prof")
    with tabs[1]:
        probes(D['geo'], [('measured', yT, False), ('predicted', yP, True)], np.arange(len(yT)) * Ts,
               f"{key}|probe", z=z0, mark=it * Ts)
    with tabs[2]:
        mae = np.mean(np.abs(yT - yP), axis=1)
        fig = go.Figure(line(np.arange(len(mae)) * Ts, mae, 'MAE all nodes', 0))
        fig.add_vline(x=it * Ts, line=dict(color=INK2, dash='dot'))
        fig.update_layout(xaxis_title='Time (s)', yaxis_title='MAE over all nodes (K)', showlegend=False)
        show(fig, height=260, key=f"{key}|mae", info="**Mean absolute error over all nodes per time step.** Peaks "
             "follow load changes; a growing error means the reduced model drifts away (missing heat sink, wrong "
             "steady state).")


def convergence(info, key):
    """Training history with the learning rate (right axis): DL per iteration, PINN per epoch."""
    loss = np.asarray(info['loss'], dtype=float)
    pinn = 'dT' in info and 'valWin' in info                            # PINN V2.1: validation as in testing
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    if pinn:
        ep = np.arange(1, len(loss) + 1)
        dT2 = float(info['dT']) ** 2
        fig.add_trace(line(ep, loss, 'L (training, weighted)', 0), secondary_y=False)
        fig.add_trace(line(ep, info['lossData'], 'L_data (training windows)', 4, dash='dash'), secondary_y=False)
        fig.add_trace(line(ep, np.asarray(info['valLoss'], dtype=float) / dT2, 'validation as testing (MSE/ΔT²)', 1,
                           mode='lines+markers'), secondary_y=False)
        fig.add_trace(line(ep, np.asarray(info['valWin'], dtype=float) / dT2,
                           'validation windows, measured start (MSE/ΔT²)', 1, dash='dot'), secondary_y=False)
        xl, xlr, lr = ep, ep, np.asarray(info.get('learnRate') or [], dtype=float)
    else:
        xl = np.asarray(info.get('iter') or np.arange(1, len(loss) + 1))
        fig.add_trace(line(xl, loss, 'training', 0), secondary_y=False)
        if info.get('valLoss'):
            vl = np.asarray(info['valLoss'], dtype=float)
            fig.add_trace(line(np.asarray(info.get('valIter') or np.arange(1, len(vl) + 1)), vl, 'validation', 1,
                               mode='lines+markers'), secondary_y=False)
        lr = np.asarray(info.get('learnRate') or [], dtype=float)
        xlr = np.asarray(info.get('lrIter') or np.arange(1, len(lr) + 1))
    if len(lr):
        fig.add_trace(line(xlr, lr, 'learning rate', 6, width=1.5, line_shape='hv', opacity=0.8),
                      secondary_y=True)
        fig.update_yaxes(type='log', title_text='learning rate', secondary_y=True, showgrid=False)
    fig.update_yaxes(type='log', title_text='loss (relative to ΔT²)' if pinn else 'loss (normalised)',
                     secondary_y=False)
    fig.update_xaxes(title_text='Epoch' if pinn or not info.get('iter') else 'Iteration')
    nDrop = int(np.sum(np.diff(lr) < 0)) if len(lr) > 1 else 0
    txt = ("**Convergence.** Left axis: losses (log), right axis: learning rate (steps). "
           + ("The learning rate is adaptive (lrAdapt = 1): it drops when the validation error stalls. "
              if info.get('lrAdapt') else "Fixed schedule (lrAdapt = 0): the learning rate drops every lrDropPr "
              "epochs. ") + "Training falling while validation rises → overfitting (early stopping keeps the best "
           "validation state). Both flat and high → model too small or learning rate too low.")
    if pinn:
        txt += (" The solid validation curve is computed exactly like the testing (every validation profile from "
                "its first measured sample); the dotted one gives every window its measured start temperature, as "
                "in the training — the gap between them is the error that builds up over a profile.")
    show(fig, height=360, key=f"{key}|loss", info=txt)
    if len(lr):
        cap = f"Learning rate {fmt(lr[0])} → {fmt(lr[-1])} ({nDrop} reduction(s)) over {len(loss)} " \
              f"{'epochs' if pinn or not info.get('iter') else 'iterations'}."
        if pinn and info.get('valLoss'):
            vl = np.asarray(info['valLoss'], dtype=float)
            b = int(np.argmin(vl))
            cap += f" Best validation RMSE (as testing) {fmt(float(np.sqrt(vl[b])))} K in epoch {b + 1}; " \
                   f"with measured window starts {fmt(float(np.sqrt(info['valWin'][b])))} K."
        st.caption(cap)


def dl_view(data, setup, key):
    tag = _tag(data)
    info = data['mdls'].get(tag, {}).get('info') or {}
    if not info.get('loss'):
        return
    convergence(info, key)
    if info.get('lossPhys'):
        ep = np.arange(1, len(info['lossData']) + 1)
        v2 = 'dT' in info                                                 # PINN V2: dimensionless losses
        tr = [line(ep, info['lossData'], 'L_data (training)', 0),
              line(ep, np.maximum(np.asarray(info['lossPhys'], dtype=float), 1e-30), 'L_phys (training)', 2,
                   yaxis=None if v2 else 'y2')]
        if v2 and info.get('valPhys'):
            tr.append(line(ep, np.maximum(np.asarray(info['valPhys'], dtype=float), 1e-30), 'L_phys (validation)', 2,
                           dash='dot'))
        if v2 and any(info.get('lossInit') or []):
            tr.append(line(ep, np.maximum(np.asarray(info['lossInit'], dtype=float), 1e-30), 'L_init (training)', 3))
        fig = go.Figure(tr)
        if v2:
            fig.update_layout(xaxis_title='Epoch', yaxis=dict(type='log', title='loss (relative to ΔT²)'))
        else:
            fig.update_layout(xaxis_title='Epoch', yaxis=dict(type='log', title='L_data (normalised K²)'),
                              yaxis2=dict(type='log', title='L_phys (W²)', overlaying='y', side='right'))
        show(fig, height=320, key=f"{key}|lossPI", info=f"**Data and physics loss** (unweighted, α = "
             f"{fmt(info.get('alpha', 1))}, β = {fmt(info.get('beta', 0))}, γ = {fmt(info.get('gamma', 0))}). "
             + (f"Both are relative to ΔT = {fmt(info['dT'])} K: L = 0.01 means an RMS of 10 % of ΔT. "
                if v2 else "") + "With β = 0 the physics loss is only monitored: it shows how well the purely "
             "data-trained network satisfies the thermal network.")
        ph = data['mdls'].get(tag, {}).get('phys') or {}
        if v2 and ph:
            lp = np.asarray(info['lossPhys'], dtype=float)
            ld = np.asarray(info['lossData'], dtype=float)
            st.caption(f"Physics: {ph.get('type')} ({ph.get('origin', '')}), {ph.get('nP', 0)} physics states, "
                       f"sources {', '.join(ph.get('src') or []) or '–'}; {info.get('nWin', '?')} training windows, "
                       f"warm-up {info.get('warm', 0)} samples. Last epoch: β·L_phys = "
                       f"{fmt(info['beta'] * lp[-1])} vs. "
                       f"α·L_data = {fmt(info['alpha'] * ld[-1])} — "
                       + ("the physics dominates the loss: the network follows the physics more than the data; "
                          "reduce β if the test error is high." if info['beta'] * lp[-1] > 3 * info['alpha'] * ld[-1]
                          else "the data dominate: the physics acts as a regulariser." if info['beta'] > 0 else
                          "β = 0: the physics has no influence."))
    if info.get('device'):
        st.caption(f"Trained on: {info['device']}")
    mdl = data['mdls'].get(tag, {})
    net = mdl.get('sys')
    body = getattr(net, 'body', net)
    if net is not None and hasattr(body, 'mu'):
        with st.expander("Architecture of the trained network", icon=':material/account_tree:'):
            try:
                from gui.core.theory_dl import arch_dot
                F = int(np.asarray(body.mu.cpu()).size)
                ph = None
                if isinstance(mdl.get('phys'), dict) and 'n' in mdl['phys']:
                    ph = dict(mdl['phys'])
                elif info.get('lossPhys') is not None and 'beta' in info:
                    ph = {'type': 'lumped', 'nP': 1, 'src': [], 'alpha': info.get('alpha', 1.0),
                          'beta': info.get('beta', 0.0)}
                st.graphviz_chart(arch_dot(body, int(mdl.get('W') or 100), F, None,
                                           list((setup or {}).get('Run', {}).get('out') or []) or None, ph),
                                  width='content')
            except Exception as e:
                st.caption(f"Architecture not available: {e}")


#######################################################################################################################
# Render functions per mode
#######################################################################################################################
def _make(*parts):
    def render(data, setup, key):
        tiles(data, setup, key)
        if not _is_field(data['grt'][_tag(data)]):
            st.divider()
            response(data, setup, key)
        for f in parts:
            st.divider()
            f(data, setup, key)
    return render


RENDER = {
    'RC': _make(rc_view, scatter),
    'SS': _make(ss_view, scatter),
    'SF': _make(sf_view, scatter),
    'PO': _make(pod_view),
    'PS': _make(pod_view, ss_view),
    'ML': _make(scatter),
    'DL': _make(dl_view, scatter),
    'PI': _make(dl_view, scatter),
}
