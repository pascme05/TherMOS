#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         networks.py
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
Thermal networks: a Foster network (table, or the result of the thermal stack tool) is converted into the equivalent
Cauer ladder, its thermal impedance Z_th(t), the cumulative and differential structure function and the response to
a power pulse train (toolkit rcSim, the RC mode's simulator) are shown.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import sketch
from src.py.tools.thermStack import zth, foster_to_cauer
from src.py.model.RC.rcSim import rcSim

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|net"
DEFAULT = pd.DataFrame({'R [K/W]': [0.02, 0.05, 0.08, 0.10], 'tau [s]': [0.001, 0.02, 0.3, 5.0]})


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Thermal networks", "Foster and Cauer networks, thermal impedance and structure function — the "
                                     "building blocks of the RC and SF modes.")
    cm.question("What does a Foster network mean physically, how is it converted into a Cauer ladder and what does "
                "the structure function tell about the heat path?")

    # ------------------------------------------
    # Inputs
    # ------------------------------------------
    cm.section(1, "Foster network", "One row per RC pair (R_i, τ_i = R_i·C_i). The values of the RC mode (Rth, tau) "
                                    "or a datasheet Z_th,jc can be pasted here.", divider=False)
    th = cm.chain().get('thermal')
    src = 'table'
    if th and th.get('R'):
        src = st.segmented_control("Source", ['table', 'thermal stack tool'], default='table', key=f"{K}|src") \
            or 'table'
    if src == 'table':
        df = st.data_editor(DEFAULT, num_rows='dynamic', key=f"{K}|tab", hide_index=True)
        df = df.dropna()
        R, tau = df['R [K/W]'].to_numpy(float), df['tau [s]'].to_numpy(float)
    else:
        R, tau = np.asarray(th['R'], float), np.asarray(th['tau'], float)
        table(pd.DataFrame({'R [K/W]': R, 'tau [s]': tau}), hide_index=True)
    m = (R > 0) & (tau > 0)
    R, tau = R[m], tau[m]
    if len(R) == 0:
        st.info("Enter at least one RC pair with R > 0 and τ > 0.", icon=':material/info:')
        return
    o = np.argsort(tau)
    R, tau = R[o], tau[o]
    Rc, Cc = foster_to_cauer(R, tau)
    sketch.show(sketch.heat_path(len(R)), "Both networks describe the same thermal impedance between the "
                "junction and the ambient. Only the Cauer ladder can be mapped onto the layers of the package.")
    Rth = R.sum()

    # ------------------------------------------
    # Result
    # ------------------------------------------
    cm.section(None, "Result")
    cm.tiles([("R_th (steady state)", f"{fmt(Rth)} K/W", "Sum of all R_i = Z_th(t → ∞)."),
              ("C_th (total)", f"{fmt(Cc.sum())} Ws/K", "Sum of the Cauer capacitances."),
              ("Slowest τ", f"{fmt(tau.max())} s", "Time to reach 63 % of the last RC pair."),
              ("t(Z_th = 0.95 R_th)", f"{fmt(_t95(R, tau))} s", "Time to reach 95 % of the steady state.")])
    c = st.columns(2)
    with c[0]:
        table(pd.DataFrame({'Foster R_i (K/W)': R, 'τ_i (s)': tau, 'C_i = τ_i/R_i (Ws/K)': tau / R}),
              hide_index=True)
    with c[1]:
        table(pd.DataFrame({'Cauer R_k (K/W)': Rc, 'C_k (Ws/K)': Cc, 'ΣR (K/W)': np.cumsum(Rc),
                            'ΣC (Ws/K)': np.cumsum(Cc)}), hide_index=True)
    tabs = st.tabs(["Z_th(t)", "Structure function", "Pulse response"])
    t = np.logspace(np.log10(tau.min() / 20), np.log10(tau.max() * 20), 300)
    with tabs[0]:
        fig = go.Figure([line(t, zth(t, R, tau), 'Z_th(t)', 0)] +
                        [line(t, zth(t, R[i:i + 1], tau[i:i + 1]), f"RC pair {i + 1}", i + 1, dash='dot', width=1)
                         for i in range(len(R))])
        fig.update_layout(xaxis=dict(type='log', title='t (s)'), yaxis_title='Z_th (K/W)')
        show(fig, key=f"{K}|z")
    with tabs[1]:
        Rs, Cs = np.r_[0, np.cumsum(Rc)], np.r_[0, np.cumsum(Cc)]
        fig = go.Figure([go.Scatter(x=Rs, y=np.maximum(Cs, Cs[1] * 1e-2), mode='lines', line_shape='hv',
                                    name='cumulative C_Σ(R_Σ)', line=dict(color=SERIES[0], width=2))])
        dCdR = Cc / Rc
        fig.add_trace(go.Bar(x=np.cumsum(Rc) - Rc / 2, y=dCdR, width=Rc, name='differential dC/dR', yaxis='y2',
                             marker_color=SERIES[1], opacity=0.35))
        fig.update_layout(xaxis_title='R_Σ from the junction (K/W)', yaxis=dict(type='log', title='C_Σ (Ws/K)'),
                          yaxis2=dict(type='log', title='dC/dR (Ws/K²)', overlaying='y', side='right'))
        show(fig, key=f"{K}|sf")
        st.caption("Each step of the cumulative structure function is one Cauer element: a flat part (large R, small "
                   "C) is a layer of poor conduction (solder, TIM), a steep part (small R, large C) a good conductor "
                   "with much heat capacity (copper, heat sink). The vertical asymptote is R_th.")
    with tabs[2]:
        c = st.columns(3)
        P = cm.num(c[0], "Pulse power P (W)", 100.0, 'net|P', min_value=0.0)
        per = cm.num(c[1], "Period (s)", round(float(tau.max()), 3), 'net|per', min_value=1e-6)
        duty = cm.num(c[2], "Duty cycle", 0.5, 'net|duty', min_value=0.01, max_value=1.0)
        Ts = min(tau.min() / 5, per / 50)
        tt = np.arange(0, 5 * per + Ts / 2, Ts)
        Pv = np.where((tt % per) < duty * per, P, 0.0)
        T, _ = rcSim(R, tau, Pv, Ts)
        fig = go.Figure([line(tt, T, 'ΔT_j (rcSim)', 0), line(tt, np.full_like(tt, P * Rth * duty),
                                                                'mean loss × R_th', 2, dash='dash')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='ΔT_j (K)')
        show(fig, key=f"{K}|pr")
        st.caption(f"Swing in the last period: {fmt(T[-int(per / Ts):].max() - T[-int(per / Ts):].min())} K, "
                   f"peak rise {fmt(T.max())} K (steady DC: {fmt(P * Rth)} K).")

    cm.equations([
        ("Foster network", r"Z_{th}(t) = \sum_{i=1}^{n} R_i\left(1 - e^{-t/\tau_i}\right),\quad \tau_i = R_i C_i",
         "Parallel RC pairs in series: a mathematical fit, the nodes have no physical meaning."),
        ("Equivalence", r"Z(s) = \sum_i \frac{R_i}{1 + s\tau_i} = \cfrac{1}{sC_1 + \cfrac{1}{R_1 + \cfrac{1}{sC_2 + "
                        r"\dots}}}", "The Cauer ladder has the same impedance; its capacitances are grounded, so its "
                                     "nodes are physical temperatures along the heat path."),
        ("Structure function", r"C_\Sigma(R_\Sigma) = \sum_{k\le m} C_k \;\text{ at }\; R_\Sigma = \sum_{k\le m} R_k,"
                               r"\quad \frac{dC_\Sigma}{dR_\Sigma} = \frac{C_k}{R_k} \propto (k\,\rho c)\,A^2",
         "Cumulative capacitance over the cumulative resistance from the junction (JESD51-14)."),
    ], sym=[('R_i, τ_i, C_i', 'Foster resistance, time constant, capacitance of pair i', 'K/W, s, Ws/K'),
            ('R_k, C_k', 'Cauer resistance and capacitance of element k', 'K/W, Ws/K'),
            ('Z_th', 'thermal impedance (step response per W)', 'K/W'), ('A', 'cross-section of the heat flow', 'm²')])
    cm.refs(['JESD51-14', 'Wintrich2015'])
    i = int(np.argmax(R))
    cm.interpret([
        f"The network has a steady-state resistance of **{fmt(Rth)} K/W**: 1 W of loss heats the junction by "
        f"{fmt(Rth)} K once all capacitances are charged, which takes about {fmt(_t95(R, tau))} s (95 %).",
        f"The largest contribution is RC pair {i + 1} with R = {fmt(R[i])} K/W and τ = {fmt(tau[i])} s — this part "
        f"of the heat path dominates both the steady state and the temperature swing at periods near τ. Pulses much "
        f"shorter than the smallest τ ({fmt(tau.min())} s) are filtered; pulses much longer reach the full P·R_th.",
        "The Foster values are what the RC mode of TherMOS fits (Rth, tau); the SF mode extends them to a matrix of "
        "Foster networks for several heat sources. Only the Cauer form can be split into package and cooling, which "
        "is why the structure function is used to find the layer that degrades (e.g. solder voids)."])


def _t95(R, tau):
    t = np.logspace(np.log10(tau.min() / 100), np.log10(tau.max() * 100), 2000)
    z = zth(t, R, tau)
    return float(t[np.argmax(z >= 0.95 * R.sum())])
