#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         handcalc.py
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
Theory & Tools › Hand calculator: closed-form thermal estimates for the design of power electronics, one tab per
question (calculations in src/py/tools/handCalc.py):
    1) Layer stack → effective h      2) Contact resistance      3) Spreading            4) Convection h
    5) Radiation → h                  6) Heat sink fins          7) Thermal vias         8) Coolant flow
    9) Pulses, trains & time constants                          10) Skin & eddy currents (self-heating)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, STATUS, MUTED, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import hand_art as art
from src.py.tools import handCalc as hc
from src.py.tools import toolData as td

#######################################################################################################################
# Settings
#######################################################################################################################
K = 'hc'                                                                   # key prefix (cm.num adds 'tl|')
MM, UM, MM2 = 1e-3, 1e-6, 1e-6
RPP = 1e6                                                                  # m² K/W → K mm²/W
TABS = ["Layer stack → effective h", "Contact resistance", "Spreading", "Convection h", "Radiation → h",
        "Heat sink fins", "Thermal vias", "Coolant flow", "Pulses, trains & τ", "Skin & eddy currents"]
MATS = list(td.MATERIALS)
STACK_COLS = {
    'layer': "Name of the layer (free text).",
    'material': "Material of the library; fills k when the k column is empty.",
    't [mm]': "Thickness of the layer in the direction of the heat flow (e.g. 0.05 mm for a grease bond line).",
    'k [W/mK]': "Thermal conductivity; empty = value of the material (use the value of your datasheet for TIMs).",
    "R''c [K·mm²/W]": "Contact (boundary) resistance on the hot side of this layer, i.e. between this layer and the "
                      "one above. 0 = perfect contact. Typical: 1–5 for grease/pad boundaries, 20–500 for a dry "
                      "metal joint (see tab Contact resistance).",
}
STACK_DEF = pd.DataFrame([
    ('Cu baseplate', 'Cu', 3.0, np.nan, 0.0),
    ('Thermal grease', 'Thermal grease', 0.05, np.nan, 2.0),
    ('Cold plate wall', 'Al die-cast (A380)', 3.0, np.nan, 2.0),
], columns=list(STACK_COLS))
COOL = {k: v for k, v in td.COOLING.items() if v}                          # presets with h > 0
FOSTER_COLS = {'R_i [K/W]': "Thermal resistance of Foster term i (datasheet Z_th table or tab Thermal networks). The "
                            "sum is R_th of the path; include the heat sink for junction → ambient.",
               'τ_i [s]': "Time constant of Foster term i (τ_i = R_i C_i). Chip µs–ms, module ms–0.1 s, heat sink "
                          "10–100 s."}
FOSTER_DEF = pd.DataFrame([(0.004, 1e-4), (0.016, 2e-3), (0.035, 0.02), (0.025, 0.1), (0.04, 8.0)],
                          columns=list(FOSTER_COLS))                       # module 0.08 K/W + heat sink 0.04 K/W


def _mat(c, label, default, key, help=None):
    """Material select box; returns (name, k, rho, cp)."""
    name = c.selectbox(label, MATS, index=MATS.index(default), key=f"tl|{K}|{key}", help=help)
    m = td.MATERIALS[name]
    return name, m[0], m[1], m[2]


def _head(n, title, text):
    """Step heading with its text on the left and a placeholder for the sketch of the tab on the right."""
    c = st.columns([3, 2], vertical_alignment='center')
    with c[0]:
        cm.section(n, title, text, divider=False)
    return c[1]


def _pic(slot, svg):
    slot.markdown(svg, unsafe_allow_html=True)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Hand calculator", "Closed-form estimates for the thermal design of power electronics: effective "
                                    "heat transfer coefficients of layer stacks, contact and spreading resistances, "
                                    "convection and radiation, fins, vias, coolant flow, pulses and pulse trains, "
                                    "skin and eddy current losses.")
    cm.question("How large is the temperature drop over a layer, a joint, a spreader, a heat sink or the coolant — "
                "before a 3D model is built?", "Every tab answers one question with textbook correlations. Use the "
                "results as inputs of the Thermal stack, the Board designer (cooling h) or the modes, and as a "
                "plausibility check of simulations.")
    tabs = st.tabs(TABS)
    for t, f in zip(tabs, [_stack, _contact, _spreading, _convection, _radiation, _fins, _vias, _coolant, _pulse,
                           _skin]):
        with t:
            f()


#######################################################################################################################
# 1) Layer stack
#######################################################################################################################
def _stack():
    pic = _head(1, "Layers from the heat source to the cooled surface", "One row per layer (top = hot side). Each row "
               "can carry the contact resistance of the interface above it. The heat flows through all layers on "
               "the same area (1D) — see the tab Spreading when the source is smaller than the layers.")
    cfg = {c: st.column_config.NumberColumn(c, help=h, min_value=0.0, format='%g') for c, h in STACK_COLS.items()}
    cfg['layer'] = st.column_config.TextColumn('layer', help=STACK_COLS['layer'])
    cfg['material'] = st.column_config.SelectboxColumn('material', options=MATS, required=True,
                                                       help=STACK_COLS['material'], width='medium')
    df = st.data_editor(STACK_DEF, num_rows='dynamic', hide_index=True, column_config=cfg, key=f"{K}|st|tab",
                        width='stretch')
    layers = []
    for _, r in df.iterrows():
        if r['material'] not in td.MATERIALS or pd.isna(r['t [mm]']):
            continue
        k = float(r['k [W/mK]']) if pd.notna(r['k [W/mK]']) and r['k [W/mK]'] > 0 else td.MATERIALS[r['material']][0]
        Rc = float(r["R''c [K·mm²/W]"]) if pd.notna(r["R''c [K·mm²/W]"]) else 0.0
        layers.append({'name': str(r['layer']) if pd.notna(r['layer']) and r['layer'] else r['material'],
                       't': float(r['t [mm]']) * MM, 'k': k, 'Rc': Rc / RPP, 'mat': r['material']})
    if not layers:
        st.info("Enter at least one layer.", icon=':material/info:')
        return

    cm.section(2, "Cooling of the far side and load")
    c = st.columns(4)
    opts = list(COOL) + ['Manual h', 'Fixed temperature (ideal sink)']
    ch = cm.chain().get('h_conv')
    if ch:
        opts.insert(0, f"From Convection / Radiation ({fmt(ch['h'])} W/m²K)")
    pre = c[0].selectbox("Far side", opts, index=opts.index('Liquid cold plate (via grease)') if not ch else 0,
                         key=f"{K}|st|pre", help="Boundary on the far side of the last layer: a heat transfer "
                         "coefficient h to the fluid at T_ref, or a surface held at T_ref (h → ∞, e.g. the "
                         "datasheet case temperature).")
    if pre.startswith('Fixed'):
        h = None
        c[1].markdown("h → ∞")
    else:
        h0 = ch['h'] if pre.startswith('From') else COOL.get(pre, 1000.0)
        h = cm.num(c[1], "h (W/m²K)", h0, f"{K}|st|h|{pre}", min_value=1e-3,
                   help=f"Heat transfer coefficient to the fluid. Typical range of the preset: "
                        f"{td.COOLING_RANGE.get(pre, '–')} W/m²K.")
    a = cm.num(c[2], "Heat flux area w × l (mm²)", 900.0, f"{K}|st|A", min_value=1e-3,
               help="Area through which the heat flows (e.g. the baseplate footprint 30 × 30 mm = 900 mm²).")
    P = cm.num(c[3], "Power P (W)", 200.0, f"{K}|st|P", min_value=0.0, help="Heat flowing through the stack.")
    T0 = cm.num(c[0], "T_ref (°C)", 40.0, f"{K}|st|T0", help="Fluid temperature (or the fixed far-side temperature).")
    res = hc.stack(layers, h, a * MM2, P)
    cm.chain()['h_eff'] = res['h_eff']
    _pic(pic, art.stack(layers, h))

    cm.section(None, "Result")
    cm.tiles([("h_eff", f"{fmt(res['h_eff'])} W/m²K", "Effective heat transfer coefficient from the hot side of the "
               "first layer to the fluid: 1/R''_tot. Use it as the cooling h of a model that does not resolve these "
               "layers (Board designer, Thermal stack)."),
              ("R''_tot", f"{fmt(res['R_tot'] * RPP)} K·mm²/W", "Area-specific resistance (independent of the "
               "area) = sum of all layers, contacts and 1/h."),
              ("R_th", f"{fmt(res['R_th'])} K/W", "Resistance of the area A: R''_tot / A."),
              ("T hot side", f"{fmt(T0 + res['dT'])} °C", f"T_ref + q''·R''_tot with q'' = P/A = "
               f"{fmt(res['q'] / 1e4)} W/cm².")])
    it = pd.DataFrame(res['items'], columns=['element', 'kind', 'R', 't'])
    it["R'' [K·mm²/W]"] = it['R'] * RPP
    it['share [%]'] = 100 * it['R'] / res['R_tot']
    it['ΔT [K]'] = res['q'] * it['R']
    tabs = st.tabs(["Where is the temperature lost?", "Temperature through the stack", "Table", "Unit converter"])
    with tabs[0]:
        col = {'layer': SERIES[0], 'contact': SERIES[1], 'convection': SERIES[2]}
        fig = go.Figure(go.Bar(y=it['element'], x=it['ΔT [K]'], orientation='h',
                               marker_color=[col[k] for k in it['kind']],
                               text=[f"{fmt(v)} K ({s:.0f} %)" for v, s in zip(it['ΔT [K]'], it['share [%]'])],
                               textposition='auto', hovertemplate='%{y}: %{x:.3g} K<extra></extra>'))
        fig.update_layout(xaxis_title='temperature drop ΔT (K)', yaxis=dict(autorange='reversed'))
        show(fig, height=120 + 36 * len(it), key=f"{K}|st|bar", info="Temperature drop over every element at the "
             "power P (blue: layer t/k, orange: contact R''c, green: convection 1/h). The longest bar is the "
             "element to improve first.")
    with tabs[1]:
        z, T = hc.profile(layers, res['q'], h)
        fig = go.Figure(line(z / MM, T0 + T, 'T(z)', 0, mode='lines+markers'))
        x0 = 0.0
        for x in layers:
            fig.add_vrect(x0=x0 / MM, x1=(x0 + x['t']) / MM, line_width=0, fillcolor=MUTED,
                          opacity=0.08 if layers.index(x) % 2 else 0.16, annotation_text=x['name'],
                          annotation_position='top left', annotation_font_size=11)
            x0 += x['t']
        fig.update_layout(xaxis_title='depth z from the hot side (mm)', yaxis_title='T (°C)', showlegend=False)
        show(fig, key=f"{K}|st|prof", info="Temperature along the heat path. A sloped segment is conduction "
             "through a layer (slope q''/k), a vertical jump at a boundary is a contact resistance, the last jump "
             "is the convection film 1/h down to T_ref.")
    with tabs[2]:
        table(it[['element', 'kind', "R'' [K·mm²/W]", 'share [%]', 'ΔT [K]']], hide_index=True)
        st.caption("R'': area-specific resistance of the element · share: fraction of R''_tot · ΔT: temperature "
                   "drop at q'' = P/A.")
    with tabs[3]:
        _converter()
    cm.equations([
        ("Area-specific resistances", r"R''_i = \frac{t_i}{k_i},\qquad R''_{c,i},\qquad R''_{conv} = \frac{1}{h}",
         "Layer i conducts like a resistor of thickness over conductivity; contacts and the convection film add "
         "their own R''."),
        ("Series connection", r"R''_{tot} = \sum_i \left(R''_{c,i} + \frac{t_i}{k_i}\right) + \frac{1}{h},\qquad "
                              r"h_{eff} = \frac{1}{R''_{tot}}", "The effective h replaces the whole stack by one "
                                                                "boundary condition on the hot side."),
        ("Temperatures", r"R_{th} = \frac{R''_{tot}}{A},\quad q'' = \frac{P}{A},\quad "
                         r"T_{hot} = T_{ref} + q''\,R''_{tot}", None),
    ], sym=[("t_i", "thickness of layer i (table)", "m"), ("k_i", "thermal conductivity of layer i (table/material)",
                                                            "W/(m K)"),
            ("R''_c,i", "contact resistance on the hot side of layer i (table)", "m² K/W (shown in K mm²/W)"),
            ("h", "heat transfer coefficient on the far side (input)", "W/(m² K)"),
            ("h_eff", "effective heat transfer coefficient of the stack incl. h (result)", "W/(m² K)"),
            ("A", "area of the heat flux (input)", "m²"), ("P", "power through the stack (input)", "W"),
            ("q''", "heat flux density", "W/m²"), ("T_ref", "fluid temperature (input)", "°C")])
    cm.refs(['Incropera2007', 'Wintrich2015'])
    i = int(it['share [%]'].idxmax())
    pr = [f"The stack conducts like a single boundary with **h_eff = {fmt(res['h_eff'])} W/m²K**: at "
          f"{fmt(res['q'] / 1e4)} W/cm² the hot side is {fmt(res['dT'])} K above T_ref ({fmt(T0 + res['dT'])} °C)."]
    if h:
        pr.append(f"The layers reduce the cooling coefficient from h = {fmt(h)} to {fmt(res['h_eff'])} W/m²K "
                  f"({100 * (1 - res['h_eff'] / h):.0f} % less): {100 * (1 - 1 / (h * res['R_tot'])):.0f} % of "
                  f"the drop happens in the layers and contacts, the rest in the fluid film.")
    pr.append(f"The largest contribution is **{it.at[i, 'element']}** with {it.at[i, 'share [%]']:.0f} % "
              f"({fmt(it.at[i, 'ΔT [K]'])} K). " + {
                  'layer': "A thinner layer or a material with higher k acts most here.",
                  'contact': "A higher clamping pressure, flatter surfaces or a TIM (tab Contact resistance) act "
                             "most here.",
                  'convection': "The fluid film dominates: more flow, fins or a liquid (tabs Convection h, Heat sink "
                                "fins) act more than better layers."}[it.at[i, 'kind']])
    pr.append("Caveat: 1D — all layers carry the heat on the same area. A small chip on a large plate has an "
              "additional spreading resistance (tab Spreading), which this sum does not contain.")
    cm.interpret(pr)


def _converter():
    c = st.columns(2)
    v = cm.num(c[0], "Value", 1.0, f"{K}|cv|v")
    u = c[1].selectbox("Unit", ["K·mm²/W", "K·cm²/W", "K·in²/W", "m²K/W", "W/m²K (h, as conductance)"],
                       key=f"{K}|cv|u")
    f = {"K·mm²/W": 1e-6, "K·cm²/W": 1e-4, "K·in²/W": 6.4516e-4, "m²K/W": 1.0}
    R = 1 / v if u.startswith('W/') and v > 0 else v * f.get(u, 1.0)
    table(pd.DataFrame({'unit': ["m²K/W", "K·mm²/W", "K·cm²/W", "K·in²/W", "h = 1/R'' (W/m²K)"],
                        'value': [R, R * 1e6, R * 1e4, R / 6.4516e-4, 1 / R if R > 0 else np.inf]}), hide_index=True)
    st.caption("TIM datasheets give the thermal impedance in K·cm²/W or K·in²/W; 1 K·cm²/W = 100 K·mm²/W = "
               "1e-4 m²K/W ≙ h = 10,000 W/m²K.")


#######################################################################################################################
# 2) Contact resistance
#######################################################################################################################
def _contact():
    pic = _head(1, "Two surfaces pressed together", "Two nominally flat, rough metal surfaces touch only at the "
               "asperities (a few % of the area). The heat flows through these spots and through the gas or "
               "fill in the gaps between them.")
    c = st.columns(4)
    mats, fins, gaps = list(hc.CONTACT_MATERIALS), list(hc.FINISHES), list(hc.GAPS)
    m1 = c[0].selectbox("Surface 1", mats, index=mats.index('Cu'), key=f"{K}|ct|m1",
                        help="Material of surface 1 (e.g. the baseplate).")
    f1 = c[1].selectbox("Finish 1", fins, index=fins.index('Ground'), key=f"{K}|ct|f1",
                        help="Surface finish; sets the arithmetic roughness Ra (rms σ = 1.25 Ra).")
    m2 = c[2].selectbox("Surface 2", mats, index=mats.index('Al alloy'), key=f"{K}|ct|m2",
                        help="Material of surface 2 (e.g. the heat sink).")
    f2 = c[3].selectbox("Finish 2", fins, index=fins.index('Milled / extruded'), key=f"{K}|ct|f2")
    c = st.columns(4)
    P = cm.num(c[0], "Contact pressure p (MPa)", 0.5, f"{K}|ct|P", min_value=1e-4,
               help="Apparent pressure = clamping force / contact area. Screwed modules 0.1–1 MPa, spring clips "
                    "0.05–0.3 MPa, press-pack 10–20 MPa.")
    gap = c[1].selectbox("Gap medium", gaps, index=gaps.index('Air (1 atm)'), key=f"{K}|ct|gap",
                         help="What fills the voids between the asperities.")
    A = cm.num(c[2], "Contact area (mm²)", 900.0, f"{K}|ct|A", min_value=1e-3)
    Q = cm.num(c[3], "Power through the joint (W)", 200.0, f"{K}|ct|Q", min_value=0.0)
    k1, H1 = hc.CONTACT_MATERIALS[m1]
    k2, H2 = hc.CONTACT_MATERIALS[m2]
    s1, s2 = 1.25 * hc.FINISHES[f1] * UM, 1.25 * hc.FINISHES[f2] * UM
    Hc = min(H1, H2)
    kg, M = hc.GAPS[gap]
    mm = None
    if cm.expert():
        c = st.columns(5)
        s1 = cm.num(c[0], "σ₁ rms (µm)", s1 / UM, f"{K}|ct|s1|{f1}", min_value=1e-3) * UM
        s2 = cm.num(c[1], "σ₂ rms (µm)", s2 / UM, f"{K}|ct|s2|{f2}", min_value=1e-3) * UM
        Hc = cm.num(c[2], "H_c (GPa)", Hc / 1e9, f"{K}|ct|H|{m1}{m2}", min_value=1e-3,
                    help="Micro-hardness of the softer surface.") * 1e9
        kg = cm.num(c[3], "k_gap (W/mK)", kg, f"{K}|ct|kg|{gap}", min_value=0.0)
        M = cm.num(c[4], "M (µm)", M / UM, f"{K}|ct|M|{gap}", min_value=0.0,
                   help="Gas rarefaction parameter (≈ 0.26 µm air, 1.6 µm He at 1 atm; 0 for liquids).") * UM
    cm.hidden(5, "surface parameters (σ, H_c, k_gap, M)")
    r = hc.contact(P * 1e6, k1, k2, Hc, s1, s2, kg, M, mm)
    hj, Rj = float(r['h_j']), float(r['R_j'])
    _pic(pic, art.contact(m1, m2, gap, P))

    cm.section(None, "Result")
    cm.tiles([("h_j", f"{fmt(hj)} W/m²K", "Joint conductance = contact spots + gap."),
              ("R''_j", f"{fmt(Rj * RPP)} K·mm²/W", "Area-specific contact resistance; copy into the column R''c "
                                                    "of the layer stack."),
              ("R_j", f"{fmt(Rj / (A * MM2))} K/W", "Resistance of the contact area."),
              ("ΔT_j", f"{fmt(Q * Rj / (A * MM2))} K", "Temperature jump over the joint at the given power.")])
    ratio = float(r['ratio'])
    if not 1e-5 <= ratio <= 2e-2:
        st.warning(f"p/H_c = {ratio:.2g} lies outside the range of the correlation (1e-5 … 2e-2).",
                   icon=':material/warning:')
    tabs = st.tabs(["How does pressure help?", "Dry joint or TIM?", "Measured values"])
    p = np.logspace(-2, np.log10(30), 120)
    rr = hc.contact(p * 1e6, k1, k2, Hc, s1, s2, kg, M, mm)
    with tabs[0]:
        tr = [line(p, RPP / rr['h_j'], "joint R''_j", 0), line(p, RPP / rr['h_c'], "contact spots only 1/h_c", 1,
                                                                dash='dash')]
        if kg > 0:
            tr.append(line(p, RPP / rr['h_g'], "gap only 1/h_g", 2, dash='dot'))
        fig = go.Figure(tr + [go.Scatter(x=[P], y=[Rj * RPP], mode='markers', name='operating point',
                                         marker=dict(size=10, color=STATUS['critical']))])
        fig.update_layout(xaxis=dict(type='log', title='contact pressure p (MPa)'),
                          yaxis=dict(type='log', title="R'' (K·mm²/W)"))
        show(fig, key=f"{K}|ct|p", info="Contact resistance over the clamping pressure (log-log). The spot "
             "conductance grows almost linearly with p (slope −0.95), the gap conductance hardly depends on p. "
             "The joint is the parallel connection of both.")
        st.caption(f"At {fmt(P)} MPa the spots conduct {100 * float(r['h_c']) / hj:.0f} % and the gap "
                   f"{100 * float(r['h_g']) / hj:.0f} % of the heat; doubling the pressure lowers R''_j to "
                   f"{fmt(RPP / float(hc.contact(2e6 * P, k1, k2, Hc, s1, s2, kg, M, mm)['h_j']))} K·mm²/W.")
    with tabs[1]:
        c = st.columns(3)
        blt = cm.num(c[0], "Bond line BLT (µm)", 50.0, f"{K}|ct|blt", min_value=0.0,
                     help="Thickness of the TIM after assembly (grease 25–100 µm, PCM 25–50 µm, pads 100–3000 µm).")
        kt = cm.num(c[1], "k_TIM (W/mK)", 3.0, f"{K}|ct|kt", min_value=1e-3)
        rb = cm.num(c[2], "R''_boundary per side (K·mm²/W)", 2.0, f"{K}|ct|rb", min_value=0.0,
                    help="Boundary resistance between TIM and each surface (1–10 K·mm²/W).")
        Rt = hc.tim(blt * UM, kt, rb / RPP)
        fig = go.Figure(go.Bar(x=["dry joint (" + gap + ")", "with TIM"], y=[Rj * RPP, Rt * RPP],
                               marker_color=[SERIES[1], SERIES[0]], text=[fmt(Rj * RPP), fmt(Rt * RPP)]))
        fig.update_layout(yaxis_title="R'' (K·mm²/W)")
        show(fig, height=300, key=f"{K}|ct|tim", info="Area-specific resistance of the dry metal joint and of the "
             "same joint with a TIM (BLT/k_TIM + 2 boundary resistances).")
        st.caption(f"The TIM gives {fmt(Rt * RPP)} K·mm²/W ({fmt(blt / kt)} in the bond line, {fmt(2 * rb)} at "
                   f"the boundaries) — {'better' if Rt < Rj else 'worse'} than the dry joint by a factor "
                   f"{fmt(max(Rj, Rt) / min(Rj, Rt))}. A TIM also keeps the joint stable over pressure loss and "
                   f"bowing, which the dry joint does not.")
    with tabs[2]:
        table(pd.DataFrame(hc.INTERFACES, columns=['interface', "R'' (K·mm²/W)"]), hide_index=True)
        st.caption("Measured contact resistances (Incropera et al., Tables 3.1–3.2) for orientation.")
    cm.equations([
        ("Effective surface", r"k_s = \frac{2k_1k_2}{k_1+k_2},\quad \sigma=\sqrt{\sigma_1^2+\sigma_2^2},\quad "
                              r"m = 0.125\,(\sigma/\mu m)^{0.402}", "Two rough surfaces are replaced by one rough "
                                                                    "surface against a smooth flat."),
        ("Contact spots", r"h_c = 1.25\,k_s\,\frac{m}{\sigma}\left(\frac{p}{H_c}\right)^{0.95}",
         "Plastic deformation of the asperities: the real contact area grows with p/H_c."),
        ("Gap", r"Y = 1.53\,\sigma\left(\frac{p}{H_c}\right)^{-0.097},\qquad h_g = \frac{k_{gap}}{Y + M}",
         "Conduction through the gas or fill across the mean gap Y."),
        ("Joint", r"h_j = h_c + h_g,\qquad R''_j = 1/h_j,\qquad R''_{TIM} = \frac{BLT}{k_{TIM}} + 2R''_b", None),
    ], sym=[("k₁, k₂", "conductivities of the two solids (material)", "W/(m K)"),
            ("k_s", "harmonic mean conductivity", "W/(m K)"),
            ("σ₁, σ₂, σ", "rms roughness of the surfaces and combined (finish; σ = 1.25 Ra)", "m"),
            ("m", "mean absolute asperity slope (Antonetti correlation)", "–"),
            ("p", "apparent contact pressure (input)", "Pa"), ("H_c", "micro-hardness of the softer solid", "Pa"),
            ("Y", "mean plane separation (gap)", "m"), ("M", "gas rarefaction parameter (gap medium)", "m"),
            ("k_gap", "conductivity of the gap medium", "W/(m K)"),
            ("h_c, h_g, h_j", "conductance of the spots, the gap, the joint", "W/(m² K)"),
            ("BLT", "bond line thickness of the TIM", "m"), ("R''_b", "boundary resistance TIM/surface", "m² K/W")])
    st.caption("Literature: Mikic 1974; Yovanovich 1981; Antonetti et al. 1991; Incropera et al. 2007.")
    cm.interpret([
        f"The joint {m1} ({f1.lower()}) / {m2} ({f2.lower()}) at {fmt(P)} MPa has **R''_j = {fmt(Rj * RPP)} "
        f"K·mm²/W** (h_j = {fmt(hj)} W/m²K); over {fmt(A)} mm² this is {fmt(Rj / (A * MM2))} K/W and "
        f"{fmt(Q * Rj / (A * MM2))} K at {fmt(Q)} W.",
        f"Only p/H_c = {ratio:.1e} of the area touches. The roughness σ = {fmt(r['sigma'] / UM)} µm sets the gap "
        f"Y = {fmt(float(r['Y']) / UM)} µm; smoother surfaces and higher pressure both act, a softer material "
        f"(lower H_c) acts like more pressure.",
        "The model assumes flat surfaces. Real heat sinks are bowed (flatness 20–100 µm), so dry joints are usually "
        "worse than computed — this is why a TIM is used in practice."])


#######################################################################################################################
# 3) Spreading
#######################################################################################################################
def _spreading():
    pic = _head(1, "Small source on a larger plate", "A chip (source) heats the centre of a plate that is cooled by h "
               "on its back. The heat must spread sideways first, which adds the spreading resistance R_sp.")
    c = st.columns(4)
    a = cm.num(c[0], "Source w (mm)", 10.0, f"{K}|sp|a", min_value=1e-3, help="Chip or heat source edge length.")
    b = cm.num(c[1], "Source l (mm)", 10.0, f"{K}|sp|b", min_value=1e-3)
    W = cm.num(c[2], "Plate W (mm)", 50.0, f"{K}|sp|W", min_value=1e-3, help="Plate (spreader/baseplate) size.")
    L = cm.num(c[3], "Plate L (mm)", 50.0, f"{K}|sp|L", min_value=1e-3)
    c = st.columns(4)
    t = cm.num(c[0], "Thickness t (mm)", 3.0, f"{K}|sp|t", min_value=1e-3)
    name, k, _, _ = _mat(c[1], "Plate material", 'Cu', 'sp|m')
    h = cm.num(c[2], "h on the back (W/m²K)", 5000.0, f"{K}|sp|h", min_value=1e-3,
               help="Can be the h_eff of the layer stack below the plate.")
    P = cm.num(c[3], "Power P (W)", 200.0, f"{K}|sp|P", min_value=0.0)
    As, Ap = a * b * MM2, W * L * MM2
    if As >= Ap:
        st.info("The source must be smaller than the plate.", icon=':material/info:')
        return
    r = hc.spreading(As, Ap, t * MM, k, h)
    R45 = hc.spread_angle(a * MM, b * MM, t * MM, k)
    _pic(pic, art.spreading(a, W, t))
    cm.section(None, "Result")
    cm.tiles([("R_tot", f"{fmt(r['R_tot'])} K/W", "Mean source temperature above the fluid per W."),
              ("R_sp (spreading)", f"{fmt(r['R_sp'])} K/W", "Extra resistance because the heat enters on A_s only."),
              ("R_1D + R_conv", f"{fmt(r['R_1D'] + r['R_conv'])} K/W", "What a 1D estimate on the plate area gives."),
              ("ΔT source", f"{fmt(P * r['R_tot'])} K", "Mean temperature rise of the source at P.")])
    ts_ = np.logspace(-1, np.log10(30), 120)
    rs = [hc.spreading(As, Ap, x * MM, k, h) for x in ts_]
    tot = np.array([x['R_tot'] for x in rs])
    io = int(np.argmin(tot))
    fig = go.Figure([line(ts_, tot, 'R_tot', 0), line(ts_, [x['R_sp'] for x in rs], 'R_sp (spreading)', 1, dash='dash'),
                     line(ts_, [x['R_1D'] for x in rs], 'R_1D = t/(k A_p)', 2, dash='dot'),
                     line(ts_, [x['R_conv'] for x in rs], 'R_conv = 1/(h A_p)', 3, dash='dot'),
                     go.Scatter(x=[t], y=[r['R_tot']], mode='markers', name='current t',
                                marker=dict(size=10, color=STATUS['critical']))])
    fig.update_layout(xaxis=dict(type='log', title='plate thickness t (mm)'), yaxis=dict(type='log', title='R (K/W)'))
    show(fig, key=f"{K}|sp|t", info="Resistance parts over the plate thickness. A thicker plate spreads better (R_sp "
         "falls) but conducts worse through its thickness (R_1D grows): R_tot has a minimum. R_conv does not depend "
         "on t.")
    st.caption(f"Optimum thickness ≈ {fmt(ts_[io])} mm with R_tot = {fmt(tot[io])} K/W (current {fmt(t)} mm: "
               f"{fmt(r['R_tot'])} K/W). The 45° rule gives a conduction resistance of {fmt(R45)} K/W for the "
               f"current t (Lee: {fmt(r['R_1D'] + r['R_sp'])} K/W); it ignores h and the plate edge.")
    cm.equations([
        ("Geometry", r"a = \sqrt{A_s/\pi},\ b = \sqrt{A_p/\pi},\ \varepsilon = a/b,\ \tau = t/b,\ Bi = \frac{h\,b}{k}",
         "Equivalent circular source and plate."),
        ("Spreading (Lee et al. 1995)", [r"\lambda_c = \pi + \frac{1}{\sqrt{\pi}\,\varepsilon},\quad \Phi_c = "
                                         r"\frac{\tanh(\lambda_c\tau) + \lambda_c/Bi}{1 + \frac{\lambda_c}{Bi}"
                                         r"\tanh(\lambda_c\tau)}",
                                         r"\psi = \frac{\varepsilon\tau}{\sqrt{\pi}} + \tfrac12(1-\varepsilon)^{3/2}"
                                         r"\Phi_c,\quad R_{sp} + R_{1D} = \frac{\psi}{\sqrt{\pi}\,k\,a}"], None),
        ("Total", r"R_{tot} = R_{sp} + \frac{t}{k A_p} + \frac{1}{h A_p}", None),
    ], sym=[("A_s, A_p", "area of the source and of the plate (inputs)", "m²"), ("t", "plate thickness", "m"),
            ("k", "conductivity of the plate (material)", "W/(m K)"), ("h", "heat transfer on the back", "W/(m² K)"),
            ("a, b", "radii of the equivalent circles", "m"), ("ε, τ", "relative source size and thickness", "–"),
            ("Bi", "Biot number of the plate", "–"), ("ψ", "dimensionless constriction resistance", "–")])
    cm.refs(['Lee1995', 'Incropera2007'])
    cm.interpret([
        f"The source of {fmt(a)} × {fmt(b)} mm on the {fmt(W)} × {fmt(L)} × {fmt(t)} mm {name} plate sees "
        f"**R_tot = {fmt(r['R_tot'])} K/W**, of which {100 * r['R_sp'] / r['R_tot']:.0f} % is spreading. A 1D "
        f"estimate on the plate area would give {fmt(r['R_1D'] + r['R_conv'])} K/W and under-estimate ΔT by "
        f"{fmt(P * r['R_sp'])} K.",
        "Spreading is large when the source is small (ε → 0), the plate thin and the cooling strong (high Bi). "
        f"Levers: a thicker or better-conducting spreader up to ≈ {fmt(ts_[io])} mm, or a larger source "
        "(several chips side by side).",
        "Caveat: the Lee model gives the mean source temperature for a centred source; the hot spot in the middle "
        "is higher, edge sources spread worse."])


#######################################################################################################################
# 4) Convection
#######################################################################################################################
def _convection():
    case = st.segmented_control("Case", ["Forced air over a plate", "Natural convection + radiation",
                                         "Liquid in channels (cold plate)"], default="Liquid in channels (cold plate)",
                                key=f"{K}|cv|case") or "Liquid in channels (cold plate)"
    pic = _head(1, case, {'F': "Air flows along a heated plate; a boundary layer grows from the leading edge and "
                               "carries the heat away.",
                          'N': "Without a fan the warm air rises by buoyancy; the surface also radiates to the "
                               "surroundings.",
                          'L': "Coolant flows through parallel channels of a cold plate below the module."}[case[0]])
    nch = 10
    if case.startswith('Forced'):
        c = st.columns(4)
        u = cm.num(c[0], "Air velocity u (m/s)", 3.0, f"{K}|fa|u", min_value=1e-3,
                   help="Free-stream velocity along the plate (fans 1–6 m/s).")
        L = cm.num(c[1], "Plate length in flow L (mm)", 100.0, f"{K}|fa|L", min_value=1e-3)
        T = cm.num(c[2], "Air temperature (°C)", 40.0, f"{K}|fa|T")
        A = cm.num(c[3], "Surface area (cm²)", 100.0, f"{K}|fa|A", min_value=1e-3)
        p = hc.fluid('Air (1 atm)', T)
        r = hc.h_forced_plate(u, L * MM, p)
        h = r['h']
        cm.tiles([("h", f"{fmt(h)} W/m²K", "Average over the plate length."), ("Re", fmt(r['Re']), r['regime']),
                  ("Nu", fmt(r['Nu']), "Nusselt number h L/k."), ("R_conv", f"{fmt(1 / (h * A * 1e-4))} K/W",
                                                                  "1/(h A) of the surface area.")])
        uu = np.linspace(0.2, 10, 80)
        fig = go.Figure(line(uu, [hc.h_forced_plate(x, L * MM, p)['h'] for x in uu], 'h(u)', 0))
        fig.update_layout(xaxis_title='u (m/s)', yaxis_title='h (W/m²K)')
        show(fig, height=300, key=f"{K}|fa|c", info="Average h over the velocity; laminar flow grows with √u.")
        eqs = [("Forced flow over a plate", [r"Re = \frac{u L}{\nu},\quad \overline{Nu} = 0.664\,Re^{1/2}Pr^{1/3}"
                                             r"\ (Re\le 5\cdot10^5)", r"\overline{Nu} = (0.037\,Re^{4/5} - 871)\,"
                                                                      r"Pr^{1/3},\quad h = \overline{Nu}\,k/L"], None)]
        txt = (f"Forced air at {fmt(u)} m/s along {fmt(L)} mm gives **h = {fmt(h)} W/m²K** ({r['regime']}, "
               f"Re = {fmt(r['Re'])}). h grows with √u and falls with √L: short plates in the flow direction "
               "cool better. Fins multiply the area (tab Heat sink fins).")
    elif case.startswith('Natural'):
        c = st.columns(5)
        dT = cm.num(c[0], "Surface − air ΔT (K)", 40.0, f"{K}|nc|dT", min_value=1e-3)
        L = cm.num(c[1], "Height / A·perimeter⁻¹ (mm)", 100.0, f"{K}|nc|L", min_value=1e-3,
                   help="Vertical plate: height. Horizontal plate: area / perimeter.")
        Ti = cm.num(c[2], "Air temperature (°C)", 25.0, f"{K}|nc|T")
        o = c[3].selectbox("Orientation", ['vertical', 'horizontal, hot side up', 'horizontal, hot side down'],
                           key=f"{K}|nc|o")
        eps = cm.num(c[4], "Emissivity ε", 0.85, f"{K}|nc|e", min_value=0.0, max_value=1.0,
                     help="Anodised/painted 0.8–0.95, bare Al 0.05–0.1, FR4/solder mask ≈ 0.9.")
        p = hc.fluid('Air (1 atm)', Ti + dT / 2)
        r = hc.h_natural(dT, L * MM, p, Ti, o)
        hr = hc.h_radiation(eps, Ti + dT, Ti)
        h = r['h'] + hr
        cm.tiles([("h (conv + rad)", f"{fmt(h)} W/m²K", "Total, use as h of the surface."),
                  ("h_conv", f"{fmt(r['h'])} W/m²K", "Natural convection."), ("h_rad", f"{fmt(hr)} W/m²K",
                                                                                   "Linearised radiation."),
                  ("Ra", fmt(r['Ra']), "Rayleigh number (turbulent above ≈ 1e9).")])
        dd = np.linspace(2, 100, 80)
        fig = go.Figure([line(dd, [hc.h_natural(x, L * MM, hc.fluid('Air (1 atm)', Ti + x / 2), Ti, o)['h']
                                   for x in dd], 'h_conv', 0),
                         line(dd, [hc.h_radiation(eps, Ti + x, Ti) for x in dd], 'h_rad', 1, dash='dash')])
        fig.update_layout(xaxis_title='ΔT surface − air (K)', yaxis_title='h (W/m²K)')
        show(fig, height=300, key=f"{K}|nc|c", info="Natural convection and radiation over the surface "
             "overtemperature; both grow with ΔT, so the cooling is non-linear.")
        eqs = [("Natural convection", [r"Ra = \frac{g\beta\,\Delta T\,L^3}{\nu\alpha},\quad \beta = 1/T_{film}",
                                       r"\text{vertical: } \overline{Nu} = \left\{0.825 + \frac{0.387\,Ra^{1/6}}"
                                       r"{[1+(0.492/Pr)^{9/16}]^{8/27}}\right\}^2",
                                       r"\text{horizontal up: } 0.54Ra^{1/4}\ (0.15Ra^{1/3}),\ \text{down: } "
                                       r"0.52Ra^{1/5}"], None),
               ("Radiation", r"h_{rad} = \varepsilon\sigma_{SB}(T_s^2+T_\infty^2)(T_s+T_\infty)", "Temperatures in K.")]
        txt = (f"At {fmt(dT)} K overtemperature the surface reaches **h = {fmt(h)} W/m²K**, "
               f"{100 * hr / h:.0f} % of it by radiation — a high emissivity (anodised, painted) matters as much as "
               "the convection. Natural convection alone stays below ≈ 10 W/m²K; it is only enough for a few W per "
               "100 cm².")
    else:
        c = st.columns(4)
        fl = c[0].selectbox("Coolant", list(hc.FLUIDS)[1:], index=1, key=f"{K}|lc|f")
        T = cm.num(c[1], "Coolant temperature (°C)", 65.0, f"{K}|lc|T", help="Automotive 65 °C (WEG), industry "
                                                                              "25–40 °C (water).")
        Vl = cm.num(c[2], "Flow rate (L/min)", 8.0, f"{K}|lc|V", min_value=1e-3)
        n = cm.num(c[3], "Parallel channels n", 10, f"{K}|lc|n", min_value=1, format='%d')
        nch = n
        c = st.columns(4)
        w = cm.num(c[0], "Channel width (mm)", 3.0, f"{K}|lc|w", min_value=1e-3)
        hg = cm.num(c[1], "Channel height (mm)", 8.0, f"{K}|lc|hh", min_value=1e-3)
        L = cm.num(c[2], "Channel length (mm)", 150.0, f"{K}|lc|L", min_value=1e-3)
        Ab = cm.num(c[3], "Cooled base area (cm²)", 150.0, f"{K}|lc|Ab", min_value=1e-3,
                    help="Footprint of the cold plate; the effective h refers the wetted area to it.")
        p = hc.fluid(fl, T)
        r = hc.h_duct(Vl / 60000, w * MM, hg * MM, n, L * MM, p)
        h = r['h']
        h_b = h * r['A_w'] / (Ab * 1e-4)
        cm.tiles([("h (wetted wall)", f"{fmt(h)} W/m²K", "h on the channel walls."),
                  ("h on the base (≈)", f"{fmt(h_b)} W/m²K", "h × wetted area / base area (fin efficiency 1); upper "
                                                            "bound of the effective h of the cold plate."),
                  ("Re", f"{fmt(r['Re'])}", f"{r['regime']}; u = {fmt(r['u'])} m/s, D_h = {fmt(r['Dh'] / MM)} mm"),
                  ("Δp", f"{fmt(r['dp'] / 100)} mbar", "Friction pressure drop of the straight channels (without "
                                                      "inlet, bends, manifolds).")])
        VV = np.logspace(-0.5, 1.5, 60)
        rr = [hc.h_duct(x / 60000, w * MM, hg * MM, n, L * MM, p) for x in VV]
        fig = go.Figure([line(VV, [x['h'] for x in rr], 'h (W/m²K)', 0),
                         line(VV, [x['dp'] / 100 for x in rr], 'Δp (mbar)', 1, dash='dash', yaxis='y2')])
        fig.update_layout(xaxis=dict(type='log', title='flow rate (L/min)'), yaxis=dict(title='h (W/m²K)'),
                          yaxis2=dict(title='Δp (mbar)', overlaying='y', side='right', type='log'))
        show(fig, height=320, key=f"{K}|lc|c", info="Heat transfer and pressure drop over the flow rate. The jump "
             "of h is the laminar → turbulent transition (Re 2300–3000); Δp grows with u² in turbulent flow.")
        eqs = [("Duct flow", [r"D_h = \frac{4A}{P_w},\ u = \frac{\dot V}{nA},\ Re = \frac{u D_h}{\nu},\ "
                              r"Gz = Re\,Pr\,\frac{D_h}{L}",
                              r"\text{laminar: } Nu = 3.66 + \frac{0.0668\,Gz}{1+0.04\,Gz^{2/3}},\ f = 64/Re",
                              r"\text{turbulent: } f = (0.790\ln Re - 1.64)^{-2},\ Nu = \frac{(f/8)(Re-1000)Pr}"
                              r"{1+12.7\sqrt{f/8}(Pr^{2/3}-1)}",
                              r"h = Nu\,k/D_h,\quad \Delta p = f\frac{L}{D_h}\frac{\rho u^2}{2}"], None)]
        txt = (f"{fmt(Vl)} L/min {fl} at {fmt(T)} °C through {n} channels of {fmt(w)} × {fmt(hg)} mm flows "
               f"{r['regime']} (Re = {fmt(r['Re'])}) and gives **h = {fmt(h)} W/m²K** on the walls, ≈ "
               f"{fmt(h_b)} W/m²K referred to the base, at Δp = {fmt(r['dp'] / 100)} mbar. In laminar flow h hardly "
               "grows with the flow — narrower channels (smaller D_h) act more; turbulent flow raises h at a steep "
               "price in Δp.")
    _pic(pic, art.convection(case, nch))
    st.button("Use this h in the layer stack", icon=':material/arrow_upward:', key=f"{K}|cv|use",
              on_click=_use_h, args=(float(h), case), help="Sets the far-side h of the tab Layer stack to this "
                                                            "value.")
    cm.equations(eqs, sym=[("u", "velocity", "m/s"), ("L", "length in flow direction / height / channel length",
                                                       "m"),
                           ("ν, α, k, Pr", "kinematic viscosity, diffusivity, conductivity, Prandtl number of the "
                                           "fluid (property table at the film temperature)", "m²/s, m²/s, W/(m K), –"),
                           ("Re, Nu, Ra, Gz", "Reynolds, Nusselt, Rayleigh, Graetz numbers", "–"),
                           ("β", "volumetric expansion coefficient", "1/K"), ("ε", "emissivity", "–"),
                           ("D_h, A, P_w", "hydraulic diameter, cross-section, wetted perimeter of a channel",
                            "m, m², m"), ("f", "Darcy friction factor", "–"), ("Δp", "pressure drop", "Pa")])
    cm.refs(['Incropera2007'])
    st.caption("Correlations: Churchill & Chu 1975 (natural, vertical), Hausen 1943 (laminar developing), "
               "Gnielinski 1976 (turbulent).")
    cm.interpret([txt, "Caveat: the correlations hold for clean geometries (±20 %). Real cold plates with "
                       "turbulators or pin fins, and heat sinks in a housing, need the data of the manufacturer or "
                       "a CFD run."])


def _use_h(h, case):
    cm.chain()['h_conv'] = {'h': h, 'case': case}
    st.session_state[f"{K}|st|pre"] = f"From Convection / Radiation ({fmt(h)} W/m²K)"
    st.toast(f"h = {fmt(h)} W/m²K is used in the tab Layer stack", icon=':material/check:')


#######################################################################################################################
# 4b) Radiation as an equivalent convection coefficient
#######################################################################################################################
def _radiation():
    case = st.segmented_control("Geometry", ["Small body in large surroundings", "Two parallel plates",
                                             "Custom (view factor, areas)"], default="Small body in large surroundings",
                                key=f"{K}|ra|case") or "Small body in large surroundings"
    pic = _head(1, "Radiation → equivalent h", "Radiation grows with T⁴, convection with ΔT. Writing the radiation "
               "as q'' = h_rad (T₁ − T₂) gives one coefficient that simply adds to the convection h: h_tot = h_conv + "
               "h_rad. Every tool that takes an h (Layer stack, Board designer, heat sink) can then include radiation.")
    em = list(hc.EMISSIVITY)
    c = st.columns(4)
    s1 = c[0].selectbox("Surface 1 (hot)", em, index=em.index('Black anodised Al'), key=f"{K}|ra|s1",
                        help="Sets ε₁; choose 'Manual' to type it.")
    e1 = cm.num(c[1], "ε₁", hc.EMISSIVITY[s1] or 0.9, f"{K}|ra|e1|{s1}", min_value=0.01, max_value=1.0,
                help="Emissivity of the hot surface (0 = mirror, 1 = black body). Metals: bare 0.03–0.1, oxidised "
                     "0.3–0.6; anodised, painted, solder mask, ceramics 0.8–0.95.")
    T1 = cm.num(c[2], "T₁ hot surface (°C)", 80.0, f"{K}|ra|T1")
    T2 = cm.num(c[3], "T₂ surroundings (°C)", 40.0, f"{K}|ra|T2", help="Temperature of the walls the surface sees "
                                                                      "(housing, sky ≈ air − 10 K, other boards).")
    e2, F, Ar = 1.0, 1.0, 0.0
    if not case.startswith('Small'):
        c = st.columns(4)
        s2 = c[0].selectbox("Surface 2 (cold)", em, index=em.index('FR4 / solder mask'), key=f"{K}|ra|s2")
        e2 = cm.num(c[1], "ε₂", hc.EMISSIVITY[s2] or 0.9, f"{K}|ra|e2|{s2}", min_value=0.01, max_value=1.0,
                    help="Emissivity of the cold surface.")
        if case.startswith('Custom'):
            F = cm.num(c[2], "View factor F₁₂", 0.5, f"{K}|ra|F", min_value=1e-3, max_value=1.0,
                       help="Share of the radiation leaving 1 that hits 2 (1 = surface 1 sees only 2; fins facing "
                            "each other ≈ 0.2–0.5 to the outside).")
            Ar = cm.num(c[3], "Area ratio A₁/A₂", 1.0, f"{K}|ra|Ar", min_value=0.0,
                        help="0 = surface 2 is very large (enclosure), 1 = equal areas (parallel plates).")
        else:
            Ar = 1.0
    hconv0 = (cm.chain().get('h_conv') or {}).get('h', 5.0)
    c = st.columns(4)
    hcv = cm.num(c[0], "h_conv to add (W/m²K)", hconv0, f"{K}|ra|hc", min_value=0.0,
                 help="Convection on the same surface (default: the h of the Convection tab if it was used, else "
                      "natural convection ≈ 5 W/m²K).")
    r = hc.radiation(e1, T1, T2, e2, F, Ar)
    h_r, h_t = float(r['h_rad']), float(r['h_rad']) + hcv
    err = 100 * (float(r['h_lin']) / h_r - 1)
    _pic(pic, art.radiation(case, e1, e2))
    cm.section(None, "Result")
    cm.tiles([("h_rad", f"{fmt(h_r)} W/m²K", "Exact equivalent convection coefficient of the radiation between T₁ "
                                             "and T₂."),
              ("h_tot = h_conv + h_rad", f"{fmt(h_t)} W/m²K", "Use this h for the surface."),
              ("ε_eff", fmt(r['eps_eff']), "Effective emissivity of the pair (geometry included)."),
              ("q'' radiated", f"{fmt(float(r['q']))} W/m²", f"Linearised 4ε_effσT_m³ = {fmt(float(r['h_lin']))} "
                                                             f"W/m²K ({err:+.1f} %).")])
    TT = np.linspace(T2 + 1, max(T2 + 150, T1 + 30), 120)
    fig = go.Figure([line(TT, hc.radiation(e, TT, T2, e2, F, Ar)['h_rad'], f"ε₁ = {e:g}", i,
                          dash=None if e == e1 else 'dot') for i, e in enumerate(sorted({0.1, 0.5, 0.9, e1}))])
    fig.add_hline(y=hcv, line_dash='dash', line_color=MUTED, annotation_text=f"h_conv = {fmt(hcv)}")
    fig.add_scatter(x=[T1], y=[h_r], mode='markers', name='this surface', marker=dict(size=10,
                                                                                       color=STATUS['critical']))
    fig.update_layout(xaxis_title='T₁ hot surface (°C)', yaxis_title='h_rad (W/m²K)')
    show(fig, height=320, key=f"{K}|ra|c", info="Equivalent radiation coefficient over the surface temperature for "
         "several emissivities (solid: your ε₁). The dashed line is the convection it adds to: where a curve is "
         "above it, radiation removes more heat than the air.")
    st.caption(f"At T₁ = {fmt(T1)} °C and T₂ = {fmt(T2)} °C the radiation is worth h_rad = {fmt(h_r)} W/m²K, "
               f"{100 * h_r / h_t:.0f} % of the total {fmt(h_t)} W/m²K.")
    st.button("Use h_tot in the layer stack", icon=':material/arrow_upward:', key=f"{K}|ra|use", on_click=_use_h,
              args=(h_t, f"radiation + convection, {case}"), help="Sets the far-side h of the tab Layer stack.")
    cm.equations([
        ("Grey-body exchange", [r"q'' = \varepsilon_{eff}\,\sigma_{SB}\,(T_1^4 - T_2^4),\quad \frac{1}"
                                r"{\varepsilon_{eff}} = \frac{1-\varepsilon_1}{\varepsilon_1} + \frac{1}{F_{12}} + "
                                r"\frac{1-\varepsilon_2}{\varepsilon_2}\frac{A_1}{A_2}"],
         "Small body: A₁/A₂ → 0, F = 1 → ε_eff = ε₁. Parallel plates: F = 1, A₁ = A₂ → 1/(1/ε₁ + 1/ε₂ − 1)."),
        ("Equivalent h", [r"h_{rad} = \frac{q''}{T_1 - T_2} = \varepsilon_{eff}\,\sigma_{SB}\,(T_1^2 + T_2^2)"
                          r"(T_1 + T_2)", r"h_{rad} \approx 4\,\varepsilon_{eff}\,\sigma_{SB}\,T_m^3,\quad T_m = "
                                          r"\tfrac{T_1 + T_2}{2}", r"h_{tot} = h_{conv} + h_{rad}"],
         "Temperatures in K. The first form is exact (factorisation of T₁⁴ − T₂⁴)."),
    ], sym=[("ε₁, ε₂", "emissivities of the hot and the cold surface", "–"),
            ("ε_eff", "effective emissivity of the pair incl. geometry", "–"),
            ("F₁₂", "view factor: share of the radiation of 1 that reaches 2", "–"),
            ("A₁, A₂", "areas of the surfaces", "m²"), ("σ_SB", "Stefan-Boltzmann constant 5.67e-8", "W/(m² K⁴)"),
            ("T₁, T₂, T_m", "surface temperatures and their mean (absolute)", "K"),
            ("h_rad, h_conv, h_tot", "equivalent radiation, convection and total heat transfer coefficient",
             "W/(m² K)")])
    cm.refs(['Incropera2007'])
    cm.interpret([
        f"The surface at {fmt(T1)} °C radiates like a convection with **h_rad = {fmt(h_r)} W/m²K** "
        f"(ε_eff = {fmt(r['eps_eff'])}); together with h_conv = {fmt(hcv)} W/m²K the surface sees "
        f"**h_tot = {fmt(h_t)} W/m²K**.",
        ("Radiation is a major part of the cooling here: a black finish (anodising, paint) is the cheapest lever."
         if h_r > 0.3 * h_t else
         "Radiation is a minor part of the cooling here (forced or liquid convection dominates, or ε is low)."),
        "Caveats: h_rad depends on both temperatures — recompute it when T₁ changes by more than ≈ 20 K. Fins "
        "facing each other see mostly each other: only the envelope of a heat sink radiates to the outside (use "
        "Custom with F < 1). Glass, FR4 and most paints are grey only in the infrared — the colour in visible light "
        "does not matter."])


#######################################################################################################################
# 5) Fins
#######################################################################################################################
def _fins():
    pic = _head(1, "Plate-fin heat sink", "N straight fins on a base plate, cooled by h on all wetted surfaces.")
    c = st.columns(4)
    W = cm.num(c[0], "Base width W (mm)", 100.0, f"{K}|hs|W", min_value=1e-3, help="Across the fins.")
    L = cm.num(c[1], "Base length L (mm)", 100.0, f"{K}|hs|L", min_value=1e-3, help="Along the fins.")
    tb = cm.num(c[2], "Base thickness (mm)", 6.0, f"{K}|hs|tb", min_value=1e-3)
    name, k, _, _ = _mat(c[3], "Material", 'Al alloy (6063)', 'hs|m')
    c = st.columns(4)
    N = cm.num(c[0], "Number of fins N", 15, f"{K}|hs|N", min_value=1, format='%d')
    tf = cm.num(c[1], "Fin thickness (mm)", 1.5, f"{K}|hs|tf", min_value=1e-3)
    H = cm.num(c[2], "Fin height (mm)", 30.0, f"{K}|hs|H", min_value=1e-3)
    h = cm.num(c[3], "h (W/m²K)", 40.0, f"{K}|hs|h", min_value=1e-3, help="Forced air 20–100, natural 5–10 "
                                                                           "(tab Convection h).")
    c = st.columns(4)
    As = cm.num(c[0], "Source area (mm²)", 900.0, f"{K}|hs|As", min_value=1e-3, help="Footprint of the module on "
                                                                                      "the base (spreading).")
    P = cm.num(c[1], "Power P (W)", 100.0, f"{K}|hs|P", min_value=0.0)
    Ta = cm.num(c[2], "Air temperature (°C)", 40.0, f"{K}|hs|Ta")
    if N * tf >= W:
        st.info("The fins do not fit on the base (N × t_f ≥ W).", icon=':material/info:')
        return
    r = hc.heatsink(h, k, W * MM, L * MM, tb * MM, N, tf * MM, H * MM, As * MM2)
    _pic(pic, art.fins(N, W, H, tf, tb))
    cm.section(None, "Result")
    cm.tiles([("R_sa", f"{fmt(r['R_sa'])} K/W", "Heat sink to air: fins + base incl. spreading from the source."),
              ("T_s (source)", f"{fmt(Ta + P * r['R_sa'])} °C", "Mean temperature under the source."),
              ("η_fin / η_o", f"{r['eta'] * 100:.0f} % / {r['eta_o'] * 100:.0f} %", "Fin efficiency and overall "
               "surface efficiency."), ("Fin spacing", f"{fmt(r['s'] / MM)} mm", "Clear gap between two fins.")])
    NN = np.arange(2, max(int(W / tf * 0.8), 3))
    RR = [hc.heatsink(h, k, W * MM, L * MM, tb * MM, int(n), tf * MM, H * MM, As * MM2) for n in NN]
    fig = go.Figure([line(NN, [x['R_sa'] for x in RR], 'R_sa (K/W)', 0),
                     line(NN, [x['s'] / MM for x in RR], 'spacing (mm)', 1, dash='dash', yaxis='y2'),
                     go.Scatter(x=[N], y=[r['R_sa']], mode='markers', name='current N',
                                marker=dict(size=10, color=STATUS['critical']))])
    fig.update_layout(xaxis_title='number of fins N', yaxis=dict(title='R_sa (K/W)', type='log'),
                      yaxis2=dict(title='fin spacing (mm)', overlaying='y', side='right', type='log'))
    show(fig, key=f"{K}|hs|N", info="Heat sink resistance and fin spacing over the number of fins at constant h. "
         "More fins always help at constant h — in reality h falls when the gaps close (boundary layers merge, the "
         "fan delivers less flow), so stay above the spacing limit.")
    s_min = 6.0 if h < 15 else 2.0
    st.caption(f"Spacing {fmt(r['s'] / MM)} mm; keep ≳ {fmt(s_min)} mm for "
               f"{'natural convection' if h < 15 else 'forced air'}, otherwise the constant-h assumption fails. "
               f"m·H_c = {fmt(r['mH'])}: above ≈ 1.5 the fin tip hardly works (η < 60 %).")
    cm.equations([
        ("Fin", r"m = \sqrt{\frac{2h}{k\,t_f}},\quad H_c = H + \frac{t_f}{2},\quad \eta_f = \frac{\tanh(mH_c)}{mH_c},"
                r"\quad A_f = 2LH_c", "Fin efficiency: actual heat of the fin / heat if the whole fin were at "
                                      "base temperature."),
        ("Heat sink", [r"A_b = WL - N t_f L,\quad A_{tot} = A_b + NA_f,\quad \eta_o = 1 - \frac{NA_f}{A_{tot}}"
                       r"(1-\eta_f)", r"R_{sa} = \frac{1}{\eta_o h A_{tot}} + R_{base}"],
         "R_base: conduction through the base incl. spreading from the source (Lee et al.)."),
    ], sym=[("W, L, t_b", "base width, length (along the fins), thickness", "m"),
            ("N, t_f, H", "number, thickness and height of the fins", "–, m, m"),
            ("k", "conductivity of the heat sink (material)", "W/(m K)"), ("h", "heat transfer coefficient", "W/(m² K)"),
            ("m", "fin parameter", "1/m"), ("η_f, η_o", "fin and overall surface efficiency", "–"),
            ("A_f, A_b, A_tot", "area of one fin, unfinned base, total wetted area", "m²")])
    cm.refs(['Incropera2007', 'Lee1995'])
    cm.interpret([
        f"The {name} heat sink {fmt(W)} × {fmt(L)} mm with {N} fins of {fmt(H)} mm has **R_sa = {fmt(r['R_sa'])} K/W**:"
        f" at {fmt(P)} W the base under the source is {fmt(P * r['R_sa'])} K above the air.",
        f"The fins multiply the wetted area to {fmt(r['A_tot'] * 1e4)} cm² ({fmt(r['A_tot'] / (W * L * MM2))}× the "
        f"base) at η_o = {r['eta_o'] * 100:.0f} %; the base with spreading adds {fmt(r['R_base'])} K/W "
        f"({100 * r['R_base'] / r['R_sa']:.0f} %). " +
        ("Taller fins still pay off." if r['eta'] > 0.8 else "The fins are already long for this h — thicker fins or "
                                                             "a better material act more than taller ones."),
        "Caveat: h is assumed uniform and independent of the spacing; the air heats up along the fins (tab Coolant "
        "flow) and lowers the effective cooling further."])


#######################################################################################################################
# 6) Vias
#######################################################################################################################
def _vias():
    pic = _head(1, "Thermal vias under a pad", "Plated through-holes conduct heat through the FR4 board under an SMD "
               "power device (e.g. D²PAK, TOLL, QFN exposed pad).")
    c = st.columns(4)
    pt = cm.num(c[0], "Pitch (mm)", 1.0, f"{K}|vi|p", min_value=1e-3, help="Distance of the vias (square array).")
    d = cm.num(c[1], "Drill diameter (mm)", 0.3, f"{K}|vi|d", min_value=1e-3)
    pl = cm.num(c[2], "Plating (µm)", 25.0, f"{K}|vi|pl", min_value=0.0, help="Copper on the barrel (IPC class 2: "
                                                                                "20–25 µm).")
    t = cm.num(c[3], "Board thickness (mm)", 1.6, f"{K}|vi|t", min_value=1e-3)
    c = st.columns(4)
    w = cm.num(c[0], "Pad w (mm)", 5.0, f"{K}|vi|w", min_value=1e-3)
    l = cm.num(c[1], "Pad l (mm)", 5.0, f"{K}|vi|l", min_value=1e-3)
    filled = c[2].toggle("Filled vias", key=f"{K}|vi|fill", help="Filled with epoxy (k ≈ 0.3–1) or copper "
                                                                 "(set k_fill = 390).")
    kf = cm.num(c[3], "k_fill (W/mK)", 0.3, f"{K}|vi|kf", min_value=0.0) if filled else 0.3
    if pl * UM >= d * MM / 2 or d >= pt:
        st.info("Check the geometry: plating < drill/2 and drill < pitch.", icon=':material/info:')
        return
    r = hc.vias(pt * MM, d * MM, pl * UM, t * MM, w * MM, l * MM, filled, 0.3, kf)
    _pic(pic, art.vias(w, pt, filled))
    cm.section(None, "Result")
    cm.tiles([("k_eff", f"{fmt(r['k_eff'])} W/mK", "Through-plane conductivity of the via field (FR4: 0.3)."),
              ("R_pad", f"{fmt(r['R_pad'])} K/W", "Through the board under the pad."),
              ("Without vias", f"{fmt(r['R_board'])} K/W", "FR4 only."),
              ("Vias in the pad", f"{r['n']}", f"One via: {fmt(r['R_via'])} K/W.")])
    pp = np.linspace(max(d * 1.2, 0.4), 3.0, 60)
    tr = []
    for i, fl in enumerate([False, True]):
        tr.append(line(pp, [hc.vias(x * MM, d * MM, pl * UM, t * MM, w * MM, l * MM, fl, 0.3, 390.0 if fl else 0.3)
                            ['R_pad'] for x in pp], 'Cu-filled' if fl else 'open / epoxy', i))
    fig = go.Figure(tr + [go.Scatter(x=[pt], y=[r['R_pad']], mode='markers', name='current',
                                     marker=dict(size=10, color=STATUS['critical']))])
    fig.update_layout(xaxis_title='via pitch (mm)', yaxis=dict(type='log', title='R_pad (K/W)'))
    show(fig, height=320, key=f"{K}|vi|p", info="Resistance through the board under the pad over the via pitch, for "
         "open/epoxy-filled and copper-filled vias. A denser array lowers R roughly with the square of the pitch.")
    cm.equations([
        ("Area fractions", r"\varphi_{Cu} = \frac{\pi(r_o^2 - r_i^2)}{p^2},\quad \varphi_{h} = \frac{\pi r_i^2}{p^2}",
         "Copper barrel and hole per via cell p × p."),
        ("Effective k", r"k_{eff} = \varphi_{Cu}k_{Cu} + \varphi_h k_{fill} + (1-\varphi_{Cu}-\varphi_h)k_{FR4},"
                        r"\quad R_{pad} = \frac{t}{k_{eff}\,w\,l}", "Parallel conduction (no spreading in the copper "
                                                                    "planes, which helps further)."),
    ], sym=[("p", "via pitch", "m"), ("r_o, r_i", "outer radius (drill/2) and inner radius (− plating)", "m"),
            ("k_Cu, k_fill, k_FR4", "conductivity of copper, fill, laminate", "W/(m K)"),
            ("t", "board thickness", "m"), ("w, l", "pad size", "m")])
    cm.refs(['Incropera2007'])
    cm.interpret([
        f"{r['n']} vias at {fmt(pt)} mm pitch raise the through-plane conductivity from 0.3 to **{fmt(r['k_eff'])} "
        f"W/mK** and lower the board resistance under the pad from {fmt(r['R_board'])} to {fmt(r['R_pad'])} K/W "
        f"({fmt(r['R_board'] / r['R_pad'])}× better).",
        "The copper barrels carry almost all the heat. Levers: smaller pitch, thicker plating, copper filling "
        "(via-in-pad) — or an IMS/inlay board when R_pad is still too high.",
        "Use k_eff as the k_z of the board layer in the layer stack or the Board designer (which averages the vias "
        "the same way)."])


#######################################################################################################################
# 7) Coolant
#######################################################################################################################
def _coolant():
    pic = _head(1, "Energy balance of the coolant", "The coolant heats up while it takes the losses away; the last "
               "component in the flow sees the warmest coolant.")
    c = st.columns(4)
    fl = c[0].selectbox("Fluid", list(hc.FLUIDS), index=2, key=f"{K}|co|f")
    T = cm.num(c[1], "Inlet temperature (°C)", 65.0, f"{K}|co|T")
    P = cm.num(c[2], "Losses P (W)", 2000.0, f"{K}|co|P", min_value=0.0)
    mode = c[3].radio("Given", ["flow rate", "temperature rise"], key=f"{K}|co|m", horizontal=True)
    p = hc.fluid(fl, T)
    gas = p['gas']
    _pic(pic, art.coolant(fl))
    if mode == 'flow rate':
        V = cm.num(st, "Flow rate (m³/h)" if gas else "Flow rate (L/min)", 100.0 if gas else 8.0, f"{K}|co|V|{gas}",
                   min_value=1e-6)
        r = hc.coolant(P, p, V_dot=V / 3600 if gas else V / 60000)
    else:
        dT = cm.num(st, "Allowed temperature rise ΔT (K)", 10.0 if gas else 5.0, f"{K}|co|dT", min_value=1e-3)
        r = hc.coolant(P, p, dT=dT)
    cm.section(None, "Result")
    cm.tiles([("ΔT coolant", f"{fmt(r['dT'])} K", "Outlet − inlet."),
              ("Outlet", f"{fmt(T + r['dT'])} °C", "Temperature after the losses."),
              ("Flow rate", f"{fmt(r['V_dot'] * 3600)} m³/h ({fmt(r['V_dot'] * 3600 / 1.699)} CFM)" if gas else
               f"{fmt(r['V_dot'] * 60000)} L/min", "Volume flow."), ("Mass flow", f"{fmt(r['m_dot'] * 1000)} g/s",
                                                                    "ṁ = ρ V̇.")])
    sc = 3600 if gas else 60000
    VV = np.linspace(0.2, 3, 60) * r['V_dot']
    fig = go.Figure(line(VV * sc, P / (p['rho'] * p['cp'] * VV), 'ΔT(V̇)', 0))
    fig.add_scatter(x=[r['V_dot'] * sc], y=[r['dT']], mode='markers', name='operating point',
                    marker=dict(size=10, color=STATUS['critical']))
    fig.update_layout(xaxis_title='flow rate (m³/h)' if gas else 'flow rate (L/min)', yaxis_title='ΔT (K)')
    show(fig, height=300, key=f"{K}|co|c", info="Temperature rise of the coolant over the flow rate (hyperbola): "
         "halving the flow doubles ΔT.")
    cm.equations([("Energy balance", r"P = \dot m\,c_p\,\Delta T,\quad \dot m = \rho\dot V", None)],
                 sym=[("P", "losses taken by the coolant", "W"), ("ṁ, V̇", "mass and volume flow", "kg/s, m³/s"),
                      ("ρ, c_p", f"density and heat capacity of {fl} at {fmt(T)} °C: {fmt(p['rho'])} kg/m³, "
                                 f"{fmt(p['cp'])} J/(kg K)", "kg/m³, J/(kg K)"), ("ΔT", "outlet − inlet", "K")])
    cm.refs(['Incropera2007'])
    cm.interpret([
        f"{fmt(P)} W heat {fl} by **{fmt(r['dT'])} K** at the given flow; a component at the outlet sees "
        f"{fmt(T + r['dT'])} °C instead of {fmt(T)} °C. Add half of ΔT to the coolant temperature of a component in "
        "the middle of the flow path.",
        "Air takes ≈ 3500× less heat per volume than water: air cooling needs large flows and a ΔT budget of "
        "10–20 K, liquid cooling stays within a few K."])


#######################################################################################################################
# 8) Short pulses and time constants
#######################################################################################################################
def _pulse():
    pic = _head(1, "Short power pulse into a chip", "For pulses shorter than the diffusion time through the chip the "
               "heat stays near the surface: the chip acts as a semi-infinite body and only its heat capacity limits "
               "the temperature (short circuit, surge, avalanche).")
    c = st.columns(5)
    name, k, rho, cp = _mat(c[0], "Chip material", 'Si', 'pu|m')
    a = cm.num(c[1], "Active area (mm²)", 100.0, f"{K}|pu|A", min_value=1e-3)
    d = cm.num(c[2], "Chip thickness (µm)", 120.0, f"{K}|pu|d", min_value=1e-3)
    P = cm.num(c[3], "Pulse power (kW)", 240.0, f"{K}|pu|P", min_value=0.0,
               help="E.g. short circuit: V_DC × I_sc = 600 V × 400 A = 240 kW.")
    tp = cm.num(c[4], "Pulse length (µs)", 10.0, f"{K}|pu|tp", min_value=1e-3)
    q = P * 1e3 / (a * MM2)
    dT = float(hc.semi_inf(q, tp * 1e-6, k, rho, cp))
    td_ = hc.diffusion_time(d * UM, k, rho, cp)
    delta = math.sqrt(k / (rho * cp) * tp * 1e-6)
    _pic(pic, art.pulse(delta, d * UM))
    cm.section(None, "Result")
    cm.tiles([("ΔT surface", f"{fmt(dT)} K", "Semi-infinite estimate at the end of the pulse."),
              ("Penetration δ", f"{fmt(delta / UM)} µm", "√(α t_p): depth the heat has reached."),
              ("Diffusion time t_d", f"{fmt(td_ * 1e6)} µs", "d²/α: time to heat through the chip; the estimate holds "
                                                             "for t_p ≲ t_d/4."),
              ("Energy", f"{fmt(P * tp)} mJ", "E = P t_p.")])
    tt = np.logspace(-8, np.log10(max(tp * 1e-6 * 100, td_ * 10)), 200)
    fig = go.Figure(line(tt * 1e6, hc.semi_inf(q, tt, k, rho, cp), 'ΔT semi-infinite', 0))
    fig.add_vline(x=td_ / 4 * 1e6, line_dash='dash', line_color=MUTED, annotation_text='t_d/4 (limit)')
    fig.add_scatter(x=[tp], y=[dT], mode='markers', name='end of pulse', marker=dict(size=10, color=STATUS['critical']))
    fig.update_layout(xaxis=dict(type='log', title='t (µs)'), yaxis=dict(type='log', title='ΔT (K)'))
    show(fig, height=320, key=f"{K}|pu|c", info="Surface temperature rise over the pulse length (log-log, slope ½: ΔT "
         "∝ √t). Right of the dashed line the heat reaches the die attach and the estimate is too high — use the "
         "Z_th of the package there.")
    st.caption("ΔT is the rise above the temperature before the pulse. The Al metallisation melts at 660 °C — a short "
               "circuit must be switched off well before the surface gets close to it.")
    _train()
    cm.section(3, "Lumped body (heat sink, cold plate)", "A body with good internal conduction (Biot < 0.1) heats "
               "with one time constant τ = R C.")
    c = st.columns(4)
    _, _, _, cp2 = _mat(c[0], "Material", 'Al alloy (6063)', 'pu|m2')
    m = cm.num(c[1], "Mass (kg)", 0.5, f"{K}|pu|mass", min_value=1e-6)
    R = cm.num(c[2], "R to ambient (K/W)", 0.4, f"{K}|pu|R", min_value=1e-6, help="E.g. R_sa of the tab Heat sink fins.")
    lb = hc.lumped(m, cp2, R)
    c[3].metric("τ = R C", f"{fmt(lb['tau'])} s", help=f"C = m c_p = {fmt(lb['C'])} J/K; 95 % of the steady state "
                                                       f"after 3τ = {fmt(3 * lb['tau'] / 60)} min.")
    cm.equations([
        ("Semi-infinite body", r"\Delta T_s(t) = \frac{2q''}{k}\sqrt{\frac{\alpha t}{\pi}} = 2q''\sqrt{\frac{t}"
                               r"{\pi k\rho c_p}},\quad q'' = \frac{P}{A}", "Constant heat flux on the surface."),
        ("Diffusion", r"\alpha = \frac{k}{\rho c_p},\quad \delta = \sqrt{\alpha t},\quad t_d = \frac{d^2}{\alpha}", None),
        ("Lumped body", r"C = m c_p,\quad \tau = R\,C,\quad \Delta T(t) = PR(1 - e^{-t/\tau})", None),
    ], sym=[("q''", "heat flux on the active area", "W/m²"), ("k, ρ, c_p", f"properties of {name}", "W/(m K), kg/m³, "
                                                                                                  "J/(kg K)"),
            ("α", "thermal diffusivity", "m²/s"), ("δ", "penetration depth", "m"), ("d", "chip thickness", "m"),
            ("t_d", "diffusion time through the chip", "s"), ("m, R, τ", "mass, resistance to ambient, time constant",
                                                              "kg, K/W, s")])
    cm.refs(['Incropera2007', 'Wintrich2015'])
    ok = tp * 1e-6 <= td_ / 4
    cm.interpret([
        f"A {fmt(P)} kW pulse of {fmt(tp)} µs on {fmt(a)} mm² {name} ({fmt(q / 1e4)} W/cm²) heats the surface by **{fmt(dT)} K**. The heat has penetrated {fmt(delta / UM)} µm "
        f"of the {fmt(d)} µm chip.",
        ("The pulse is shorter than t_d/4: only the chip itself buffers the energy, the package and the cooling do not "
         "matter. Levers: a larger active area or a material with higher √(k ρ c_p) (SiC > Si) — or a top-side copper "
         "layer / sinter buffer." if ok else
         "The pulse is longer than t_d/4: the heat reaches the die attach, the semi-infinite estimate is too high. "
         "Use the Z_th of the package (Thermal stack, Thermal networks) for this pulse length."),
        f"The {fmt(m)} kg heat sink needs ≈ {fmt(3 * lb['tau'] / 60)} min to reach 95 % of its steady state — load "
        "steps shorter than that are buffered by its heat capacity."])


def _train():
    """Pulse train on the Foster network of the device (section 2 of the tab Short pulses)."""
    slot = _head(2, "Pulse train on the Foster network", "Repeated pulses (pulsed load, low output frequency, "
                 "start-stop) heat up from pulse to pulse until the heat removed in each pause equals the energy "
                 "of a pulse. The Foster network R_i, τ_i of the datasheet (Z_th junction → ambient) gives the "
                 "exact peak of every pulse.")
    cfg = {c: st.column_config.NumberColumn(c, help=h, min_value=0.0, format='%g') for c, h in FOSTER_COLS.items()}
    df = st.data_editor(FOSTER_DEF, num_rows='dynamic', hide_index=True, column_config=cfg, key=f"{K}|pt|tab",
                        width='stretch')
    df = df.dropna()
    df = df[(df['R_i [K/W]'] > 0) & (df['τ_i [s]'] > 0)]
    if df.empty:
        st.info("Enter at least one Foster term with R_i > 0 and τ_i > 0.", icon=':material/info:')
        return
    R, tau = df['R_i [K/W]'].to_numpy(float), df['τ_i [s]'].to_numpy(float)
    c = st.columns(3)
    P = cm.num(c[0], "Pulse power P (W)", 1000.0, f"{K}|pt|P", min_value=0.0,
               help="Loss of the device during a pulse (conduction + switching).")
    tp = cm.num(c[1], "Pulse length t_p (ms)", 10.0, f"{K}|pt|tp", min_value=1e-6)
    T = cm.num(c[2], "Period T (ms)", 100.0, f"{K}|pt|T", min_value=1e-6, help="Time from the start of one pulse "
                                                                               "to the next (T > t_p).")
    if T <= tp:
        st.warning("The period must be longer than the pulse (T > t_p); for T = t_p the load is constant.",
                   icon=':material/warning:')
        return
    r = hc.pulse_train(P, tp * 1e-3, T * 1e-3, R, tau)
    _pic(slot, art.pulse_train(tp, T, r['first'], r['peak'], r['valley']))
    cm.tiles([("1st pulse peak", f"{fmt(r['first'])} K", "ΔT at the end of the first pulse = P Z_th(t_p)."),
              ("Periodic peak", f"{fmt(r['peak'])} K", f"ΔT at the end of every pulse after ≈ {fmt(r['N'])} pulses "
                                                     "(99 % of the steady state)."),
              ("Valley / swing", f"{fmt(r['valley'])} K / {fmt(r['peak'] - r['valley'])} K",
               "Lowest ΔT before a pulse and the swing ΔT_peak − ΔT_valley (drives the power cycling)."),
              ("Mean D P R_th", f"{fmt(r['mean'])} K", f"Average losses on the total R_th = {fmt(r['Rth'])} K/W "
                                                       f"(duty D = {fmt(100 * r['D'])} %).")])
    t1, T1 = r['w_first']
    t2, T2 = r['w_last']
    n_last = max(r['N'] - 3, 0)
    fig = go.Figure([line(t1 * 1e3, T1, "first 3 periods", 0),
                     line((t2 - n_last * T * 1e-3) * 1e3, T2, f"periods {n_last + 1}–{n_last + 3} (steady)", 1)])
    fig.add_hline(y=r['mean'], line_dash='dash', line_color=MUTED, annotation_text='mean D P R_th')
    fig.update_layout(xaxis_title='t within the shown periods (ms)', yaxis_title='ΔT junction (K)')
    show(fig, height=300, key=f"{K}|pt|w", info="Junction temperature rise during the first three periods (blue) "
         "and during three periods in the periodic steady state (shifted to the same time axis). The gap between the "
         "two curves is the heat stored in the slow terms (heat sink).")
    fig = go.Figure(line(r['n'], r['peaks'], 'peak after pulse n', 0, mode='lines+markers', marker=dict(size=4)))
    fig.add_hline(y=r['peak'], line_dash='dash', line_color=STATUS['critical'], annotation_text='periodic peak')
    fig.add_hline(y=r['mean'], line_dash='dot', line_color=MUTED, annotation_text='mean')
    fig.update_layout(xaxis=dict(type='log', title='pulse number n'), yaxis_title='ΔT at the end of the pulse (K)')
    show(fig, height=280, key=f"{K}|pt|n", info="Peak temperature after each pulse (log scale of n). Each Foster "
         "term saturates after ≈ 3 τ_i/T pulses — the fast terms after the first pulse, the heat sink much later.")
    err = 100 * (r['approx'] / r['peak'] - 1)
    st.caption(f"Peak after the 1st pulse {fmt(r['first'])} K, periodic peak {fmt(r['peak'])} K (+"
               f"{fmt(r['peak'] - r['first'])} K by the build-up), mean {fmt(r['mean'])} K. The datasheet "
               f"approximation gives {fmt(r['approx'])} K ({err:+.1f} %).")
    cm.equations([
        ("Foster network", r"Z_{th}(t) = \sum_i R_i\,(1 - e^{-t/\tau_i}),\quad R_{th} = \sum_i R_i", None),
        ("Exact peak after pulse n", [r"a_i = e^{-t_p/\tau_i},\ b_i = e^{-T/\tau_i}",
                                      r"\Delta T(n) = \sum_i P R_i\,(1 - a_i)\,\frac{1 - b_i^{\,n}}{1 - b_i}",
                                      r"\Delta T_{\infty} = \sum_i P R_i\,\frac{1 - a_i}{1 - b_i},\quad "
                                      r"\Delta T_{valley} = \sum_i \Delta T_{\infty,i}\,e^{-(T - t_p)/\tau_i}"],
         "Superposition of the step responses; exact for rectangular pulses of constant power."),
        ("Datasheet approximation", r"\Delta T_{\infty} \approx P\,[D R_{th} + (1-D) Z_{th}(T + t_p) - Z_{th}(T) + "
                                    r"Z_{th}(t_p)],\quad D = t_p/T", "Common in application notes (Wintrich 2015)."),
    ], sym=[("R_i, τ_i", "thermal resistance and time constant of Foster term i (datasheet or Thermal networks)",
             "K/W, s"), ("i", "index of the Foster term", "–"), ("P", "power during a pulse", "W"),
            ("t_p, T, D", "pulse length, period, duty cycle t_p/T", "s, s, –"), ("n", "number of the pulse", "–"),
            ("ΔT(n), ΔT_∞", "temperature rise at the end of pulse n and in the periodic steady state", "K"),
            ("Z_th", "thermal impedance (step response per watt)", "K/W")])
    slow = float(np.max(tau))
    cm.interpret([
        f"A train of {fmt(P)} W pulses of {fmt(tp)} ms every {fmt(T)} ms reaches **{fmt(r['peak'])} K** at the end "
        f"of each pulse in the steady state — {fmt(r['peak'] / max(r['first'], 1e-12))}× the first pulse and "
        f"{fmt(r['peak'] / max(r['mean'], 1e-12))}× the mean value D P R_th = {fmt(r['mean'])} K.",
        ("The pulses are short against the fast time constants: the junction hardly cools in the pause and the train "
         "acts almost like the average loss." if r['peak'] - r['valley'] < 0.2 * r['peak'] else
         f"Each pulse swings the junction by {fmt(r['peak'] - r['valley'])} K: rating with the average losses would "
         f"miss {fmt(r['peak'] - r['mean'])} K — use the peak for T_j,max and the swing for power cycling (tab "
         "Lifetime)."),
        f"The slowest term (τ = {fmt(slow)} s) needs ≈ {fmt(3 * slow / 60)} min to settle; a burst shorter than that "
        "stays between the first-pulse and the periodic peak (read it off the second chart).",
        "Caveat: the Foster network must cover the whole path to ambient (junction → case only gives ΔT_jc). Losses "
        "that depend on T_j (R_DS,on) raise the peak further."])


#######################################################################################################################
# 9) Skin and eddy currents
#######################################################################################################################
def _skin():
    case = st.segmented_control("Conductor", ["Round wire", "Square / rectangular bar",
                                              "Trace over a metal substrate"], default="Round wire",
                                key=f"{K}|sk|case") or "Round wire"
    if case.startswith('Trace'):
        return _plate()
    rnd = case.startswith('Round')
    pic = _head(1, f"{case}: skin effect and self-heating", "An AC current is pushed to the surface of the conductor "
               "by its own magnetic field (skin effect); neighbouring conductors of a winding or bundle push it "
               "further to one side (proximity effect). Both shrink the used cross-section: R_ac > R_dc, more loss, "
               "a hotter conductor.")
    mats = list(hc.CONDUCTORS)
    c = st.columns(4)
    mat = c[0].selectbox("Material", mats, key=f"{K}|sk|mat")
    if rnd:
        w = cm.num(c[1], "Diameter d (mm)", 2.0, f"{K}|sk|d", min_value=1e-3) * MM
        t = w
    else:
        w = cm.num(c[1], "Width w (mm)", 10.0, f"{K}|sk|w", min_value=1e-3) * MM
        t = cm.num(c[2], "Thickness t (mm)", 2.0, f"{K}|sk|t", min_value=1e-3) * MM
    I = cm.num(c[3], "Current I_rms (A)", 20.0 if rnd else 150.0, f"{K}|sk|I|{rnd}", min_value=0.0,
               help="RMS value of the AC component at the frequency f (add the DC part as a second run).")
    c = st.columns(4)
    f = cm.num(c[0], "Frequency f (kHz)", 100.0 if rnd else 20.0, f"{K}|sk|f|{rnd}", min_value=1e-6,
               help="Switching frequency for a ripple current, output frequency for a load current (50 Hz = 0.05).")
    Ta = cm.num(c[1], "Ambient T (°C)", 40.0, f"{K}|sk|Ta")
    h = cm.num(c[2], "h on the surface (W/m²K)", 15.0, f"{K}|sk|h", min_value=1e-3,
               help="Cooling of the conductor surface: still air 8–15, fan 30–80, potted/on a heat sink 100–500 "
                    "(tabs Convection h and Radiation → h).")
    m = cm.num(c[3], "Layers m (proximity)", 1, f"{K}|sk|m", min_value=1, format='%d',
               help="Number of conductor layers in a winding or bundle that carry the same current. 1 = isolated "
                    "conductor (skin effect only).")
    pitch = None
    if m > 1:
        pitch = cm.num(st.columns(4)[0], "Pitch of the conductors (mm)", (w if rnd else t) * 1.1 / MM,
                       f"{K}|sk|p|{rnd}", min_value=1e-3,
                       help="Centre distance of neighbouring conductors in a layer (porosity η = width / pitch).") * MM
    shp = 'round' if rnd else 'rect'
    r = hc.conductor(shp, w, t, I, f * 1e3, mat, Ta, h, m, pitch)
    r0 = hc.conductor(shp, w, t, I, 1e-9, mat, Ta, h)
    _pic(pic, art.skin(shp, w, t, r['delta']) +
         (art.proximity(shp, w, t, m, pitch, r['Delta'], r['F_prox']) if m > 1 else ''))
    cm.section(None, "Result")
    run = not math.isfinite(r['T'])
    cm.tiles([("Skin depth δ", f"{fmt(r['delta'] / MM)} mm", "Depth at which J has dropped to 1/e = 37 %."),
              ("R_ac/R_dc", fmt(r['F']), f"skin {fmt(r['F_skin'])} + proximity {fmt(r['F_prox'])}"),
              ("Loss P' (AC / DC)", f"{fmt(r['P'])} / {fmt(r0['P'])} W/m", "Loss per metre at the conductor "
                                                                           "temperature; DC = same I_rms at f = 0."),
              ("Conductor T (AC / DC)", "runaway" if run else f"{fmt(r['T'])} / {fmt(r0['T'])} °C",
               "Steady temperature with ρ(T) and the cooling h on the perimeter.")])
    ff = np.logspace(np.log10(max(f * 1e3, 1) / 1e3), np.log10(f * 1e3 * 1e2), 80)
    rr = [hc.conductor(shp, w, t, I, x, mat, Ta, h, m, pitch) for x in ff]
    fig = go.Figure([line(ff / 1e3, [x['F_skin'] for x in rr], 'skin only', 0),
                     line(ff / 1e3, [x['F'] for x in rr], 'skin + proximity', 1, dash='dash')])
    fig.add_scatter(x=[f], y=[r['F']], mode='markers', name='this f', marker=dict(size=10, color=STATUS['critical']))
    fig.update_layout(xaxis=dict(type='log', title='f (kHz)'), yaxis=dict(type='log', title='R_ac/R_dc'))
    show(fig, height=300, key=f"{K}|sk|c", info="AC resistance factor over the frequency. It stays at 1 while δ is "
         "larger than the conductor and rises with √f once the current flows only in the skin. The dashed line "
         "adds the proximity effect of the m layers.")
    Ts = [x['T'] if math.isfinite(x['T']) else np.nan for x in rr]
    fig = go.Figure(line(ff / 1e3, Ts, 'AC', 0))
    fig.add_hline(y=r0['T'], line_dash='dash', line_color=MUTED, annotation_text=f"DC {fmt(r0['T'])} °C")
    fig.update_layout(xaxis=dict(type='log', title='f (kHz)'), yaxis_title='conductor T (°C)')
    show(fig, height=260, key=f"{K}|sk|T", info="Steady conductor temperature over the frequency for the same RMS "
         "current. Gaps mean thermal runaway (> 1000 °C).")
    cm.equations([
        ("Skin depth", r"\delta = \sqrt{\frac{\rho}{\pi f \mu_0 \mu_r}},\quad \rho(T) = \rho_{20}\,[1 + \alpha"
                       r"(T - 20\,°C)]", "Cu at 20 °C: δ = 66 mm/√(f/Hz) → 9.3 mm at 50 Hz, 0.21 mm at 100 kHz."),
        ("Round wire (exact)", [r"k = \frac{1 - j}{\delta},\quad \frac{R_{ac}}{R_{dc}} = \mathrm{Re}\!\left["
                                r"\frac{k a}{2}\,\frac{J_0(k a)}{J_1(k a)}\right],\ a = d/2",
                                r"\to 1 + \frac{(a/\delta)^4}{48}\ (a \ll \delta),\qquad \to \frac{a}{2\delta} + "
                                r"\frac{1}{4}\ (a \gg \delta)"], None),
        ("Bar w × t", r"\frac{R_{ac}}{R_{dc}} \approx \text{round wire at } d_h = \frac{4A}{U} = \frac{2wt}{w+t}",
         "Exact high-frequency limit A/(U δ) for any shape, ±10 % in the transition, not for very thin wide strips "
         "with corner crowding."),
        ("Proximity (Dowell)", [r"h = d\sqrt{\pi}/2\ (\text{round}),\ t\ (\text{bar}),\quad \Delta = \frac{h}{\delta}"
                                r"\sqrt{\eta},\ \eta = \frac{\text{width}}{\text{pitch}}",
                                r"F_{prox} = \Delta\,\frac{2(m^2-1)}{3}\,\frac{\sinh\Delta - \sin\Delta}{\cosh\Delta "
                                r"+ \cos\Delta}"], None),
        ("Self-heating", r"P' = I^2 \frac{R_{ac}}{R_{dc}} \frac{\rho(T)}{A},\quad T = T_{amb} + \frac{P'}{h\,U}",
         "Solved by fixed-point iteration; no solution (runaway) when the loss grows faster with T than the cooling."),
    ], sym=[("δ", "skin depth", "m"), ("ρ, ρ₂₀, α", f"resistivity of {mat} at T and at 20 °C, temperature coefficient",
                                       "Ω m, Ω m, 1/K"), ("f", "frequency of the current", "Hz"),
            ("μ₀, μ_r", "vacuum permeability 4π·1e-7, relative permeability (1: non-magnetic)", "H/m, –"),
            ("d, a, w, t", "diameter, radius, width, thickness of the conductor", "m"),
            ("J₀, J₁", "Bessel functions of the first kind (complex argument)", "–"),
            ("A, U, d_h", "cross-section, cooled perimeter, hydraulic diameter", "m², m, m"),
            ("m, η, Δ", "number of layers, porosity of a layer, normalised conductor height", "–"),
            ("I", "RMS current", "A"), ("P'", "loss per length", "W/m"), ("h", "heat transfer coefficient on the "
                                                                              "surface", "W/(m² K)")])
    cm.refs(['Kazimierczuk2014', 'Dowell1966', 'Incropera2007'])
    size = w if rnd else min(w, t)
    txt = [f"At {fmt(f)} kHz the skin depth in {mat} is {fmt(r['delta'] / MM)} mm against a conductor of "
           f"{fmt(size / MM)} mm{' diameter' if rnd else ' thickness'}: **R_ac/R_dc = {fmt(r['F'])}**, the loss "
           f"rises from {fmt(r0['P'])} to {fmt(r['P'])} W/m" +
           (". The conductor runs away thermally — larger cross-section or better cooling needed." if run else
            f" and the conductor from {fmt(r0['T'])} to **{fmt(r['T'])} °C**.")]
    if r['F'] < 1.1:
        txt.append("The conductor is thin against δ: the skin effect is negligible, the whole cross-section is used.")
    else:
        txt.append(f"Only a shell of ≈ δ carries the current; a conductor thicker than ≈ 2δ = {fmt(2 * r['delta'] / MM)}"
                   " mm adds copper but hardly reduces R_ac. Levers: litz wire or several thin strands (each < 2δ), "
                   "flat bars (more perimeter per area), or a lower frequency.")
    if m > 1:
        txt.append(f"With {m} layers the proximity effect adds {fmt(r['F_prox'])} to the factor "
                   f"({fmt(100 * r['F_prox'] / r['F'])} % of the AC resistance) — it grows with m²; interleaving the "
                   "winding halves the effective m.")
    txt.append("Caveats: sinusoidal current (for a ripple add the harmonics, each with its own δ), non-magnetic "
               "conductor, the conductor is cooled only on its surface (no conduction along the length).")
    cm.interpret(txt)


def _plate():
    pic = _head(1, "Trace over a metal substrate: eddy losses", "On an insulated metal substrate (IMS), a DBC "
               "baseplate or close to a housing, the AC field of a trace induces an image current in the metal "
               "below. It flows within the skin depth of the metal and heats it; at the same time the current in "
               "the trace crowds to its bottom face.")
    c = st.columns(4)
    I = cm.num(c[0], "AC current I_rms (A)", 20.0, f"{K}|pl|I", min_value=0.0,
               help="RMS of the AC component (e.g. the ripple of a converter) at the frequency f.")
    f = cm.num(c[1], "Frequency f (kHz)", 100.0, f"{K}|pl|f", min_value=1e-6)
    w = cm.num(c[2], "Trace width w (mm)", 8.0, f"{K}|pl|w", min_value=1e-3) * MM
    tc = cm.num(c[3], "Copper thickness (µm)", 70.0, f"{K}|pl|tc", min_value=1e-3,
                help="35 µm = 1 oz, 70 µm = 2 oz, 105 µm = 3 oz, DBC 300 µm.") * UM
    c = st.columns(4)
    h = cm.num(c[0], "Dielectric thickness h (mm)", 0.1, f"{K}|pl|h", min_value=1e-4,
               help="Distance trace → metal (IMS 0.05–0.15 mm, DBC ceramic 0.32–0.63 mm).") * MM
    kd = cm.num(c[1], "Dielectric k (W/mK)", 2.0, f"{K}|pl|kd", min_value=1e-3,
                help="IMS dielectric 1–3, Al2O3 24, AlN 170, FR4 0.3.")
    mats = list(hc.CONDUCTORS)
    pm = c[2].selectbox("Base metal", mats, index=mats.index('Al 5052 (IMS base)'), key=f"{K}|pl|m")
    tp = cm.num(c[3], "Base thickness t (mm)", 1.5, f"{K}|pl|tp", min_value=1e-3) * MM
    c = st.columns(4)
    ret = c[0].selectbox("Return current", ["Neighbouring trace (distance s)", "Far away (worst case)"],
                         key=f"{K}|pl|ret", help="A return next to the trace cancels most of the field below; a "
                                                 "far return (other board, cable) induces the full image current.")
    s = cm.num(c[1], "Centre distance s (mm)", 12.0, f"{K}|pl|s", min_value=1e-3,
               disabled=ret.startswith('Far')) * MM
    Lt = cm.num(c[2], "Trace length (mm)", 50.0, f"{K}|pl|L", min_value=1e-3) * MM
    Tc = cm.num(c[3], "Copper temperature (°C)", 80.0, f"{K}|pl|T")
    pair = not ret.startswith('Far')
    rho_p = float(hc.rho_T(*hc.CONDUCTORS[pm], Tc))
    rho_c = float(hc.rho_T(*hc.CONDUCTORS['Cu (annealed)'], Tc))
    e = hc.plate_eddy(I, f * 1e3, w, h, tp, rho_p, s if pair else None)
    dc = float(hc.skin_depth(rho_c, f * 1e3))
    Ft = float(hc._F1(tc / dc))
    Pt_ = I ** 2 * rho_c / (w * tc) * Ft                                   # trace loss per length, per trace
    ntr = 2 if pair else 1
    Pp, Ptr = e['P'] * Lt, Pt_ * Lt * ntr
    dT_d = Pt_ * h / (kd * (w + h))
    _pic(pic, art.plate(h, tp, e['delta'], pair))
    cm.section(None, "Result")
    cm.tiles([("δ in the base", f"{fmt(e['delta'] / MM)} mm", f"Skin depth of {pm}; Δ = t/δ = {fmt(e['Delta'])}."),
              ("Eddy loss in the base", f"{fmt(Pp)} W", f"{fmt(e['P'])} W/m over {fmt(Lt / MM)} mm "
                                                        f"(β = {fmt(e['beta'])} of the full image current)."),
              ("Trace loss (AC)", f"{fmt(Ptr)} W", f"{ntr} trace(s); R_ac/R_dc = {fmt(Ft)} by the current crowding "
                                                   "to the bottom face."),
              ("ΔT trace → base", f"{fmt(dT_d)} K", "Temperature drop of the trace loss through the dielectric "
                                                   "(45° spreading).")])
    ff = np.logspace(np.log10(max(f, 1e-3) / 100), np.log10(f * 100), 80)
    fig = go.Figure([line(ff, [hc.plate_eddy(I, x * 1e3, w, h, tp, rho_p, s if pair else None)['P'] * Lt
                               for x in ff], 'eddy loss in the base', 0),
                     line(ff, [I ** 2 * rho_c / (w * tc) * float(hc._F1(tc / float(hc.skin_depth(rho_c, x * 1e3))))
                               * Lt * ntr for x in ff], 'loss in the trace(s)', 1, dash='dash')])
    fig.add_scatter(x=[f], y=[Pp], mode='markers', name='this f', marker=dict(size=10, color=STATUS['critical']))
    fig.update_layout(xaxis=dict(type='log', title='f (kHz)'), yaxis=dict(type='log', title='P (W)'))
    show(fig, height=300, key=f"{K}|pl|c", info="Eddy loss in the metal base and loss in the trace over the "
         "frequency. The base loss rises with f² while the base is thinner than δ (partial shielding) and with √f "
         "once δ < t.")
    cm.equations([
        ("Image current in the base", [r"h_{eff} = h + \frac{w}{2\pi},\quad K(x) = \beta\,\frac{I h_{eff}}{\pi}\left["
                                       r"\frac{1}{h_{eff}^2 + x^2} - \frac{1}{h_{eff}^2 + (x - s)^2}\right]",
                                       r"\beta = \frac{X}{\sqrt{1 + X^2}},\quad X = \frac{2 h_{eff} t}{\delta^2}"],
         "The second term is the return trace; it is dropped for a far return."),
        ("Loss per length", [r"R_s = \frac{\rho}{\delta}\,\frac{\sinh 2\Delta + \sin 2\Delta}{\cosh 2\Delta - \cos "
                             r"2\Delta},\quad \Delta = t/\delta",
                             r"P' = R_s\!\int K^2 dx = \frac{\beta^2 R_s I^2}{2\pi h_{eff}}\cdot\begin{cases}1 & "
                             r"\text{far return}\\ \frac{2 s^2}{s^2 + 4 h_{eff}^2} & \text{return at } s\end{cases}"],
         None),
        ("Trace", [r"\frac{R_{ac}}{R_{dc}} = \Delta_c\,\frac{\sinh 2\Delta_c + \sin 2\Delta_c}{\cosh 2\Delta_c - "
                   r"\cos 2\Delta_c},\ \Delta_c = t_{Cu}/\delta_{Cu}",
                   r"\Delta T_d = \frac{P'_{trace}\,h}{k_d\,(w + h)}"],
         "The trace over the base behaves like one winding layer with the field on one side (Dowell, m = 1)."),
    ], sym=[("I", "RMS of the AC current", "A"), ("f", "frequency", "Hz"),
            ("w, t_Cu", "width and copper thickness of the trace", "m"),
            ("h", "dielectric thickness (trace → base)", "m"), ("h_eff", "effective height of the line current", "m"),
            ("s", "centre distance of the return trace", "m"), ("x", "position across the base", "m"),
            ("K(x)", "induced surface current density in the base", "A/m"),
            ("t, δ, Δ", "base thickness, skin depth of the base, t/δ", "m, m, –"),
            ("β", "share of the full image current that a thin base carries", "–"),
            ("R_s", "AC surface resistance of the base", "Ω"), ("ρ", f"resistivity of {pm} at {fmt(Tc)} °C", "Ω m"),
            ("P'", "loss per length", "W/m"), ("k_d", "conductivity of the dielectric", "W/(m K)"),
            ("ΔT_d", "temperature drop trace → base", "K")])
    cm.refs(['Kazimierczuk2014', 'Dowell1966'])
    sh = 100 * Pp / max(Pp + Ptr, 1e-12)
    cm.interpret([
        f"{fmt(I)} A at {fmt(f)} kHz induce **{fmt(Pp)} W** in the {fmt(tp / MM)} mm {pm} base below "
        f"{fmt(Lt / MM)} mm of trace, against {fmt(Ptr)} W in the copper itself — the base adds {sh:.0f} % to the "
        "heat of this current path.",
        ("The return trace close by cancels most of the field: keep go and return traces adjacent (or stacked) — the "
         "loss scales with s²/(s² + 4h_eff²) for s ≲ 2h_eff." if pair else
         "With the return far away the base carries the full image current. Routing the return next to the trace "
         "(distance s ≈ w) cuts this loss by an order of magnitude."),
        f"The trace loss heats the trace by {fmt(dT_d)} K above the base; the eddy loss is generated in the base "
        "itself and only adds to the load of the heat sink.",
        "Caveats: sinusoidal current, non-magnetic base (steel: μ_r ≫ 1, much higher loss), infinite base width, "
        "image approximation ±30 % when h ≈ w. For a ripple current add the loss of each harmonic."])
