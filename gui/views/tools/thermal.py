#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         thermal.py
# Date:         05.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Theory & Tools › Thermal: thermal stack designer for power semiconductors, organised like a datasheet:
    inside the package   junction → case: datasheet Foster network Z_th,jc or the layers of the package
    outside the package  case → coolant: freely configurable N layers (TIM, insulation, board, housing, ...) + cooling
    optional             extra paths (plastic housing, pins / board) and the second side of a double-side cooled part
The inside Foster network is converted into a Cauer ladder, so that it can be connected in series with the outside
layers; the result is the exact Foster network of Z_th,ja, Z_th(t) of the inside, the outside and the whole stack,
and the junction temperature for a power profile.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, STATUS, INK, INK2, MUTED, SEQ, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import pkg_art
from src.py.tools import thermStack as ts
from src.py.tools import toolData as td

K = cm.KEY + '|th'
COLS = ['layer', 'role', 'material', 't [mm]', 'w [mm]', 'l [mm]', 'k [W/mK]', 'rho [kg/m³]', 'cp [J/kgK]',
        'angle [°]']
PROPS = ['k [W/mK]', 'rho [kg/m³]', 'cp [J/kgK]']
COL_HELP = {
    'layer': "Name of the layer (free text).",
    'role': "attach: solder / sinter · package: leadframe, substrate · iso: electrical insulation · board: PCB · "
            "tim: thermal interface material · housing · cooler: heat sink / cold plate (the last layer is cooled by "
            "h on its outer face).",
    'material': "Material of the library; changing it fills k, ρ and c_p with the library values (you can then "
                "overwrite them).",
    't [mm]': "Thickness of the layer in the heat flow direction z.",
    'w [mm]': "Width of the layer footprint (0 = unlimited, the heat spreads freely).",
    'l [mm]': "Length of the layer footprint (0 = unlimited).",
    'k [W/mK]': "Thermal conductivity (prefilled from the material; estimates for TIMs and pads — use the value of "
                "your datasheet).",
    'rho [kg/m³]': "Density (only for the heat capacity, i.e. the transient).",
    'cp [J/kgK]': "Specific heat capacity (only for the transient).",
    'angle [°]': "Spreading angle θ: the heated area grows by 2 t tan θ per side (45° rule).",
}
ROLES = ['attach', 'package', 'iso', 'board', 'tim', 'housing', 'cooler', 'chip']
EXTRA_DEF = {'R_jtop': 30.0, 'A_top': 100.0, 'h_top': 10.0, 'R_jpins': 20.0, 'R_pins_amb': 40.0}


#######################################################################################################################
# Helpers
#######################################################################################################################
def _df(layers):
    return pd.DataFrame([{c: x.get(c) for c in COLS} for x in layers], columns=COLS)


def _layers(df):
    out = []
    for _, r in df.iterrows():
        if r['material'] not in td.MATERIALS or pd.isna(r['t [mm]']) or float(r['t [mm]']) <= 0:
            continue
        m = td.MATERIALS[r['material']]
        x = {'layer': str(r['layer']) if pd.notna(r['layer']) and r['layer'] else r['material'],
             'role': str(r['role']) if pd.notna(r['role']) else '', 'material': r['material'],
             't [mm]': float(r['t [mm]'])}
        for c in ('w [mm]', 'l [mm]'):
            x[c] = float(r[c]) if pd.notna(r[c]) else 0.0
        x['angle [°]'] = float(r['angle [°]']) if pd.notna(r['angle [°]']) else 45.0
        for c, v in zip(PROPS, m[:3]):
            x[c] = float(r[c]) if pd.notna(r[c]) and float(r[c]) > 0 else float(v)
        out.append(x)
    return out


def _editor(layers, key, height='auto'):
    """
    Layer table with add / delete rows. When the material of a row changes and its k, ρ, c_p were not edited, they
    are replaced by the library values of the new material (the table is re-drawn with a new revision key).
    """
    bk, rk = f"{key}|base", f"{key}|rev"
    if bk not in st.session_state:
        st.session_state[bk] = _df(layers)
    base = st.session_state[bk]
    rev = st.session_state.get(rk, 0)
    cfg = {c: st.column_config.Column(help=COL_HELP[c]) for c in COLS}
    cfg['material'] = st.column_config.SelectboxColumn('material', options=list(td.MATERIALS), required=True,
                                                       help=COL_HELP['material'], width='medium')
    cfg['role'] = st.column_config.SelectboxColumn('role', options=ROLES, help=COL_HELP['role'])
    for c in COLS[3:]:
        cfg[c] = st.column_config.NumberColumn(c, help=COL_HELP[c], min_value=0.0, format='%g')
    order = COLS if cm.expert() else [c for c in COLS if c != 'angle [°]']
    kw = {} if height == 'auto' else {'height': height}
    df = st.data_editor(base, key=f"{key}|ed{rev}", num_rows='dynamic', hide_index=True, column_config=cfg,
                        column_order=order, width='stretch', **kw)
    new, fill = df.copy(), False
    for i in new.index:
        mat = new.at[i, 'material']
        if mat not in td.MATERIALS:
            continue
        lib = td.MATERIALS[mat][:3]
        if i in base.index and base.at[i, 'material'] != mat and base.at[i, 'material'] in td.MATERIALS:
            old = td.MATERIALS[base.at[i, 'material']][:3]
            for c, vo, vn in zip(PROPS, old, lib):
                if pd.isna(new.at[i, c]) or abs(float(new.at[i, c]) - float(base.at[i, c] if pd.notna(
                        base.at[i, c]) else vo)) < 1e-12:
                    new.at[i, c] = vn
                    fill = True
        for c, vn in zip(PROPS, lib):
            if pd.isna(new.at[i, c]):
                new.at[i, c] = vn
                fill = True
    if fill:
        st.session_state[bk] = new.reset_index(drop=True)
        st.session_state[rk] = rev + 1
        st.rerun()
    return _layers(df)


def _with_board(layers, board):
    """Replaces the first contiguous run of 'board' layers by the board technology."""
    if board is None:
        return [dict(x) for x in layers]
    idx = [i for i, x in enumerate(layers) if x['role'] == 'board']
    if not idx:
        return [dict(x) for x in layers]
    i0 = i1 = idx[0]
    while i1 + 1 < len(layers) and layers[i1 + 1]['role'] == 'board':
        i1 += 1
    return [dict(x) for x in layers[:i0]] + [dict(x) for x in board] + [dict(x) for x in layers[i1 + 1:]]


@st.cache_data(show_spinner=False)
def _typical_foster(pk_name, which='main', N=4):
    """Typical Foster Z_th,jc of a preset, derived from its inside layers (case ideally cooled)."""
    main, sec = td.preset_spec(pk_name, dsc=True)
    spec = main if which == 'main' else sec
    sd = ts.side(dict(spec, outer=[], h=0.0, R_sa=1e-9), td.MATERIALS)
    R, tau, R0 = sd['zjc']
    Rf, tf = ts.foster_fit(R, tau, R0, N=N)
    return [float(f"{x:.3g}") for x in Rf], [float(f"{x:.3g}") for x in tf], float(f"{sd['A_case']:.3g}")


def _side_ui(key, pk_name, which, chip, inner, outer, h, label):
    """Inputs of one cooled side: inside (Foster or layers), outside layers, cooling. Returns the side spec."""
    ex = cm.expert()

    # ------------------------------------------
    # Inside the package
    # ------------------------------------------
    st.markdown(f"**Inside the package ({label}): junction → case**")
    mode = st.radio("Inside model", ['foster', 'layers'], horizontal=True, key=f"{key}|mode",
                    format_func={'foster': "Datasheet Foster network Z_th,jc",
                                 'layers': "Layers inside the package (chip, attach, leadframe / substrate)"}.get,
                    help="Datasheets give Z_th,jc as a Foster network (R_i, τ_i) from the junction to the case (the "
                         "cooling area). If it is not available, build the inside from its layers.",
                    label_visibility='collapsed')
    Rf, tf, A0 = _typical_foster(pk_name, which)
    if mode == 'foster':
        c1, c2 = st.columns([3, 2])
        with c1:
            fd = st.data_editor(pd.DataFrame({'R_i [K/W]': Rf, 'τ_i [s]': tf}), key=f"{key}|fst", num_rows='dynamic',
                                hide_index=False, width='stretch', column_config={
                    'R_i [K/W]': st.column_config.NumberColumn(min_value=0.0, format='%g', help="Thermal resistance "
                                                               "of the Foster term i (datasheet table or fit)."),
                    'τ_i [s]': st.column_config.NumberColumn(min_value=0.0, format='%g', help="Time constant τ_i = "
                                                             "R_i C_i of the term i.")})
        fd = fd.dropna()
        fd = fd[(fd['R_i [K/W]'] > 0) & (fd['τ_i [s]'] > 0)]
        with c2:
            A_case = cm.num(st, "Cooling area of the case A_case [mm²]", A0, f"{key}|Ac", min_value=0.01,
                            help="Area through which the heat leaves the package (exposed pad, tab or baseplate "
                                 "under the chip). The outside layers start with this footprint; the default is the "
                                 "heated area at the case of the preset.")
            st.caption(f"R_th,jc = Σ R_i = **{fd['R_i [K/W]'].sum():.3g} K/W**. Prefilled with the typical Z_th,jc of "
                       f"the preset (derived from its inside layers) — replace it with the datasheet values.")
        if fd.empty:
            st.warning("Enter at least one Foster term with R_i > 0 and τ_i > 0.")
            st.stop()
        inn = {'mode': 'foster', 'R': fd['R_i [K/W]'].to_numpy(float), 'tau': fd['τ_i [s]'].to_numpy(float)}
    else:
        c = st.columns(3)
        mats = list(td.MATERIALS)
        cmat = c[0].selectbox("Chip material", mats, index=mats.index(chip['material']), key=f"{key}|cm",
                              help="Semiconductor of the heat source (GaN-on-Si: Si substrate).")
        ct = cm.num(c[1], "Chip thickness [mm]", chip['t'], f"{key}|ct", min_value=1e-4)
        cA = cm.num(c[2], "Active chip area A_chip [mm²]", chip['A'], f"{key}|cA", min_value=0.01,
                    help="Heat source area (active area of the die, datasheet or die size); the heat is generated "
                         "on its top surface (junction).")
        inner_l = _editor(inner, f"{key}|in")
        inn = {'mode': 'layers', 'chip': {'material': cmat, 't': ct, 'A': cA}, 'layers': inner_l}
        A_case = None

    # ------------------------------------------
    # Outside the package
    # ------------------------------------------
    st.markdown(f"**Outside the package ({label}): case → coolant**")
    outer_l = _editor(outer, f"{key}|out")
    c = st.columns([2.2, 1.3, 1.3])
    cool = c[0].selectbox("Cooling of the last layer", list(td.COOLING), key=f"{key}|cool",
                          help="Heat transfer coefficient on the outer face of the last outside layer (with fins / "
                               "pin-fins as an effective value). Typical: " + "; ".join(
                              f"{k} {v}" for k, v in td.COOLING_RANGE.items()))
    h0 = h if td.COOLING[cool] is None else td.COOLING[cool]
    use_rsa = c[2].toggle("Given cooler R_th", value=False, key=f"{key}|rsa_on",
                          help="Use the thermal resistance of the heat sink / cold plate from its datasheet "
                               "(surface → coolant) instead of h on the area of the last layer.")
    if use_rsa:
        R_sa = cm.num(c[1], "R_th,sa [K/W]", 0.5, f"{key}|rsa", min_value=1e-6)
        hh = 0.0
    else:
        hh = cm.num(c[1], "h [W/m²K]", h0, f"{key}|{cool}|h", min_value=0.0)
        R_sa = None
    spec = {'inner': inn, 'outer': outer_l, 'h': hh, 'R_sa': R_sa}
    if A_case is not None:
        spec['A_case'] = A_case
    return spec


def _cross_section(res):
    """Schematic cross-section: one row per layer (heights not to scale), width = footprint, heated area inside."""
    shapes, annos = [], []
    allR = []
    rows = {}
    for k, sd in res['sides'].items():
        rr = []
        if sd['mode'] == 'layers':
            rr += [(r['layer'], r['A_in [mm²]'], r['A_out [mm²]'], r['R [K/W]'], r['t [mm]'], 'in') for r in sd['inner']]
        else:
            a = sd['A_case']
            rr.append(("Datasheet Z_th,jc", a * 0.3, a, sd['R_jc'], None, 'in'))
        rr += [(r['layer'], r['A_in [mm²]'], r['A_out [mm²]'], r['R [K/W]'], r['t [mm]'], 'out') for r in sd['outer']]
        rows[k] = rr
        allR += [x[3] for x in rr]
    rmax = max(allR) if allR else 1.0
    for k, sign in (('main', -1), ('second', 1)):
        if k not in rows:
            continue
        y = 0.0
        prev = 'in'
        for lab, a_in, a_out, R, t, part in rows[k]:
            if part != prev:
                annos.append(dict(x=0, y=y, showarrow=False, text="– – – case – – –", yshift=0,
                                  font=dict(size=11, color=STATUS['critical'])))
                prev = part
            a_in, a_out = np.sqrt(a_in), np.sqrt(a_out)
            wfoot = max(a_out, a_in) * 1.15
            y0, y1 = (y - 1, y) if sign < 0 else (y, y + 1)
            col = SEQ[int(np.clip(round(R / rmax * 4), 0, 4))][1]
            shapes.append(dict(type='rect', x0=-wfoot / 2, x1=wfoot / 2, y0=y0, y1=y1, fillcolor='#f3f2ee',
                               line=dict(color=INK2, width=0.5)))
            ys, ye = (y1, y0) if sign < 0 else (y0, y1)
            shapes.append(dict(type='path', path=f"M {-a_in / 2},{ys} L {a_in / 2},{ys} L {a_out / 2},{ye} "
                                                 f"L {-a_out / 2},{ye} Z", fillcolor=col, opacity=0.85,
                               line=dict(color=INK2, width=0.5)))
            annos.append(dict(x=wfoot / 2 + 0.5, y=(y0 + y1) / 2, xanchor='left', showarrow=False,
                              text=f"{lab}" + (f" · {t:g} mm" if t else "") + f" · {R:.3g} K/W",
                              font=dict(size=11, color=INK)))
            y = y0 if sign < 0 else y1
        Rc = res['sides'][k]['R_conv']
        if np.isfinite(Rc):
            annos.append(dict(x=0, y=y + (-0.45 if sign < 0 else 0.45), showarrow=False,
                              text=f"coolant · R_conv = {Rc:.3g} K/W", font=dict(size=11, color=SERIES[0])))
    W = max([np.sqrt(x[2]) for rr in rows.values() for x in rr] + [10])
    annos.append(dict(x=-W * 0.62, y=0, xanchor='left', showarrow=False, text="junction ▸",
                      font=dict(size=12, color=STATUS['critical'])))
    nb, nt = len(rows.get('main', [])), len(rows.get('second', []))
    fig = go.Figure()
    fig.update_layout(shapes=shapes, annotations=annos, showlegend=False,
                      title="Stack cross-section (layer heights not to scale)", xaxis=dict(visible=False,
                                                                                           range=[-W * 0.7, W * 0.9]),
                      yaxis=dict(visible=False, range=[-nb - 1, nt + 1]), margin=dict(l=10, r=260, t=60, b=10))
    return fig, 60 + 34 * (nb + nt + 2)


#######################################################################################################################
# Stack designer
#######################################################################################################################
def stack_designer():
    cm.question("What is the thermal resistance from the junction to the coolant of my part, which layer dominates, "
                "and how hot does the junction get?",
                "The model follows the datasheet: inside the package the junction-to-case network Z_th,jc (datasheet "
                "Foster network or the layers of the package), outside the package any number of layers (TIM, "
                "insulation, board, housing, …) up to the cooling. Heat flows in z and spreads in every layer.")
    ex = cm.expert()

    # ==============================================================================
    # 1 Package and operating point
    # ==============================================================================
    cm.section(1, "Package and operating point")
    c1, c2 = st.columns([3, 2])
    with c1:
        pk_name = st.selectbox("Package / module", list(td.PACKAGES), key=f"{K}|pk")
        pk = td.PACKAGES[pk_name]
        st.caption(pk['note'])
        c = st.columns(2)
        P = cm.num(c[0], "Power loss P [W]", 50.0, "th|P", min_value=0.0, help="Losses of the chip (Power losses › "
                                                                               "Loss calculator gives them).")
        Tc = cm.num(c[1], "Coolant / ambient [°C]", 40.0, "th|Tc", help="Reference temperature of all paths.")
    with c2:
        st.markdown(pkg_art.svg(pk['kind']), unsafe_allow_html=True)

    # ==============================================================================
    # 2 Main cooled side
    # ==============================================================================
    cm.section(2, "Cooled side", "Inside the package (junction → case) and outside (case → coolant). The table rows "
                                 "can be added, deleted and edited; k, ρ and c_p follow the material.")
    has_board = any(x['role'] == 'board' for x in pk['outer'])
    bd_name = 'Keep the board of the package preset'
    if has_board:
        bd_name = st.selectbox("Board technology", list(td.BOARDS), key=f"{K}|bd",
                               help="Replaces the board layers of the outside stack.")
    outer = _with_board(pk['outer'], td.BOARDS[bd_name])
    with st.container(border=True):
        main = _side_ui(f"{K}|{pk_name}|{bd_name}|m", pk_name, 'main', pk['chip'], pk['inner'], outer, pk['h'],
                        "cooled side")

    # ==============================================================================
    # 3 Options
    # ==============================================================================
    cm.section(3, "Further heat paths (optional)", "Single-side cooling is the default. Switch on the paths that "
                                                   "your design has.")
    xd = dict(EXTRA_DEF, **pk.get('extra', {}))
    c = st.columns(3)
    dsc = c[0].toggle("Double-side cooling", value='second' in pk, key=f"{K}|{pk_name}|dsc",
                      help="Adds a second cooled side above the chip (inside + outside + cooling).")
    top_on = c[1].toggle("Path through the plastic housing", value=False, key=f"{K}|{pk_name}|top",
                         help="R_th,j-top of the datasheet (junction → top of the mold) and the convection of the "
                              "housing surface.")
    pin_on = c[2].toggle("Path through the pins / board", value=False, key=f"{K}|{pk_name}|pin",
                         help="R_th,j-pins (junction → solder point of the leads) and the board to the ambient.")
    second, extra = None, []
    if dsc:
        sec = pk.get('second') or {'inner': pk['inner'], 'outer': pk['outer'], 'h': pk['h']}
        with st.container(border=True):
            second = _side_ui(f"{K}|{pk_name}|s", pk_name, 'second', dict(pk['chip'], t=1e-3), sec['inner'],
                              sec['outer'], sec['h'], "second side")
    if top_on or pin_on:
        with st.container(border=True):
            c = st.columns(5)
            if top_on:
                Rjt = cm.num(c[0], "R_th,j-top [K/W]", xd['R_jtop'], f"th|{pk_name}|rjt", min_value=1e-6,
                             help="Junction → top surface of the plastic housing (datasheet Ψ_jt / R_th,j-top; "
                                  "the default is an estimate).")
                At = cm.num(c[1], "Top area [mm²]", xd['A_top'], f"th|{pk_name}|at", min_value=0.01)
                ht = cm.num(c[2], "h top [W/m²K]", xd['h_top'], f"th|{pk_name}|ht", min_value=1e-3,
                            help="Natural convection 5–25, forced air 25–250.")
                extra.append({'name': 'housing (top)', 'R': Rjt + 1 / (ht * At * 1e-6)})
            if pin_on:
                Rjp = cm.num(c[3], "R_th,j-pins [K/W]", xd['R_jpins'], f"th|{pk_name}|rjp", min_value=1e-6,
                             help="Junction → solder point of the pins (datasheet R_th,j-sp; the default is an "
                                  "estimate).")
                Rpa = cm.num(c[4], "R_th pins → ambient [K/W]", xd['R_pins_amb'], f"th|{pk_name}|rpa",
                             min_value=1e-6, help="Board copper and its convection from the solder points.")
                extra.append({'name': 'pins / board', 'R': Rjp + Rpa})
    spread, lee, Nf = 'angle', True, 5
    if ex:
        c = st.columns([2, 2, 1])
        spread = c[0].radio("Heat spreading in the layers", ['angle', 'ratio'], horizontal=True, key=f"{K}|sp",
                            format_func={'angle': 'fixed angle (column)', 'ratio': 'conductivity ratio'}.get,
                            help="fixed angle: θ from the column 'angle [°]' (45° rule). conductivity ratio: θ = "
                                 "arctan(k_i / k_i+1), 10–80°.")
        lee = c[1].toggle("Lee spreading in the cooled plate", value=True, key=f"{K}|lee",
                          help="The last outside layer is a plate cooled by h: its spreading resistance follows Lee "
                               "et al. (1995) on the full plate area.")
        Nf = int(cm.num(c[2], "Foster terms", 5, "th|N", min_value=1, max_value=8, format='%d',
                        help="Terms of the exported Foster network (the model workbook holds up to 5)."))
    cm.hidden(0 if ex else 3, "model settings (spreading rule, Lee spreading, Foster terms)")
    try:
        res = ts.stackModel(main, td.MATERIALS, second, extra, P=P, T_cool=Tc, N_fit=Nf, spread=spread,
                            lee_last=lee)
    except (ValueError, np.linalg.LinAlgError, KeyError) as e:
        st.error(f"The stack cannot be computed: {e}")
        return None
    st.session_state[f"{K}|res"] = res
    f = res['fit']
    tau_eq = float(np.sum(f['R'] * f['tau']) / np.sum(f['R']))
    sm = res['sides']['main']
    tim = next((r for r in sm['outer'] if r.get('role') in ('tim', 'iso')), None)          # first interface layer
    cm.chain()['thermal'] = {'R_ja': res['R_ja'], 'R': f['R'].tolist(), 'tau': f['tau'].tolist(),
                             'package': pk_name, 'kind': td.PACKAGES[pk_name].get('kind', ''), 'T_cool': Tc, 'R_jc': sm['R_jc'], 'R_ca': sm['R_ca'],
                             'tim': None if tim is None else {'layer': tim['layer'], 'material': tim['material'],
                                                              't_um': 1e3 * tim['t [mm]'], 'k': tim['k [W/mK]'],
                                                              'A': tim['A_in [mm²]'], 'R': tim['R [K/W]']},
                             'paths': len(res['sides']) + len(res.get('extra') or [])}

    # ==============================================================================
    # Result
    # ==============================================================================
    cm.section(None, "Result")
    cm.tiles([("R_th junction → coolant", f"{res['R_ja']:.3g} K/W", "All paths in parallel: 1/R = Σ 1/R_path."),
              ("Junction temperature", f"{res['Tj']:.1f} °C", f"T_j = T_coolant + P R_th = {Tc:g} + {P:g} · "
                                                               f"{res['R_ja']:.3g}"),
              ("Inside R_th,jc", f"{sm['R_jc']:.3g} K/W", "Junction → case of the cooled side (datasheet value)."),
              ("Outside R_th,ca", f"{sm['R_ca']:.3g} K/W", "Case → coolant: outside layers and cooling."),
              ("Case temperature", f"{res['profile']['main']['T_case']:.1f} °C", "Temperature of the cooling area "
                                                                                 "of the package."),
              ("Mean time constant", f"{fmt(tau_eq)} s", "R-weighted mean τ: the junction reaches ≈ 63 % of its "
                                                         "rise after about this time.")])
    tabs = st.tabs(["Z_th inside / outside / total", "Cross-section", "Resistance per layer", "Temperature profile",
                    "Equations & materials"])
    with tabs[0]:
        fo = res['foster']
        taus = np.r_[fo['tau'], sm['zjc'][1], sm['zca'][1] if len(sm['zca'][1]) else []]
        tt = np.logspace(np.log10(max(taus.min() / 10, 1e-7)), np.log10(taus.max() * 10), 300)
        z_ex = ts.zth(tt, fo['R'], fo['tau'], fo['R0'])
        z_fit = ts.zth(tt, f['R'], f['tau'])
        z_jc = ts.zth(tt, sm['zjc'][0], sm['zjc'][1], sm['zjc'][2])
        tr = [line(tt, z_jc, "inside Z_th,jc (case ideally cooled)", 0),
              line(tt, z_ex, "total Z_th,ja (exact)", 1, width=3),
              line(tt, z_fit, f"total, Foster fit ({len(f['R'])} terms)", 1, dash='dash')]
        if len(sm['zca'][1]):
            tr.insert(1, line(tt, ts.zth(tt, sm['zca'][0], sm['zca'][1], 0.0), "outside Z_th,ca (heat into the "
                                                                               "case)", 2))
        if 'second' in res['sides']:
            s2 = res['sides']['second']
            tr.append(line(tt, ts.zth(tt, s2['zjc'][0], s2['zjc'][1], s2['zjc'][2]), "second side Z_th,jc", 3,
                           dash='dot'))
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(tr)
            fig.update_layout(title="Thermal impedance: inside, outside and the whole stack",
                              xaxis=dict(title="t [s]", type='log'), yaxis=dict(title="Z_th [K/W]", type='log'),
                              hovermode='x unified')
            show(fig, 380, key=f"{K}|zth", legend_below=True, info=(
                "Z_th(t): temperature rise of the junction per watt after a power step. Blue: the inside of the "
                "package alone (the datasheet curve, case held at the coolant temperature). Green: the outside stack "
                "alone when the heat is fed into the case. Orange: the whole stack. Up to the time constant of the "
                "package the total follows the inside curve; at long times it reaches R_th,jc + R_th,ca. In between "
                "the total is smaller than the sum of both curves, because the outside layers only heat up once the "
                "heat has passed the package."))
        with c2:
            ft = pd.DataFrame({'term': np.arange(1, len(f['R']) + 1), 'R_i [K/W]': f['R'], 'τ_i [s]': f['tau'],
                               'C_i [Ws/K]': f['C']})
            table(ft, hide_index=True)
            m = z_ex > 0.02 * res['R_ja']
            err = float(np.max(np.abs(z_fit - z_ex)[m] / z_ex[m]))
            st.caption(f"Foster network of Z_th,ja junction → coolant. Largest deviation of the fit: {100 * err:.1f} % "
                       f"(above 2 % of R_th). Σ R_i = {f['R'].sum():.4g} K/W.")
            cm.export(ft.rename(columns={'R_i [K/W]': 'Rth [K/W]', 'C_i [Ws/K]': 'Cth [Ws/K]'})[
                          ['term', 'Rth [K/W]', 'Cth [Ws/K]']], "foster_network", label="Foster Z_th,ja (CSV)",
                      note="Paste Rth/Cth into the model workbook, sheet Mdl, block 'Thermal Model (Foster)', rows "
                           "Node 1–5 (Thermal Model 3 in Reliability › Thermal).")
            Rj, tj = (sm['zjc'][0], sm['zjc'][1]) if sm['mode'] == 'foster' else ts.foster_fit(*sm['zjc'], N=4)
            cm.export(pd.DataFrame({'term': np.arange(1, len(Rj) + 1), 'Rth [K/W]': Rj, 'tau [s]': tj}),
                      "foster_zthjc", label="Foster Z_th,jc (CSV)", key=f"{K}|dljc")
            prow = pd.DataFrame([{'Rth[K/W]': res['R_ja'], 'Cth[Ws/K]': tau_eq / res['R_ja']}])
            cm.export(prow, "parts_rth_cth", label="Lumped R_th/C_th (CSV)",
                      note="Single RC for the Parts sheet of the architecture workbook (Thermal Model 2).")
            cm.export_json({'package': pk_name, 'board': bd_name, 'main': main, 'second': second, 'extra': extra},
                           "thermal_stack", label="Stack (JSON)")
    with tabs[1]:
        fig, hgt = _cross_section(res)
        show(fig, max(hgt, 320), key=f"{K}|xs", info=(
            "Each row is a layer (heights equal for readability, the thickness is in the label). The grey box is the "
            "footprint of the layer, the coloured trapezoid the area that the heat uses: it starts at the heated area "
            "of the layer above and grows with the spreading angle. Darker = larger resistance. The red dashed line "
            "is the case: above it the inside of the package (datasheet Z_th,jc), below it the outside stack."))
    with tabs[2]:
        rows = []
        for k, sd in res['sides'].items():
            nm = 'cooled side' if k == 'main' else 'second side'
            if sd['mode'] == 'layers':
                rows += [{'path': nm, 'part': 'inside', 'layer': r['layer'], 'material': r['material'],
                          't [mm]': r['t [mm]'], 'k [W/mK]': r['k [W/mK]'], 'A_in [mm²]': r['A_in [mm²]'],
                          'A_out [mm²]': r['A_out [mm²]'], 'R [K/W]': r['R [K/W]'], 'C [J/K]': r['C [J/K]']}
                         for r in sd['inner']]
            else:
                rows.append({'path': nm, 'part': 'inside', 'layer': 'datasheet Z_th,jc', 'material': '–',
                             't [mm]': np.nan, 'k [W/mK]': np.nan, 'A_in [mm²]': np.nan,
                             'A_out [mm²]': sd['A_case'], 'R [K/W]': sd['R_jc'],
                             'C [J/K]': float(np.sum(sd['foster_in'][1] / sd['foster_in'][0]))})
            rows += [{'path': nm, 'part': 'outside', 'layer': r['layer'], 'material': r['material'],
                      't [mm]': r['t [mm]'], 'k [W/mK]': r['k [W/mK]'], 'A_in [mm²]': r['A_in [mm²]'],
                      'A_out [mm²]': r['A_out [mm²]'], 'R [K/W]': r['R [K/W]'], 'C [J/K]': r['C [J/K]']}
                     for r in sd['outer']]
            if np.isfinite(sd['R_conv']):
                rows.append({'path': nm, 'part': 'outside', 'layer': 'cooling (1/hA or R_sa)', 'material': '–',
                             't [mm]': np.nan, 'k [W/mK]': np.nan, 'A_in [mm²]': np.nan, 'A_out [mm²]': np.nan,
                             'R [K/W]': sd['R_conv'], 'C [J/K]': 0.0})
        for e in res['extra']:
            rows.append({'path': e['name'], 'part': 'extra', 'layer': 'lumped path', 'material': '–',
                         't [mm]': np.nan, 'k [W/mK]': np.nan, 'A_in [mm²]': np.nan, 'A_out [mm²]': np.nan,
                         'R [K/W]': e['R'], 'C [J/K]': 0.0})
        df = pd.DataFrame(rows)
        for p_ in df['path'].unique():
            m = df['path'] == p_
            df.loc[m, 'share of path [%]'] = 100 * df.loc[m, 'R [K/W]'] / df.loc[m, 'R [K/W]'].sum()
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure()
            for i, part in enumerate(('inside', 'outside', 'extra')):
                d = df[df['part'] == part]
                if len(d):
                    fig.add_trace(go.Bar(y=[f"{a}: {b}" for a, b in zip(d['path'], d['layer'])], x=d['R [K/W]'],
                                         orientation='h', name=part, marker=dict(color=SERIES[i], cornerradius=3)))
            fig.update_layout(title="Thermal resistance per layer", xaxis_title="R [K/W]",
                              yaxis=dict(autorange='reversed'))
            show(fig, 80 + 24 * len(df), key=f"{K}|rbar", legend_below=True, info=(
                "Resistance of every layer: blue inside the package, orange outside, green the extra paths. The "
                "layers of one path are in series, so the longest bars set the junction temperature."))
        with c2:
            table(df, hide_index=True, height=60 + 35 * len(df))
    with tabs[3]:
        fig = go.Figure()
        for i, (k, pr) in enumerate(res['profile'].items()):
            sgn = -1 if k == 'main' else 1
            z = np.array(pr['z [mm]']) * sgn
            fig.add_trace(go.Scatter(x=np.r_[z, z[-1]], y=np.r_[pr['T [°C]'], Tc], mode='lines+markers',
                                     name='cooled side' if k == 'main' else 'second side',
                                     line=dict(color=SERIES[i], width=2), text=pr['interface'] + ['coolant'],
                                     hovertemplate="%{text}<br>z = %{x:.3f} mm<br>T = %{y:.1f} °C<extra></extra>"))
        fig.update_layout(title=f"Steady temperature at the layer interfaces (P = {P:g} W)",
                          xaxis_title="z from the junction [mm] (cooled side < 0)", yaxis_title="T [°C]")
        show(fig, 340, key=f"{K}|prof", info=(
            "Temperature at every interface for the steady power P: the drop over a layer is the heat through it × "
            "its resistance. With a datasheet Z_th,jc the inside is one step from the junction to the case. The last "
            "step is the cooling to the coolant."))
    with tabs[4]:
        cm.equations([
            ("Layer resistance and capacity with heat spreading",
             [r"A(z)=\big(w_0+2z\tan\theta\big)\big(l_0+2z\tan\theta\big)",
              r"R_i=\int_0^{t_i}\frac{dz}{k_i\,A(z)},\qquad C_i=\rho_i c_{p,i}\int_0^{t_i}A(z)\,dz"],
             "The heated area grows from the footprint of the layer above with the spreading angle θ, limited by the "
             "footprint of the layer."),
            ("Cooling of the last layer (Lee et al.)",
             [r"R=\frac{\psi}{\sqrt{\pi}\,k\,a}+\frac{t}{kA_p}+\frac{1}{hA_p}",
              r"\psi=\frac{\varepsilon\tau}{\sqrt{\pi}}+\tfrac12(1-\varepsilon)^{3/2}\Phi_c,\quad "
              r"\Phi_c=\frac{\tanh(\lambda_c\tau)+\lambda_c/Bi}{1+\lambda_c/Bi\,\tanh(\lambda_c\tau)}"],
             "Spreading in a plate heated by a smaller source and cooled by h on its full area A_p."),
            ("Inside: datasheet Foster network → Cauer ladder",
             [r"Z_{th,jc}(s)=\sum_i\frac{R_i}{1+s\tau_i}=\cfrac{1}{sC_1+\cfrac{1}{R_1+\cfrac{1}{sC_2+\dots}}}"],
             "A Foster network is a curve fit, its nodes are not physical; only the equivalent Cauer ladder (Lanczos "
             "tridiagonalisation, exact) can be connected in series with the outside layers."),
            ("Whole stack", [r"\frac{1}{R_{th,ja}}=\sum_{paths}\frac{1}{R_{jc}+R_{ca}},\qquad "
                             r"Z_{th,ja}(t)=\sum_i R_i\left(1-e^{-t/\tau_i}\right),\ \tau_i=1/\lambda_i"],
             "The paths (cooled side, second side, housing, pins) are in parallel. The eigenvalues λ of the network "
             "give the exact Foster network of the junction, Σ R_i = R_th,ja."),
        ], [('t_i, k_i', "thickness and thermal conductivity of layer i", 'm, W/(m K)'),
            ('ρ_i, c_p,i', "density and specific heat", 'kg/m³, J/(kg K)'),
            ('A(z), w_0, l_0', "heated area at depth z, its width and length at the top of the layer", 'm²'),
            ('θ', "spreading angle", '°'), ('h', "heat transfer coefficient to the coolant", 'W/(m² K)'),
            ('R_i, C_i, τ_i', "Foster terms: resistance, capacity, time constant", 'K/W, J/K, s'),
            ('R_jc, R_ca', "junction → case (inside), case → coolant (outside)", 'K/W'),
            ('a, b, A_p', "radii of the heat source and the plate (equal-area circles), plate area", 'm, m²'),
            ('ε, τ, Bi, λ_c', "a/b, t/b, h b/k, π + 1/(√π ε) (Lee et al.)", '-')], expanded=True)
        mats = pd.DataFrame([{'material': k, 'k [W/mK]': v[0], 'ρ [kg/m³]': v[1], 'c_p [J/kgK]': v[2],
                              'group': v[3], 'note': v[4]} for k, v in td.MATERIALS.items()])
        st.markdown("**Material library** (TIM, pad and potting values are estimates — use your datasheet)")
        table(mats, hide_index=True, column_config={'note': st.column_config.TextColumn(width='large')})
        st.caption(f"Effective k of the default via field (0.3 mm drill, 1 mm pitch, 25 µm plating, unfilled): "
                   f"{td.via_k(1.0, 0.3, 25):.2f} W/(m K).")
        cm.refs(td.MATERIAL_REFS + td.PACKAGE_REFS + ['Gerstenmaier2007'])

    # ==============================================================================
    # Interpretation
    # ==============================================================================
    lay = [(r['layer'], r['R [K/W]'], r['role'], 'inside') for r in sm['inner']] + \
          [(r['layer'], r['R [K/W]'], r['role'], 'outside') for r in sm['outer']]
    worst = max(lay, key=lambda x: x[1]) if lay else None
    R_main = sm['R']
    s_in = sm['R_jc'] / R_main
    p1 = (f"With {P:g} W the junction of the {pk_name} reaches {res['Tj']:.1f} °C, i.e. {res['Tj'] - Tc:.1f} K above "
          f"the coolant at {Tc:g} °C. The thermal resistance from the junction to the coolant is "
          f"{res['R_ja']:.3g} K/W, of which {sm['R_jc']:.3g} K/W ({100 * s_in:.0f} %) lie inside the package and "
          f"{sm['R_ca']:.3g} K/W ({100 * (1 - s_in):.0f} %) outside. The case, i.e. the cooling area of the package, "
          f"runs at {res['profile']['main']['T_case']:.1f} °C.")
    if s_in < 0.3:
        p2 = ("The outside stack dominates: the package itself is not the bottleneck, and the junction temperature "
              "is set by the interface materials, the board and the cooling. A better TIM, a thinner insulation or a "
              "larger cooled area pays off more than a better package.")
    elif s_in > 0.6:
        p2 = ("The inside of the package dominates: the outside stack is already good, and further improvements of "
              "the cooling have little effect. A package with a lower R_th,jc (sintered attach, better substrate, "
              "larger chip) is the lever.")
    else:
        p2 = "Inside and outside contribute similarly, so both the package and the outside stack matter."
    if worst:
        p2 += (f" The largest single resistance is the layer '{worst[0]}' ({worst[3]}, {worst[1]:.3g} K/W, "
               f"{100 * worst[1] / R_main:.0f} % of the cooled side).")
    sh = res['share']
    p3 = ""
    if len(sh) > 1:
        p3 = ("The heat splits over the paths as follows: " + ", ".join(
            f"{'cooled side' if k == 'main' else 'second side' if k == 'second' else k} {100 * v:.0f} %"
            for k, v in sh.items()) + ". ")
        if any(k not in ('main', 'second') and v < 0.05 for k, v in sh.items()):
            p3 += ("It should be noted that the extra paths through the housing or the pins carry only a few percent: "
                   "they matter for small SMD parts without a heat sink, but not for a cooled power part. ")
    p3 += (f"Thermally, the stack settles after about {fmt(3 * tau_eq)} s (three mean time constants). Power pulses "
           f"shorter than {fmt(f['tau'][0])} s only heat the chip and its attach; such pulses are covered by the "
           f"datasheet Z_th,jc alone, whereas load changes in the range of seconds to minutes also heat the outside "
           f"stack.")
    cm.interpret([p1, p2, p3])
    return res


#######################################################################################################################
# Transient junction temperature
#######################################################################################################################
def transient():
    cm.question("How does the junction temperature follow a power profile (step, pulse train, own profile), and how "
                "large are the temperature swings that drive the lifetime?",
                "Uses the Foster network of the stack designer (tab 'Stack designer').")
    res = st.session_state.get(f"{K}|res")
    if res is None:
        st.info("Open the tab 'Stack designer' first: its Foster network is used here.")
        return
    f = res['fit']
    c = st.columns(5)
    kind = c[0].selectbox("Power profile", ['Pulse train', 'Step', 'Own CSV (t [s], P [W])'], key=f"{K}|tr|k")
    P = cm.num(c[1], "Power in the pulse [W]", 50.0, "tr|P", min_value=0.0)
    ton = cm.num(c[2], "On-time t_on [s]", 2.0, "tr|on", min_value=1e-6)
    per = cm.num(c[3], "Period [s]", 6.0, "tr|per", min_value=1e-6)
    Tc = cm.chain().get('thermal', {}).get('T_cool', 40.0)                     # coolant of the stack designer
    tmax = max(10 * per, 5 * float(np.max(f['tau'])))
    t = np.linspace(0, tmax, 4000)
    if kind == 'Step':
        Pt = np.full_like(t, P)
    elif kind == 'Pulse train':
        Pt = np.where((t % per) < ton, P, 0.0)
    else:
        up = c[4].file_uploader("CSV", type=['csv'], key=f"{K}|tr|up", label_visibility='collapsed')
        if up is None:
            st.info("Upload a CSV with the columns t [s] and P [W].")
            return
        d = pd.read_csv(up)
        t, Pt = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    dT = ts.response(t, Pt, f['R'], f['tau'])
    Tj = Tc + dT
    last = t >= t[-1] - per if kind == 'Pulse train' else t >= 0
    Tmax, Tmin = float(Tj[last].max()), float(Tj[last].min())
    cm.tiles([("T_j,max", f"{Tmax:.1f} °C"), ("T_j,min (periodic)", f"{Tmin:.1f} °C"),
              ("ΔT_j swing", f"{Tmax - Tmin:.1f} K", "Swing of the junction temperature in the settled pulse train: "
                                                    "the ΔT_j of the power cycling models (LESIT, CIPS08)."),
              ("Mean T_j", f"{float(np.mean(Tj[last])):.1f} °C")])
    s = max(1, len(t) // 3000)
    fig = go.Figure([line(t[::s], Tj[::s], 'T_j', 1)])
    fig.add_trace(go.Scatter(x=t[::s], y=Pt[::s], name='P', yaxis='y2', line=dict(color=MUTED, width=1, shape='hv')))
    fig.update_layout(title="Junction temperature for the power profile", xaxis_title="t [s]",
                      yaxis_title="T_j [°C]", yaxis2=dict(title="P [W]", overlaying='y', side='right',
                                                          showgrid=False), hovermode='x unified')
    show(fig, 360, key=f"{K}|tr", info=(
        "Junction temperature (orange) computed with the Foster network of the stack for the power profile (grey, "
        "right axis). Each Foster term heats up and cools down with its time constant; short pulses only lift the "
        "fast terms (die, attach), long ones the whole stack."))
    cm.chain()['swing'] = {'dTj': Tmax - Tmin, 'Tjmin': Tmin, 'Tjmax': Tmax, 'ton': ton}
    cm.interpret([
        f"For the chosen profile the junction swings by {Tmax - Tmin:.1f} K between {Tmin:.1f} °C and {Tmax:.1f} °C "
        f"once the pulse train has settled. This swing ΔT_j, together with T_j,min and the on-time t_on = {ton:g} s, "
        f"is the load of the power cycling models (LESIT, CIPS08) and can be used directly in Reliability › Lifetime "
        f"calculator or Testing › Quick test design.",
        f"A longer on-time at the same power heats more of the slow Foster terms (substrate, baseplate, cooler) and "
        f"therefore increases ΔT_j, whereas pulses much shorter than the smallest time constant ({fmt(f['tau'][0])} s) "
        f"are filtered by the chip and hardly change its temperature. It should be noted that the swing, not the "
        f"mean temperature, drives bond wire and solder fatigue.",
    ])


#######################################################################################################################
# Die attach & voids
#######################################################################################################################
def die_attach():
    cm.question("How much does the die attach layer add to the thermal resistance, and how much do voids in it (or "
                "their growth during power cycling) raise the junction temperature?",
                "The solder or sinter layer under the die conducts the heat through its area A over its thickness t. "
                "Voids take away a part of that area; the heat has to pass through the rest. Power cycling grows "
                "cracks and voids, which is why R_th,jc is the end-of-life criterion of power cycling tests.")
    cm.section(1, "Die and layer")
    with st.container(border=True):
        c = st.columns(5)
        a = cm.num(c[0], "Die edge a [mm]", 10.0, "da|a", min_value=0.1, help="Edge of a square die, e.g. 10 mm "
                                                                            "for a 200 A IGBT.")
        t = cm.num(c[1], "Layer thickness t [µm]", 70.0, "da|t", min_value=1.0,
                   help="Solder 50–100 µm, silver sinter 20–50 µm.")
        k = cm.num(c[2], "Conductivity k [W/(m K)]", 58.0, "da|k", min_value=0.01,
                   help="SAC305 ≈ 58, SnPb ≈ 50, silver sinter ≈ 150–250 W/(m K).")
        v = cm.num(c[3], "Void fraction φ [%]", 20.0, "da|v", min_value=0.0, max_value=95.0,
                   help="Share of the die area without solder (X-ray inspection: < 5–10 % is a typical limit).") / 100
        P = cm.num(c[4], "Power per die P [W]", 150.0, "da|P", min_value=0.0)
    A = (a * 1e-3) ** 2
    R0 = t * 1e-6 / (k * A)
    Rv = R0 / (1 - v)
    cm.section(None, "Result")
    cm.tiles([("R without voids", f"{1e3 * R0:.2f} mK/W"), ("R with voids", f"{1e3 * Rv:.2f} mK/W",
                                                             f"+{100 * (Rv / R0 - 1):.0f} %"),
              ("Extra ΔT_j", f"{P * (Rv - R0):.2f} K", "P · (R_void − R) at the given power."),
              ("Layer ΔT", f"{P * Rv:.2f} K", "Temperature drop across the layer with voids.")])
    vv = np.linspace(0, 0.8, 200)
    fig = go.Figure([line(100 * vv, 1e3 * R0 / (1 - vv), "R of the layer", 0)])
    fig.add_vline(x=100 * v, line=dict(color=MUTED, dash='dot'), annotation_text='design',
                  annotation_font_color=INK2)
    fig.add_hline(y=1.2e3 * R0, line=dict(color=STATUS['warning'], dash='dash'), annotation_text='+20 %',
                  annotation_font_color=STATUS['warning'])
    fig.update_layout(title="Resistance of the layer over the void fraction", xaxis_title="void fraction φ [%]",
                      yaxis_title="R [mK/W]", showlegend=False)
    show(fig, 300, key=f"{cm.KEY}|th|void", info=(
        "Resistance of the die attach layer over the void fraction (evenly spread voids, 1D). It rises slowly at "
        "first and steeply above about 50 %; the dashed line marks +20 %, the end-of-life criterion of R_th in "
        "power cycling (reached at φ ≈ 17 % of the layer alone)."))
    cm.equations([("Layer resistance", r"R=\frac{t}{k\,A},\qquad R_{void}=\frac{t}{k\,A\,(1-\varphi)}",
                   "1D, voids evenly spread; a void under the hot spot is worse because the heat cannot spread "
                   "sideways in the thin layer.")],
                 [('t', "thickness of the layer", 'm'), ('k', "thermal conductivity of the layer", 'W/(m K)'),
                  ('A', "die area a²", 'm²'), ('φ', "void fraction of the area", '-'),
                  ('R, R_void', "thermal resistance of the layer without and with voids", 'K/W')])
    cm.interpret([
        f"The {t:g} µm layer under the {a:g} × {a:g} mm² die has R = {t:g}·10⁻⁶ m / ({k:g} W/(m K) · "
        f"{A * 1e6:g}·10⁻⁶ m²) = {1e3 * R0:.2f} mK/W. With {100 * v:g} % voids the heat has only "
        f"{100 * (1 - v):g} % of the area, so R = {1e3 * Rv:.2f} mK/W (+{100 * (Rv / R0 - 1):.0f} %), which at "
        f"{P:g} W raises the junction by {P * (Rv - R0):.2f} K.",
        "This is small against the whole stack (tab 'Stack designer'), but the die attach is the layer that ages: "
        "cracks grow from the corners during power cycling and the voids under the hot spot block the heat locally. "
        "Therefore R_th,jc is monitored in power cycling tests (end of life at +20 %), and the X-ray void limit of "
        "the production keeps the start value low.",
    ])
    cm.refs(['Wintrich2015', 'AQG324', 'JESD51-14'])


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Thermal", "Thermal stack of power semiconductors: packages, boards, interface materials and "
                            "cooling in one 1D model with heat spreading, Foster networks and junction temperature.")
    tabs = st.tabs(["Stack designer", "Transient T_j", "Die attach & voids"])
    with tabs[0]:
        stack_designer()
    with tabs[1]:
        transient()
    with tabs[2]:
        die_attach()
