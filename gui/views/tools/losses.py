#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         losses.py
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
Theory & Tools › Power losses: conduction, switching, recovery and dead-time losses of Si IGBT/diode, Si MOSFET, SiC
MOSFET and GaN HEMT in an inverter (B6, B4), a buck / boost converter or directly at one device (general mode, no
topology), with electro-thermal coupling, a schematic and the waveforms of every topology, technology comparison and
export of the losses to the workbooks of the reliability mode.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, STATUS, INK2, MUTED, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools import topo_art
from src.py.tools import lossCalc as lc
from src.py.tools import toolData as td

K = cm.KEY + '|ls'
PARTS = ['T cond', 'D cond', 'dead time', 'T on', 'T off', 'D rec']
PART_TXT = {'T cond': 'transistor conduction', 'D cond': 'diode conduction', 'dead time': 'dead-time conduction',
            'T on': 'turn-on', 'T off': 'turn-off', 'D rec': 'reverse recovery'}
EXPERT_PAR = {'dV0', 'dr', 'dVF0', 'drF', 'kR', 'Kv', 'Ki', 'Kv_rec', 'Ki_rec', 'TcE', 't_dead'}


#######################################################################################################################
# Inputs
#######################################################################################################################
def _device(key, default=0):
    c1, c2 = st.columns([2, 3])
    name = c1.selectbox("Device", list(td.DEVICES), index=default, key=f"{key}|dev",
                        format_func=lambda n: f"{td.DEVICES[n]['tech']} · {n}")
    d0 = td.DEVICES[name]
    c2.caption(d0['src'] + f". Rated {d0['V_rated']} V / {d0['I_rated']} A.")
    ex = cm.expert()
    with st.expander("Device parameters (editable)", icon=':material/tune:'):
        rows = [{'parameter': sym, 'value': float(d0[k]), 'unit': u, 'meaning': txt, 'key': k}
                for k, sym, u, txt in td.DEVICE_PARAMS
                if not (d0['type'] == 'IGBT' and k in ('R25', 'kR', 'VSD')) and
                not (d0['type'] == 'MOSFET' and k in ('V0', 'r', 'dV0', 'dr', 'VF0', 'rF', 'dVF0', 'drF')) and
                (ex or k not in EXPERT_PAR)]
        df = st.data_editor(pd.DataFrame(rows), key=f"{key}|{name}|par|{int(ex)}", hide_index=True,
                            disabled=['parameter', 'unit', 'meaning', 'key'],
                            column_config={'value': st.column_config.NumberColumn(format='%g'),
                                           'meaning': st.column_config.TextColumn(width='large'), 'key': None})
        d = dict(d0)
        for _, r in df.iterrows():
            if pd.notna(r['value']):
                d[r['key']] = float(r['value'])
        st.caption("Values at the reference temperature T_ref (on-state) and at the datasheet test point (switching "
                   "energies). Take them from the datasheet of the part; the library values are typical examples.")
        cm.hidden(0 if ex else len([k for k, *_ in td.DEVICE_PARAMS if k in EXPERT_PAR]),
                  "parameters (temperature coefficients, exponents K_v / K_i, dead time)")
    return name, d


def _operating_point(key, d, name):
    """Topology, its inputs, the schematic and the waveforms. Returns (topology, operating point dict)."""
    topo = st.selectbox("Topology", list(lc.TOPOLOGIES), format_func=lc.TOPOLOGIES.get, key=f"{key}|topo",
                        help="'General' needs no topology: enter the currents and the voltage directly at the "
                             "device, e.g. from a simulation or a measurement.")
    V = 0.6 * d['V_rated']
    c = st.columns(6)
    if topo in ('B6', 'B4'):
        op = {'Vdc': cm.num(c[0], "DC-link voltage V_dc [V]", round(V), f"ls|{name}|{topo}|V", min_value=0.0),
              'I': cm.num(c[1], "Phase current amplitude Î [A]", round(0.5 * d['I_rated'], 1), f"ls|{name}|{topo}|I",
                          min_value=0.0, help="Peak value of the sinusoidal phase current (I_rms = Î / √2)."),
              'M': cm.num(c[2], "Modulation index M", 0.9, f"ls|{name}|{topo}|M", min_value=0.0, max_value=1.15,
                          help="Phase voltage amplitude / (V_dc / 2); the duty of the upper switch is "
                               "½ (1 + M sin ωt)."),
              'cosphi': cm.num(c[3], "Power factor cos φ", 0.85, f"ls|{name}|{topo}|cp", min_value=-1.0, max_value=1.0,
                               help="cos φ < 0: generator / braking operation, the diodes conduct more."),
              'fs': 1e3 * cm.num(c[4], "Switching frequency f_s [kHz]",
                                 10.0 if d['type'] == 'IGBT' else 40.0, f"ls|{name}|{topo}|fs", min_value=0.01)}
    elif topo in ('Buck', 'Boost'):
        Vin = cm.num(c[0], "Input voltage V_in [V]", round(V if topo == 'Buck' else V / 2), f"ls|{name}|{topo}|Vi",
                     min_value=0.1)
        Vout = cm.num(c[1], "Output voltage V_out [V]", round(V / 2 if topo == 'Buck' else V),
                      f"ls|{name}|{topo}|Vo", min_value=0.1)
        op = {'Vin': Vin, 'Vout': Vout,
              'I': cm.num(c[2], "Inductor current I [A]", round(0.5 * d['I_rated'], 1), f"ls|{name}|{topo}|I",
                          min_value=0.0, help="Mean inductor current (output current of the buck, input current of "
                                              "the boost)."),
              'L': 1e-6 * cm.num(c[3], "Inductance L [µH]", 47.0, f"ls|{name}|{topo}|L", min_value=0.01),
              'fs': 1e3 * cm.num(c[4], "Switching frequency f_s [kHz]", 10.0 if d['type'] == 'IGBT' else 100.0,
                                 f"ls|{name}|{topo}|fs", min_value=0.01),
              'sync': c[5].toggle("Synchronous rectifier", value=d['type'] == 'MOSFET', key=f"{K}|{name}|{topo}|sync",
                                  disabled=d['type'] == 'IGBT',
                                  help="On: the rectifier is a MOSFET with its gate on (channel conduction, only "
                                       "the dead time in the body diode). Off: the body diode conducts.")}
        if topo == 'Buck' and Vout >= Vin or topo == 'Boost' and Vout <= Vin:
            st.warning("Buck needs V_out < V_in, boost needs V_out > V_in.")
    else:
        op = _general_inputs(name, d)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(topo_art.svg(topo), unsafe_allow_html=True)
    with c2:
        _waveform(topo, d, op)
    return topo, op


def _general_inputs(name, d):
    """Inputs of the general device mode: currents as pulses, as mean / RMS values or as a sampled waveform."""
    k = f"ls|{name}|dev"
    kind = st.radio("Current given as", ['pulse', 'avg', 'wave'], horizontal=True, key=f"{K}|{name}|gkind",
                    format_func={'pulse': "rectangular pulses", 'avg': "mean and RMS values",
                                 'wave': "sampled waveform i(t) over one period"}.get)
    c = st.columns(6)
    I0 = round(0.5 * d['I_rated'], 1)
    wave = None
    if kind == 'pulse':
        IF = cm.num(c[0], "Forward current I_F [A]", I0, f"{k}|IF", min_value=0.0,
                    help="Current through the transistor / channel while it is on.")
        DF = cm.num(c[1], "Forward duty D_F", 0.5, f"{k}|DF", min_value=0.0, max_value=1.0,
                    help="Share of the period with forward conduction.")
        IR = cm.num(c[2], "Reverse current I_R [A]", 0.0, f"{k}|IR", min_value=0.0,
                    help="Current in the reverse direction (antiparallel diode, body diode or third quadrant).")
        DR = cm.num(c[3], "Reverse duty D_R", 0.0, f"{k}|DR", min_value=0.0, max_value=1.0)
        cur = lc.currents_pulse(IF, DF, IR, DR)
        Ion = Ioff = IF
        wave = ('pulse', IF, DF, IR, DR)
    elif kind == 'avg':
        cur = {'I_avg_T': cm.num(c[0], "I_avg forward [A]", I0 / 2, f"{k}|ia", min_value=0.0),
               'I_rms_T': cm.num(c[1], "I_rms forward [A]", I0 / math.sqrt(2), f"{k}|ir", min_value=0.0),
               'I_avg_D': cm.num(c[2], "I_avg reverse [A]", 0.0, f"{k}|da", min_value=0.0),
               'I_rms_D': cm.num(c[3], "I_rms reverse [A]", 0.0, f"{k}|dr", min_value=0.0)}
        Ion = Ioff = I0
        if cur['I_rms_T'] < cur['I_avg_T'] or cur['I_rms_D'] < cur['I_avg_D']:
            st.warning("The RMS value cannot be smaller than the mean value of the same current.")
    else:
        txt = st.text_area("i(t) over one period: one value per line or 't, i' pairs (uniform sampling)",
                           "\n".join(f"{x:.3g}" for x in I0 * np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False))),
                           key=f"{K}|{name}|wave", height=110)
        try:
            df = pd.read_csv(io.StringIO(txt), header=None)
            i = df.iloc[:, -1].to_numpy(float)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            st.error("The waveform cannot be read: enter numbers, one per line (or 't, i' per line).")
            st.stop()
        cur = lc.currents_wave(i)
        Ion = Ioff = float(np.max(np.abs(i))) if len(i) else 0.0
        wave = ('wave', i)
    Vsw = cm.num(c[4], "Switched voltage V_sw [V]", round(0.6 * d['V_rated']), f"{k}|V", min_value=0.0,
                 help="Voltage that the device blocks before turn-on and after turn-off.")
    fs = 1e3 * cm.num(c[5], "Switching frequency f_s [kHz]", 10.0 if d['type'] == 'IGBT' else 50.0, f"{k}|fs",
                      min_value=0.0, help="Turn-on / turn-off events per second (0 = conduction only).")
    c = st.columns(4)
    Ion = cm.num(c[0], "Current at turn-on I_on [A]", Ion, f"{k}|Ion|{kind}", min_value=0.0,
                 help="0 = soft (zero-current) turn-on.")
    Ioff = cm.num(c[1], "Current at turn-off I_off [A]", Ioff, f"{k}|Ioff|{kind}", min_value=0.0)
    Irec = cm.num(c[2], "Diode current before recovery I_rec [A]", 0.0, f"{k}|Irec", min_value=0.0,
                  help="Current of this device's diode when the opposite switch turns on (hard commutation); "
                       "0 = no recovery in this device.")
    sync = c[3].toggle("Reverse current in the channel", value=d['type'] == 'MOSFET', key=f"{K}|{name}|gsync",
                       disabled=d['type'] == 'IGBT', help="MOSFET / GaN: the gate is on during reverse conduction "
                                                          "(synchronous rectification); off: body diode / V_SD.")
    st.session_state[f"{K}|wave"] = wave
    return dict(cur, V=Vsw, I_on=Ion, I_off=Ioff, I_rec=Irec, fs=fs, sync=sync)


def _waveform(topo, d, op):
    """Current waveforms of the topology for the current inputs."""
    fig = go.Figure()
    if topo in ('B6', 'B4'):
        th = np.linspace(0, 2 * np.pi, 361)
        phi = math.acos(np.clip(op['cosphi'], -1, 1))
        i = op['I'] * np.sin(th - phi)
        duty = 0.5 * (1 + op['M'] * np.sin(th))
        deg = np.degrees(th)
        fig.add_trace(line(deg, i, "phase current i(t)", 0))
        fig.add_trace(line(deg, np.where(i > 0, duty * i, 0), "T1: local mean δ·i (forward)", 1))
        fig.add_trace(line(deg, np.where(i < 0, duty * i, 0), "D1 / channel reverse: δ·i", 2))
        fig.update_layout(title="One fundamental period", xaxis_title="ωt [°]", yaxis_title="i [A]")
        info = ("Phase current (blue) and the share that flows through the upper switch T1 (orange: forward, green: "
                "reverse through its diode or channel), averaged over each switching period with the duty "
                "δ = ½ (1 + M sin ωt). The phase shift φ moves current from the transistor to the diode.")
    elif topo in ('Buck', 'Boost'):
        Vin, Vout, I, L, fs = op['Vin'], op['Vout'], op['I'], op['L'], op['fs']
        D = float(np.clip(Vout / Vin if topo == 'Buck' else 1 - Vin / Vout, 0, 1))
        dI = ((Vin - Vout) if topo == 'Buck' else Vin) * D / (L * fs)
        t = np.linspace(0, 1, 401)
        iL = np.where(t < D, I - dI / 2 + dI * t / max(D, 1e-9), I + dI / 2 - dI * (t - D) / max(1 - D, 1e-9))
        fig.add_trace(line(t, iL, "inductor current i_L", 0))
        fig.add_trace(line(t, np.where(t < D, iL, 0), "switch S", 1))
        fig.add_trace(line(t, np.where(t >= D, iL, 0), "rectifier R", 2, dash='dot'))
        fig.update_layout(title=f"One switching period (D = {D:.2f}, ΔI = {fmt(dI)} A)", xaxis_title="t / T",
                          yaxis_title="i [A]")
        info = ("Inductor current (blue) with its triangular ripple ΔI; the switch S carries it during D·T, the "
                "rectifier R during (1 − D)·T. S turns on at I − ΔI/2 and off at I + ΔI/2; if I − ΔI/2 < 0 (large "
                "ripple), the turn-on is soft.")
    else:
        w = st.session_state.get(f"{K}|wave")
        if w and w[0] == 'pulse':
            _, IF, DF, IR, DR = w
            t = np.linspace(0, 1, 401)
            i = np.where(t < DF, IF, np.where((t >= DF) & (t < DF + DR), -IR, 0.0))
        elif w:
            i = np.asarray(w[1], float)
            t = np.arange(len(i)) / max(len(i), 1)
        else:
            t = np.linspace(0, 1, 2)
            i = np.zeros(2)
        fig.add_trace(go.Scatter(x=t, y=i, name="device current", line=dict(color=SERIES[0], width=2, shape='hv')))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.update_layout(title="Device current over one period (forward > 0, reverse < 0)", xaxis_title="t / T",
                          yaxis_title="i [A]")
        info = ("Current through the device: positive = forward through the transistor / channel, negative = reverse "
                "through the diode or the channel. Only its mean and RMS values enter the conduction losses; the "
                "switching losses use the currents at turn-on and turn-off.")
    show(fig, 260, key=f"{K}|wave|{topo}", legend_below=True, info=info)


def _thermal(key, d):
    c = st.columns(4)
    T_hs = cm.num(c[0], "Heat sink / case temperature T_hs [°C]", 80.0, "ls|ths")
    th = cm.chain().get('thermal')
    srcs = ['device value', 'own value'] + (['thermal stack designer'] if th else [])
    src = c[1].selectbox("R_th,jh from", srcs, key=f"{key}|rsrc",
                         help="Thermal resistance junction → heat sink per die. 'thermal stack designer' takes R_th "
                              "junction → coolant of Theory & Tools › Thermal (then T_hs is the coolant).")
    if src == 'own value':
        Rth = cm.num(c[2], "R_th,jh [K/W]", d['Rth_jh'], "ls|rth", min_value=0.0)
    elif src == 'thermal stack designer':
        Rth = th['R_ja']
        c[2].metric("R_th from the stack", f"{Rth:.3g} K/W", help=th['package'])
    else:
        Rth = d['Rth_jh']
        c[2].metric("R_th,jh (device)", f"{Rth:.3g} K/W")
    coupled = c[3].toggle("Electro-thermal coupling", value=True, key=f"{key}|cpl",
                          help="On: the losses are computed at the junction temperature they cause (iteration). Off: "
                               "at T_j = T_hs.")
    return T_hs, Rth, coupled


#######################################################################################################################
# Equations per topology
#######################################################################################################################
SYM = [('V_0, r (V_F0, r_F)', "threshold voltage and resistance of the IGBT (diode) at T_j", 'V, Ω'),
       ('R_ds,on(T_j)', "on-resistance of the MOSFET/HEMT at T_j = R25 ((T_j + 273)/298)^k_R", 'Ω'),
       ('Î, M, cos φ', "phase current amplitude, modulation index, power factor", 'A, -, -'),
       ('f_s, V_dc', "switching frequency, DC-link voltage", 'Hz, V'),
       ('E_ref, V_ref, I_ref', "datasheet switching energy and its test point", 'J, V, A'),
       ('K_v, K_i, TC_E', "voltage and current exponents and temperature coefficient of the switching energy",
        '-, -, 1/K'),
       ('t_dead, V_SD', "dead time and reverse voltage during the dead time", 's, V'),
       ('D, ΔI, L', "duty cycle, inductor current ripple, inductance", '-, A, H'),
       ('I_avg, I_rms', "mean and RMS value of a current over one period (forward T, reverse D)", 'A'),
       ('I_on, I_off, I_rec', "current at turn-on, turn-off and before the diode recovery", 'A'),
       ('R_th,jh, T_hs', "thermal resistance junction → heat sink, heat sink temperature", 'K/W, °C')]
E_SW = r"E(V,I,T_j)=E_{ref}\left(\frac{V}{V_{ref}}\right)^{K_v}\left(\frac{I}{I_{ref}}\right)^{K_i}\big(1+TC_E\,(T_j-T_{ref})\big)"
TJ = ("Electro-thermal coupling", [r"T_j=T_{hs}+P\,R_{th,jh}", r"V_0(T_j),\ r(T_j),\ R_{ds,on}(T_j)\ \rightarrow\ P\ "
                                                              r"\rightarrow\ T_j\quad\text{(iteration)}"],
      "The losses are recomputed at the junction temperature they cause until T_j no longer changes.")


def _equations(topo, d):
    igbt = d['type'] == 'IGBT'
    if topo in ('B6', 'B4'):
        st_ = [("Operating point", [r"i(t)=\hat I\sin(\omega t-\varphi),\qquad \delta(t)=\tfrac12\big(1+M\sin\omega t"
                                    r"\big)"], "Phase current and duty of the upper switch (sinusoidal PWM)."),
               ("Transistor conduction",
                r"P_{T,cond}=V_0\hat I\left(\tfrac{1}{2\pi}+\tfrac{M\cos\varphi}{8}\right)+r\hat I^2\left(\tfrac18+"
                r"\tfrac{M\cos\varphi}{3\pi}\right)" if igbt else r"P_{T,cond}=\tfrac14 R_{ds,on}(T_j)\,\hat I^2",
                "Mean of δ(t)·v(i)·i over the positive half wave." if igbt else
                "The channel conducts in both directions (synchronous rectification)."),
               ("Diode conduction / dead time",
                r"P_{D,cond}=V_{F0}\hat I\left(\tfrac{1}{2\pi}-\tfrac{M\cos\varphi}{8}\right)+r_F\hat I^2\left("
                r"\tfrac18-\tfrac{M\cos\varphi}{3\pi}\right)" if igbt else
                r"P_{dt}=\tfrac{2}{\pi}\,f_s\,t_{dead}\,V_{SD}\,\hat I",
                "The diode conducts the rest of the switching period." if igbt else
                "Only during the dead time the body diode (or third quadrant) conducts."),
               ("Switching", [E_SW, r"P_{on}+P_{off}=f_s\,(E_{on}+E_{off})(V_{dc},\hat I)\;\tfrac{1}{2\pi}\int_0^{\pi}"
                                    r"\sin^{K_i}x\,dx"], "Every switching period has one turn-on and one turn-off at "
                                                         "the instantaneous current."),
               ("Reverse recovery", r"P_{rec}=f_s\,E_{rec}(V_{dc},\hat I)\;\tfrac{1}{2\pi}\int_0^{\pi}\sin^{K_{i,rec}}"
                                    r"x\,dx", "Recovery of the diode when the opposite transistor turns on (0 for GaN)."),
               ("Total", r"P_{switch}=P_{T,cond}+P_{D,cond}+P_{dt}+P_{on}+P_{off}+P_{rec},\qquad P_{tot}=n\,P_{switch}",
                f"n = {lc.N_SWITCH[topo]} switch positions."), TJ]
    elif topo in ('Buck', 'Boost'):
        st_ = [("Operating point", [r"D_{buck}=\frac{V_{out}}{V_{in}},\qquad D_{boost}=1-\frac{V_{in}}{V_{out}}",
                                    r"\Delta I=\frac{V_L\,D}{L\,f_s},\quad V_L=V_{in}-V_{out}\ \text{(buck)},\ V_{in}\ "
                                    r"\text{(boost)}"], "Duty cycle and inductor current ripple in continuous "
                                                        "conduction (CCM)."),
               ("Switch conduction", r"I_{rms,S}^2=D\left(I^2+\tfrac{\Delta I^2}{12}\right),\quad P_{S,cond}=" +
                (r"D\,(V_0 I+r\,(I^2+\tfrac{\Delta I^2}{12}))" if igbt else r"R_{ds,on}(T_j)\,I_{rms,S}^2"),
                "Mean square of the triangular current during D·T."),
               ("Rectifier conduction", r"P_{R,cond}=(1-D)\,(V_{F0}I+r_F(I^2+\tfrac{\Delta I^2}{12}))\ \text{or}\ "
                                        r"R_{ds,on}(1-D)(I^2+\tfrac{\Delta I^2}{12})",
                "Diode, or a synchronous MOSFET in its channel (plus the dead time in the body diode)."),
               ("Switching", [E_SW, r"P_{sw}=f_s\left[E_{on}\big(I-\tfrac{\Delta I}{2}\big)+E_{off}\big(I+\tfrac{"
                                    r"\Delta I}{2}\big)\right]"], "Turn-on at the valley, turn-off at the peak of "
                                                                 "the inductor current."),
               ("Reverse recovery", r"P_{rec}=f_s\,E_{rec}\big(I-\tfrac{\Delta I}{2}\big)", "Recovery of the "
                                                                                           "rectifier diode."),
               ("Total", r"P_{tot}=P_S+P_R,\qquad \eta=\frac{P_{out}}{P_{out}+P_{tot}}", ""), TJ]
    else:
        st_ = [("Current values", [r"I_{avg}=\frac1T\int_0^T i\,dt,\qquad I_{rms}=\sqrt{\frac1T\int_0^T i^2\,dt}",
                                   r"\text{pulses: }I_{avg}=D\,I,\ I_{rms}=\sqrt{D}\,I"],
                "Separately for the forward (T) and the reverse (D) part of the current."),
               ("Forward conduction", r"P_{T,cond}=V_0 I_{avg,T}+r\,I_{rms,T}^2" if igbt else
                r"P_{T,cond}=R_{ds,on}(T_j)\,I_{rms,T}^2", "A resistance only sees I_rms, a threshold voltage only "
                                                          "I_avg."),
               ("Reverse conduction", r"P_{D,cond}=V_{F0}I_{avg,D}+r_F I_{rms,D}^2" if igbt else
                r"P=R_{ds,on}\,I_{rms,D}^2\ \text{(channel)}\quad\text{or}\quad P_{D,cond}=V_{SD}\,I_{avg,D}",
                "Diode, body diode or third-quadrant channel conduction."),
               ("Switching and recovery", [E_SW, r"P_{sw}=f_s\left[E_{on}(V_{sw},I_{on})+E_{off}(V_{sw},I_{off})"
                                                 r"\right],\qquad P_{rec}=f_s\,E_{rec}(V_{sw},I_{rec})"], ""),
               ("Total", r"P=P_{T,cond}+P_{D,cond}+P_{sw}+P_{rec}", ""), TJ]
    cm.equations(st_, SYM)


#######################################################################################################################
# Loss calculator
#######################################################################################################################
def calculator():
    cm.question("How much power does every switch dissipate at my operating point, split into conduction and "
                "switching, how hot does it get and what is the efficiency?",
                "Closed-form average losses with the datasheet parameters of the device (no time simulation). For "
                "detailed waveforms use PyPowerSim.")
    cm.section(1, "Device", "Pick a device of the library and adapt its datasheet parameters.")
    name, d = _device(K, 0)
    cm.section(2, "Topology and operating point", "The schematic shows the notation, the chart the currents for the "
                                                  "current inputs.")
    topo, op = _operating_point(K, d, name)
    cm.section(3, "Cooling", "The junction temperature changes the on-state parameters, so the losses are computed at "
                             "the temperature they cause.")
    T_hs, Rth, coupled = _thermal(K, d)
    r = lc.lossCalc(d, topo, op, T_hs, Rth, coupled)
    V_blk = op['Vdc'] if topo in ('B6', 'B4') else op['V'] if topo == 'Device' else max(op['Vin'], op['Vout'])
    if V_blk > 0.8 * d['V_rated']:
        st.warning(f"The blocking voltage is above 80 % of the rated {d['V_rated']} V: overvoltage at turn-off and "
                   f"cosmic ray failure rate (FIT) rise steeply — choose a higher voltage class.")
    if r['runaway']:
        st.error("Thermal runaway: the losses rise faster with T_j than the cooling removes them. Lower the current "
                 "or f_s, or improve the cooling.")

    # ==============================================================================
    # Result
    # ==============================================================================
    cm.section(None, "Result")
    pos = r['pos']
    per = {k: sum(v.values()) for k, v in pos.items()}
    Tj_max = max(r['Tj'].values())
    cm.tiles([("Total losses", f"{fmt(r['P_tot'])} W", f"All {r['n']} switch position(s)."),
              ("Efficiency", f"{100 * r['eta']:.2f} %" if np.isfinite(r['eta']) else '–',
               "η = P_out / (P_out + P_loss) (semiconductors only; not defined for a single device)."),
              ("Output power", f"{fmt(r['P_out'] / 1e3)} kW" if r['P_out'] else '–'),
              ("Hottest junction", f"{Tj_max:.1f} °C", " · ".join(f"{k}: {v:.1f} °C" for k, v in r['Tj'].items())),
              ("Losses per position", " · ".join(f"{fmt(v)} W" for v in per.values()), ", ".join(per))])
    _equations(topo, d)
    tabs = st.tabs(["Loss split", "Over switching frequency", "Over current"])
    with tabs[0]:
        fig = go.Figure()
        for i, p in enumerate(PARTS):
            v = [pos[k][p] for k in pos]
            if max(v) > 0:
                fig.add_trace(go.Bar(x=list(pos), y=v, name=PART_TXT[p], marker=dict(color=SERIES[i],
                                                                                       cornerradius=2)))
        fig.update_layout(barmode='stack', title="Losses per switch position", yaxis_title="P [W]")
        show(fig, 330, key=f"{K}|split", info=(
            "Average losses of one switch position split by cause. Conduction depends on the current (and on T_j), "
            "switching on f_s × energy per transition. For an inverter all positions are equal (multiply by the "
            "number of switches for the total)."))
    with tabs[1]:
        fs0 = op['fs'] if op['fs'] > 0 else 10e3
        fs = np.linspace(0.1, 3.0, 40) * fs0
        P_c, P_s, eta = [], [], []
        for f_ in fs:
            rr = lc.lossCalc(d, topo, dict(op, fs=f_), T_hs, Rth, coupled)
            P_c.append(sum(sum(x[p] for p in ('T cond', 'D cond', 'dead time')) for x in rr['pos'].values()))
            P_s.append(sum(sum(x[p] for p in ('T on', 'T off', 'D rec')) for x in rr['pos'].values()))
            eta.append(rr['eta'])
        tr = [line(fs / 1e3, P_c, 'conduction', 0), line(fs / 1e3, P_s, 'switching', 1)]
        if np.isfinite(eta).any():
            tr.append(go.Scatter(x=fs / 1e3, y=100 * np.array(eta), name='efficiency', yaxis='y2',
                                 line=dict(color=INK2, dash='dot')))
        fig = go.Figure(tr)
        fig.add_vline(x=op['fs'] / 1e3, line=dict(color=MUTED, dash='dot'))
        fig.update_layout(title="Losses over the switching frequency", xaxis_title="f_s [kHz]",
                          yaxis_title="P of the positions shown [W]", yaxis2=dict(title="η [%]", overlaying='y',
                                                                                  side='right', showgrid=False),
                          hovermode='x unified')
        show(fig, 340, key=f"{K}|fs", info=(
            "Conduction losses stay (nearly) constant, switching losses rise linearly with f_s. Where they cross, a "
            "higher frequency costs as much as the conduction; wide bandgap devices (SiC, GaN) move this point to much "
            "higher frequencies."))
    with tabs[2]:
        if topo == 'Device':
            st.info("In the general mode the currents are entered directly; scale them above to see the trend.")
        else:
            I0 = op['I']
            Is = np.linspace(0.05, 1.5, 40) * max(I0, 1e-3)
            Ptot, Tjs = [], []
            for i_ in Is:
                rr = lc.lossCalc(d, topo, dict(op, I=i_), T_hs, Rth, coupled)
                Ptot.append(max(sum(x.values()) for x in rr['pos'].values()))
                Tjs.append(max(rr['Tj'].values()))
            fig = go.Figure([line(Is, Ptot, 'losses of the hottest position', 0),
                             go.Scatter(x=Is, y=Tjs, name='T_j', yaxis='y2', line=dict(color=SERIES[1],
                                                                                       dash='dash'))])
            fig.add_vline(x=I0, line=dict(color=MUTED, dash='dot'))
            fig.update_layout(title="Losses and junction temperature over the current", xaxis_title="I [A]",
                              yaxis_title="P [W]", yaxis2=dict(title="T_j [°C]", overlaying='y', side='right',
                                                               showgrid=False), hovermode='x unified')
            show(fig, 340, key=f"{K}|cur", info=(
                "Losses of the hottest switch position and its junction temperature over the current. The curve bends "
                "upwards: conduction rises with I² (MOSFET) and R_ds,on rises with T_j."))

    # ==============================================================================
    # Interpretation and export
    # ==============================================================================
    tot = {p: sum(x[p] for x in pos.values()) for p in PARTS}
    sw = tot['T on'] + tot['T off'] + tot['D rec']
    cond = tot['T cond'] + tot['D cond'] + tot['dead time']
    big = max(tot, key=tot.get)
    s_sw = sw / max(sw + cond, 1e-12)
    where = {'B6': "each of the six switch positions of the inverter", 'B4': "each of the four switch positions",
             'Buck': "the active switch and the rectifier", 'Boost': "the active switch and the rectifier",
             'Device': "the device"}[topo]
    p1 = (f"The {d['tech']} '{name}' dissipates {fmt(max(per.values()))} W in {where} at the chosen operating point, "
          f"i.e. {fmt(r['P_tot'])} W for all semiconductors" +
          (f", which corresponds to a semiconductor efficiency of {100 * r['eta']:.2f} %" if np.isfinite(r['eta'])
           else "") + f". The hottest junction reaches {Tj_max:.1f} °C, {Tj_max - T_hs:.1f} K above the heat sink.")
    p2 = (f"Switching causes {100 * s_sw:.0f} % of the losses and conduction {100 * (1 - s_sw):.0f} %; the largest "
          f"single part is the {PART_TXT[big]}. " +
          ("Switching therefore dominates: a lower switching frequency, a faster device (SiC, GaN) or soft switching "
           "pays off most." if s_sw > 0.5 else
           "Conduction therefore dominates: a device with a lower R_ds,on or V_CE,sat, or parallel devices, pay off "
           "most, whereas a higher switching frequency would be comparatively cheap."))
    p3 = (("It should be noted that the electro-thermal iteration includes the rise of R_ds,on and of the on-state "
           "voltage with temperature; without it the losses would be underestimated. " if coupled else
           "The electro-thermal coupling is switched off, so the losses are computed at T_hs and are "
           "underestimated for a hot junction. ") +
          "The losses feed the reliability chain: they set T_j and its swing ΔT_j, which drive the power cycling "
          "life (Thermal › Transient T_j, Reliability › Lifetime calculator).")
    cm.interpret([p1, p2, p3])
    cm.chain()['losses'] = {'P': max(per.values()), 'device': name, 'Tj': Tj_max, 'T_hs': T_hs, 'topo': topo,
                            'op': dict(op), 'd': dict(d)}
    cm.refs(td.DEVICE_REFS)
    with st.expander("Export to the workbooks of the reliability mode", icon=':material/upload_file:'):
        st.markdown(f"**Constant loss** of the hottest position: P_v = **{fmt(max(per.values()))} W** (Parts sheet, "
                    f"column Pv[W], Loss Model 1).")
        cm.export(pd.DataFrame([{'Pv[W]': max(per.values()), 'device': name, 'topology': topo}]), "parts_pv",
                  label="Constant loss (CSV)")
        if topo in ('B6', 'B4'):
            I_ax = np.linspace(0, 1.5 * max(op['I'], 1.0), 10)
            V_ax = np.linspace(0.5, 1.0, 5) * op['Vdc']
            if st.toggle("Compute the loss table P(I, V_dc)", key=f"{K}|lut",
                         help="Loss of the hottest position on a grid of current amplitudes (rows) and DC voltages "
                              "(columns), for the Power Loss LUT of the model workbook (Loss Model 2)."):
                lut = lc.lossLut(d, topo, op, I_ax, V_ax, T_hs, Rth)
                df = pd.DataFrame(lut, columns=[f"{v:.0f} V" for v in V_ax])
                df.insert(0, 'Î [A]', I_ax)
                table(df, hide_index=True)
                cm.export(df, "loss_lut", label="Loss table (CSV)",
                          note="Axis-1 = current (rows), Axis-2 = DC voltage (columns), as the 'Power Loss LUT' block "
                               "of the model workbook. Check that the current axis of your mission profile has the "
                               "same meaning (amplitude vs. RMS: I_rms = Î / √2).")


#######################################################################################################################
# Technology comparison
#######################################################################################################################
def comparison():
    cm.question("How do Si, SiC and GaN compare at the same operating point, and up to which switching frequency does "
                "each technology make sense?")
    default = ['Si IGBT discrete 1200 V / 75 A (TO-247)', 'Si superjunction MOSFET 650 V / 40 mΩ',
               'SiC MOSFET discrete 750 V / 50 mΩ', 'GaN HEMT 650 V / 50 mΩ']
    pick = st.multiselect("Devices", list(td.DEVICES), default=default, key=f"{K}|cmp|dev")
    if not pick:
        return
    c = st.columns(6)
    Vdc = cm.num(c[0], "V_dc [V]", 400.0, "cmp|V", min_value=0.0)
    I = cm.num(c[1], "Î [A]", 15.0, "cmp|I", min_value=0.0)
    M = cm.num(c[2], "M", 0.9, "cmp|M", min_value=0.0, max_value=1.15)
    cp = cm.num(c[3], "cos φ", 0.9, "cmp|cp", min_value=-1.0, max_value=1.0)
    fs = 1e3 * cm.num(c[4], "f_s [kHz]", 20.0, "cmp|fs", min_value=0.01)
    T_hs = cm.num(c[5], "T_hs [°C]", 80.0, "cmp|Th")
    rows = []
    for n in pick:
        d = td.DEVICES[n]
        rr = lc.lossCalc(d, 'B6', dict(Vdc=Vdc, I=I, M=M, cosphi=cp, fs=fs), T_hs)
        p = rr['pos']['switch']
        rows.append({'device': n, 'technology': d['tech'], 'conduction [W]': p['T cond'] + p['D cond'] + p['dead time'],
                     'switching [W]': p['T on'] + p['T off'] + p['D rec'], 'per switch [W]': sum(p.values()),
                     'total B6 [W]': rr['P_tot'], 'η [%]': 100 * rr['eta'], 'T_j [°C]': max(rr['Tj'].values()),
                     'V_dc / V_rated [%]': 100 * Vdc / d['V_rated']})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure([go.Bar(x=df['device'], y=df['conduction [W]'], name='conduction',
                                marker=dict(color=SERIES[0], cornerradius=2)),
                         go.Bar(x=df['device'], y=df['switching [W]'], name='switching',
                                marker=dict(color=SERIES[1], cornerradius=2))])
        fig.update_layout(barmode='stack', title=f"Losses per switch at {fs / 1e3:g} kHz (B6 inverter)",
                          yaxis_title="P [W]")
        show(fig, 340, key=f"{K}|cmp|bar", info=(
            "Losses of one switch position of a B6 inverter for every device at the same operating point. Devices of a "
            "different voltage class or current rating are not directly comparable — check the column V_dc / V_rated "
            "in the table."))
    with c2:
        fsx = np.logspace(np.log10(2e3), np.log10(500e3), 40)
        fig = go.Figure()
        for i, n in enumerate(pick):
            d = td.DEVICES[n]
            e = [100 * lc.lossCalc(d, 'B6', dict(Vdc=Vdc, I=I, M=M, cosphi=cp, fs=f_), T_hs)['eta'] for f_ in fsx]
            fig.add_trace(line(fsx / 1e3, e, d['tech'] + ' · ' + n.split(' ')[-1], i))
        fig.update_layout(title="Efficiency over the switching frequency", xaxis=dict(title="f_s [kHz]", type='log'),
                          yaxis=dict(title="η [%]"), hovermode='x unified')
        show(fig, 340, key=f"{K}|cmp|eta", legend_below=True, info=(
            "Semiconductor efficiency of the B6 inverter over f_s. Si IGBTs drop fastest because of their tail "
            "current and diode recovery, SiC and GaN keep a high efficiency to much higher frequencies, which allows "
            "smaller filters and magnetics."))
    table(df, hide_index=True)
    best = df.loc[df['per switch [W]'].idxmin()]
    worst = df.loc[df['per switch [W]'].idxmax()]
    cm.interpret([
        f"At {fs / 1e3:g} kHz and Î = {I:g} A, the {best['device']} has the lowest losses with "
        f"{fmt(best['per switch [W]'])} W per switch, whereas the {worst['device']} dissipates "
        f"{fmt(worst['per switch [W]'])} W, i.e. a factor of {worst['per switch [W]'] / max(best['per switch [W]'], 1e-9):.1f} "
        f"more. Lower losses also mean a lower junction temperature and smaller swings ΔT_j, i.e. a longer power "
        f"cycling life.",
        "As a rule of thumb, Si IGBTs are competitive at low switching frequencies (below 10–20 kHz) and high currents, "
        "SiC MOSFETs from about 10 kHz to 100 kHz at high voltages, and GaN HEMTs at low to medium power up to the MHz "
        "range. Superjunction Si MOSFETs suffer from the large reverse recovery charge of their body diode in "
        "hard-switched bridges. It should be noted that devices of different voltage classes or current ratings are "
        "not directly comparable (column V_dc / V_rated).",
    ])
    cm.refs(td.DEVICE_REFS)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Power losses", "Conduction and switching losses of Si, SiC and GaN power semiconductors in "
                                 "inverters and DC/DC converters with electro-thermal coupling.")
    tabs = st.tabs(["Loss calculator", "Technology comparison"])
    with tabs[0]:
        calculator()
    with tabs[1]:
        comparison()
