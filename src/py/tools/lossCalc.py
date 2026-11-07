#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         lossCalc.py
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
Power loss calculator (Theory & Tools › Power losses): closed-form average losses of Si IGBT/diode, Si MOSFET, SiC
MOSFET and GaN HEMT switches in an inverter with sinusoidal PWM and in a buck / boost converter, with
electro-thermal coupling of the junction temperature.

Device model (parameters at the reference temperature T_ref, see src/tools/toolData.DEVICE_PARAMS):
    IGBT / diode:  v = V_0(T) + r(T) i,   V_0(T) = V_0 + dV_0 (T - T_ref),  r(T) = r + dr (T - T_ref)
    MOSFET / HEMT: v = R(T) i,            R(T) = R_25 ((T + 273.15) / 298.15)^k_R   (both current directions)
    switching:     E(V, I, T) = E_ref (V / V_ref)^K_v (I / I_ref)^K_i (1 + TC_E (T - T_ref))
Inverter, sinusoidal PWM with modulation index M, power factor cos(phi) and current amplitude I (per switch) [1, 2]:
    P_T,cond = V_0 I (1/(2 pi) + M cos(phi) / 8) + r I^2 (1/8 + M cos(phi) / (3 pi))
    P_D,cond = V_F0 I (1/(2 pi) - M cos(phi) / 8) + r_F I^2 (1/8 - M cos(phi) / (3 pi))
    MOSFET:    P_cond = R I^2 / 4 (synchronous rectification) + dead time P_dt = 2 f_s t_dead V_SD I / pi
    P_sw = f_s E_ref (V / V_ref)^K_v (I / I_ref)^K_i / (2 pi) int_0^pi sin^K_i(x) dx
Buck / boost (CCM, inductor current I with ripple dI): RMS and mean currents of the switch (duty D) and the rectifier
(1 - D), turn-on at I - dI/2 (soft if <= 0) and turn-off at I + dI/2.
General device (no topology): mean and RMS values of the forward (transistor / channel) and the reverse (diode / third
quadrant) current over one period, the switched voltage and the currents at turn-on, turn-off and recovery:
    IGBT:      P_T = V_0 I_avg,T + r I_rms,T^2,   P_D = V_F0 I_avg,D + r_F I_rms,D^2
    MOSFET:    P_T = R(T) (I_rms,T^2 + I_rms,D^2) (reverse current in the channel) or P_D = V_SD I_avg,D (body diode)
    switching: P_sw = f_s [E_on(V, I_on) + E_off(V, I_off)],  P_rec = f_s E_rec(V, I_rec)

Inputs:     device dict, operating point, heat sink temperature and R_th,jh
Outputs:    losses per switch (conduction / switching / recovery / dead time), junction temperatures, efficiency

Copied from OPTELIA (RAISE, src/tools) for the Theory & Tools pages of the TherMOS GUI.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

import numpy as np

#######################################################################################################################
# Parameter
#######################################################################################################################
TOPOLOGIES = {'B6': "Three-phase inverter (B6), sinusoidal PWM",
              'B4': "Single-phase H-bridge (B4), bipolar sinusoidal PWM",
              'Buck': "Buck converter (synchronous / diode rectifier), CCM",
              'Boost': "Boost converter (synchronous / diode rectifier), CCM",
              'Device': "General: currents and voltage directly at one device (no topology)"}
N_SWITCH = {'B6': 6, 'B4': 4, 'Buck': 2, 'Boost': 2, 'Device': 1}


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def sin_int(k):
    """(1 / 2 pi) int_0^pi sin^k(x) dx  (= 1/pi for k = 1)."""
    return math.sqrt(math.pi) * math.gamma((k + 1) / 2) / math.gamma(k / 2 + 1) / (2 * math.pi)


def cond_par(d, T):
    """On-state parameters at the junction temperature T [°C]."""
    dT = T - d['T_ref']
    if d['type'] == 'IGBT':
        return {'V0': max(d['V0'] + d['dV0'] * dT, 0.0), 'r': max(d['r'] + d['dr'] * dT, 0.0),
                'VF0': max(d['VF0'] + d['dVF0'] * dT, 0.0), 'rF': max(d['rF'] + d['drF'] * dT, 0.0)}
    return {'R': d['R25'] * ((T + 273.15) / 298.15) ** d['kR']}


def energy(d, E, V, I, T, rec=False):
    """Switching energy scaled to V, I, T (I <= 0: soft switching, no energy)."""
    if I <= 0 or E <= 0:
        return 0.0
    Kv, Ki = (d['Kv_rec'], d['Ki_rec']) if rec else (d['Kv'], d['Ki'])
    return E * (V / d['V_ref']) ** Kv * (I / d['I_ref']) ** Ki * max(1 + d['TcE'] * (T - d['T_ref']), 0.0)


# ==============================================================================
# Inverter with sinusoidal PWM (one switch = transistor + antiparallel / body diode)
# ==============================================================================
def inverter(d, Vdc, I, M, cosphi, fs, Tj_T, Tj_D):
    """Average losses of one switch position [W] for current amplitude I [A]."""
    c = cond_par(d, Tj_T)
    out = {}
    if d['type'] == 'IGBT':
        cd = cond_par(d, Tj_D)
        out['T cond'] = c['V0'] * I * (1 / (2 * math.pi) + M * cosphi / 8) + c['r'] * I ** 2 * (1 / 8 + M * cosphi /
                                                                                                (3 * math.pi))
        out['D cond'] = max(cd['VF0'] * I * (1 / (2 * math.pi) - M * cosphi / 8) + cd['rF'] * I ** 2 *
                            (1 / 8 - M * cosphi / (3 * math.pi)), 0.0)
        out['dead time'] = 0.0
    else:
        out['T cond'] = c['R'] * I ** 2 / 4
        out['D cond'] = 0.0
        out['dead time'] = 2 * fs * d['t_dead'] * d['VSD'] * I / math.pi
    kV = (Vdc / d['V_ref']) ** d['Kv']
    tf = max(1 + d['TcE'] * (Tj_T - d['T_ref']), 0.0)
    out['T on'] = fs * d['Eon'] * kV * tf * (I / d['I_ref']) ** d['Ki'] * sin_int(d['Ki'])
    out['T off'] = fs * d['Eoff'] * kV * tf * (I / d['I_ref']) ** d['Ki'] * sin_int(d['Ki'])
    out['D rec'] = (fs * d['Erec'] * (Vdc / d['V_ref']) ** d['Kv_rec'] * max(1 + d['TcE'] * (Tj_D - d['T_ref']), 0.0)
                    * (I / d['I_ref']) ** d['Ki_rec'] * sin_int(d['Ki_rec']))
    return out


def inverter_numeric(d, Vdc, I, M, cosphi, fs, Tj_T, Tj_D, n=20000):
    """Numeric integration of the upper switch over one fundamental period (verification of inverter())."""
    th = (np.arange(n) + 0.5) / n * 2 * math.pi
    phi = math.acos(np.clip(cosphi, -1, 1))
    duty = 0.5 * (1 + M * np.sin(th))
    i = I * np.sin(th - phi)
    pos, neg = i > 0, i < 0
    c = cond_par(d, Tj_T)
    out = {}
    if d['type'] == 'IGBT':
        cd = cond_par(d, Tj_D)
        out['T cond'] = float(np.mean(pos * duty * (c['V0'] * i + c['r'] * i ** 2)))
        out['D cond'] = float(np.mean(neg * duty * (cd['VF0'] * -i + cd['rF'] * i ** 2)))
        out['dead time'] = 0.0
    else:
        out['T cond'] = float(np.mean(duty * c['R'] * i ** 2))
        out['D cond'] = 0.0
        out['dead time'] = float(np.mean(neg * 2 * fs * d['t_dead'] * d['VSD'] * -i))
    out['T on'] = float(np.mean([energy(d, d['Eon'], Vdc, x, Tj_T) for x in np.where(pos, i, 0.0)])) * fs
    out['T off'] = float(np.mean([energy(d, d['Eoff'], Vdc, x, Tj_T) for x in np.where(pos, i, 0.0)])) * fs
    out['D rec'] = float(np.mean([energy(d, d['Erec'], Vdc, x, Tj_D, rec=True) for x in np.where(neg, -i, 0.0)])) * fs
    return out


# ==============================================================================
# DC/DC half bridge (active switch S with duty D, rectifier R with 1 - D)
# ==============================================================================
def dcdc(d, V, I, dI, D, fs, Tj_S, Tj_R, sync=True):
    """Average losses of the active switch S and the rectifier R [W] (inductor current I, ripple dI)."""
    I_lo, I_hi = I - dI / 2, I + dI / 2
    i2 = I ** 2 + dI ** 2 / 12                                                 # mean square of the triangle
    cS, cR = cond_par(d, Tj_S), cond_par(d, Tj_R)
    S, R = {}, {}
    if d['type'] == 'IGBT':
        S['T cond'] = D * (cS['V0'] * I + cS['r'] * i2)
        R['D cond'] = (1 - D) * (cR['VF0'] * I + cR['rF'] * i2)
        R['T cond'] = R['dead time'] = 0.0
    else:
        S['T cond'] = D * cS['R'] * i2
        if sync:
            R['T cond'] = (1 - D) * cR['R'] * i2
            R['dead time'] = fs * d['t_dead'] * d['VSD'] * (max(I_lo, 0.0) + max(I_hi, 0.0))
            R['D cond'] = 0.0
        else:                                                                 # body diode only (gate off)
            R['T cond'] = R['dead time'] = 0.0
            R['D cond'] = (1 - D) * d['VSD'] * I
    S['T on'] = fs * energy(d, d['Eon'], V, I_lo, Tj_S)
    S['T off'] = fs * energy(d, d['Eoff'], V, I_hi, Tj_S)
    R['D rec'] = fs * energy(d, d['Erec'], V, I_lo, Tj_R, rec=True)
    for x in (S, R):
        for k in ('T cond', 'D cond', 'dead time', 'T on', 'T off', 'D rec'):
            x.setdefault(k, 0.0)
    return S, R


# ==============================================================================
# General device (no topology)
# ==============================================================================
def general(d, op, Tj_T, Tj_D):
    """
    Average losses of one device from its current values [W].
    op: I_avg_T, I_rms_T (forward / channel), I_avg_D, I_rms_D (reverse / diode), V (switched voltage), I_on, I_off,
        I_rec (currents at turn-on, turn-off and diode recovery), fs, sync (MOSFET: reverse current in the channel)
    """
    c = cond_par(d, Tj_T)
    out = {'dead time': 0.0}
    if d['type'] == 'IGBT':
        cd = cond_par(d, Tj_D)
        out['T cond'] = c['V0'] * op['I_avg_T'] + c['r'] * op['I_rms_T'] ** 2
        out['D cond'] = cd['VF0'] * op['I_avg_D'] + cd['rF'] * op['I_rms_D'] ** 2
    elif op.get('sync', True):
        out['T cond'] = c['R'] * (op['I_rms_T'] ** 2 + op['I_rms_D'] ** 2)
        out['D cond'] = 0.0
    else:
        out['T cond'] = c['R'] * op['I_rms_T'] ** 2
        out['D cond'] = d['VSD'] * op['I_avg_D']
    out['T on'] = op['fs'] * energy(d, d['Eon'], op['V'], op['I_on'], Tj_T)
    out['T off'] = op['fs'] * energy(d, d['Eoff'], op['V'], op['I_off'], Tj_T)
    out['D rec'] = op['fs'] * energy(d, d['Erec'], op['V'], op.get('I_rec', 0.0), Tj_D, rec=True)
    return out


def currents_pulse(I_F, D_F, I_R=0.0, D_R=0.0):
    """Mean and RMS values of rectangular forward (I_F during D_F) and reverse (I_R during D_R) current pulses."""
    return {'I_avg_T': D_F * I_F, 'I_rms_T': math.sqrt(max(D_F, 0.0)) * I_F, 'I_avg_D': D_R * I_R,
            'I_rms_D': math.sqrt(max(D_R, 0.0)) * I_R}


def currents_wave(i):
    """Mean and RMS values of the positive (forward) and negative (reverse) parts of a sampled current i(t)."""
    i = np.asarray(i, float)
    p, n = np.maximum(i, 0.0), np.maximum(-i, 0.0)
    return {'I_avg_T': float(p.mean()), 'I_rms_T': float(np.sqrt((p ** 2).mean())), 'I_avg_D': float(n.mean()),
            'I_rms_D': float(np.sqrt((n ** 2).mean()))}


#######################################################################################################################
# Main Function
#######################################################################################################################
def lossCalc(d, topo, op, T_hs=80.0, Rth=None, coupled=True, iterMax=50, tol=0.01):
    """
    Losses with electro-thermal coupling T_j = T_hs + P R_th,jh (per die; IGBT and diode on separate dies, MOSFET
    transistor and body diode on one die).

    Inputs:     d: device dict, topo: 'B6' | 'B4' | 'Buck' | 'Boost' | 'Device'
                op: B6/B4: Vdc [V], I [A amplitude], M [-], cosphi [-], fs [Hz]
                    Buck: Vin, Vout, I (output current), L [H], fs, sync     Boost: Vin, Vout, I (input current), L, fs
                    Device: see general()
                T_hs: heat sink temperature [°C], Rth: R_th,jh per die [K/W] (default: device value)
    Outputs:    dict with 'pos' (losses per switch position), 'Tj', 'P_tot', 'P_out', 'eta', 'runaway'
    """
    Rth = d['Rth_jh'] if Rth is None else Rth
    T1 = T2 = T_hs
    runaway = False

    ###################################################################################################################
    # Operating point
    ###################################################################################################################
    if topo in ('B6', 'B4'):
        Vdc, I, M, cphi, fs = op['Vdc'], op['I'], op['M'], op['cosphi'], op['fs']
        P_out = (0.75 if topo == 'B6' else 0.5) * M * Vdc * I * cphi
    elif topo == 'Device':
        P_out = None
        Vdc = I = M = cphi = fs = None
    else:
        Vin, Vout, I, L, fs = op['Vin'], op['Vout'], op['I'], op['L'], op['fs']
        if topo == 'Buck':
            D, V = Vout / Vin, Vin
            dI = (Vin - Vout) * D / (L * fs)
            P_out = Vout * I
        else:
            D, V = 1 - Vin / Vout, Vout
            dI = Vin * D / (L * fs)
            P_out = None
        D = float(np.clip(D, 0.0, 1.0))

    ###################################################################################################################
    # Electro-thermal iteration
    ###################################################################################################################
    for _ in range(iterMax if coupled else 1):
        if topo in ('B6', 'B4', 'Device'):
            p = inverter(d, Vdc, I, M, cphi, fs, T1, T2) if topo != 'Device' else general(d, op, T1, T2)
            pos = {'switch' if topo != 'Device' else 'device': p}
            if d['type'] == 'IGBT':
                P1 = p['T cond'] + p['T on'] + p['T off']
                P2 = p['D cond'] + p['D rec']
            else:
                P1 = P2 = sum(p.values())
        else:
            S, R = dcdc(d, V, I, dI, D, fs, T1, T2, op.get('sync', True))
            pos = {'active switch': S, 'rectifier': R}
            P1, P2 = sum(S.values()), sum(R.values())
        n1, n2 = T_hs + P1 * Rth, T_hs + P2 * Rth
        done = abs(n1 - T1) < tol and abs(n2 - T2) < tol
        T1, T2 = (0.5 * T1 + 0.5 * n1, 0.5 * T2 + 0.5 * n2) if coupled else (T1, T2)
        if max(T1, T2) > 400:
            runaway = True
            break
        if done:
            break
    if not coupled:
        T1, T2 = T_hs + P1 * Rth, T_hs + P2 * Rth

    ###################################################################################################################
    # Totals
    ###################################################################################################################
    n = N_SWITCH[topo]
    P_tot = (n * sum(pos['switch'].values()) if topo in ('B6', 'B4') else sum(pos['device'].values())
             if topo == 'Device' else sum(pos['active switch'].values()) + sum(pos['rectifier'].values()))
    if topo == 'Device':
        eta = np.nan
    elif topo == 'Boost':
        P_in = Vin * I
        P_out = P_in - P_tot
        eta = P_out / P_in if P_in > 0 else np.nan
    else:
        eta = P_out / (P_out + P_tot) if P_out > 0 else np.nan
    single = topo in ('B6', 'B4', 'Device')
    Tj = ({'transistor': T1, 'diode': T2} if single and d['type'] == 'IGBT' else
          {'die': T1} if single else {'active switch': T1, 'rectifier': T2})
    extra = {} if single else {'D': D, 'dI': dI}
    return {'pos': pos, 'Tj': Tj, 'P_tot': P_tot, 'P_out': P_out, 'eta': eta, 'runaway': runaway, 'n': n, **extra}


def lossLut(d, topo, op, I_axis, V_axis, T_hs=80.0, Rth=None):
    """Loss of the hottest switch position over current (rows) and DC voltage (columns) for the mdl workbook."""
    out = np.zeros((len(I_axis), len(V_axis)))
    for a, I in enumerate(I_axis):
        for b, V in enumerate(V_axis):
            q = dict(op)
            q['I'] = I
            q['Vdc' if topo in ('B6', 'B4') else 'Vin'] = V
            r = lossCalc(d, topo, q, T_hs, Rth)
            out[a, b] = max(sum(x.values()) for x in r['pos'].values())
    return out


#######################################################################################################################
# References
#######################################################################################################################
# [1]: D. Graovac and M. Pürschel, "IGBT power losses calculation using the data-sheet parameters," Infineon AN, 2009.
# [2]: A. Wintrich et al., Application Manual Power Semiconductors, SEMIKRON, 2015.
# [3]: F. Bierhoff and F. W. Fuchs, "Semiconductor losses in voltage source and current source IGBT converters based
#      on analytical derivation," IEEE PESC, 2004.
