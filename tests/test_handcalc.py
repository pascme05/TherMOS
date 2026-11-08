#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_handcalc
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
Unit tests of the thermal hand calculator (src/py/tools/handCalc.py, run with 'python -m pytest tests'). Every test
compares a routine with a hand calculation, a limit case or a numerical integration.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

import numpy as np

from src.py.tools import handCalc as hc


#######################################################################################################################
# Tests
#######################################################################################################################
def test_stack_series():
    L = [{'name': 'a', 't': 2e-3, 'k': 200.0, 'Rc': 0.0}, {'name': 'b', 't': 1e-4, 'k': 2.0, 'Rc': 5e-6}]
    r = hc.stack(L, h=1000.0, A=1e-3, P=10.0)
    R = 2e-3 / 200 + 5e-6 + 1e-4 / 2 + 1e-3
    assert math.isclose(r['R_tot'], R) and math.isclose(r['h_eff'], 1 / R)
    assert math.isclose(r['R_th'], R / 1e-3) and math.isclose(r['dT'], 10 / 1e-3 * R)
    z, T = hc.profile(L, 1e4, 1000.0)
    assert math.isclose(T[0], 1e4 * R) and abs(T[-1]) < 1e-9 and math.isclose(z[-1], 2.1e-3)
    assert math.isinf(hc.stack(L)['h_eff']) is False and hc.stack(L)['R_tot'] < R      # ideal sink: no 1/h


def test_contact():
    r = hc.contact(1e6, 200.0, 200.0, 1e9, 1e-6, 1e-6)                                  # vacuum
    sig = math.sqrt(2) * 1e-6
    h_c = 1.25 * 200 * hc.slope(sig) / sig * 1e-3 ** 0.95
    assert math.isclose(float(r['h_c']), h_c) and float(r['h_g']) == 0
    lo, hi = hc.contact(np.array([1e5, 1e7]), 200.0, 200.0, 1e9, 1e-6, 1e-6, 0.026, 0.26e-6)['R_j']
    assert hi < lo                                                                        # pressure helps
    assert math.isclose(hc.tim(50e-6, 3.0, 2e-6), 50e-6 / 3 + 4e-6)


def test_spreading_limits():
    k, t, h, A = 390.0, 3e-3, 5000.0, 2.5e-3
    r = hc.spreading(A * 0.999999, A, t, k, h)                                            # source = plate: 1D
    assert r['R_sp'] / r['R_1D'] < 1e-2
    small, big = hc.spreading(1e-4, A, t, k, h), hc.spreading(4e-4, A, t, k, h)
    assert small['R_sp'] > big['R_sp'] > 0
    a, b, t = 5e-3, 8e-3, 2e-3                                                            # angle rule vs quadrature
    z = np.linspace(0, t, 20001)
    f = 1 / (k * (a + 2 * z) * (b + 2 * z))
    num = float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(z)))
    assert math.isclose(hc.spread_angle(a, b, t, k), num, rel_tol=1e-6)
    assert math.isclose(hc.spread_angle(a, a, t, k), t / (k * a * (a + 2 * t)))


def test_convection():
    w = hc.fluid('Water', 26.85)
    assert math.isclose(w['Pr'], 855e-6 * 4179 / 0.613)
    r = hc.h_duct(1e-7, 2e-3, 2e-3, 1, 10.0, w)                                           # long laminar duct
    assert r['regime'] == 'laminar' and abs(r['Nu'] - 3.66) < 0.05 and math.isclose(r['f'], 64 / r['Re'])
    r = hc.h_duct(5e-4, 1e-2, 1e-2, 1, 1.0, w)                                            # turbulent: ~ Dittus-Boelter
    db = 0.023 * r['Re'] ** 0.8 * w['Pr'] ** 0.4
    assert r['regime'] == 'turbulent' and abs(r['Nu'] / db - 1) < 0.25
    a = hc.fluid('Air (1 atm)', 40)
    assert 3 < hc.h_natural(40, 0.1, a, 25)['h'] < 10                                    # textbook range
    assert math.isclose(hc.h_radiation(1.0, 0.0, 0.0), 4 * hc.SIGMA_SB * 273.15 ** 3)


def test_fins_vias_coolant_transient():
    f = hc.fin(1e-3, 200.0, 1e-3, 1e-2, 0.1)
    assert f['eta'] > 0.999                                                               # weak cooling: η → 1
    r = hc.heatsink(50.0, 200.0, 0.1, 0.1, 5e-3, 10, 2e-3, 3e-2)
    assert 0 < r['eta_o'] <= 1 and r['R_sa'] > r['R_fins']
    v = hc.vias(1e-3, 0.3e-3, 25e-6, 1.6e-3, 5e-3, 5e-3)
    assert 0.3 < v['k_eff'] < 390 and v['R_pad'] < v['R_board'] and v['n'] == 25
    p = hc.fluid('Water-glycol 50/50', 60)
    V = hc.coolant(1000.0, p, dT=5.0)['V_dot']
    assert math.isclose(hc.coolant(1000.0, p, V_dot=V)['dT'], 5.0)
    T1, T4 = hc.semi_inf(1e7, [1e-5, 4e-5], 148.0, 2330.0, 705.0)
    assert math.isclose(T4 / T1, 2.0)                                                     # ΔT ∝ √t
    assert math.isclose(hc.lumped(1.0, 900.0, 0.5)['tau'], 450.0)


def test_radiation():
    r = hc.radiation(0.8, 100.0, 20.0)                                                    # small body: ε_eff = ε1
    T1, T2 = 373.15, 293.15
    assert math.isclose(r['eps_eff'], 0.8) and math.isclose(float(r['q']), 0.8 * hc.SIGMA_SB * (T1 ** 4 - T2 ** 4))
    assert math.isclose(float(r['h_rad']), float(r['q']) / 80)
    p = hc.radiation(0.8, 100.0, 20.0, 0.5, 1.0, 1.0)                                     # parallel plates
    assert math.isclose(p['eps_eff'], 1 / (1 / 0.8 + 1 / 0.5 - 1))
    assert abs(float(r['h_lin']) / float(r['h_rad']) - 1) < 0.02                          # ≈ (ΔT/T_m)²/4


def test_pulse_train():
    R, tau, P, tp, T = [0.1], [0.05], 100.0, 0.01, 0.04
    r = hc.pulse_train(P, tp, T, R, tau)
    assert math.isclose(r['first'], P * float(hc.foster_Z(R, tau, tp)))                  # 1st pulse = P Z(t_p)
    x = 0.0                                                                               # brute force: 2000 periods
    for _ in range(2000):
        x = P * 0.1 + (x - P * 0.1) * math.exp(-tp / 0.05)
        pk = x
        x *= math.exp(-(T - tp) / 0.05)
    assert math.isclose(r['peak'], pk) and math.isclose(r['valley'], x)
    f = hc.pulse_train(P, 1e-5, 4e-5, R, [10.0])                                          # fast train → mean
    assert math.isclose(f['peak'], f['mean'], rel_tol=1e-4)


def test_skin_eddy():
    d = 1e-3
    assert abs(float(hc.skin_round(d, 1.0)) - 1) < 1e-9                                  # δ ≫ d: DC
    a = 25 * d
    assert math.isclose(float(hc.skin_round(2 * a, d)), a / (2 * d) + 0.25, rel_tol=1e-3)  # δ ≪ d: a/(2δ) + 1/4
    assert math.isclose(float(hc.skin('rect', 2e-3, 2e-3, 3e-4)), float(hc.skin_round(2e-3, 3e-4)))
    assert math.isclose(float(hc.skin_depth(1.72e-8, 50.0)), 9.33e-3, rel_tol=1e-3)       # Cu, 50 Hz
    c = hc.conductor('round', 2e-3, 0, 10.0, 1e-9, 'Cu (annealed)', 40.0, 20.0)           # DC: energy balance
    assert math.isclose(c['T'] - 40, c['P'] / (20.0 * c['U']), rel_tol=1e-6) and math.isclose(c['F'], 1, abs_tol=1e-6)
    I, f, w, h, t, rho = 10.0, 1e6, 1e-3, 2e-4, 5e-3, 2.8e-8                              # thick plate: R_s = ρ/δ
    e = hc.plate_eddy(I, f, w, h, t, rho)
    assert math.isclose(e['R_s'], rho / e['delta'], rel_tol=1e-6) and e['beta'] > 0.999
    s, he = 3e-3, e['h_eff']                                                              # pair vs quadrature
    x = np.linspace(-2, 2, 2000001)
    K = e['beta'] * I * he / math.pi * (1 / (he ** 2 + x ** 2) - 1 / (he ** 2 + (x - s) ** 2))
    num = e['R_s'] * float(np.sum((K[1:] ** 2 + K[:-1] ** 2) / 2 * np.diff(x)))
    assert math.isclose(hc.plate_eddy(I, f, w, h, t, rho, s)['P'], num, rel_tol=1e-3)
