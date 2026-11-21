#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Tests
# File:         test_designcalc.py
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Tests of src/py/tools/designCalc.py (Theory & Tools › Design effects): where a parameter enters C and G, steady state
against superposition, time constants against the dense eigenproblem limit, Krylov transient against the steady
state and against a finely sub-stepped implicit Euler, sweeps and sensitivities against finite differences.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from src.py.tools import designCalc as dc
from src.py.model.PF.pfParam import NOM, CHIPS

POS = [(c['x'], c['y']) for c in CHIPS]
P = (40.0, 15.0, 20.0)


#######################################################################################################################
# Where a change acts
#######################################################################################################################
def test_operator_change_rows():
    h = dc.operatorChange(NOM, POS, 'h')
    assert h['nOff'] == 0 and h['nC'] == 0 and h['nDiag'] > 0
    assert h['rows'][0] == h['nDiag'] and np.all(h['rows'][1:] == 0)            # cooling: bottom row only
    rc = dc.operatorChange(NOM, POS, 'sRc')
    assert rc['nDiag'] == rc['nOff'] == 0 and rc['nC'] > 0                       # heat capacity: C only
    kd = dc.operatorChange(NOM, POS, 'kDiel')
    assert kd['nOff'] > 0 and kd['nC'] == 0 and kd['rows'][3] > 0              # dielectric row touched
    at = dc.operatorChange(NOM, POS, 'sAtt')
    assert 0 < at['rows'][5] < at['nodesRow'][4] and at['rows'][0] == 0         # die attach: under the chips only
    ps = dc.operatorChange(NOM, POS, 'pos', dpos=(2.0, 0.0), chip=0)
    assert ps['sameNodes'] and ps['nOff'] > 0 and ps['nC'] == 0


#######################################################################################################################
# Steady state, coupling, sensitivities
#######################################################################################################################
def test_steady_superposition_and_coupling():
    B = dc.board(NOM, POS)
    th = dc.steady(B, P)
    assert np.allclose(B['G'] @ th, B['Bp'] @ np.asarray(P), rtol=1e-8, atol=1e-8)
    R = dc.coupling(NOM, POS)
    tj = dc.junction(B, th)
    assert R.shape == (3, 3) and np.all(R > 0) and np.all(np.diag(R) > R.max(axis=1) - 1e-12)
    assert np.all(R @ np.asarray(P) >= tj - 1e-9)                                # max over sources ≥ max of the sum
    assert np.allclose(dc.junction(B, dc.steady(B, (40.0, 0, 0))), R[:, 0] * 40.0)


def test_sweeps_and_sensitivity():
    T, tau = dc.sweepParam(NOM, POS, P, 'h', [750.0, 1500.0, 3000.0])
    assert np.all(np.diff(T[:, 0]) < 0) and np.all(np.diff(tau) < 0)            # better cooling: cooler, faster
    T2, tau2 = dc.sweepParam(NOM, POS, P, 'sRc', [0.8, 1.2])
    assert np.allclose(T2[0], T2[1]) and tau2[1] > tau2[0]                        # ρc: transient only
    S = dc.sensitivity(NOM, POS, P)
    f = 1.05
    lo, hi = [dc.junction(b, dc.steady(b, P)) for b in (dc.board(dict(NOM, h=1500 / f), POS),
                                                       dc.board(dict(NOM, h=1500 * f), POS))]
    assert np.allclose(S['h'], (hi - lo) / (2 * np.log(f)), rtol=0.02)
    assert np.allclose(S['sRc'], 0.0, atol=1e-8)
    Tp = dc.sweepPos(NOM, POS, P, 0, 0, [1.0, POS[0][0]])
    assert np.all(np.isnan(Tp[0])) and np.allclose(Tp[1], dc.junction(dc.board(NOM, POS), dc.steady(
        dc.board(NOM, POS), P)))


#######################################################################################################################
# Time constants and transient
#######################################################################################################################
def test_time_constants_and_transient():
    B = dc.board(NOM, POS)
    tau = dc.timeConstants(B, 3)
    assert np.all(np.diff(tau) < 0) and 1.0 < tau[0] < 10.0
    Ts, n = 0.05, 1200                                                           # 60 s ≈ 20 τ₁: steady
    PP = np.tile(np.asarray(P), (n, 1))
    PP[0] = 0.0
    y = dc.transient(B, PP, Ts)
    assert np.allclose(y[-1], dc.junction(B, dc.steady(B, P)), rtol=2e-3)
    m = 20                                                                       # implicit Euler, 20 sub-steps
    C = sp.diags(B['C'])
    lu = spl.splu((C / (Ts / m) + B['G']).tocsc())
    th, ref = np.zeros(len(B['C'])), [np.zeros(3)]
    for k in range(1, 80):
        for _ in range(m):
            th = lu.solve(C @ th / (Ts / m) + B['Bp'] @ PP[k])
        ref.append(dc.junction(B, th))
    assert np.max(np.abs(y[:80] - np.array(ref))) < 1.5                          # Euler error ~ O(Ts/m) at the step
    assert np.sqrt(np.mean((y[:80] - np.array(ref)) ** 2)) < 0.3


def test_in_range():
    assert dc.inRange(NOM) == {}
    assert set(dc.inRange(dict(NOM, h=250.0, kDiel=12.0))) == {'h', 'kDiel'}
