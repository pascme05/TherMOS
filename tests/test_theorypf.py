#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Tests
# File:         test_theorypf.py
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
Tests of src/py/tools/theoryPF.py (worked examples of the PF theory view): POD error falls with K and the Galerkin
model approaches the projection, the exact ZOH lag matches the continuous lag while explicit Euler diverges for
Ts > 2τ, chip superposition equals the direct solve, the physics residual vanishes for the exact solution and equals
the error in steady phases, the steady-state anchor removes the long-term drift, and moving sources need more POD
modes than a varying cooling.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np

from src.py.tools import theoryPF as tp


#######################################################################################################################
# Tests
#######################################################################################################################
def test_pod_and_galerkin():
    d = tp.pod(T=20.0, Kmax=8)
    assert np.all(np.diff(d['proj']) < 0) and d['proj'][-1] < 0.05 * d['proj'][0]
    assert np.all(d['gal'] >= d['proj'] - 1e-9)                                  # projection is the best possible
    assert d['gal'][-1] < 0.1 and d['E'][0] > 0.95
    assert np.allclose(d['Phi'].T @ d['Phi'], np.eye(8), atol=1e-8)


def test_lag_zoh_vs_euler():
    a = tp.lag(0.5)
    assert a['errZoh'] < 0.01 and a['errEuler'] > 3 * a['errZoh']
    b = tp.lag(0.02)                                                             # Ts = 2.5 τ: Euler unstable
    assert not b['stable'] and b['errEuler'] > 10 and b['errZoh'] < 0.1


def test_superposition():
    s = tp.superposition(np.array([40.0, 15.0, 20.0]))
    assert s['diff'] < 1e-9
    assert np.all(s['R'] @ np.array([40.0, 15.0, 20.0]) >= s['tj'] - 1e-9)      # sum of maxima >= max of sum
    assert np.allclose(np.diag(s['tjPart']), np.diag(s['R']) * np.array([40.0, 15.0, 20.0]))


def test_residual():
    r = tp.residual()
    assert r['rmsE'] < 1e-9 and r['rmsR'] < 0.05
    e = tp.residual(eps=0.05)
    m = (e['t'] % 8.0 > 3.5) & (e['t'] % 8.0 < 4.0)                             # end of a pulse: almost steady
    assert np.allclose(e['r'][m], e['e'][m], atol=0.15 * np.max(np.abs(e['e'])))
    assert tp.residual(ftau=1.5)['rmsR'] > 1.0


def test_drift_and_nwidth():
    d = tp.drift()
    assert d['allAn'] < 0.5 * d['allNo'] and d['trainAn'] > d['trainNo']        # anchor: visible in training
    assert tp.drift(tauCap=20.0)['visible'] > d['visible']
    n = tp.nwidth(n=16)
    assert n['h']['k1'] <= 3 and n['x']['k1'] > 3 * n['h']['k1']
