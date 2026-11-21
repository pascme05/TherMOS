#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_krylov
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################
"""Tests of the KR mode (block Krylov reduction) and of the comparison helpers of the MOR page."""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.py.model.KR.calcKrylov import basis, simulate                                       # noqa: E402
from src.py.tools import theoryPF as tp                                                      # noqa: E402


def _rod(N=60):
    """1D rod: C (N), G (sparse, cooled at the end), B (heat at the first node)."""
    g = 2.0
    G = sp.diags([np.full(N - 1, -g), np.r_[g, np.full(N - 2, 2 * g), g + 0.5], np.full(N - 1, -g)], [-1, 0, 1])
    return np.full(N, 0.3), G.tocsr(), np.eye(N)[:, [0]]


def test_steady_state_exact():
    """The first moment block is the steady state: the reduced model reproduces G⁻¹·B exactly for any q."""
    C, G, B = _rod()
    for q in (1, 3):
        R = basis(C, G, B, q)
        xss = np.linalg.solve(R['G'], R['B'][:, 0])
        assert np.allclose(R['V'] @ xss, sp.linalg.spsolve(G.tocsc(), B[:, 0]), rtol=1e-8, atol=1e-10)


def test_zoh_exact_and_converges():
    """Full basis (q = N) equals the exact ZOH solution of the full model; error falls with q."""
    from scipy.linalg import expm
    C, G, B = _rod(20)
    Ts, p = 0.2, np.r_[0.0, np.ones(49)][:, None]
    A = -np.diag(1 / C) @ G.toarray()
    Ad = expm(A * Ts)
    Bd = np.linalg.solve(A, (Ad - np.eye(20))) @ (np.diag(1 / C) @ B)
    x, X = np.zeros(20), np.zeros((50, 20))
    for k in range(1, 50):
        x = Ad @ x + Bd[:, 0] * p[k, 0]
        X[k] = x
    full = {'C': np.diag(C), 'G': G.toarray(), 'B': B}                     # identity basis: exact solution
    assert np.abs(simulate(full, p, Ts) - X).max() < 1e-9 * np.abs(X).max()
    err = []
    for q in (1, 2, 4):
        R = basis(C, G, B, q)
        err.append(np.abs(simulate(R, p, Ts) @ R['V'].T - X).max())
    assert err[0] > err[1] > err[2]


def test_reduced_stable():
    C, G, B = _rod()
    R = basis(C, G, B, 5)
    assert np.all(R['tau'] > 0) and np.allclose(R['C'], R['C'].T)


def test_compare_mor_moving_chip():
    """Krylov (own operator) stays accurate when the chip moves, a fixed POD basis does not."""
    b = tp.cmpBases(r=8)
    r0 = tp.compareMor(0.0, 1500.0, b, 4, T=8.0)
    r6 = tp.compareMor(6.0, 1500.0, b, 4, T=8.0)
    assert r0['models']['Krylov']['rmse'] < 0.1 and r6['models']['Krylov']['rmse'] < 0.1
    assert r6['models']['POD']['rmse'] > 10 * r0['models']['POD']['rmse']
    assert tp.compareMor(30.0, 1500.0, b, 4) is None                      # invalid layout


def test_krylov_order_decreases():
    rows = tp.krylovOrder(qs=(1, 2, 4), T=8.0)['rows']
    assert rows[0]['rmse'] > rows[1]['rmse'] > rows[2]['rmse']


def test_needs_all_modes():
    """Every mode of the GUI states what it needs (build, predict, new design, information level)."""
    from gui import adapter as A
    from gui.core.needs import NEEDS
    for m in A.MODES:
        assert m in NEEDS and all(len(t) > 5 for t in NEEDS[m])


@pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, 'data', 'board', 'tutorial_pf_test.mat')),
                    reason="PF tutorial data not generated")
def test_kr_mode_on_tutorial_board():
    import scipy.io as sio
    from src.py.model.KR.krFit import reduce
    m = sio.loadmat(os.path.join(ROOT, 'data', 'board', 'tutorial_pf_test.mat'), simplify_cells=True)
    th = np.asarray(m['y'], float) - np.asarray(m['r'], float)
    _, p, R = reduce({'Data': m, 'X2': [m['X']]}, 4, 1e-6)
    e = simulate(R, p[0], float(m['Ts'])) @ R['V'].T - th
    assert np.sqrt(np.mean(e ** 2)) < 0.01 * th.max()
