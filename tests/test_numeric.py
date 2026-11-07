#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_numeric
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
This file contains unit tests of the numerical building blocks of the Python port (run with 'python -m pytest tests').
Every test compares a routine with an exact reference (analytic solution, plain recursion or known system).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.numeric.simSS import simSS
from src.py.numeric.fitN4sid import fitN4sid
from src.py.numeric.solveLinOde import solveLinOde
from src.py.numeric.calcInterpMat import calcInterpMat
from src.py.numeric.calcMultilinMat import calcMultilinMat
from src.py.model.RC.rcSim import rcSim
from src.py.model.DL.dlWin import dlWin
from src.py.general.calcError import calcError
from src.py.general.calcTheta import calcTheta

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Tests
#######################################################################################################################
def test_simSS():
    rng = np.random.default_rng(0)
    A = np.array([[0.9, 0.05], [0.0, 0.7]])
    B = np.array([[0.1], [0.3]])
    C = np.array([[1.0, 0.5]])
    u = rng.standard_normal((200, 1))
    x, y = np.zeros(2), []
    for k in range(0, 200):
        y.append(C @ x)
        x = A @ x + B @ u[k]
    assert np.allclose(simSS(A, B, C, None, u)[0], np.array(y), atol=1e-12)


def test_fitN4sid():
    rng = np.random.default_rng(1)
    A = np.array([[0.95, 0.02], [0.0, 0.8]])
    B = np.array([[0.1, 0.0], [0.05, 0.2]])
    C = np.eye(2)
    us = [rng.standard_normal((400, 2)) for _ in range(0, 2)]
    ys = [simSS(A, B, C, None, u)[0] for u in us]
    s = fitN4sid(us, ys, 2)
    assert np.allclose(np.sort(np.linalg.eigvals(s['A']).real), [0.8, 0.95], atol=1e-6)


def test_solveLinOde():
    t = np.arange(50) * 0.1
    a = solveLinOde(np.array([[2.0]]), np.array([[0.5]]), np.ones((50, 1)), 0.1)
    assert np.allclose(a[:, 0], (1 - np.exp(-0.25 * t)) / 0.5, atol=1e-12)


def test_rcSim():
    Pv = np.r_[np.zeros(5), np.ones(100) * 3]
    T = rcSim([1.0, 2.0], [5.0, 50.0], Pv, 1.0)[0]
    Tn, ref = np.zeros(2), []
    for i in range(0, len(Pv)):
        if i > 0:
            Tn = (2 * np.array([5.0, 50.0]) - 1) / (2 * np.array([5.0, 50.0]) + 1) * Tn + \
                 np.array([1.0, 2.0]) / (2 * np.array([5.0, 50.0]) + 1) * (Pv[i] + Pv[i - 1])
        ref.append(Tn.sum())
    assert np.allclose(T, ref, atol=1e-12)


def test_interpolation():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, (400, 2))
    q = rng.uniform(0.2, 0.8, (50, 2))
    f = lambda p: 2 * p[:, 0] - 3 * p[:, 1] + 1
    assert np.allclose(calcInterpMat(pts, q, 2) @ f(pts), f(q), atol=1e-10)
    ax = [np.linspace(0, 1, 11), np.linspace(0, 2, 21)]
    G = np.meshgrid(*ax, indexing='ij')
    assert np.allclose(calcMultilinMat(ax, q) @ (2 * G[0] - G[1]).ravel(), 2 * q[:, 0] - q[:, 1], atol=1e-12)


def test_dlWin():
    X = np.arange(20, dtype=float)[:, None]
    Xw, yw = dlWin(X, X, 5, 3, 1, 1)
    assert Xw.shape == (20, 5, 1) and np.allclose(Xw[:, 2, 0], X[:, 0]) and np.allclose(yw[:, 0], X[:, 0])
    Xw, yw = dlWin(X, X, 5, 3, 1, 0)
    assert np.allclose(Xw[:, 2, 0], yw[:, 0])


def test_calcError_calcTheta():
    e = calcError([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert np.isclose(e['MAE'], 1 / 3) and np.isclose(e['MAX'], 1.0) and np.isclose(e['NRMSE'], 100 * np.sqrt(1 / 3) / 2)
    para = {'Par': {'loss': {'Rcon': 0.1, 'Rohm': 0.9, 'Rslope': 0.004, 'Tref': 25}, 'gen': {'eps': 1e-12}}}
    assert np.isclose(calcTheta(25.0, para), 1.0)
