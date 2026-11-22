#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_pffoster
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################
"""Tests of the coupled Foster model (pfFoster) and of the half-bridge losses (genBoardPFHB.hbLoss)."""

import numpy as np

from src.py.model.PF import pfFoster as pf
from src.py.model.PF.pfParam import NOM
from setup.tutorial import genBoardPFHB as hb
from src.py.tools.lossCalc import inverter


def _system(rng, L=801, Ts=0.05):
    """Synthetic coupled system: 2 sources, 3 outputs, 3 time constants."""
    tau = np.array([0.2, 2.0, 12.0])
    R = rng.uniform(0.05, 0.8, (2, 3, 3))
    P = np.repeat(rng.uniform(0, 30, (L // 20 + 1, 2)), 20, axis=0)[0:L]
    return tau, R, P, np.einsum('lck,cko->lo', pf.lags(P, tau, Ts), R)


def test_fit_board_exact():
    rng = np.random.default_rng(0)
    tau, R, P, y = _system(rng)
    f = pf.fitBoard([P], [y], K=3)
    assert f['rmse'] < 1e-3
    assert np.allclose(f['tau'], tau, rtol=1e-2)
    assert np.allclose(f['R'].sum(axis=1), R.sum(axis=1), rtol=1e-3)         # steady-state R_th matrix


def test_family_interpolates_condition():
    rng = np.random.default_rng(1)
    tau, R, P, _ = _system(rng)
    boards = []
    for h in (500.0, 1000.0, 2000.0, 4000.0):
        mu = dict(NOM, h=h)
        Rh = R * (1500.0 / h) ** 0.5
        y = np.einsum('lck,cko->lo', pf.lags(P, tau, 0.05), Rh)
        boards.append({'mu': mu, 'P': [P], 'ths': [y]})
    mdl = pf.fitFam(boards, 3, K=3)
    yp = pf.predict(mdl, boards[1]['mu'], P)
    assert np.sqrt(np.mean((yp - boards[1]['ths'][0]) ** 2)) < 0.05 * np.abs(boards[1]['ths'][0]).max()


def _family(rng, hs, L=401):
    """Boards of a synthetic family: τ and R scale with log h (R linear in log h, as the joint model)."""
    tau0, R, P, _ = _system(rng, L)
    out = []
    for h in hs:
        a = np.log(h / 1500.0)
        tau, Rh = tau0 * np.exp(-0.3 * a), R * (1.0 - 0.3 * a)
        out.append({'mu': dict(NOM, h=h), 'P': [P], 'ths': [np.einsum('lck,cko->lo', pf.lags(P, tau, 0.05), Rh)]})
    return out


def test_joint_fit_family():
    rng = np.random.default_rng(2)
    tr = _family(rng, (500.0, 800.0, 1300.0, 2000.0, 3200.0, 5000.0))
    te = _family(np.random.default_rng(2), (1000.0,))
    m0 = pf.fitFam(tr, 3, K=3)
    mj = pf.fitJoint(tr, 3, K=3, mdl0=m0)
    assert mj['rmseJoint'] < 0.05
    assert pf.rmse(mj, te) <= pf.rmse(m0, te) + 1e-6
    assert pf.rmse(mj, te) < 0.02 * np.abs(te[0]['ths'][0]).max()


def test_select_uses_validation():
    rng = np.random.default_rng(3)
    tr, va = _family(rng, (500.0, 1000.0, 2000.0, 4000.0)), _family(np.random.default_rng(3), (1400.0,))
    m = pf.select(tr, va, 3, K=3)
    assert m['variant'] in m['valRmse'] and m['valRmse'][m['variant']] == min(m['valRmse'].values())


def test_hbloss_matches_lossCalc():
    t = np.arange(0.0, 10.0, 0.05)
    for cp in (0.9, -0.5):
        P = hb.hbLoss(t, 25.0, cp, 1.0, M=0.8)
        ref = inverter(hb.DEV, hb.OP['Vdc'], 25.0, 0.8, cp, hb.OP['fs'], hb.OP['Tj'], hb.OP['Tj'])
        assert np.isclose(P[:, 0].mean(), ref['T cond'] + ref['T on'] + ref['T off'], rtol=1e-3)
        assert np.isclose(P[:, 3].mean(), ref['D cond'] + ref['D rec'], rtol=1e-3)
        assert np.allclose(P[:, 0].mean(), P[:, 2].mean(), rtol=1e-3)          # high and low side symmetric
