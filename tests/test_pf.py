#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_pf
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Unit tests of the PF mode (physics-informed field network): source encoding, sensor selection, both physics residuals
against known solutions, the differentiable solve with the conductance matrix and a short training of both variants on
a small board (finite-volume data of the board designer).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.saveBoard import saveBoard
from src.py.board.genProfile import genProfile
from src.py.data.DataLoader import DataLoader
from src.py.model.PF.pfData import encodeSrc, projectSrc, selSensors, geoScale, toHat, material, pfPara
from src.py.model.PF.pfPhys import basisLap, resOperator, resOperatorFV, coordDeriv, coordRate, solveG
from src.py.model.PF.pfFit import pfFit
from src.py.model.PF.pfSol import pfSol

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import torch

#######################################################################################################################
# Settings
#######################################################################################################################
LAY = {'view': 'xz', 'W': 12, 'L': 8, 'board': 'IMS, Al base, 100 µm dielectric', 'h': 1500, 'Ta': 25, 'dx': 1.0,
       'yCut': 4, 'chips': [{'name': 'A', 'tech': 'Si MOSFET', 'x': 3, 'y': 4, 'w': 3, 'l': 3, 'P': 10},
                            {'name': 'B', 'tech': 'Si MOSFET', 'x': 9, 'y': 4, 'w': 3, 'l': 3, 'P': 6}]}
TS = 0.05


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _split(tmp, view, tag, kind, seed, n=161):
    """One data split (structure of loadData/concatData) of the small board, written and read with the loaders."""
    D = genBoard(dict(LAY, view=view))
    [C, G, Bp] = calcBoardMat(D)
    t = np.arange(n) * TS
    P = np.column_stack([genProfile(kind, c['P'], t, {'period': 2.0, 'hold': 0.5}, seed=seed + i)
                         for i, c in enumerate(D['chips'])])
    [T, _] = simBoard(C, G, Bp, P, TS, LAY['Ta'], 4, steady=False)
    saveBoard(D, t, P, T, str(tmp / (tag + '.mat')))
    raw = DataLoader(TS, 'mat', 3 if view == 'xz' else 2, 1, 1, [], [], {'data': str(tmp)}).loadData(tag, None, 1, {})
    return {'X2': [raw['X']], 'y2': [raw['y']], 'r2': [raw['r']], 'Ts': TS, 'Data': raw['Data'],
            'X': raw['X'], 'y': raw['y'], 'r': raw['r'], 'Nt': [len(t)]}


def _para(pinf, epoch=3):
    return {'Exp': {'gen': {'init': 1}}, 'Par': {'gen': {'normX': 1, 'normY': 1}},
            'Mdl': {'gen': {'dev': 0}, 'pinf': pinf,
                    'dl': {'W': 60, 'stride': 10, 'epoch': epoch, 'batch': 8, 'initLr': 3e-3, 'valPat': 50,
                           'gradTh': 1, 'lrDropPr': 50, 'lrDropFa': 0.5, 'lrAdapt': 0}}}


#######################################################################################################################
# Tests
#######################################################################################################################
def test_source_encoding_exact():
    rng = np.random.default_rng(0)
    B = np.zeros((50, 3))
    B[5:9, 0], B[20:26, 1], B[40:43, 2] = 1.0, 2.0, 0.5                 # three fixed footprints
    P = rng.uniform(0, 1, (30, 3))
    [U, sv] = encodeSrc([P @ B.T], 1e-6)
    assert U.shape == (50, 3) and np.allclose(U.T @ U, np.eye(3), atol=1e-10)
    [p, err] = projectSrc(rng.uniform(0, 1, (7, 3)) @ B.T, U)           # other amplitudes, same footprints
    assert p.shape == (7, 3) and err < 1e-10
    assert projectSrc(np.ones((2, 50)), U)[1] > 0.5                     # sources outside the footprints
    assert encodeSrc([np.zeros((4, 50))])[0].shape == (50, 0)


def test_sensors():
    g = np.random.default_rng(1).uniform(-1, 1, (400, 2))
    idx = selSensors(g, 20, first=7)
    assert len(np.unique(idx)) == 20 and 7 in idx
    d = np.linalg.norm(g[idx][:, None] - g[idx][None], axis=2) + 10 * np.eye(20)
    assert d.min() > 0.15                                               # evenly spread (random: about 0.03)
    assert len(selSensors(g, 0)) == 400


def test_parameters_default():
    pp = pfPara({'Mdl': {'pinf': {'K': 8, 'beta': ''}}})
    assert pp['K'] == 8 and pp['beta'] == 1.0 and isinstance(pp['nCol'], int)


def test_residual_autodiff_analytic():
    """θ = exp(−a·t)·cos(π·x̂)·cos(π·ŷ) solves ρc·θ_t = k_x·θ_xx + k_y·θ_yy − h_v·θ for a = (π²·(κ_x + κ_y) + h_v)/ρc."""
    torch.manual_seed(0)
    rc, hv, kap = 2.0e6, 500.0, torch.tensor([3.0e4, 1.2e4], dtype=torch.float64)       # κ = k·s²
    a = float((np.pi ** 2 * kap.sum() + hv) / rc)
    x = torch.rand(40, 2, dtype=torch.float64) * 2 - 1
    one = torch.ones(40, dtype=torch.float64)

    # Operator network: one basis function φ = cos(π·x̂)·cos(π·ŷ), coefficient b = exp(−a·t)
    trunk = lambda v: (torch.cos(np.pi * v[..., 0]) * torch.cos(np.pi * v[..., 1]))[..., None]
    t = torch.linspace(0, 1, 5, dtype=torch.float64)
    b, db = torch.exp(-a * t)[None, :, None], (-a * torch.exp(-a * t))[None, :, None]
    r = resOperator(b, db, trunk(x), basisLap(trunk, x, kap[None].expand(40, 2)), torch.zeros(1, 5, 0, dtype=x.dtype),
                    rc * one, hv * one, torch.zeros(40, 0, dtype=x.dtype), 1e4 * one, 1.0)
    assert float(r.abs().max()) < 1e-9
    r = resOperator(b, 0 * db, trunk(x), basisLap(trunk, x, kap[None].expand(40, 2)),
                    torch.zeros(1, 5, 0, dtype=x.dtype), rc * one, hv * one, torch.zeros(40, 0, dtype=x.dtype),
                    1e4 * one, 1.0)
    assert float(r.abs().max()) > 1e-3                                  # a wrong time derivative is detected

    # Coordinate network: f(x̂, ŷ, t̂) with t = (t̂ + 1)/sT
    sT = 2.0
    net = lambda z: torch.exp(-a * (z[..., 2] + 1) / sT) * torch.cos(np.pi * z[..., 0]) * torch.cos(np.pi * z[..., 1])
    z = torch.cat([x, torch.rand(40, 1, dtype=torch.float64) * 2 - 1], dim=1)
    [f, g, h2] = coordDeriv(net, z)
    r = (rc * sT * g[:, 2] - torch.sum(kap * h2[:, 0:2], dim=1) + hv * f) / 1e4
    assert float(r.abs().max()) < 1e-9
    [f2, ft] = coordRate(net, x, torch.tensor([-0.3, 0.4], dtype=torch.float64))
    assert f2.shape == (2, 40) and torch.allclose(ft, -a / sT * f2)


def test_residual_fv_simulation(tmp_path):
    """The finite-volume residual of the simulated field is zero (up to the time discretisation)."""
    d = _split(tmp_path, 'xz', 'a', 'random', 3)
    mat = material(d['Data'], 3)
    assert mat['op'] is not None and mat['k'].shape[1] == 3
    import scipy.sparse.linalg as spl
    lu = spl.splu(mat['op']['G'].tocsc())
    th = d['y2'][0] - d['r2'][0]
    dT = float(np.max(th))
    u = mat['op']['V'][None, :] * d['X2'][0]
    k = np.arange(1, len(th))                                           # step k − 1 → k, sources of sample k
    r = (lu.solve((mat['op']['C'] * (th[k] - th[k - 1]) / TS).T).T + (th[k] + th[k - 1]) / 2 - lu.solve(u[k].T).T) / dT
    assert np.sqrt(np.mean(r ** 2)) < 0.01
    r0 = ((th[k] + th[k - 1]) / 2 - lu.solve(u[k].T).T) / dT            # without the storage term
    assert np.sqrt(np.mean(r0 ** 2)) > 5 * np.sqrt(np.mean(r ** 2))

    # Same residual through the functions of the model: θ = ΔT·φ·b with φ = identity on a few nodes
    idn = np.array([0, 50, len(th[0]) - 1])
    phi = torch.zeros(len(th[0]), 3, dtype=torch.float64)
    phi[idn, [0, 1, 2]] = 1.0
    sol = solveG(lu)
    M = sol(torch.as_tensor(mat['op']['C'])[:, None] * phi)
    assert torch.allclose(M, torch.as_tensor(lu.solve(mat['op']['C'][:, None] * phi.numpy())))
    x = torch.rand(len(th[0]), 2, dtype=torch.float64, requires_grad=True)
    (sol(x) ** 2).sum().backward()                                      # gradient of the solve: 2·G⁻ᵀ·G⁻¹·x
    assert torch.allclose(x.grad, torch.as_tensor(2 * lu.solve(lu.solve(x.detach().numpy()), trans='T')))
    b = torch.ones(1, 2, 3, dtype=torch.float64)
    r = resOperatorFV(b, 0 * b, phi[idn], M[idn], torch.zeros(1, 2, 0, dtype=torch.float64),
                      torch.zeros(3, 0, dtype=torch.float64), 1.0)
    assert torch.allclose(r, torch.ones(1, 2, 3, dtype=torch.float64))


def test_fit_operator(tmp_path):
    tr = _split(tmp_path, 'xz', 'tr', 'random', 3)
    vl = _split(tmp_path, 'xz', 'vl', 'random', 9)
    te = _split(tmp_path, 'xz', 'te', 'pulse', 0)
    for phys in (0, 1, 2):
        para = _para({'phys': phys, 'K': 6, 'width': 16, 'depth': 2, 'warm': 20, 'nCol': 64, 'nDat': 256}, epoch=4)
        mdl = pfFit(dict(tr, _te=te), vl, para, {'selPF': 1})
        info = mdl['info']
        assert mdl['var'] == 1 and mdl['S'] == 2 and len(info['loss']) == 4 and np.all(np.isfinite(info['loss']))
        assert info['loss'][-1] < info['loss'][0]                       # the weighted loss falls
        assert (info['lossPhys'][-1] > 0) == (phys > 0)
        out = pfSol(mdl, te, para, {'selPF': 1})
        assert out['y'].shape == te['y'].shape and np.all(np.isfinite(out['y']))
        assert out['theta_hat'].shape == (len(te['y']), 6)
    torch.save(mdl, str(tmp_path / 'm.pt'))                             # as training/testing store the model
    m2 = torch.load(str(tmp_path / 'm.pt'), weights_only=False)
    assert np.allclose(pfSol(m2, te, para, {'selPF': 1})['y'], out['y'], atol=1e-4)


def test_fit_coordinate(tmp_path):
    te = _split(tmp_path, 'xy', 'te', 'random', 5, n=81)                # lumped 2D board
    assert material(te['Data'], 2)['op'] is not None
    for phys in (1, 2):
        para = _para({'phys': phys, 'nSens': 20, 'width': 16, 'depth': 2, 'steps': 5, 'nCol': 64, 'nDat': 256},
                     epoch=3)
        mdl = pfFit(dict(te, _te=te), te, para, {'selPF': 2})
        assert mdl['var'] == 2 and len(mdl['sens']) == 16 and len(mdl['sensVal']) == 4
        assert np.all(np.isfinite(mdl['info']['loss'])) and mdl['info']['lossPhys'][-1] > 0
        out = pfSol(mdl, te, para, {'selPF': 2})
        assert out['y'].shape == te['y'].shape and np.all(np.isfinite(out['y']))


#######################################################################################################################
# References
#######################################################################################################################
