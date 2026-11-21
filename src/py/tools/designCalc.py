#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         designCalc
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
Calculations of the tool 'Design effects' (gui/views/tools/design.py): what happens to the temperatures of the IMS
board of the parametric PF study (src/py/model/PF/pfParam.py, 30 x 20 mm, 3 chips, 3D finite-volume model) when the
cooling, a material or a chip position changes. Every design enters the finite-volume model

    C·dθ/dt = −G·θ + Bp·P(t)

at a specific place: h → diagonal of G at the cooled nodes, k of a layer → series conductances (off-diagonal of G),
ρ·c → C, chip position → Bp (and the nodes under the chip). All results are exact for the model (no network):
    steady state        θ_ss = G⁻¹·Bp·P                     (one sparse LU)
    time constants      G·v = λ·C·v, τ = 1/λ (slowest)      (shift-invert Lanczos)
    transients          block Krylov reduction V = orth[Θ, G⁻¹C·Θ, …] (q blocks), Galerkin projection and the exact
                        zero-order-hold solution of the reduced model (q = 6: junction RMSE 0.01 K against the FV model with 40
                        sub-steps; the default FV integration with 4 sub-steps deviates by up to 2 K at load steps)
    sensitivities       central differences of T_j in ln μ (1 %) and in the chip position (±1 mm)
The optional parametric network (results/pfparam/<run>.pt of setup/tutorial/studyPFParam.py) predicts the same
quantities without any matrix solve.
Inputs:     mu: dict(h W/(m²K), kDiel W/(mK), sAtt -, sRc -), pos: list of chip centres (mm), P: losses (W, per chip)
Outputs:    dicts with the intermediate values (see the functions)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import os

import numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import eigh

from src.py.model.PF.pfParam import buildBoard, CHIPS, NOM, MU, W_MM, L_MM, _ok

#######################################################################################################################
# Settings
#######################################################################################################################
PARAMS = {'h': ("Cooling h", "W/(m²K)", "Heat transfer coefficient at the bottom of the Al base (cold plate)."),
          'kDiel': ("Dielectric k", "W/(mK)", "Thermal conductivity of the 100 µm IMS dielectric."),
          'sAtt': ("Die-attach factor", "-", "Factor on the conductivity of the die attach (solder / sintered Ag)."),
          'sRc': ("Base ρc factor", "-", "Factor on the heat capacity ρ·c of the 1.5 mm Al base.")}
WHERE = {'h': "diagonal of G at the bottom nodes (cooling term h·A)",
         'kDiel': "series conductances through the dielectric (off-diagonal of G, and its diagonal sum)",
         'sAtt': "series conductances through the die attach under each chip",
         'sRc': "capacitances C of the Al base nodes",
         'pos': "load distribution Bp and the die / attach nodes under the chip"}


#######################################################################################################################
# Board
#######################################################################################################################
def board(mu=None, pos=None):
    """Board (pfParam.buildBoard) of the condition mu and the chip centres pos (default: nominal)."""
    mu = dict(NOM, **(mu or {}))
    pos = [tuple(map(float, p)) for p in (pos or [(c['x'], c['y']) for c in CHIPS])]
    return buildBoard(mu, pos)


def valid(pos):
    """True if the chips lie on the board (margin MARGIN) and keep the spacing GAP."""
    return _ok([tuple(p) for p in pos], CHIPS)


def layers(B):
    """Heights of the node rows (mm) and the row index of every node."""
    z = np.round(np.asarray(B['geo'])[:, 2] * 1e3, 4)
    zr = np.unique(z)
    return zr, np.searchsorted(zr, z)


def operatorChange(mu, pos, key, rel=0.2, dpos=(2.0, 0.0), chip=0):
    """
    Which entries of C, G, Bp change when parameter key (h, kDiel, sAtt, sRc) is raised by rel, or (key = 'pos')
    chip is moved by dpos (mm). Returns the number of changed entries of G (diagonal, off-diagonal), C and Bp, the
    node rows (heights, mm) that are touched, and the relative change of the touched entries.
    """
    B0 = board(mu, pos)
    if key == 'pos':
        p1 = [tuple(p) for p in pos]
        p1[chip] = (p1[chip][0] + dpos[0], p1[chip][1] + dpos[1])
        B1 = board(mu, p1)
    else:
        B1 = board(dict(mu, **{key: mu[key] * (1 + rel)}), pos)
    zr, row = layers(B0)
    out = {'zRows': zr, 'nodesRow': np.bincount(row, minlength=len(zr))}
    if len(B1['C']) != len(B0['C']):
        out.update(sameNodes=False)
        return out
    dG = (B1['G'] - B0['G']).tocoo()
    m = np.abs(dG.data) > 1e-12 * max(abs(B0['G']).max(), 1e-30)
    r, c = dG.row[m], dG.col[m]
    dC = np.flatnonzero(np.abs(B1['C'] - B0['C']) > 1e-12 * np.max(B0['C']))
    dB = np.flatnonzero(np.any(np.abs(B1['Bp'] - B0['Bp']) > 1e-12, axis=1))
    touched = np.unique(np.concatenate([r, dC, dB]))
    out.update(sameNodes=True, nDiag=int(np.sum(r == c)), nOff=int(np.sum(r != c)), nC=len(dC), nB=len(dB),
               rows=np.bincount(row[touched], minlength=len(zr)) if len(touched) else np.zeros(len(zr), int),
               relG=_relChange(dG.data[m], np.asarray(B0['G'][r, c]).ravel()) if m.any() else 0.0,
               nnzG=int(B0['G'].nnz), N=len(B0['C']))
    return out


def _relChange(d, g):
    """Median relative change |ΔG/G| of the changed entries that were non-zero before (new entries excluded)."""
    ok = np.abs(g) > 0
    return float(np.median(np.abs(d[ok] / g[ok]))) if ok.any() else float('inf')


#######################################################################################################################
# Steady state, time constants, transients
#######################################################################################################################
def steady(B, P):
    """Steady-state rise θ_ss (N) (K) for the constant losses P (nChips) (W): θ_ss = G⁻¹·Bp·P."""
    return spl.splu(B['G'].tocsc()).solve(np.asarray(B['Bp'], float) @ np.asarray(P, float))


def junction(B, th):
    """Junction rises (nChips) or (L x nChips): maximum over the die nodes of every chip."""
    th = np.asarray(th)
    return np.stack([th[..., d].max(axis=-1) for d in B['die']], axis=-1)


def topMap(B, th):
    """Hottest node of every (x, y) column (21 x 31 array, K) and the axes x, y (mm)."""
    g = np.asarray(B['geo']) * 1e3
    ix, iy = np.round(g[:, 0]).astype(int), np.round(g[:, 1]).astype(int)
    M = np.full((int(L_MM) + 1, int(W_MM) + 1), -np.inf)
    np.maximum.at(M, (iy, ix), np.asarray(th))
    return np.arange(0, W_MM + 1), np.arange(0, L_MM + 1), M


def timeConstants(B, n=3):
    """The n slowest time constants τ = 1/λ (s) of G·v = λ·C·v."""
    w = spl.eigsh(B['G'].tocsc(), k=n, M=_diag(B['C']), sigma=0, which='LM')[0]
    return np.sort(1.0 / w)[::-1]


def _diag(c):
    import scipy.sparse as sp
    return sp.diags(np.asarray(c, float))


def rom(B, q=4):
    """Block Krylov reduction (r = q·nChips): V (N x r), reduced C, G, Bp."""
    lu = spl.splu(B['G'].tocsc())
    K = [lu.solve(np.asarray(B['Bp'], float))]
    for _ in range(1, q):
        K.append(lu.solve(B['C'][:, None] * K[-1]))
    V, _ = np.linalg.qr(np.column_stack(K))
    return {'V': V, 'C': V.T @ (B['C'][:, None] * V), 'G': V.T @ (B['G'] @ V), 'B': V.T @ np.asarray(B['Bp'], float)}


def transient(B, P, Ts, q=6, nodes=None):
    """
    Rises (L x n) (K) at the nodes (default: die nodes of every chip → junction rises L x nChips) for the losses P
    (L x nChips) (W, zero-order hold, θ(0) = 0): exact discrete solution of the Krylov model, x_k = Ad·x_{k−1} + Bd·P_k.
    """
    R = rom(B, q)
    w, U = eigh(R['G'], R['C'])                                         # modal form of the reduced model
    Ad = U @ np.diag(np.exp(-w * Ts)) @ U.T @ R['C']
    Bd = U @ np.diag((1 - np.exp(-w * Ts)) / w) @ U.T @ R['B']
    P = np.asarray(P, float)
    X = np.zeros((len(P), R['V'].shape[1]))
    for k in range(1, len(P)):
        X[k] = Ad @ X[k - 1] + Bd @ P[k]
    if nodes is None:
        return np.stack([(X @ R['V'][d].T).max(axis=1) for d in B['die']], axis=1)
    return X @ R['V'][nodes].T


#######################################################################################################################
# Sweeps and sensitivities
#######################################################################################################################
def sweepParam(mu, pos, P, key, values):
    """Steady junction rises (n x nChips) (K) and slowest τ (n) (s) over the values of parameter key."""
    tj, tau = [], []
    for v in values:
        B = board(dict(mu, **{key: float(v)}), pos)
        tj.append(junction(B, steady(B, P)))
        tau.append(timeConstants(B, 1)[0])
    return np.array(tj), np.array(tau)


def sweepPos(mu, pos, P, chip, axis, values):
    """
    Steady junction rises (n x nChips) (K) when chip is moved along axis (0: x, 1: y) to the values (mm); NaN where
    the layout is not valid (edge margin, spacing).
    """
    out = []
    for v in values:
        p = [tuple(x) for x in pos]
        q = list(p[chip])
        q[axis] = float(v)
        p[chip] = tuple(q)
        if not valid(p):
            out.append(np.full(len(CHIPS), np.nan))
            continue
        B = board(mu, p)
        out.append(junction(B, steady(B, P)))
    return np.array(out)


def coupling(mu, pos):
    """Steady junction rise of every chip per W of every chip (nChips x nChips) (K/W): R_th matrix of the layout."""
    B = board(mu, pos)
    Th = spl.splu(B['G'].tocsc()).solve(np.asarray(B['Bp'], float))
    return np.stack([Th[d].max(axis=0) for d in B['die']])


def sensitivity(mu, pos, P, rel=0.01):
    """
    Sensitivities of the steady junction rises: dT_j/d ln μ (K per 100 % change, central difference rel) for every
    parameter and dT_j/dx, dT_j/dy (K/mm, ±1 mm) of every chip position. Returns {name: (nChips) array}.
    """
    out = {}
    for k in PARAMS:
        lo, hi = [board(dict(mu, **{k: mu[k] * f}), pos) for f in (1 - rel, 1 + rel)]
        out[k] = (junction(hi, steady(hi, P)) - junction(lo, steady(lo, P))) / (np.log(1 + rel) - np.log(1 - rel))
    for c, ch in enumerate(CHIPS):
        for a, name in ((0, 'x'), (1, 'y')):
            r = []
            for d in (-1.0, 1.0):
                p = [tuple(x) for x in pos]
                q = list(p[c])
                q[a] += d
                p[c] = tuple(q)
                r.append(junction(board(mu, p), steady(board(mu, p), P)) if valid(p) else None)
            if r[0] is not None and r[1] is not None:
                out[f"{ch['name']} {name}"] = (r[1] - r[0]) / 2.0
            elif r[0] is not None or r[1] is not None:
                base = junction(board(mu, pos), steady(board(mu, pos), P))
                out[f"{ch['name']} {name}"] = (r[1] - base) if r[1] is not None else (base - r[0])
    return out


#######################################################################################################################
# Parametric network (optional)
#######################################################################################################################
def loadNet(path):
    """Network of studyPFParam (<run>.pt) or None: (net, ΔT)."""
    if not path or not os.path.isfile(path):
        return None
    import torch
    from src.py.model.PF.pfNet import pfModal
    ck = torch.load(path, map_location='cpu')
    c = ck['cfg']
    net = pfModal(5, 11, c['K'], c['width'], c['depth'], c['nF'], rel=c['rel'], tauCap=c.get('tauCap'))
    net.load_state_dict(ck['state'])
    net.eval()
    return net, float(ck['dT'])


def netJunction(nd, B, P, Ts):
    """Junction rises (L x nChips) (K) predicted by the network for the board B and the losses P (L x nChips)."""
    import torch
    from src.py.model.PF.pfParamFit import prepBoard, predict
    net, dT = nd
    die = np.concatenate(B['die'])
    with torch.no_grad():
        y = predict(net, prepBoard(B, torch.device('cpu')), np.asarray(P, float), dT, Ts,
                    rows=torch.as_tensor(die)).numpy()
    n = np.cumsum([0] + [len(d) for d in B['die']])
    return np.stack([y[:, n[c]:n[c + 1]].max(axis=1) for c in range(len(B['die']))], axis=1)


def inRange(mu):
    """Parameters outside the training range of the network: {name: (value, lo, hi)}."""
    return {k: (mu[k], MU[k][0], MU[k][1]) for k in MU if not MU[k][0] <= mu[k] <= MU[k][1]}
