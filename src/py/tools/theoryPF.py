#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         theoryPF.py
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
Small worked examples behind the equations of the PF theory view (gui/core/theory_pf.py, section "The equations as
tools"). Each function takes a few inputs and returns the numbers of one chart; all use the IMS board of the
parametric PF study (pfParam, 30 x 20 mm, 3 chips, 3453 nodes, nominal condition) and its finite-volume model

    C·dθ/dt = −G·θ + Bp·P(t)

    pod             space × time:   θ(t) ≈ Φ·a(t), Φ from the SVD of snapshots, Galerkin C_r·da/dt = −G_r·a + B_r·P
    lag             one lag:        db/dt = λ·(p̂ − b), exact zero-order hold vs explicit Euler
    superposition   chips add up:   θ_ss = Σ_c Θ_c·P_c, Θ = G⁻¹·Bp
    residual        physics loss:   r = G⁻¹·C·dθ̂/dt + θ̂ − Θ·P = e + G⁻¹·C·de/dt (e = θ̂ − θ)
    drift           steady anchor:  a slow, unseen mode over a long profile, with and without Σ_k φ_k = Θ·P_ref
    nwidth          why positions are hard for POD: singular values of steady fields over h vs over a chip position
Inputs/outputs in SI units except mm for positions, K for rises, s for times.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import eigh

from src.py.model.PF.pfParam import CHIPS, NOM, P_REF, randProfile
from src.py.tools.designCalc import board, steady, junction, topMap, rom, valid, transient

#######################################################################################################################
# Settings
#######################################################################################################################
TS = 0.05                                                               # sampling time (s)
POS = [(float(c['x']), float(c['y'])) for c in CHIPS]                   # nominal chip centres (mm)
P_NOM = np.array([c['P'] for c in CHIPS], dtype=float)                  # rated losses (W)


#######################################################################################################################
# Helper
#######################################################################################################################
def zohReduced(Cr, Gr, Br, P, Ts=TS):
    """Exact zero-order-hold solution x (L x r) of Cr·dx/dt = −Gr·x + Br·P, x(0) = 0 (Cr, Gr symmetric pos. def.)."""
    w, U = eigh(Gr, Cr)
    Ad = U @ np.diag(np.exp(-w * Ts)) @ U.T @ Cr
    Bd = U @ np.diag((1 - np.exp(-w * Ts)) / w) @ U.T @ Br
    X = np.zeros((len(P), len(w)))
    for k in range(1, len(P)):
        X[k] = Ad @ X[k - 1] + Bd @ P[k]
    return X


def layerMap(B, v):
    """Values v (N) on the copper layer (top layer that covers the whole board) as a 21 x 31 array (y, x)."""
    g = np.asarray(B['geo']) * 1e3
    z = np.round(g[:, 2], 4)
    zr, cnt = np.unique(z, return_counts=True)
    zc = zr[cnt == cnt.max()].max()
    m = z == zc
    M = np.full((int(round(g[:, 1].max())) + 1, int(round(g[:, 0].max())) + 1), np.nan)
    M[np.round(g[m, 1]).astype(int), np.round(g[m, 0]).astype(int)] = np.asarray(v)[m]
    return M


def lagZoh(p, lam, Ts=TS):
    """Exact discrete lag b[n] = a·b[n−1] + (1 − a)·p[n], a = exp(−λ·Ts), b[0] = 0 (p: L or L x m, lam: scalar/m)."""
    p = np.asarray(p, float)
    a = np.exp(-np.asarray(lam, float) * Ts)
    b = np.zeros_like(p)
    for n in range(1, len(p)):
        b[n] = a * b[n - 1] + (1 - a) * p[n]
    return b


#######################################################################################################################
# 1) Space x time: POD and Galerkin projection
#######################################################################################################################
def pod(T=40.0, seed=1, Kmax=12):
    """
    FV snapshots of a random loss profile (T s) on the nominal board and their POD. Returns the singular values σ
    (relative to σ₁), the captured energy E_K, the error of the best K-mode approximation (projection) and of the
    Galerkin model with K modes (RMSE over all nodes, max junction error), the modes Φ, the time coefficients a(t)
    and the board.
    """
    B = board()
    t = np.arange(0.0, T + 1e-9, TS)
    P = randProfile(t, np.random.default_rng(seed), (0.5, 8.0))
    S = transient(B, P, TS, q=10, nodes=np.arange(len(B['C'])))          # L x N, Krylov r = 30 (≈ FV, 20x faster)
    w, Q = np.linalg.eigh(S @ S.T)                                       # method of snapshots: S·Sᵀ = W·Σ²·Wᵀ
    w, Q = np.maximum(w[::-1], 0.0), Q[:, ::-1]
    s = np.sqrt(w)
    U = (S.T @ Q[:, :Kmax]) / s[:Kmax]                                   # Φ = Sᵀ·W·Σ⁻¹ (first Kmax modes)
    Wt = Q.T
    E = np.cumsum(w) / np.sum(w)
    Ks = np.arange(1, Kmax + 1)
    proj, gal, galJ = [], [], []
    tj = junction(B, S)
    for K in Ks:
        Phi = U[:, :K]
        proj.append(float(np.sqrt(np.mean((S - (S @ Phi) @ Phi.T) ** 2))))
        X = zohReduced(Phi.T @ (B['C'][:, None] * Phi), Phi.T @ (B['G'] @ Phi), Phi.T @ np.asarray(B['Bp'], float), P)
        th = X @ Phi.T
        gal.append(float(np.sqrt(np.mean((th - S) ** 2))))
        galJ.append(float(np.max(np.abs(junction(B, th) - tj))))
    return {'t': t, 'P': P, 'S': S, 'sv': s / s[0], 'E': E, 'K': Ks, 'proj': np.array(proj), 'gal': np.array(gal),
            'galJ': np.array(galJ), 'Phi': U[:, :Kmax], 'a': (Wt[:Kmax].T * s[:Kmax]), 'B': B, 'tj': tj,
            'rise': float(S.max())}


#######################################################################################################################
# 2) One lag: exact zero-order hold vs explicit Euler
#######################################################################################################################
def lag(tau, Ts=TS, period=4.0, T=12.0, duty=0.5):
    """
    Lag db/dt = (p̂ − b)/τ for a pulse train p̂ (0/1, period s, duty) sampled with Ts: continuous reference (fine
    grid), exact ZOH (eq. of the network) and explicit Euler b[n] = b[n−1] + Ts/τ·(p̂[n−1] − b[n−1]).
    """
    tf = np.arange(0.0, T + 1e-9, Ts / 50)
    pf = ((tf % period) < duty * period).astype(float)
    ref = lagZoh(np.concatenate([[0.0], pf[:-1]]), 1.0 / tau, Ts / 50)      # exact for the piece-wise constant loss
    t = np.arange(0.0, T + 1e-9, Ts)
    p = ((t % period) < duty * period).astype(float)
    pz = np.concatenate([[0.0], p[:-1]])                                 # level held over (t_{n−1}, t_n]
    zoh = lagZoh(pz, 1.0 / tau, Ts)
    eul = np.zeros_like(t)
    for n in range(1, len(t)):
        eul[n] = eul[n - 1] + Ts / tau * (pz[n] - eul[n - 1])
    a = np.exp(-Ts / tau)
    refS = np.interp(t, tf, ref)
    return {'t': t, 'p': p, 'tf': tf, 'ref': ref, 'zoh': zoh, 'euler': eul, 'a': a, 'aEuler': 1 - Ts / tau,
            'errZoh': float(np.max(np.abs(zoh - refS))), 'errEuler': float(np.max(np.abs(eul - refS))),
            'stable': Ts < 2 * tau}


#######################################################################################################################
# 3) Superposition over the chips
#######################################################################################################################
def superposition(P):
    """Steady rise per chip Θ_c·P_c (top maps, K), their sum and the direct solve G⁻¹·Bp·P; junction rises."""
    B = board()
    lu = spl.splu(B['G'].tocsc())
    Th = lu.solve(np.asarray(B['Bp'], float))                           # N x nc (K/W)
    parts = [Th[:, c] * P[c] for c in range(len(P))]
    total = np.sum(parts, axis=0)
    direct = steady(B, P)
    return {'x': topMap(B, total)[0], 'y': topMap(B, total)[1], 'maps': [topMap(B, q)[2] for q in parts],
            'sum': topMap(B, total)[2], 'diff': float(np.max(np.abs(total - direct))), 'tj': junction(B, total),
            'tjPart': np.array([junction(B, q) for q in parts]), 'R': np.array([junction(B, Th[:, c])
                                                                                for c in range(len(P))]).T}


#######################################################################################################################
# 4) Physics residual
#######################################################################################################################
def residual(eps=0.0, ftau=1.0, T=20.0, q=6):
    """
    Exact field of a 40 W pulse train on T1 (period 8 s) in the modal form of the Krylov model and a wrong
    "prediction": amplitudes × (1 + eps), time constants × ftau. Returns, at the hottest die node of T1, the true
    rise θ, the prediction θ̂, the error e = θ̂ − θ and the physics residual r = G⁻¹·C·dθ̂/dt + θ̂ − Θ·P (all K).
    """
    B = board()
    R = rom(B, q)
    lam, W = eigh(R['G'], R['C'])                                       # modes of the reduced model
    t = np.arange(0.0, T + 1e-9, TS)
    P = np.zeros((len(t), len(CHIPS)))
    P[:, 0] = 40.0 * ((t % 8.0) < 4.0)
    P[0] = 0.0
    p = P / P_REF
    node = B['die'][0][np.argmax(steady(B, P_NOM)[B['die'][0]])]
    lu = spl.splu(B['G'].tocsc())
    GiCV = lu.solve(B['C'][:, None] * R['V'])                            # G⁻¹·C·V (N x r)
    Th = lu.solve(np.asarray(B['Bp'], float))

    def field(lamU, scale):
        # x(t) = Σ_c Σ_k w_k (w_kᵀ B_r,c) P_ref / λ_k · b_c,k(t); dx/dt from db/dt = λ_U (p̂ − b)
        x, dx = np.zeros((len(t), len(lam))), np.zeros((len(t), len(lam)))
        for c in range(len(CHIPS)):
            g = W.T @ R['B'][:, c] * P_REF / lam                         # amplitude per mode (exact steady state)
            b = lagZoh(np.repeat(p[:, c:c + 1], len(lam), axis=1), lamU)
            db = lamU * (np.repeat(p[:, c:c + 1], len(lam), axis=1) - b)
            x += scale * (b * g) @ W.T
            dx += scale * (db * g) @ W.T
        return x, dx
    x, _ = field(lam, 1.0)
    xh, dxh = field(lam / ftau, 1.0 + eps)
    th, thh = x @ R['V'][node], xh @ R['V'][node]
    r = dxh @ GiCV[node] + thh - P @ Th[node]
    return {'t': t, 'P': P[:, 0], 'th': th, 'thh': thh, 'e': thh - th, 'r': r, 'tau': 1.0 / lam[:3],
            'rmsE': float(np.sqrt(np.mean((thh - th) ** 2))), 'rmsR': float(np.sqrt(np.mean(r ** 2)))}


#######################################################################################################################
# 5) Drift of a slow mode and the steady-state anchor
#######################################################################################################################
def drift(tauSlow=300.0, tauCap=80.0, amp=5.0, T=1800.0, Ttrain=40.0, seed=4):
    """
    Error that one spurious space function with amplitude amp (K at P_ref) and time constant min(tauSlow, tauCap)
    adds to an otherwise exact prediction of T1 over a mission profile (random levels, holds 10–120 s):
        no anchor    e = amp·b_slow                (its steady sum is wrong by amp)
        anchor       e = amp·(b_slow − b_fast)     (Σ_k φ_k = Θ·P_ref: a fast mode carries −amp, τ_fast = 0.5 s)
    Returns the errors and their RMS within the training window (what the data loss sees) and over the profile.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T + 1e-9, 0.5)
    p = np.zeros_like(t)
    k = 0
    while k < len(t):
        n = max(int(rng.uniform(10, 120) / 0.5), 1)
        p[k:k + n] = rng.uniform(0.1, 1.0)
        k += n
    tauE = min(tauSlow, tauCap)
    bs, bf = lagZoh(p, 1.0 / tauE, 0.5), lagZoh(p, 1.0 / 0.5, 0.5)
    e0, e1 = amp * bs, amp * (bs - bf)
    w = t <= Ttrain

    def rms(e, m):
        return float(np.sqrt(np.mean(e[m] ** 2)))
    return {'t': t, 'p': p, 'noAnchor': e0, 'anchor': e1, 'tau': tauE,
            'trainNo': rms(e0, w), 'trainAn': rms(e1, w), 'allNo': rms(e0, slice(None)), 'allAn': rms(e1, slice(None)),
            'visible': float(1 - np.exp(-Ttrain / tauE))}


#######################################################################################################################
# 6) Why chip positions are hard for a global POD basis
#######################################################################################################################
def nwidth(n=24, seed=0):
    """
    Steady fields (top map, 651 values, per W of T1) of n boards: (i) h log-spaced 500–5000 W/(m²K) at the nominal
    layout, (ii) T1 at n random valid positions (nominal condition). Returns the relative singular values of both
    snapshot sets and the number of modes for a relative projection error below 1 % and 0.1 %.
    """
    rng = np.random.default_rng(seed)
    P = np.array([P_REF, 0.0, 0.0])

    def snap(mu, pos):
        B = board(mu, pos)
        return topMap(B, steady(B, P))[2].ravel()
    Sh = np.array([snap(dict(NOM, h=float(h)), POS) for h in np.geomspace(500.0, 5000.0, n)]).T
    poss = []
    while len(poss) < n:
        p = [tuple(x) for x in POS]
        p[0] = (float(rng.integers(4, 27)), float(rng.integers(4, 17)))
        if valid(p):
            poss.append(p)
    Sx = np.array([snap(NOM, p) for p in poss]).T
    out = {'pos': [p[0] for p in poss]}
    for tag, S in (('h', Sh), ('x', Sx)):
        s = np.linalg.svd(S, compute_uv=False)
        err = np.sqrt(np.maximum(1 - np.cumsum(s ** 2) / np.sum(s ** 2), 0))  # relative projection error (Frobenius)
        out[tag] = {'sv': s / s[0], 'err': err, 'k1': int(np.argmax(err < 1e-2) + 1), 'k01': int(np.argmax(err < 1e-3)
                                                                                                    + 1)}
    return out
