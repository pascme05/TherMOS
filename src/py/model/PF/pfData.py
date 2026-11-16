#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfData
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
This module prepares the field data of the PF mode (physics-informed field network, pfFit/pfSol).
1) Parameters: pfPara(para) returns the settings of Mdl › pinf with their defaults (also used by the GUI).
2) Coordinates: geoScale maps the node positions to x̂ = 2·(x − x_min)/(x_max − x_min) − 1 ∈ [−1, 1] per axis; the
   factors s_i = 2/(x_max,i − x_min,i) (1/m) convert the derivatives, ∂/∂x_i = s_i·∂/∂x̂_i.
3) Temperature rise: rise returns θ = T − T_ref (K) of every profile. T_ref is the reference channel r (ambient) if the
   offset was not removed (Exp › init = 1), else the data are already a rise (init = 2: over the first sample,
   init = 3: over r).
4) Source encoding: the heat generation Q (Nt x N) (W/m³) of the training data is factorised with the singular value
   decomposition

       Q(x, t) ≈ Σ_j U_j(x)·p_j(t),      p = Q·U,      UᵀU = I,      j = 1 … S

   S is the number of singular values above srcTol·σ_1 (at most 32). For sources with fixed footprints (chips) S is the
   number of independently driven sources and the encoding is exact. U_j (W/m³ per unit of p_j) is the footprint of
   source j, p_j(t) its amplitude and the input of the network. encodeSrc builds U, projectSrc gives p and the
   relative error of the encoding for other data (a large error: sources that the training data do not contain).
5) Sensors: selSensors returns the nodes whose temperatures are used in the data loss: all nodes (nSens = 0) or nSens
   nodes chosen by farthest-point sampling (every new sensor is the node farthest from the sensors chosen so far, in
   scaled coordinates, so that they cover the domain evenly; the first one is the hottest node).
6) Material: material returns per node the heat capacity ρc (J/m³K), the conductivities k_x, k_y(, k_z) (W/mK), the
   distributed heat sink h_v (W/m³K) and, if the data file stores it, the finite-volume operator (C, G, V).
Functions:  pfPara(para), geoScale(geo), toHat(geo, sc), rise(d, init), offset(d, init), encodeSrc(Q2, tol),
            projectSrc(Q, U), mapNodes(geoA, geoB), selSensors(geoHat, n, seed, first), material(D, d),
            checkNorm(para)

V1.0: 16.11.2026, Pascal Schirmer
- Initial version
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import sys
import numpy as np

#######################################################################################################################
# Settings
#######################################################################################################################
VARIANTS = {1: 'operator network (branch x trunk)', 2: 'coordinate network T(x, t)'}
PHYS = {0: 'none (data only)', 1: 'heat equation, automatic differentiation', 2: 'finite-volume operator of the data'}
SMAX = 32                                                               # maximum number of source amplitudes
DEFAULTS = {'phys': 1, 'alpha': 1.0, 'beta': 1.0, 'gamma': 0.1, 'K': 16, 'nSens': 0, 'nDat': 4096, 'nCol': 1024,
            'srcTol': 1e-6, 'width': 64, 'depth': 4, 'fourier': 4, 'warm': 50, 'branch': 2, 'steps': 100, 'seed': 42,
            'col': 1}
INTS = ('phys', 'K', 'nSens', 'nDat', 'nCol', 'width', 'depth', 'fourier', 'warm', 'branch', 'steps', 'seed', 'col')


#######################################################################################################################
# Parameters
#######################################################################################################################
def pfPara(para):
    """Settings of the PF mode (Mdl › pinf) with defaults for parameters that the config workbook does not hold."""
    p = para.get('Mdl', {}).get('pinf', {}) or {}
    out = {}
    for k, d in DEFAULTS.items():
        v = p.get(k)
        v = d if v is None or (isinstance(v, str) and not v.strip()) else v
        out[k] = int(v) if k in INTS else float(v)
    return out


def checkNorm(para):
    """The PF mode works in raw units (the physics needs K, W/m³): the data normalisation must be off."""
    g = para.get('Par', {}).get('gen', {})
    if int(g.get('normX', 1) or 1) != 1 or int(g.get('normY', 1) or 1) != 1:
        sys.exit("ERROR: PF mode: set Par › Data Normalisation X and y to 1 (none); the network scales the "
                 "coordinates, sources and temperatures itself and the physics loss needs raw units")


#######################################################################################################################
# Coordinates
#######################################################################################################################
def geoScale(geo):
    """Scaling of the coordinates: {'lo', 'hi' (m), 's' (1/m), 'd'}; axes without extent get s = 1."""
    geo = np.asarray(geo, dtype=float)
    lo, hi = geo.min(axis=0), geo.max(axis=0)
    ext = np.where(hi - lo > 0, hi - lo, 2.0)
    return {'lo': lo, 'hi': hi, 's': 2.0 / ext, 'd': geo.shape[1]}


def toHat(geo, sc):
    """Scaled coordinates x̂ = (x − lo)·s − 1 (N x d, float32)."""
    return ((np.asarray(geo, dtype=float) - sc['lo']) * sc['s'] - 1.0).astype(np.float32)


#######################################################################################################################
# Temperature rise
#######################################################################################################################
def offset(d, init):
    """Reference of every profile (list of Nt x N or 0.0): r for init = 1, else 0 (the offset was removed)."""
    return [np.asarray(r, dtype=float) for r in d['r2']] if int(init) == 1 else [0.0] * len(d['y2'])


def rise(d, init):
    """Temperature rise θ = T − T_ref (K) of every profile (list of Nt x N)."""
    return [np.asarray(y, dtype=float) - o for y, o in zip(d['y2'], offset(d, init))]


#######################################################################################################################
# Source encoding
#######################################################################################################################
def encodeSrc(Q2, tol=1e-6):
    """
    Basis U (N x S, orthonormal columns) of the heat generation of the profiles Q2 (list of Nt x N) (W/m³) and the
    relative singular values σ_j/σ_1 (all). S = 0 if there are no sources.
    """
    Q = np.concatenate([np.asarray(q, dtype=float) for q in Q2])
    N = Q.shape[1]
    act = np.flatnonzero(np.max(np.abs(Q), axis=0) > 0)
    if len(act) == 0:
        return np.zeros((N, 0)), np.zeros(0)
    Qa = Q[:, act]
    if len(Qa) > 4000:                                                  # rank of the sources: a subset of the samples
        Qa = Qa[np.linspace(0, len(Qa) - 1, 4000).astype(int)]
    [_, sv, Vt] = np.linalg.svd(Qa, full_matrices=False)
    S = int(min(np.sum(sv > tol * sv[0]), SMAX))
    U = np.zeros((N, S))
    U[act] = Vt[0:S].T
    sgn = np.sign(np.sum(U, axis=0))                                    # footprints positive
    sgn[sgn == 0] = 1.0
    return U * sgn[None, :], sv / sv[0]


def projectSrc(Q, U):
    """Amplitudes p = Q·U⁺ (Nt x S, least squares; U⁺ = Uᵀ for the orthonormal basis of the training mesh) and the
    relative error ‖Q − p·Uᵀ‖/‖Q‖ of the encoding."""
    Q = np.asarray(Q, dtype=float)
    if U.shape[1] == 0:
        return np.zeros((len(Q), 0)), float(np.any(Q != 0))
    p = Q @ np.linalg.pinv(U).T
    nQ = np.linalg.norm(Q)
    err = float(np.linalg.norm(Q - p @ U.T) / nQ) if nQ > 0 else 0.0
    return p, err


def mapNodes(geoA, geoB):
    """Index of the nearest node of mesh A for every node of mesh B (identity if the meshes are equal)."""
    geoA, geoB = np.asarray(geoA, dtype=float), np.asarray(geoB, dtype=float)
    if geoA.shape == geoB.shape and np.allclose(geoA, geoB):
        return np.arange(len(geoA))
    from scipy.spatial import cKDTree
    return cKDTree(geoA).query(geoB)[1]


#######################################################################################################################
# Sensors
#######################################################################################################################
def selSensors(geoHat, n, seed=42, first=None):
    """Sensor nodes: all (n <= 0 or n >= N) or n nodes by farthest-point sampling from the node 'first'."""
    N = len(geoHat)
    if n <= 0 or n >= N:
        return np.arange(N)
    g = np.asarray(geoHat, dtype=float)
    idx = np.zeros(n, dtype=int)
    idx[0] = int(np.random.default_rng(seed).integers(N)) if first is None else int(first)
    dist = np.sum((g - g[idx[0]]) ** 2, axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.sum((g - g[idx[i]]) ** 2, axis=1))
    return np.sort(idx)


#######################################################################################################################
# Material
#######################################################################################################################
def material(D, d):
    """
    Material of the nodes: {'rc' (N) (J/m³K), 'k' (N x d) (W/mK), 'hv' (N) (W/m³K), 'h' (d) node spacing (m),
    'op': None or {'C' (N) (Ws/K), 'G' (scipy sparse N x N) (W/K), 'V' (N) (m³)}}.
    """
    N = np.asarray(D['geo']).shape[0]

    def vec(k, default):
        v = D.get(k)
        if v is None:
            return np.full(N, float(default)) if np.ndim(default) == 0 else np.asarray(default, dtype=float)
        v = np.asarray(v, dtype=float).ravel()
        return np.full(N, float(v[0])) if v.size == 1 else v
    kx = vec('k', 1.0)
    ks = [kx, vec('ky', kx), vec('kz', kx)][0:d]
    out = {'rc': vec('rho', 1.0) * vec('Cp', 1.0), 'k': np.column_stack(ks), 'hv': vec('hv', 0.0),
           'h': np.array([float(np.ravel(D.get(k, 0.0))[0]) for k in ('dx', 'dy', 'dz')[0:d]]), 'op': None}
    if all(k in D for k in ('Cn', 'Gi', 'Gj', 'Gv', 'Vn')) and np.size(D['Cn']) == N:
        import scipy.sparse as sp
        G = sp.coo_matrix((np.ravel(D['Gv']), (np.ravel(D['Gi']).astype(int), np.ravel(D['Gj']).astype(int))),
                          shape=(N, N)).tocsr()
        out['op'] = {'C': np.ravel(D['Cn']).astype(float), 'G': G, 'V': np.ravel(D['Vn']).astype(float)}
    return out


#######################################################################################################################
# References
#######################################################################################################################
