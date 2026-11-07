#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcInterpMat
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
This function calculates a sparse interpolation matrix M that maps values at scattered points (e.g. FE nodes) to
query points (e.g. a regular grid): v_q = M · v. For linear interpolation the barycentric weights of the Delaunay
simplex that contains the query point are used, points outside the convex hull take the value of the nearest
scattered point (as scatteredInterpolant(..., 'nearest') in MATLAB). The matrix is calculated once and cached, so that
all time steps and all modes are interpolated with one sparse matrix product instead of one interpolation each.
Inputs:     1) pts:     scattered points (N x d) (m)
            2) qpts:    query points (Nq x d) (m)
            3) method:  (1) nearest, (2) linear, (3) natural (linear is used)
Outputs:    1) M:       sparse interpolation matrix (Nq x N)
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
import hashlib
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix

#######################################################################################################################
# Additional Functions
#######################################################################################################################
_CACHE = {}
_TRI = {}


def _key(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


#######################################################################################################################
# Main Function
#######################################################################################################################
def calcInterpMat(pts, qpts, method=2):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    pts = np.asarray(pts, dtype=float)
    qpts = np.asarray(qpts, dtype=float)
    Nq, d = qpts.shape
    N = pts.shape[0]
    key = _key(pts, qpts, np.array([method]))
    if key in _CACHE:
        return _CACHE[key]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Nearest Neighbour
    # ==============================================================================
    idxNear = cKDTree(pts).query(qpts)[1]
    if method == 1:
        M = csr_matrix((np.ones(Nq), (np.arange(Nq), idxNear)), shape=(Nq, N))

    # ==============================================================================
    # Linear (Barycentric)
    # ==============================================================================
    else:
        kTri = _key(pts)
        if kTri not in _TRI:
            _TRI[kTri] = Delaunay(pts)
        tri = _TRI[kTri]
        simp = tri.find_simplex(qpts)
        inside = simp >= 0
        T = tri.transform[simp[inside]]
        b = np.einsum('ijk,ik->ij', T[:, :d, :], qpts[inside] - T[:, d, :])
        bary = np.hstack([b, 1 - b.sum(axis=1, keepdims=True)])
        verts = tri.simplices[simp[inside]]
        rows = np.repeat(np.where(inside)[0], d + 1)
        rowsOut = np.where(~inside)[0]
        M = csr_matrix((np.concatenate([bary.ravel(), np.ones(len(rowsOut))]),
                        (np.concatenate([rows, rowsOut]), np.concatenate([verts.ravel(), idxNear[~inside]]))),
                       shape=(Nq, N))

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    _CACHE[key] = M

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return M


#######################################################################################################################
# References
#######################################################################################################################
