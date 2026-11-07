#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         fitN4sid
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
This function identifies a discrete-time state-space model (D = 0) from input/output data of one or more experiments
with a subspace method (past-output MOESP, a member of the N4SID family). The block Hankel matrices of the past
inputs/outputs (Wp), future inputs (Uf) and future outputs (Yf) of all experiments are stacked, the part of Yf that
is explained by Wp but not by Uf is extracted with an LQ decomposition and its SVD gives the extended observability
matrix Γ = [C; CA; CA²; ...]. A and C follow from the shift structure of Γ, B and x0 from linear least squares
(fitBx0). If n is a list, the order is selected from the largest gap of the singular values (log scale). If no
horizon is given, the horizon with the smallest simulation error on the identification data is selected from a set
of candidates (as N4Horizon 'auto' in MATLAB).
Inputs:     1) uList:   list of inputs (Nt_i x m)
            2) yList:   list of outputs (Nt_i x p)
            3) n:       model order or list of candidate orders
            4) estX0:   (0) x0 = 0, (1) x0 estimated
            5) hor:     number of block rows (horizon), default max(10, 2·max(n))
Outputs:    1) sys:     dictionary with A, B, C, D, x0 (list), sv (singular values) and n
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.numeric.fitBx0 import fitBx0
from src.py.numeric.simSS import simSS

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _hankel(x, i, j, start):
    # rows: block rows start..start+i-1, columns: j samples
    return np.vstack([x[start + k:start + k + j].T for k in range(0, i)])


def _simErr(sys, uList, yList):
    err = 0.0
    for e in range(0, len(uList)):
        yp = simSS(sys['A'], sys['B'], sys['C'], None, uList[e], sys['x0'][e])[0]
        err = err + np.sum((yp - yList[e]) ** 2)
    return err if np.isfinite(err) else np.inf


#######################################################################################################################
# Main Function
#######################################################################################################################
def fitN4sid(uList, yList, n, estX0=0, hor=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    orders = list(np.atleast_1d(n))

    ###################################################################################################################
    # Horizon Selection
    ###################################################################################################################
    if hor is None:
        Nmin = min(len(u) for u in uList)
        cand = sorted({max(k, max(orders) + 1) for k in (2 * max(orders), 3 * max(orders), 5, 8, 10, 15, 20, 30)})
        cand = [h for h in cand if Nmin - 2 * h + 1 >= 2 * h * (uList[0].shape[1] + yList[0].shape[1])]
        if len(cand) > 1:
            best = None
            for h in cand:
                s = fitN4sid(uList, yList, n, estX0, h)
                e = _simErr(s, uList, yList)
                if best is None or e < best[0]:
                    best = (e, h, s)
            print("INFO: Selected subspace horizon %d (simulation error)" % best[1])
            return best[2]
    i = hor if hor is not None else max(10, 2 * max(orders))
    m = uList[0].shape[1]
    p = yList[0].shape[1]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    Uf, Wp, Yf = [], [], []
    for u, y in zip(uList, yList):
        j = u.shape[0] - 2 * i + 1
        if j < 1:
            continue
        Up, Yp = _hankel(u, i, j, 0), _hankel(y, i, j, 0)
        Uf.append(_hankel(u, i, j, i))
        Yf.append(_hankel(y, i, j, i))
        Wp.append(np.vstack([Up, Yp]))
    if len(Uf) == 0:
        raise ValueError('Data too short for subspace identification with horizon ' + str(i))
    Uf, Wp, Yf = np.hstack(Uf), np.hstack(Wp), np.hstack(Yf)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # LQ Decomposition
    # ==============================================================================
    H = np.vstack([Uf, Wp, Yf]) / np.sqrt(Uf.shape[1])
    L = np.linalg.qr(H.T, mode='r').T
    mU, mW = Uf.shape[0], Wp.shape[0]
    L32 = L[mU + mW:, mU:mU + mW]

    # ==============================================================================
    # Observability Matrix
    # ==============================================================================
    U, S, _ = np.linalg.svd(L32, full_matrices=False)
    if len(orders) > 1:
        cand = [k for k in orders if 0 < k < len(S)]
        gap = [np.log10(S[k - 1] + 1e-300) - np.log10(S[k] + 1e-300) for k in cand]
        nSel = cand[int(np.argmax(gap))]
        print("INFO: Selected model order n = %d (singular value gap)" % nSel)
    else:
        nSel = int(orders[0])
    Gam = U[:, 0:nSel] * np.sqrt(S[0:nSel])

    # ==============================================================================
    # System Matrices
    # ==============================================================================
    C = Gam[0:p, :]
    A = np.linalg.lstsq(Gam[0:-p, :], Gam[p:, :], rcond=None)[0]
    [B, x0] = fitBx0(A, C, uList, yList, estX0)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'A': A, 'B': B, 'C': C, 'D': np.zeros((p, m)), 'x0': x0, 'sv': S, 'n': nSel}


#######################################################################################################################
# References
#######################################################################################################################
# [1] M. Verhaegen, "Identification of the deterministic part of MIMO state space models given in innovations form
#     from input-output data", Automatica, vol. 30, no. 1, pp. 61-74, 1994.
# [2] P. Van Overschee and B. De Moor, "Subspace Identification for Linear Systems", Kluwer, 1996.
