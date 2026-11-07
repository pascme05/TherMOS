#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         fitPem
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
This function refines a discrete-time state-space model (D = 0) by minimising the simulation (output) error over all
experiments with a nonlinear least squares solver (prediction error method with output-error structure, as MATLAB
ssest with DisturbanceModel 'none'):

    min_{A,B,C,x0}  Σ_e Σ_k ‖y_e(k) - ŷ_e(k)‖²

The free parameters are all elements of A, B, C (and x0 of every experiment if estimated); the start values come from
fitN4sid. If stability is enforced, unstable eigenvalues λ are mirrored to 1/λ* and B, x0 are estimated again.
Inputs:     1) sys:     start model (A, B, C, x0)
            2) uList:   list of inputs (Nt_i x m)
            3) yList:   list of outputs (Nt_i x p)
            4) estX0:   (0) x0 = 0, (1) x0 estimated
            5) sta:     (1) no constraint, (2) stability enforced
            6) maxIter: maximum number of function evaluations
Outputs:    1) sys:     refined model
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.numeric.simSS import simSS
from src.py.numeric.fitBx0 import fitBx0

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.optimize import least_squares


#######################################################################################################################
# Main Function
#######################################################################################################################
def fitPem(sys, uList, yList, estX0=0, sta=1, maxIter=2000):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    A0, B0, C0 = sys['A'], sys['B'], sys['C']
    n, m, p, E = A0.shape[0], B0.shape[1], C0.shape[0], len(uList)
    sA, sB, sC = n * n, n * m, p * n

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    def unpack(th):
        A = th[0:sA].reshape(n, n)
        B = th[sA:sA + sB].reshape(n, m)
        C = th[sA + sB:sA + sB + sC].reshape(p, n)
        x0 = [th[sA + sB + sC + e * n:sA + sB + sC + (e + 1) * n] if estX0 else None for e in range(0, E)]
        return A, B, C, x0

    def res(th):
        A, B, C, x0 = unpack(th)
        r = [simSS(A, B, C, None, uList[e], x0[e])[0] - yList[e] for e in range(0, E)]
        r = np.concatenate([x.ravel() for x in r])
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    th0 = np.concatenate([A0.ravel(), B0.ravel(), C0.ravel()] + ([np.concatenate(sys['x0'])] if estX0 else []))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    with np.errstate(all="ignore"):
        sol = least_squares(res, th0, method="trf", x_scale="jac", max_nfev=maxIter)
    A, B, C, x0 = unpack(sol.x)
    x0 = [np.zeros(n) if x is None else x for x in x0]

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    # ==============================================================================
    # Stability
    # ==============================================================================
    lam = np.linalg.eigvals(A)
    if np.max(np.abs(lam)) >= 1:
        if sta == 2:
            lam, V = np.linalg.eig(A)
            lam = np.where(np.abs(lam) >= 1, 0.999 / np.conj(lam), lam)
            A = np.real(V @ np.diag(lam) @ np.linalg.inv(V))
            [B, x0] = fitBx0(A, C, uList, yList, estX0)
            print("INFO: Unstable poles mirrored into the unit circle (stability enforced)")
        else:
            print("WARN: Identified model is unstable")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'A': A, 'B': B, 'C': C, 'D': np.zeros((p, m)), 'x0': x0, 'n': n, 'cost': float(sol.cost)}


#######################################################################################################################
# References
#######################################################################################################################
# [1] L. Ljung, "System Identification: Theory for the User", 2nd ed., Prentice Hall, 1999.
