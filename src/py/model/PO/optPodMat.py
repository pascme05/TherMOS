#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         optPodMat
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
This function calibrates the reduced matrices with the projected training data (equation error). With the modal
coordinates a(t), their time derivative da/dt (central differences) and the projected heat generation q(t):

    protocol 5: min_{C,G} ‖q - C·da/dt - G·a‖²                                 (MATLAB optCthGthUB)
    protocol 6: min_{C,G} ‖q - C·da/dt - G·a‖² + λ·(‖C - C0‖² + ‖G - G0‖²)       (MATLAB optCthGth, λ = 1e-3)
    protocol 4: min_α     ‖G·a - α∘q‖²                                         (MATLAB optAlphaSS)
    protocol ≥ 5: min_α   ‖C·da/dt + G·a - α∘q‖²                               (MATLAB optAlpha)

All problems are linear least squares problems and are solved in closed form per row (mode). The row weights √λ_i of
the MATLAB version do not change the per-row solutions and are not used. Protocol 5 is underdetermined for modes with
almost no energy; as the quasi-Newton solver of MATLAB started at the projected matrices, the minimum-norm correction
of the projected matrices is used (truncated SVD). If the calibrated model is singular or unstable the projected
matrices are kept. If ids are given, the derivatives are calculated per profile (no differences across profiles).
Inputs:     1) a:           modal coordinates (Nt x K)
            2) q:           projected heat generation (Nt x K)
            3) Ts:          sampling time (sec)
            4) C:           capacitance matrix (K x K)
            5) G:           conductance matrix (K x K)
            6) protocol:    POD protocol (1-6)
            7) ids:         profile id per sample (Nt) or None
Outputs:    1) C:           calibrated capacitance matrix
            2) G:           calibrated conductance matrix
            3) alpha:       load correction vector (K)
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
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _deriv(a, Ts, ids):
    if ids is None:
        return np.gradient(a, Ts, axis=0)
    da = np.zeros_like(a)
    for i in np.unique(ids):
        m = ids == i
        if np.sum(m) > 1:
            da[m] = np.gradient(a[m], Ts, axis=0)
    return da


def _alpha(r, q):
    den = np.sum(q * q, axis=0)
    return np.where(den > 0, np.sum(q * r, axis=0) / np.where(den > 0, den, 1), 1.0)


#######################################################################################################################
# Main Function
#######################################################################################################################
def optPodMat(a, q, Ts, C, G, protocol, ids=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    K = a.shape[1]
    alpha = np.ones(K)
    lam = 1e-3

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    da = _deriv(a, Ts, ids)
    Phi = np.hstack([da, a])

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Matrices
    # ==============================================================================
    X0 = np.vstack([C.T, G.T])
    X = X0
    if protocol == 5:
        print("INFO: Optimizing Cth and Gth unbounded")
        X = X0 + np.linalg.lstsq(Phi, q - Phi @ X0, rcond=1e-8)[0]
    elif protocol == 6:
        print("INFO: Optimizing Cth and Gth with priors")
        A = Phi.T @ Phi + lam * np.eye(2 * K)
        X = np.linalg.solve(A, Phi.T @ q + lam * X0)

    # ==============================================================================
    # Stability Check
    # ==============================================================================
    if protocol in (5, 6):
        Cn, Gn = X[0:K].T, X[K:].T
        try:
            stable = np.max(np.real(np.linalg.eigvals(-np.linalg.solve(Cn, Gn)))) < 0
        except np.linalg.LinAlgError:
            stable = False
        if stable:
            C, G = Cn, Gn
        else:
            print("WARN: Calibrated matrices are singular or unstable, projected matrices are used")

    # ==============================================================================
    # Load Correction
    # ==============================================================================
    if protocol == 4:
        print("INFO: Optimizing Alpha")
        alpha = _alpha(a @ G.T, q)
    elif protocol >= 5:
        print("INFO: Optimizing Alpha")
        alpha = _alpha(da @ C.T + a @ G.T, q)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    print("INFO: Matrix conditioning cond(C) + cond(G): %.3e" % (np.linalg.cond(C) + np.linalg.cond(G)))

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [C, G, alpha]


#######################################################################################################################
# References
#######################################################################################################################
