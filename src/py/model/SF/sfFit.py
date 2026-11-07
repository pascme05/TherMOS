#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         sfFit
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
This function fits the nodal Foster model ('structure function' model in the MATLAB version) of M coupled nodes:

    C·dT/dt = P(t) - G·T

1) For every node k a Foster network with Nr RC pairs is fitted to the self-heating curve Zth_kk(t) = ΔT_k/P_k of
   experiment k (only node k is heated). The RC pairs give the branch capacitances C (diagonal, M·Nr) and the
   weights w_ik = R_ik / Σ_k R_ik.
2) The steady-state resistance matrix Rth_ik = ΔT_i(end)/P_k(end) is calculated from the last samples of every
   experiment, symmetrised and inverted (G = Rth⁻¹).
This requires one single-source experiment per node (number of profiles ≥ M). Note: in the ODE (sfSol) the branch
time constants are C_ik/(w_ik·G_ii) and not the fitted R_ik·C_ik, so the dynamics are not identical to the Foster fit
(F-F1, open point, the model is marked as experimental in the README).
Inputs:     1) data:    training data
            2) val:     validation data (not used)
            3) para:    all simulation parameters
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model (Rth, Gth, Cth, weights, Rth_rc, tau_rc, Cth_rc, Nr)
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
from scipy.optimize import least_squares


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _fncZth(p, t, K):
    R, tau = np.exp(p[0:K]), np.exp(p[K:])
    return np.sum(R[:, None] * (1 - np.exp(-t[None, :] / tau[:, None])), axis=0)


#######################################################################################################################
# Main Function
#######################################################################################################################
def sfFit(data, val, para, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting nodal Foster model")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    M = data['y'].shape[1]
    E = len(data['y2'])
    Nr = int(para['Mdl']['rc']['K'])
    maxIter = int(para['Mdl']['gen']['iterMax'])
    tol = max(para['Mdl']['gen']['eps'], 1e-15)
    Pv, Tc, Tm, t = data['X2'], data['r2'], data['y2'], data['t2']
    if legacy == 0 and para['Exp']['gen']['init'] != 1:
        Tc = [np.zeros_like(r) for r in Tc]                                     # offset already removed (F-F2)
    if E < M:
        sys.exit('ERROR: The nodal Foster model needs one single-source experiment per node (%d < %d)' % (E, M))

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    Rf = np.zeros((M, Nr))
    tauf = np.zeros((M, Nr))
    for k in range(0, M):
        Pk = Pv[k][:, k]
        dT = Tm[k][:, k] - Tc[k][:, k]
        tk = np.asarray(t[k], dtype=float).ravel()
        Zth = dT / np.maximum(Pk, 1e-9)
        Zth = Zth - Zth[0]
        w = 1 / (tk + 1e-6)
        tau0 = np.logspace(np.log10(tk[1]), np.log10(tk[-1]), Nr)
        p0 = np.log(np.concatenate([np.full(Nr, max(Zth[-1], 1e-9) / Nr), tau0]))
        sol = least_squares(lambda p: (_fncZth(p, tk, Nr) - Zth) * w, p0, method='lm', max_nfev=maxIter,
                            ftol=tol, xtol=tol)
        R, tau = np.exp(sol.x[0:Nr]), np.exp(sol.x[Nr:])
        idx = np.argsort(tau)
        Rf[k], tauf[k] = R[idx], tau[idx]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    Cf = tauf / Rf
    Cth = np.diag(Cf.ravel())
    Rth = np.zeros((M, M))
    for k in range(0, min(M, E)):
        Rth[:, k] = (Tm[k][-1, :] - Tc[k][-1, :]) / Pv[k][-1, k]
    Rth = (Rth + Rth.T) / 2
    Gth = np.linalg.inv(Rth)
    Gth = (Gth + Gth.T) / 2

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    print("INFO: Thermal resistance Rth (K/W) and capacitance Cth (Ws/K)")
    print(np.array2string(Rth, precision=4))
    print(np.array2string(np.diag(Cth), precision=4))

    mdl = {'weights': Rf / Rf.sum(axis=1, keepdims=True), 'Rth_rc': Rf, 'tau_rc': tauf, 'Cth_rc': Cf,
           'Rth': Rth, 'Gth': Gth, 'Cth': Cth, 'Nr': Nr}

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting nodal Foster model")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
