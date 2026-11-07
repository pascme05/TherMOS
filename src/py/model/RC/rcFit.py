#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         rcFit
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
This function fits the thermal resistances (Rth) and capacitances (Cth) of a 1D Foster network with K RC pairs for
the first input/output channel. The transient thermal impedance of the network is

    Zth(t) = Σ_k Rth_k · (1 - exp(-t/τ_k)),     τ_k = Rth_k · Cth_k

Corrected version (legacy=0): the network response to the measured losses (rcSim) is fitted to the measured
temperature rise ΔT(t) = T(t) - T(0) of all training profiles (output-error fit). This works for any loss profile,
not only for a constant loss step. The parameters are fitted on a log scale (always positive) and the start values
of τ are spread logarithmically between Ts and the profile length.
Legacy version (legacy=1): the profiles are averaged, Zth = ΔT/Pv is calculated and the Foster function is fitted to
Zth (only valid for a constant loss step, F-R2), start values Rth = τ = 1, bounds as in MATLAB (F-R1).
If para['Mdl']['gen']['opt'] = 1 the order K = 1..Kmax with the smallest validation error is selected.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model with Rth (K/W), tau (sec), Cth (Ws/K), K, Ts
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.RC.rcSim import rcSim

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.optimize import least_squares


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _fncZth(p, t, K):
    return np.sum(p[0:K, None] * (1 - np.exp(-t[None, :] / p[K:2 * K, None])), axis=0)


def _profiles(d):
    Pv = [x[:, 0] for x in d['X2']]
    dT = [y[:, 0] - y[0, 0] for y in d['y2']]
    return Pv, dT


def _avgZth(d):
    Nt = min(d['Nt'])
    Pv = np.mean([x[0:Nt, 0] for x in d['X2']], axis=0)
    T = np.mean([y[0:Nt, 0] for y in d['y2']], axis=0)
    t = np.arange(Nt) * d['Ts']
    with np.errstate(divide='ignore', invalid='ignore'):
        Zth = (T - T[0]) / Pv
    Zth = np.nan_to_num(Zth, nan=0.0, posinf=0.0, neginf=0.0)
    return t, Zth


def _fitLegacy(t, Zth, K, bc, iterMax, tol):
    if bc > 0:
        lb2 = np.array([[bc] * K, [bc] * K])
        ub2 = np.array([[np.max(Zth)] * K, [np.max(t)] * K])
        lb, ub = lb2.ravel(order='F'), ub2.ravel(order='F')
    else:
        lb, ub = np.zeros(2 * K), np.full(2 * K, np.inf)
    p0 = np.clip(np.ones(2 * K), lb, ub)
    sol = least_squares(lambda p: _fncZth(p, t, K) - Zth, p0, bounds=(lb, ub), max_nfev=iterMax,
                        ftol=tol, xtol=tol)
    return sol.x, sol


def _fitSim(Pv, dT, Ts, K, bc, iterMax, tol):
    tEnd = max(len(p) for p in Pv) * Ts
    Rtot = np.mean([d[-1] / p[-1] for d, p in zip(dT, Pv) if abs(p[-1]) > 0] or [1.0])
    tau0 = np.logspace(np.log10(Ts), np.log10(tEnd), K)
    p0 = np.log(np.concatenate([np.full(K, max(Rtot, 1e-6) / K), tau0]))
    if bc > 0:
        lb = np.log(np.full(2 * K, bc))
        ub = np.log(np.concatenate([np.full(K, max(10 * Rtot, bc * 10)), np.full(K, max(10 * tEnd, bc * 10))]))
        p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
    else:
        lb, ub = -np.inf, np.inf

    def res(p):
        R, tau = np.exp(p[0:K]), np.exp(p[K:])
        return np.concatenate([rcSim(R, tau, P, Ts)[0] - d for P, d in zip(Pv, dT)])

    sol = least_squares(res, p0, bounds=(lb, ub), max_nfev=iterMax, ftol=tol, xtol=tol)
    p = np.exp(sol.x)
    idx = np.argsort(p[K:])
    return np.concatenate([p[0:K][idx], p[K:][idx]]), sol


#######################################################################################################################
# Main Function
#######################################################################################################################
def rcFit(data, val, para, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting Foster Network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    K = int(para['Mdl']['rc']['K'])
    bc = para['Mdl']['rc']['bc']
    iterMax = int(para['Mdl']['gen']['iterMax'])
    tol = max(para['Mdl']['gen']['eps'], 1e-15)
    Kmax = int(para['Mdl']['gen']['Kmax'])
    Ts = data['Ts']

    # ==============================================================================
    # Variables
    # ==============================================================================
    if legacy == 1:
        t, Zth = _avgZth(data)
        tVl, ZthVl = _avgZth(val)
        fit = lambda k: _fitLegacy(t, Zth, k, bc, iterMax, tol)
        err = lambda p, k: np.sum((_fncZth(p, tVl, k) - ZthVl) ** 2)
    else:
        Pv, dT = _profiles(data)
        PvVl, dTVl = _profiles(val)
        fit = lambda k: _fitSim(Pv, dT, Ts, k, bc, iterMax, tol)
        err = lambda p, k: np.sum(np.concatenate([rcSim(p[0:k], p[k:], P, Ts)[0] - d for P, d in zip(PvVl, dTVl)]) ** 2)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Optimal Order
    # ==============================================================================
    if para['Mdl']['gen']['opt'] == 1:
        best = np.inf
        for k in range(1, Kmax + 1):
            p, _ = fit(k)
            e = err(p, k)
            print("INFO: Order K = %d, validation residual = %.4e" % (k, e))
            if e < best:
                best, K = e, k

    # ==============================================================================
    # Fixed Order
    # ==============================================================================
    p, sol = fit(K)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    if sol.success:
        print("INFO: Optimized Rth (K/W): " + np.array2string(p[0:K], precision=4))
        print("INFO: Optimized tau (sec): " + np.array2string(p[K:], precision=4))
    else:
        print("WARN: Optimization did not converge: " + sol.message)

    mdl = {'Rth': p[0:K], 'tau': p[K:2 * K], 'Cth': p[K:2 * K] / p[0:K], 'K': K, 'Ts': Ts,
           'res': sol.fun, 'resnorm': float(2 * sol.cost)}

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting Foster Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
# [1] Touzelbaev, Maxat N., et al. "High-efficiency transient temperature calculations for applications in dynamic
#     thermal management of electronic devices." Journal of Electronic Packaging 135.3 (2013): 031001.
