#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         sfFit
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function fits the nodal thermal network of M coupled heat sources (SF mode, 'structure function' model in the
MATLAB version). Two methods are available (sel):

(1) Nodal Foster model from single-source experiments (MATLAB method)

    C·dT/dt = P(t) - G·T

    a) For every node k a Foster network with Nr RC pairs is fitted to the self-heating curve Zth_kk(t) = ΔT_k/P_k of
       experiment k (only node k is heated). The RC pairs give the branch capacitances C (diagonal, M·Nr) and the
       weights w_ik = R_ik / Σ_k R_ik.
    b) The steady-state resistance matrix Rth_ik = ΔT_i(end)/P_k(end) is calculated from the last samples of every
       experiment, symmetrised and inverted (G = Rth⁻¹).
    This requires one single-source experiment per node (number of profiles ≥ M). In the ODE (sfSol) the branch time
    constants are C_ik/(w_ik·G_ii) and not the fitted R_ik·C_ik, so the dynamics are not identical to the Foster fit
    (F-F1). If there are fewer profiles than nodes, method (2) is used.

(2) Coupled Cauer network fitted to any profiles (see sfNet)

    Every source has a Cauer ladder of Nr RC stages to the reference, the stages k of the sources are coupled by the
    resistances R_ijk. All R and C are found by an output-error fit (bounded nonlinear least squares, trust region
    reflective)

        min_p Σ_n ‖T(n) - T̂(n; p)‖²,      p = log[R_ik, C_ik, 1/R_ijk],      p0 - ln(bnd) ≤ p ≤ p0 + ln(bnd)

    The log-parameters keep every R and C positive, so every candidate network is passive and stable. The bounds
    limit every parameter to a factor bnd around its initial value: R_ik = R_th,i/Nr (R_th,i: diagonal of the
    least-squares fit ΔT(end) ≈ Rth·P(end) of the last fifth of every profile),
    τ_ik log-spaced between Ts/2 and a third of the profile length, coupling 1/R_ijk = 0.1/mean(R_th,i). Long data are
    decimated to at most Nmax samples for the fit. The resistance matrix of the fitted network is Rth = [G⁻¹]_jj.

Inputs:     1) data:    training data
            2) val:     validation data (not used)
            3) para:    all simulation parameters (Mdl.rc.K stages, Mdl.sf.bnd, Mdl.sf.Nmax, Mdl.gen.iterMax)
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
            5) sel:     (1): nodal Foster model, (2): coupled Cauer network
Outputs:    1) mdl:     model (Rth, Gth, Cth, Nr, type and the parameters of the method)

V1.1: 07.11.2026, Pascal Schirmer
- Coupled Cauer network fitted to any profiles (sel = 2); the Foster method falls back to it instead of stopping
  when there are fewer single-source experiments than nodes
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.SF.sfNet import sfNet

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.optimize import least_squares


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _fncZth(p, t, K):
    R, tau = np.exp(p[0:K]), np.exp(p[K:])
    return np.sum(R[:, None] * (1 - np.exp(-t[None, :] / tau[:, None])), axis=0)


def _fitFoster(Pv, Tm, Tc, t, M, E, Nr, maxIter, tol):
    # ==============================================================================
    # Self-Heating
    # ==============================================================================
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

    # ==============================================================================
    # Resistance Matrix
    # ==============================================================================
    Cf = tauf / Rf
    Rth = np.zeros((M, M))
    for k in range(0, min(M, E)):
        Rth[:, k] = (Tm[k][-1, :] - Tc[k][-1, :]) / Pv[k][-1, k]
    Rth = (Rth + Rth.T) / 2
    Gth = np.linalg.inv(Rth)
    Gth = (Gth + Gth.T) / 2
    return {'type': 'foster', 'weights': Rf / Rf.sum(axis=1, keepdims=True), 'Rth_rc': Rf, 'tau_rc': tauf,
            'Cth_rc': Cf, 'Rth': Rth, 'Gth': Gth, 'Cth': np.diag(Cf.ravel()), 'Nr': Nr}


def _fitNet(Pv, Tm, Tc, Ts, M, Nr, maxIter, bnd, Nmax):
    # ==============================================================================
    # Data (temperature rises, decimated)
    # ==============================================================================
    Y2 = [np.asarray(y, dtype=float) - np.asarray(r, dtype=float) for y, r in zip(Tm, Tc)]
    P2 = [np.asarray(p, dtype=float) for p in Pv]
    L = sum(len(p) for p in P2)
    ds = max(1, int(np.ceil(L / max(Nmax, 100))))
    P2d = [p[::ds] for p in P2]
    Y2d = [y[::ds] for y in Y2]
    T0 = [y[0] for y in Y2d]
    yAll = np.concatenate(Y2d)
    if ds > 1:
        print("INFO: Data decimated by %d for the fit (%d samples)" % (ds, len(yAll)))

    # ==============================================================================
    # Initial Values
    # ==============================================================================
    yEnd = np.concatenate([y[-max(len(y) // 5, 1):] for y in Y2])
    pEnd = np.concatenate([p[-max(len(p) // 5, 1):] for p in P2])
    Rls = np.linalg.lstsq(pEnd, yEnd, rcond=1e-6)[0].T                      # end values ΔT ≈ Rth·P
    Rsum = np.mean(yEnd, axis=0) / max(np.mean(np.sum(pEnd, axis=1)), 1e-9)
    Rtot = np.where(np.diag(Rls) > 0, np.diag(Rls), Rsum)
    Rtot = np.where(np.isfinite(Rtot) & (Rtot > 0), Rtot, 1.0)
    tEnd = np.mean([len(p) for p in P2]) * Ts
    tau = np.logspace(np.log10(0.5 * Ts), np.log10(max(tEnd / 3, 30 * Ts)), Nr)
    R0 = np.repeat((Rtot / Nr)[:, None], Nr, axis=1)
    C0 = tau[None, :] / R0
    p0 = np.log(np.concatenate([R0.ravel(), C0.ravel(), np.full(Nr * M * (M - 1) // 2, 0.1 / np.mean(Rtot))]))

    # ==============================================================================
    # Output-Error Fit (bounded)
    # ==============================================================================
    def res(p):
        return (np.concatenate(sfNet(p, M, Nr, P2d, Ts * ds, T0)) - yAll).ravel()

    lb, ub = p0 - np.log(bnd), p0 + np.log(bnd)
    sol = least_squares(res, p0, bounds=(lb, ub), method='trf', x_scale='jac', max_nfev=min(maxIter, 2000))
    p = sol.x
    atB = np.isclose(p, lb, atol=1e-6) | np.isclose(p, ub, atol=1e-6)
    nAct = int(np.sum(atB[0:2 * M * Nr]))                                   # R and C at the bounds
    nOff = int(np.sum(np.isclose(p[2 * M * Nr:], lb[2 * M * Nr:], atol=1e-6)))
    print("INFO: Network fitted in %d evaluations, RMSE %.4f K (decimated data)" %
          (sol.nfev, np.sqrt(np.mean(sol.fun ** 2))))
    if nOff > 0:
        print("INFO: %d of %d couplings at the lower bound (negligible coupling)" % (nOff, len(p) - 2 * M * Nr))
    if nAct > 0:
        print("WARN: %d resistances/capacitances at their bounds (factor %.0e), increase Mdl.sf.bnd" % (nAct, bnd))

    # ==============================================================================
    # Network
    # ==============================================================================
    [G, C, E] = sfNet(p, M, Nr)
    n = M * Nr
    rows = np.arange(0, M) * Nr
    Rth = np.linalg.inv(G)[rows][:, rows]
    Rlad = np.exp(p[0:n]).reshape(M, Nr)
    Clad = np.exp(p[n:2 * n]).reshape(M, Nr)
    Rcpl = np.full((Nr, M, M), np.inf)
    c = 0
    for k in range(0, Nr):
        for i in range(0, M):
            for j in range(i + 1, M):
                Rcpl[k, i, j] = Rcpl[k, j, i] = np.exp(-p[2 * n + c])
                c = c + 1
    return {'type': 'net', 'p': p, 'p0': p0, 'bnd': bnd, 'nAct': nAct, 'Rth': Rth, 'Gth': G, 'Cth': np.diag(C),
            'Rlad': Rlad, 'Clad': Clad, 'Rcpl': Rcpl, 'Nr': Nr, 'M': M}


#######################################################################################################################
# Main Function
#######################################################################################################################
def sfFit(data, val, para, legacy=0, sel=1):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting nodal thermal network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    M = data['y'].shape[1]
    E = len(data['y2'])
    Nr = int(para['Mdl']['rc']['K'])
    maxIter = int(para['Mdl']['gen']['iterMax'])
    tol = max(para['Mdl']['gen']['eps'], 1e-15)
    sf = para['Mdl'].get('sf', {})
    bnd = float(sf.get('bnd', 1e4) or 1e4)
    Nmax = int(sf.get('Nmax', 4000) or 4000)

    # ==============================================================================
    # Variables
    # ==============================================================================
    Pv, Tc, Tm, t = data['X2'], data['r2'], data['y2'], data['t2']
    if legacy == 0 and para['Exp']['gen']['init'] != 1:
        Tc = [np.zeros_like(r) for r in Tc]                                     # offset already removed (F-F2)
    if sel == 1 and E < M:
        print("WARN: The nodal Foster model needs one single-source experiment per node (%d < %d), the coupled "
              "Cauer network is fitted instead" % (E, M))
        sel = 2

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    if sel == 1:
        mdl = _fitFoster(Pv, Tm, Tc, t, M, E, Nr, maxIter, tol)
    else:
        mdl = _fitNet(Pv, Tm, Tc, float(data['Ts']), M, Nr, maxIter, bnd, Nmax)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    print("INFO: Thermal resistance matrix Rth (K/W) and node capacitances Cth (Ws/K)")
    print(np.array2string(mdl['Rth'], precision=4))
    print(np.array2string(np.diag(mdl['Cth']), precision=4))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting nodal thermal network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
