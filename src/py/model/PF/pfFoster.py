#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfFoster
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
Coupled Foster model of a board family (baseline of the half-bridge study). The rise at the output o (sensor node) for
the losses P_c of the chips c is a sum of first-order lags with K time constants τ_k shared by all inputs and outputs,

    θ_o(t) = Σ_c Σ_k R_o,c,k · x_c,k(t),      τ_k·dx_c,k/dt = P_c(t) − x_c,k,

i.e. the thermal impedance Z_o,c(t) = Σ_k R_o,c,k (1 − e^(−t/τ_k)) of the step response; the self-heating (o = c) and
the coupling (o ≠ c) are both included. The lags are integrated exactly for piece-wise constant losses (zero-order
hold): x[n] = a_k·x[n−1] + (1 − a_k)·P[n], a_k = exp(−T_s/τ_k), x[0] = 0 (cold start, as the finite-volume model).
1) Fit of one board (fitBoard): output-error fit over all labelled profiles. For given τ the model is linear in R, so
   that R follows from linear least squares and only log τ is optimised (variable projection, scipy least_squares);
   neighbouring τ differ by at least the factor GAP, since equal τ give large R of opposite sign.
2) Family (fitFam): every labelled board is fitted on its own, and the parameters (log τ_k and R_o,c,k) are regressed
   linearly on the features f(μ) = (1, log h, log k_diel, log s_att, s_rc) of the condition (least squares with a small
   ridge). The chip positions are not an input: a Foster model describes the layout it was fitted on, so its error grows
   with the distance of a new layout from the training layouts.
3) Prediction (predict): parameters for the condition μ of a new board, rise at the outputs for the losses P.
4) Joint fit (fitJoint): log τ_k and R_o,c,k linear in f(μ), fitted to all boards at once (output error, variable
   projection), started from 2); optionally with the time constants bounded by TAU_MAX. Since the per-board fits are
   not unique (lags may swap or compensate each other), their regression is inconsistent; the joint fit avoids this.
   The variant (regression, joint, joint bounded) is selected with the validation boards (select).
Functions:  lags(P, tau, Ts), fitBoard(Ps, Ys, K, Ts), fitFam(boards, nS, K, Ts), fitJoint(...), select(...),
            predict(mdl, mu, P)

V1.0: 21.11.2026, Pascal Schirmer
- Initial version
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfParam import NAMES

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

#######################################################################################################################
# Settings
#######################################################################################################################
TAU0 = (0.05, 40.0)                                                     # range of the start values of τ (s)
RIDGE = 1e-6                                                            # ridge of the regression over μ
GAP = 1.3                                                               # minimum ratio of neighbouring τ (-)
TAU_MAX = 80.0                                                          # upper bound of τ of the family (s), as τ_cap of the network


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def lags(P, tau, Ts):
    """Lags x (L x nc x K) of the losses P (L x nc) (W) for the time constants tau (K) (s), x[0] = 0."""
    P = np.asarray(P, dtype=float).copy()
    P[0] = 0.0
    out = np.zeros((P.shape[0], P.shape[1], len(tau)))
    for k, tk in enumerate(tau):
        a = np.exp(-Ts / tk)
        out[:, :, k] = lfilter([1.0 - a], [1.0, -a], P, axis=0)
    return out


def _feat(mu):
    """Features of the condition: (1, log h, log k_diel, log s_att, s_rc)."""
    return np.array([1.0] + [np.log(mu[k]) if k != 'sRc' else mu[k] for k in NAMES])


#######################################################################################################################
# Fit
#######################################################################################################################
def fitBoard(Ps, Ys, K=4, Ts=0.05):
    """
    Foster model of one board: Ps list of losses (L x nc) (W), Ys list of rises at the outputs (L x nS) (K). Returns
    {'tau' (K) (s, ascending), 'R' (nc x K x nS) (K/W), 'rmse' (K)}.
    """
    Y = np.concatenate(Ys)

    def lin(lt):
        X = np.concatenate([lags(P, np.exp(lt), Ts).reshape(len(P), -1) for P in Ps])
        R = np.linalg.lstsq(X, Y, rcond=None)[0]
        return X, R

    def res(lt):
        X, R = lin(lt)
        return (X @ R - Y).ravel()

    # log τ = log τ_1 + cumulative gaps; gaps >= log(GAP) keep the lags distinct (equal τ give cancelling R)
    def lts(x):
        return x[0] + np.concatenate([[0.0], np.cumsum(x[1:])])

    d0 = (np.log(TAU0[1]) - np.log(TAU0[0])) / max(K - 1, 1)
    x0 = np.array([np.log(TAU0[0])] + [d0] * (K - 1))
    lo = [np.log(Ts / 4)] + [np.log(GAP)] * (K - 1)
    hi = [np.log(1e3)] + [np.log(1e4)] * (K - 1)
    sol = least_squares(lambda x: res(lts(x)), x0, bounds=(lo, hi), x_scale=1.0, max_nfev=200)
    lt = lts(sol.x)
    X, R = lin(lt)
    nc = Ps[0].shape[1]
    return {'tau': np.exp(lt), 'R': R.reshape(nc, K, -1), 'rmse': float(np.sqrt(np.mean((X @ R - Y) ** 2)))}


def fitFam(boards, nS, K=4, Ts=0.05):
    """
    Foster model of a family: boards with 'P', 'ths' (rises at the sensor nodes) and 'mu'; outputs are the first nS
    sensor nodes. Returns {'W' (5 x p) regression of the parameters, 'K', 'nc', 'nS', 'fits' (per board)}.
    """
    fits = [fitBoard(B['P'], [y[:, 0:nS] for y in B['ths']], K, Ts) for B in boards]
    F = np.stack([_feat(B['mu']) for B in boards])
    Th = np.stack([np.concatenate([np.log(f['tau']), f['R'].ravel()]) for f in fits])
    W = np.linalg.solve(F.T @ F + RIDGE * len(F) * np.eye(F.shape[1]), F.T @ Th)
    return {'W': W, 'K': K, 'nc': fits[0]['R'].shape[0], 'nS': nS, 'Ts': Ts,
            'fits': [{'tau': f['tau'].tolist(), 'rmse': f['rmse']} for f in fits]}


def fitJoint(boards, nS, K=4, Ts=0.05, mdl0=None, nfev=60, tauMax=np.inf):
    """
    Joint Foster model of a family: log τ_k(μ) = a_k·f(μ) and R_o,c,k(μ) = w_o,c,k·f(μ) fitted to all labelled boards at
    once (output error over all profiles). For given a, the output is linear in w (regressors x_c,k(t)·f_j(μ)), so that
    w follows from linear least squares and only a (K x 5) is optimised (variable projection); start: the regression of
    the per-board fits (fitFam, or mdl0); tauMax bounds the time constants (s). Returns the same structure as fitFam
    (W: 5 x (K + nc·K·nS)) plus 'tauMax' and the training RMSE 'rmseJoint'.
    """
    mdl0 = mdl0 or fitFam(boards, nS, K, Ts)
    F = np.stack([_feat(B['mu']) for B in boards])                                          # boards x 5
    Ps = [(i, P) for i, B in enumerate(boards) for P in B['P']]
    Y = np.concatenate([y[:, 0:nS] for B in boards for y in B['ths']])
    nc, nf = mdl0['nc'], F.shape[1]

    def design(A):
        rows = []
        for i, P in Ps:
            tau = np.minimum(np.exp(F[i] @ A), tauMax)                                      # K
            x = lags(P, tau, Ts)                                                            # L x nc x K
            rows.append((x[..., None] * F[i][None, None, None, :]).reshape(len(P), -1))     # L x (nc·K·5)
        return np.concatenate(rows)

    def solve(A):
        X = design(A)
        Wr = np.linalg.lstsq(X, Y, rcond=None)[0]                                           # (nc·K·5) x nS
        return X, Wr

    def res(a):
        X, Wr = solve(a.reshape(nf, K))
        return (X @ Wr - Y).ravel()

    a0 = mdl0['W'][:, 0:K]                                                                  # 5 x K
    sol = least_squares(res, a0.ravel(), x_scale=1.0, max_nfev=nfev)
    A = sol.x.reshape(nf, K)
    X, Wr = solve(A)
    rm = float(np.sqrt(np.mean((X @ Wr - Y) ** 2)))
    Wr = Wr.reshape(nc, K, nf, nS).transpose(2, 0, 1, 3).reshape(nf, -1)                   # 5 x (nc·K·nS)
    return {'W': np.concatenate([A, Wr], axis=1), 'K': K, 'nc': nc, 'nS': nS, 'Ts': Ts, 'fits': mdl0['fits'],
            'rmseJoint': rm, 'tauMax': tauMax}


def rmse(mdl, boards):
    """Hot-spot RMSE (K) of a model on labelled boards (outputs of the model, all profiles)."""
    e = np.concatenate([predict(mdl, B['mu'], P) - y[:, 0:mdl['nS']] for B in boards for P, y in zip(B['P'], B['ths'])])
    return float(np.sqrt(np.mean(e ** 2)))


def select(boards, val, nS, K=4, Ts=0.05):
    """
    Foster model of a family selected by the validation boards val among the regression of the per-board fits
    (fitFam), the joint fit (fitJoint) and the joint fit with time constants bounded by TAU_MAX. Returns the model
    with 'variant' and the validation RMSE of all variants ('valRmse').
    """
    m0 = fitFam(boards, nS, K, Ts)
    cand = {'regression': m0, 'joint': fitJoint(boards, nS, K, Ts, m0),
            'jointBounded': fitJoint(boards, nS, K, Ts, m0, tauMax=TAU_MAX)}
    vr = {k: rmse(m, val) for k, m in cand.items()}
    best = min(vr, key=vr.get)
    return dict(cand[best], variant=best, valRmse=vr)


#######################################################################################################################
# Prediction
#######################################################################################################################
def params(mdl, mu):
    """Time constants (K) (s) and resistances (nc x K x nS) (K/W) of the condition mu."""
    th = _feat(mu) @ mdl['W']
    K = mdl['K']
    return np.minimum(np.exp(th[0:K]), mdl.get('tauMax', np.inf)), th[K:].reshape(mdl['nc'], K, mdl['nS'])


def predict(mdl, mu, P):
    """Rise (L x nS) (K) at the outputs of a board with the condition mu for the losses P (L x nc) (W)."""
    tau, R = params(mdl, mu)
    return np.einsum('lck,cko->lo', lags(P, tau, mdl['Ts']), R)


#######################################################################################################################
# References
#######################################################################################################################
# [1] A. Wintrich et al., Application Manual Power Semiconductors, SEMIKRON, 2015 (Foster and Cauer networks).
