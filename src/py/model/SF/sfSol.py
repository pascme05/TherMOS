#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         sfSol
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
This function solves the nodal thermal network of sfFit. For the nodal Foster model (sel = 1) every node i has Nr
branch temperatures T_ik with

    dT_ik/dt = w_ik · (θ_i·P_i(t) - Σ_j G_ij·T_j) / C_ik,      T_i = Σ_k T_ik

with the weights w_ik, the branch capacitances C_ik (Ws/K), the conductance matrix G (W/K) and the loss scaling
θ_i = θ(T_i + Toff_i) (see calcTheta). The ODE is solved with a stiff solver (BDF) and linear interpolation of P.
The coupled Cauer network (sel = 2) is simulated exactly for piece-wise constant losses (see sfNet); temperature
dependent losses are included by a fixed-point iteration Pv_j = Pv·θ(T_{j-1} + Toff) as in ssSol.
Inputs:     1) mdl:     fitted model
            2) data:    testing data
            3) para:    all simulation parameters
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) out:     prediction

V1.0: 07.11.2026, Pascal Schirmer
- Python port of sfSol.m; the temperature dependent losses are used in the ODE and the offset is handled by
  offRemove/offAdd only (MATLAB subtracted/added the reference a second time, F-F2)

V1.1: 07.11.2026, Pascal Schirmer
- Simulation of the coupled Cauer network (sfNet)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.calcTheta import calcTheta
from src.py.model.SF.sfNet import sfNet

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.integrate import solve_ivp


#######################################################################################################################
# Main Function
#######################################################################################################################
def sfSol(mdl, data, para, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Solving nodal thermal network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    t = np.asarray(data['t'], dtype=float)
    Pv = np.asarray(data['X'], dtype=float)
    Toff = np.asarray(data['r'], dtype=float)
    M = data['y'].shape[1]
    Nr = mdl['Nr']
    G = mdl['Gth']
    out = dict(data)
    rise = (legacy == 1) or (para['Exp']['gen']['init'] == 1)

    ###################################################################################################################
    # Coupled Cauer Network
    ###################################################################################################################
    if mdl.get('type') == 'net':
        # ==============================================================================
        # Profiles (temperature rises)
        # ==============================================================================
        P2 = [np.asarray(x, dtype=float) for x in data.get('X2', [Pv])]
        R2 = [np.asarray(r, dtype=float) for r in data.get('r2', [Toff])]
        Y2 = [np.asarray(y, dtype=float) for y in data.get('y2', [data['y']])]
        T0 = [y[0] - r[0] if rise else y[0] for y, r in zip(Y2, R2)]
        Ts = float(data.get('Ts', t[1] - t[0] if len(t) > 1 else 1.0))
        T2 = sfNet(mdl['p'], M, Nr, P2, Ts, T0)

        # ==============================================================================
        # Temperature Dependent Losses
        # ==============================================================================
        if legacy == 0 and para['Par']['loss']['Rohm'] != 0:
            for it in range(0, int(para['Mdl']['gen']['nSub'])):
                Pj = [p * calcTheta(T + r, para, legacy)
                      for p, T, r in zip(P2, T2, R2)]
                Tn = sfNet(mdl['p'], M, Nr, Pj, Ts, T0)
                err = np.mean(np.abs(np.concatenate(Tn) - np.concatenate(T2)))
                T2 = Tn
                if err < para['Mdl']['gen']['err']:
                    break
            print("INFO: Loss iteration finished after %d iterations" % (it + 1))

        # ==============================================================================
        # Output
        # ==============================================================================
        T = np.concatenate(T2)
        out['y'] = T + np.concatenate(R2) if rise else T
        out['X'] = Pv
        print("DONE: Solving nodal thermal network")
        return out
    w = mdl['weights']
    Cvec = np.diag(mdl['Cth'])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # the ODE states are temperature rises over the reference, the offset is only removed here if offRemove did not
    T0node = data['y'][0] - Toff[0] if rise else data['y'][0]
    T0 = (w * T0node[:, None]).ravel()

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    def ode(tt, T):
        Tn = T.reshape(M, Nr).sum(axis=1)
        P = np.array([np.interp(tt, t, Pv[:, i]) for i in range(0, M)])
        if legacy == 0 and para['Par']['loss']['Rohm'] != 0:
            Tr = np.array([np.interp(tt, t, Toff[:, i]) for i in range(0, M)])
            P = P * calcTheta(Tn + Tr, para, legacy)
        return ((w * (P - G @ Tn)[:, None]).ravel()) / Cvec

    sol = solve_ivp(ode, (t[0], t[-1]), T0, t_eval=t, method='BDF', rtol=1e-6, atol=1e-8)
    T = sol.y.T.reshape(len(t), M, Nr).sum(axis=2)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    if rise:
        T = T + Toff
    out['y'] = T
    out['X'] = Pv

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Solving nodal thermal network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
