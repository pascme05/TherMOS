#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         simBoard
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
This function simulates the finite-volume board model (calcBoardMat) for the chip losses P(t). The ODE

    C·dθ/dt = −G·θ + Bp·P(t),      T = T_a + θ

is integrated with the implicit Euler method on nSub sub-steps per sampling period (unconditionally stable, the
sparse system matrix C/h + G is factorised once). The steady state G·θ = Bp·P(end) is returned as well. The
initial state is the ambient temperature (θ = 0) unless T0 is given.

Optional electro-thermal coupling (cpl): the losses of every chip rise linearly with its junction temperature

    P_i,eff = P_i·max(1 + α_i·(T_j,i − T_ref), 0),     T_j,i = T_a + W_iᵀ·θ   (W: die-average weights)

In the transient the junction temperature of the previous sub-step is used (one-step lag, h ≪ thermal time
constants); the steady state is solved exactly with the low-rank update (Woodbury identity)

    (G − U·Wᵀ)·θ = Bp·P·(1 + α·(T_a − T_ref)),     U = Bp·diag(α·P)

and a thermal runaway (loop gain ≥ 1) is reported.
Inputs:     1) C:       capacitances (N)
            2) G:       conductance matrix (sparse N x N)
            3) Bp:      input matrix (N x nChips)
            4) P:       chip losses (Nt x nChips) (W)
            5) Ts:      sampling time (sec)
            6) Ta:      ambient temperature (°C)
            7) nSub:    implicit Euler sub-steps per sampling period
            8) T0:      initial temperature (N) (°C), None: ambient
            9) cpl:     electro-thermal coupling {'W' (N x nChips), 'alpha' (nChips) (1/K), 'Tref' (°C)} or None
            10) prog:   progress callback prog(fraction) or None
            11) steady: (True): steady state is calculated, (False): Tss = None (faster)
            12) q:      constant additional losses in W per node (N), e.g. traces in the board, or None
Outputs:    1) T:       temperature at the sampling instants (Nt x N) (°C)
            2) Tss:     steady-state temperature for P(end) (N) (°C)

V1.1: 07.11.2026, Pascal Schirmer
- Electro-thermal coupling (linear temperature coefficient of the losses), progress callback

V1.2: 08.11.2026, Pascal Schirmer
- Constant additional losses q (C·dθ/dt = −G·θ + Bp·P(t) + q), not coupled to the temperature
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
import scipy.sparse as sp
import scipy.sparse.linalg as spl


#######################################################################################################################
# Main Function
#######################################################################################################################
def simBoard(C, G, Bp, P, Ts, Ta, nSub=4, T0=None, cpl=None, prog=None, steady=True, q=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Nt, N = P.shape[0], len(C)
    h = Ts / max(int(nSub), 1)
    th = np.zeros(N) if T0 is None else np.asarray(T0, dtype=float) - Ta
    T = np.zeros((Nt, N))
    if cpl is not None:
        W = np.asarray(cpl['W'], dtype=float)
        al = np.asarray(cpl['alpha'], dtype=float)
        Tref = float(cpl.get('Tref', 25.0))
    nP = max(Nt // 100, 1)
    q = np.zeros(N) if q is None else np.asarray(q, dtype=float)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    lu = spl.splu((sp.diags(C / h) + G).tocsc())

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    T[0] = th
    for k in range(1, Nt):
        u = Bp @ P[k] + q
        for _ in range(0, max(int(nSub), 1)):
            if cpl is not None:
                u = Bp @ (P[k] * np.maximum(1.0 + al * (W.T @ th + Ta - Tref), 0.0)) + q
            th = lu.solve(C / h * th + u)
        T[k] = th
        if prog is not None and k % nP == 0:
            prog(k / Nt)

    # ==============================================================================
    # Steady State
    # ==============================================================================
    if not steady:
        return [T + Ta, None]
    luG = spl.splu(G.tocsc())
    if cpl is None:
        Tss = luG.solve(Bp @ P[-1] + q)
    else:
        b = Bp @ (P[-1] * (1.0 + al * (Ta - Tref))) + q
        U = Bp * (al * P[-1])[None, :]
        x0 = luG.solve(b)
        GU = np.column_stack([luG.solve(U[:, i]) for i in range(U.shape[1])]) if U.shape[1] else np.zeros((N, 0))
        M = np.eye(U.shape[1]) - W.T @ GU
        if np.any(np.real(np.linalg.eigvals(np.eye(U.shape[1]) - M)) >= 1.0):
            print("WARN: Thermal runaway: the losses rise faster with T_j than the cooling removes them")
        Tss = x0 + GU @ np.linalg.solve(M, W.T @ x0)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [T + Ta, Tss + Ta]


#######################################################################################################################
# References
#######################################################################################################################
