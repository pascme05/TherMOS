#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         sfNet
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
This function builds and simulates the coupled Cauer network of the SF mode. Every heat source i (chip) has a Cauer
ladder of Nr RC stages from its junction node to the reference (heat sink / ambient); the junction nodes of all
sources are coupled stage by stage with the conductances g_ijk (coupling resistances R_ijk = 1/g_ijk between stage k
of source i and stage k of source j; k = 1 couples the junctions directly, deeper stages couple through the
substrate / heat sink, which delays the cross-heating):

    C·dT/dt = E·P(t) - G·T,      T_j = T[junction nodes]

    G = Σ_i,k (1/R_ik)·(e_ik - e_i,k+1)(e_ik - e_i,k+1)ᵀ + Σ_i<j,k g_ijk·(e_ik - e_jk)(e_ik - e_jk)ᵀ

with the ladder resistances R_ik (K/W), capacitances C_ik (Ws/K), the input matrix E (losses enter at the junction
nodes) and the unit vectors e of the nodes (the last stage of every ladder is connected to the reference). G is
symmetric and positive definite, so the network is passive and stable for any positive parameters. The simulation is
exact for piece-wise constant losses (zero-order hold): with S = C^½, the symmetric matrix S⁻¹·G·S⁻¹ = V·Λ·Vᵀ gives
decoupled modes z = Vᵀ·S·T,

    z_k[n+1] = exp(-λ_k·Ts)·z_k[n] + (1 - exp(-λ_k·Ts))/λ_k · b_k[n],      b = Vᵀ·S⁻¹·E·P

that are filtered with scipy.signal.lfilter (one first-order filter per mode).
Inputs:     1) p:       log-parameters [log R (M·Nr), log C (M·Nr), log g (Nr·M(M-1)/2, stage by stage)]
            2) M:       number of heat sources (junction nodes)
            3) Nr:      number of Cauer stages per source
            4) P2:      list of loss profiles (Nt x M) (W), None returns the matrices only
            5) Ts:      sampling time (sec)
            6) T0:      list of initial junction temperature rises (M) per profile or None (zero, steady start)
Outputs:    1) out:     list of junction temperature rises (Nt x M) (K) per profile, or [G, C, E] for P2 = None
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
from scipy.signal import lfilter


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _matrices(p, M, Nr):
    n = M * Nr
    R = np.exp(p[0:n]).reshape(M, Nr)
    C = np.exp(p[n:2 * n])
    g = np.exp(p[2 * n:])
    G = np.zeros((n, n))

    # Ladders
    for i in range(0, M):
        for k in range(0, Nr):
            a = i * Nr + k
            G[a, a] += 1 / R[i, k]
            if k < Nr - 1:
                G[a + 1, a + 1] += 1 / R[i, k]
                G[a, a + 1] -= 1 / R[i, k]
                G[a + 1, a] -= 1 / R[i, k]

    # Coupling (stage by stage)
    c = 0
    for k in range(0, Nr):
        for i in range(0, M):
            for j in range(i + 1, M):
                a, b = i * Nr + k, j * Nr + k
                G[a, a] += g[c]
                G[b, b] += g[c]
                G[a, b] -= g[c]
                G[b, a] -= g[c]
                c = c + 1

    # Input
    E = np.zeros((n, M))
    E[np.arange(0, M) * Nr, np.arange(0, M)] = 1
    return [G, C, E]


#######################################################################################################################
# Main Function
#######################################################################################################################
def sfNet(p, M, Nr, P2=None, Ts=1.0, T0=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    [G, C, E] = _matrices(np.asarray(p, dtype=float), M, Nr)
    if P2 is None:
        return [G, C, E]
    out = []

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    S = np.sqrt(C)
    lam, V = np.linalg.eigh(G / S[:, None] / S[None, :])
    lam = np.maximum(lam, 1e-300)
    a = np.exp(-lam * Ts)
    gain = -np.expm1(-lam * Ts) / lam
    Bz = V.T @ (E / S[:, None])                                         # mode inputs (n x M)
    rows = np.arange(0, M) * Nr                                         # junction nodes
    Rjj = np.linalg.inv(G)[rows][:, rows]                               # steady-state Rth matrix of the junctions

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for e, P in enumerate(P2):
        # Initial state (steady state of the network with the given junction rises)
        z0 = np.zeros(len(lam))
        if T0 is not None and T0[e] is not None:
            x0 = np.linalg.solve(G, E @ np.linalg.solve(Rjj, np.asarray(T0[e], dtype=float)))
            z0 = V.T @ (S * x0)

        # Modes (z[n] = a·z[n-1] + gain·b[n-1], z[0] = z0)
        b = np.asarray(P, dtype=float) @ Bz.T
        Z = np.zeros_like(b)
        for k in range(0, len(lam)):
            Z[:, k] = lfilter([0.0, gain[k]], [1.0, -a[k]], b[:, k], zi=[z0[k]])[0]

        # Junction temperatures
        T = (Z @ V.T) / S[None, :]
        out.append(T[:, rows])

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
