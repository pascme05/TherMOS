#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcKrylov
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
Block Krylov reduction (moment matching at s = 0) of the finite-volume model of a board

    C·dθ/dt = −G·θ + B·p(t),        B = V·U  (N x S)

with the capacitances C (N) (Ws/K), the conductance matrix G (N x N) (W/K, incl. the cooling), the control volumes
V (N) (m³), the source footprints U (N x S) and the source amplitudes p (Nt x S) (W/m³), θ = T − T_ref (K).
The transfer function H(s) = (G + s·C)⁻¹·B is expanded around s = 0; its moments are the blocks

    Θ = G⁻¹·B (steady-state fields of the sources),  M·Θ, M²·Θ, …,  M = G⁻¹·C (time constants, s)

and the basis V_r = orth[Θ, M·Θ, …, M^(q−1)·Θ] (N x r, r = q·S) matches the first q moments of H(s) exactly
(Odabasioglu et al., PRIMA). The Galerkin projection gives the reduced model

    C_r·dx/dt = −G_r·x + B_r·p,     C_r = V_rᵀ·diag(C)·V_r,  G_r = V_rᵀ·G·V_r,  B_r = V_rᵀ·B,   θ ≈ V_r·x

which is stable and passive for any q (C_r, G_r symmetric positive definite). It is solved exactly in its modal
form (G_r·w = λ·C_r·w) for piece-wise constant (zero-order hold) or piece-wise linear (first-order hold) sources.
The reduction needs the operator (C, G, V) of the board and one sparse LU factorisation of G, but no temperature
data: it is rebuilt from the operator of every new board.
Functions:  basis(C, G, B, q):              V_r, C_r, G_r, B_r, time constants of the reduced model
            simulate(R, p, Ts, hold):       reduced states x (Nt x r) for the amplitudes p (Nt x S)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.linalg import eigh


#######################################################################################################################
# Basis
#######################################################################################################################
def basis(C, G, B, q=4):
    """
    Block Krylov basis and reduced matrices. C: (N) capacitances (Ws/K), G: sparse (N x N) (W/K), B: (N x S) input
    matrix (W per unit amplitude), q: number of moment blocks. Returns {'V', 'C', 'G', 'B', 'tau' (s, descending),
    'r', 'q', 'tRed' (s): time of factorisation and reduction}.
    """
    t0 = time.time()
    C = np.asarray(C, dtype=float).ravel()
    B = np.atleast_2d(np.asarray(B, dtype=float).T).T
    lu = spl.splu(sp.csc_matrix(G))
    K = [lu.solve(B)]
    for _ in range(1, int(q)):
        K.append(lu.solve(C[:, None] * K[-1]))
    M = np.column_stack(K)
    M = M / np.maximum(np.linalg.norm(M, axis=0), 1e-300)[None, :]
    [V, R] = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > 1e-10 * np.abs(R[0, 0])                 # drop linearly dependent moments
    V = V[:, keep]
    Cr = V.T @ (C[:, None] * V)
    Gr = V.T @ (G @ V)
    Gr = 0.5 * (Gr + Gr.T)
    lam = eigh(Gr, Cr, eigvals_only=True)
    return {'V': V, 'C': Cr, 'G': Gr, 'B': V.T @ B, 'tau': np.sort(1.0 / lam)[::-1], 'r': int(V.shape[1]),
            'q': int(q), 'tRed': time.time() - t0}


#######################################################################################################################
# Simulation
#######################################################################################################################
def simulate(R, p, Ts, hold=0):
    """
    Exact discrete solution of C_r·dx/dt = −G_r·x + B_r·p, x(0) = 0, in the modal form (G_r·W = C_r·W·Λ,
    Wᵀ·C_r·W = I): z_k = a·z_(k−1) + b0·u_(k−1) + b1·u_k with u = Wᵀ·B_r·p, a = exp(−λ·Ts).
    hold = 0: zero-order hold (p_k acts on (t_(k−1), t_k]), hold = 1: linear between the samples.
    Returns x (Nt x r).
    """
    p = np.atleast_2d(np.asarray(p, dtype=float).T).T
    [lam, W] = eigh(R['G'], R['C'])
    u = p @ (W.T @ R['B']).T                                            # Nt x r (modal inputs)
    x = lam * Ts
    a = np.exp(-x)
    g = (1.0 - a) / lam                                                 # ∫ exp(−λ·s) ds over one step
    if hold == 1:
        b1 = (x - 1.0 + a) / (lam * x)                                  # weight of u_k (linear interpolation)
        b0 = g - b1
    else:
        b0, b1 = np.zeros_like(g), g
    z = np.zeros((len(p), len(lam)))
    for k in range(1, len(p)):
        z[k] = a * z[k - 1] + b0 * u[k - 1] + b1 * u[k]
    return z @ W.T


#######################################################################################################################
# References
#######################################################################################################################
"""
Odabasioglu, A.; Celik, M.; Pileggi, L. T. PRIMA: Passive Reduced-Order Interconnect Macromodeling Algorithm.
IEEE Trans. CAD 1998, 17, 645-654.
Antoulas, A. C. Approximation of Large-Scale Dynamical Systems. SIAM 2005.
"""
