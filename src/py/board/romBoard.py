#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         romBoard
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
This function builds and simulates a POD-Galerkin reduced-order model of the finite-volume board model (calcBoardMat).
The POD modes are the left singular vectors of the snapshot matrix of the temperature rise θ = T − T_a (training run),
the full model is projected onto the first K modes (discrete Galerkin projection, the same as calcPodProj on the FV
grid):

    θ ≈ Φ·a,   Φ = U[:, 0:K],   E_K = Σ_{i≤K} σ_i² / Σ_i σ_i²   (captured energy)
    C_r·da/dt = −G_r·a + B_r·P,   C_r = Φᵀ·diag(C)·Φ,   G_r = Φᵀ·G·Φ,   B_r = Φᵀ·Bp

The reduced model is integrated with the same implicit Euler scheme as simBoard, so the difference to the full model
is the projection error only.
Inputs:     1) C:       capacitances of the full model (N)
            2) G:       conductance matrix (sparse N x N)
            3) Bp:      input matrix (N x nChips)
            4) Tsnap:   training snapshots (Nt x N) (°C)
            5) Ta:      ambient temperature (°C)
            6) K:       number of modes
Outputs:    1) rom:     reduced model {'Phi' (N x K), 'sig' (singular values), 'E' (captured energy per K), 'Cr', 'Gr',
                        'Br', 'K'}
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
def simRom(rom, P, Ts, Ta, nSub=4):
    """
    Simulates the reduced model for the chip losses P (Nt x nChips) and returns the temperature at all nodes
    (Nt x N) (°C), starting at the ambient temperature.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    h = Ts / max(int(nSub), 1)
    A = rom['Cr'] / h + rom['Gr']
    Ainv = np.linalg.inv(A)
    a = np.zeros(rom['K'])
    out = np.zeros((P.shape[0], rom['K']))
    for k in range(1, P.shape[0]):
        u = rom['Br'] @ P[k]
        for _ in range(0, max(int(nSub), 1)):
            a = Ainv @ (rom['Cr'] / h @ a + u)
        out[k] = a
    return out @ rom['Phi'].T + Ta


#######################################################################################################################
# Main Function
#######################################################################################################################
def romBoard(C, G, Bp, Tsnap, Ta, K):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    th = np.asarray(Tsnap, dtype=float) - Ta

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # POD
    # ==============================================================================
    U, sig, _ = np.linalg.svd(th.T, full_matrices=False)
    K = int(max(1, min(K, len(sig))))
    E = np.cumsum(sig ** 2) / max(np.sum(sig ** 2), 1e-300)
    Phi = U[:, 0:K]

    # ==============================================================================
    # Galerkin Projection
    # ==============================================================================
    Cr = Phi.T @ (np.asarray(C)[:, None] * Phi)
    Gr = Phi.T @ (G @ Phi)
    Br = Phi.T @ Bp

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'Phi': Phi, 'sig': sig, 'E': E, 'Cr': Cr, 'Gr': Gr, 'Br': Br, 'K': K}


#######################################################################################################################
# References
#######################################################################################################################
# [1] P. Holmes, J. L. Lumley, G. Berkooz, Turbulence, Coherent Structures, Dynamical Systems and Symmetry, 1996.
