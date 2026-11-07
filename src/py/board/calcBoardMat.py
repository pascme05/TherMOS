#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcBoardMat
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
This function assembles the vertex-centred finite-volume model of a board field (genBoard) on its regular grid: every
node has the control volume V_i = w_i·dx·dy (per metre depth, w_i = 1 inside, 1/2 on edges, 1/4 in corners),
neighbouring nodes are connected by the series conductance of their half distances (harmonic mean of the
conductivities) through the face of the control volumes, the distributed sink h_v connects every node to the ambient:

    C·dθ/dt = −G·θ + Bp·P(t),      θ = T − T_a
    C_i  = (ρc)_i·V_i                                            (Ws/K per m)
    G_ij = −A_ij / (d/(2k_i) + d/(2k_j))     for neighbours      (W/K per m)
    G_ii = Σ_j |G_ij| + h_v,i·V_i
    Bp   = B·V                                                   (heat into a node per W of a chip)

The boundary is adiabatic except for the sink. The x direction uses k, the y direction ky (anisotropic layers).
Inputs:     1) D:       field data of genBoard
Outputs:    1) C:       capacitances (N) (Ws/K per m depth)
            2) G:       conductance matrix (sparse N x N) (W/K per m depth)
            3) Bp:      input matrix (N x nChips) (W per m depth per W)
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


#######################################################################################################################
# Main Function
#######################################################################################################################
def calcBoardMat(D):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    ny, nx = D['shape']
    dx, dy = D['dx'], D['dy']
    kx = np.asarray(D['k'], dtype=float).reshape(ny, nx)
    ky = np.asarray(D.get('ky', D['k']), dtype=float).reshape(ny, nx)
    rc = (np.asarray(D['rho'], dtype=float) * np.asarray(D['Cp'], dtype=float))
    hv = np.asarray(D['hv'], dtype=float)
    wA = np.asarray(D.get('wA', np.ones(nx * ny)), dtype=float)
    wx = np.ones(nx)
    wy = np.ones(ny)
    if nx > 1:
        wx[[0, -1]] = 0.5
    if ny > 1:
        wy[[0, -1]] = 0.5
    idx = np.arange(nx * ny).reshape(ny, nx)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Face Conductances
    # ==============================================================================
    gx = (dy * wy)[:, None] / (dx / (2 * kx[:, :-1]) + dx / (2 * kx[:, 1:]))                     # (ny, nx-1)
    gy = (dx * wx)[None, :] / (dy / (2 * ky[:-1, :]) + dy / (2 * ky[1:, :]))                     # (ny-1, nx)

    # ==============================================================================
    # Assembly
    # ==============================================================================
    i = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    j = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    g = np.concatenate([gx.ravel(), gy.ravel()])
    N = nx * ny
    Goff = sp.coo_matrix((np.concatenate([-g, -g]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(N, N))
    V = wA * dx * dy
    diag = -np.asarray(Goff.sum(axis=1)).ravel() + hv * V
    G = (Goff + sp.diags(diag)).tocsc()
    C = rc * V
    Bp = np.asarray(D['B'], dtype=float) * V[:, None]

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [C, G, Bp]


#######################################################################################################################
# References
#######################################################################################################################
# [1] S. V. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere, 1980.
