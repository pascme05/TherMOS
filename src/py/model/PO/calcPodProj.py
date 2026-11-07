#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcPodProj
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
This function calculates the Galerkin projection of the heat equation onto the POD modes φ_i:

    ρc·∂T/∂t = ∇·(k∇T) + Q,     T(x,t) = T0(x) + Σ_i φ_i(x)·a_i(t)

    C_ij = ∫ ρc·φ_i·φ_j dΩ      thermal capacitance matrix (Ws/K)
    G_ij = ∫ k·∇φ_i·∇φ_j dΩ     thermal conductance matrix (W/K)
    q_i  = ∫ Q·φ_i dΩ           projected heat generation (W)

    C·da/dt = -G·a + q

The modes are zero on Dirichlet boundaries (mean centred snapshots with fixed boundary temperature), so no boundary
term appears in the weak form. Legacy version (legacy=1, MATLAB): diffusivity form C_ij = ∫ J·φ_i·φ_j, G_ij =
∫ J·α·∇φ_i·∇φ_j + boundary terms of φ_j·∂φ_i/∂n without normal direction, q_i = ∫ J·(α/k)·Q·φ_i (F-P2, F-P3, F-P8).
Inputs:     1) quad:    quadrature data (calcPodQuad)
            2) rPhi:    POD modes at the nodes (N x K)
            3) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) proj:    dictionary with C, G (K x K), phiQ (Nq x K) and sPhi (grid modes, shape + (K,))
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
def calcPodLoad(quad, phiQ, Q, legacy=0):
    """
    This function projects the heat generation onto the modes.
    Inputs:     1) quad:    quadrature data
                2) phiQ:    modes at the quadrature points (Nq x K)
                3) Q:       volumetric heat generation at the nodes (Nt x N) (W/m³)
                4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
    Outputs:    1) q:       projected heat generation (Nt x K) (W)
    """
    Qq = (quad['Rq'] @ (quad['Mg'] @ np.asarray(Q, dtype=float).T)).T
    if legacy == 1:
        wq = quad['J'] * quad['W'] * quad['alpha'] / quad['k']
    else:
        wq = quad['W']
    return (Qq * wq) @ phiQ


#######################################################################################################################
# Main Function
#######################################################################################################################
def calcPodProj(quad, rPhi, legacy=0):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    K = rPhi.shape[1]
    shape = quad['shape']
    W, J = quad['W'], quad['J']
    Rq = quad['Rq']

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    sPhi = (quad['Mg'] @ rPhi).reshape(shape + (K,))
    grads = np.gradient(sPhi, *quad['axes'], axis=tuple(range(0, quad['d'])))
    phiQ = Rq @ sPhi.reshape(-1, K)
    dPhiQ = [Rq @ g.reshape(-1, K) for g in grads]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    if legacy == 1:
        aG = quad['alphaGrid']
        C = phiQ.T @ ((J * W)[:, None] * phiQ)
        G = sum(dq.T @ ((J * W * aG)[:, None] * dq) for dq in dPhiQ)

        # boundary terms (2D only, as in MATLAB)
        ny, nx = shape[0], shape[1]
        wy = np.polynomial.legendre.leggauss(ny)[1] * 0.5 * (quad['axes'][0][-1] - quad['axes'][0][0])
        wx = np.polynomial.legendre.leggauss(nx)[1] * 0.5 * (quad['axes'][1][-1] - quad['axes'][1][0])
        base = (aG * J)[:, None]
        tX = (base[:, :, None] * phiQ[:, None, :] * dPhiQ[1][:, :, None]).reshape(ny, nx, K, K)
        tY = (base[:, :, None] * phiQ[:, None, :] * dPhiQ[0][:, :, None]).reshape(ny, nx, K, K)
        Gx = np.einsum('y,yij->ij', wy, tX[:, 0] + tX[:, -1])
        Gy = np.einsum('x,xij->ij', wx, tY[0] + tY[-1])
        G = G + Gx + Gy
    else:
        C = phiQ.T @ ((W * quad['rhoCp'])[:, None] * phiQ)
        G = sum(dq.T @ ((W * quad['k'])[:, None] * dq) for dq in dPhiQ)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'C': C, 'G': G, 'phiQ': phiQ, 'sPhi': sPhi}


#######################################################################################################################
# References
#######################################################################################################################
