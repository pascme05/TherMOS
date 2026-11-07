#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcPodQuad
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
This function prepares the Gauss-Legendre quadrature of the POD Galerkin projection on the rectangular (2D) or
cuboid (3D) domain [0, Lx] x [0, Ly] (x [0, Lz]). The scattered FE nodes are mapped to the regular grid
x = 0:dx:Lx, y = 0:dy:Ly (, z = 0:dz:Lz) with linear interpolation and the grid values are mapped to the quadrature
points with multilinear interpolation (as map2D/map3D + interp2/interp3 in the MATLAB version). Both steps are linear,
so they are stored as sparse matrices:

    v_grid = Mg · v_nodes,      v_quad = Rq · v_grid,       ∫ f dΩ ≈ Σ W_q · f(x_q)

The order of the grid arrays is (y, x) in 2D and (y, x, z) in 3D (as in MATLAB meshgrid).
Inputs:     1) Data:    geometry and material data (geo, dx, dy, (dz), Lx, Ly, (Lz), k, rho, Cp, hc)
            2) legacy:  (0): corrected behaviour, (1): MATLAB behaviour (relative element size Jgrid, F-P8)
            3) protocol:POD protocol (Jgrid only used for protocol > 2 and legacy = 1)
            4) th:      threshold of the relative element size around one (legacy only)
Outputs:    1) quad:    dictionary with axes, shape, Mg, Rq, W, J, k, rhoCp, alpha (at the quadrature points),
                        alphaGrid, kGrid (on the grid) and d (dimension)

V1.1: 07.11.2026, Pascal Schirmer
- Optional anisotropic conductivity ky (y direction) and distributed heat sink hv (W/m³K) at the quadrature points
  for the corrected version (board designer data)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.numeric.calcInterpMat import calcInterpMat
from src.py.numeric.calcMultilinMat import calcMultilinMat
from src.py.numeric.gaussLegendre import gaussLegendre
from src.py.numeric.jacobianMesh import jacobianMesh

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Main Function
#######################################################################################################################
def calcPodQuad(Data, legacy=0, protocol=5, th=0.001):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    d = 3 if 'dz' in Data and 'Lz' in Data and np.asarray(Data['geo']).shape[1] == 3 else 2
    names = ['x', 'y', 'z'][0:d]
    vec = {n: np.arange(0, Data['L' + n] + Data['d' + n] * 1e-6, Data['d' + n]) for n in names}
    geo = np.asarray(Data['geo'], dtype=float)[:, 0:d]
    geo = geo - geo.min(axis=0)

    # ==============================================================================
    # Variables
    # ==============================================================================
    order = ['y', 'x', 'z'][0:d]
    axes = [vec[n] for n in order]
    shape = tuple(len(a) for a in axes)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Quadrature Points
    # ==============================================================================
    gl = {n: gaussLegendre(len(vec[n]), vec[n][0], vec[n][-1]) for n in names}
    G = np.meshgrid(*[gl[n][0] for n in order], indexing='ij')
    Wl = np.meshgrid(*[gl[n][1] for n in order], indexing='ij')
    W = np.prod(np.stack([w.ravel() for w in Wl]), axis=0)
    qpts = np.column_stack([g.ravel() for g in G])

    # ==============================================================================
    # Interpolation Matrices
    # ==============================================================================
    Gg = np.meshgrid(*axes, indexing='ij')
    gpts = np.column_stack([g.ravel() for g in Gg])
    colsGeo = [names.index(n) for n in order]
    Mg = calcInterpMat(geo[:, colsGeo], gpts, 2)
    Rq = calcMultilinMat(axes, qpts)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Material
    # ==============================================================================
    k = np.asarray(Data['k'], dtype=float).ravel()
    rhoCp = np.asarray(Data['rho'], dtype=float).ravel() * np.asarray(Data['Cp'], dtype=float).ravel()
    kGrid = Mg @ k
    alphaGrid = Mg @ (k / rhoCp)

    # ==============================================================================
    # Relative Element Size (legacy)
    # ==============================================================================
    J = np.ones(len(W))
    if legacy == 1 and protocol > 2:
        Jm = jacobianMesh(geo, [gl['x'][0], gl['y'][0]] + ([gl['z'][0]] if d == 3 else []), th)
        J = (Jm if d == 2 else np.transpose(Jm, (1, 0, 2))).ravel()

    # ==============================================================================
    # Boundary Conditions
    # ==============================================================================
    if 'hc' in Data and np.any(np.asarray(Data['hc']) > 0):
        print("WARN: Convective boundary conditions (hc > 0) are not included in the POD projection")

    # ==============================================================================
    # Anisotropy and Heat Sink (corrected version only)
    # ==============================================================================
    kDir, hv = None, None
    if legacy == 0 and 'ky' in Data:
        ky = np.asarray(Data['ky'], dtype=float).ravel()
        kDir = [Rq @ (Mg @ ky), Rq @ kGrid] + ([Rq @ kGrid] if d == 3 else [])            # order of the axes y, x, z
    if legacy == 0 and 'hv' in Data:
        hv = Rq @ (Mg @ np.asarray(Data['hv'], dtype=float).ravel())

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'d': d, 'axes': axes, 'shape': shape, 'Mg': Mg, 'Rq': Rq, 'W': W, 'J': J,
            'k': Rq @ kGrid, 'rhoCp': Rq @ (Mg @ rhoCp), 'alpha': Rq @ alphaGrid,
            'kGrid': kGrid, 'alphaGrid': alphaGrid, 'kDir': kDir, 'hv': hv}


#######################################################################################################################
# References
#######################################################################################################################
