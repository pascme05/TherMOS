#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         jacobianMesh
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
This function calculates the relative element size of the FE input mesh on the quadrature grid (MATLAB jacobian2D.m
and jacobian3D.m). The input points are triangulated (2D) or tetrahedralised (3D), the area/volume of every element is
divided by the median element size and interpolated to the quadrature points. Values within 1 ± th are set to one.
This is not the Jacobian of a coordinate transformation; it is only used for legacy=1 (F-P8).
Inputs:     1) pts:     input points (N x d) (m)
            2) qGrid:   list of quadrature vectors [x, y] or [x, y, z] (m)
            3) th:      threshold around one
Outputs:    1) Jgrid:   relative element size on the grid, shape (Ny x Nx) or (Nx x Ny x Nz) as in MATLAB
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
from math import factorial
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


#######################################################################################################################
# Main Function
#######################################################################################################################
def jacobianMesh(pts, qGrid, th):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    pts = np.asarray(pts, dtype=float)
    d = pts.shape[1]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    tri = Delaunay(pts)
    v = pts[tri.simplices]
    E = v[:, 1:, :] - v[:, 0:1, :]
    size = np.abs(np.linalg.det(E)) / factorial(d)
    jac = size / np.median(size)
    cen = v.mean(axis=1)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    if d == 2:
        Q = np.meshgrid(qGrid[0], qGrid[1])
    else:
        Q = np.meshgrid(qGrid[0], qGrid[1], qGrid[2], indexing='ij')
    qp = np.column_stack([q.ravel() for q in Q])
    J = LinearNDInterpolator(cen, jac)(qp)
    nan = np.isnan(J)
    J[nan] = NearestNDInterpolator(cen, jac)(qp[nan])
    J[(J < 1 + th) & (J > 1 - th)] = 1

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return np.abs(J).reshape(Q[0].shape)


#######################################################################################################################
# References
#######################################################################################################################
