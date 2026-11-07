#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         saveBoard
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
This function writes a simulated board profile in the 2D data format of TherMOS (as the FEM data in data/tutorial),
so that the PO and PS models can be trained and tested on it:
    X (Nt x N) heat generation (W/m³), y (Nt x N) temperature (°C), r (Nt x N) reference (ambient, °C), t, Ts,
    geo (N x 2), dx, dy, Lx, Ly (m), k, ky (W/mK), rho (kg/m³), Cp (J/kgK), hv (W/m³K), hc = 0, Ta (°C), fl = 0
In contrast to the FEM data (F-P16) the heat sink is stored (hv: distributed sink of the lumped board or the
convective rows of the cross-section), so that the pure Galerkin projection (POD protocol 2/3) has a heat sink, too.
With a heat sink plane below the board (top view) only the board temperatures are stored and hv is the equivalent
series sink hvEq (no spreading in the heat sink), the calibrated protocols correct the difference.
Inputs:     1) D:       field data of genBoard
            2) t:       time vector (Nt) (sec)
            3) P:       chip losses (Nt x nChips) (W)
            4) T:       temperature (Nt x N) (°C)
            5) fileName: file name incl. path (.mat)
Outputs:    1) fileName: written file

V1.1: 07.11.2026, Pascal Schirmer
- Board plane only and equivalent sink hvEq for boards with a heat sink plane
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
import os
import numpy as np
import scipy.io as sio


#######################################################################################################################
# Main Function
#######################################################################################################################
def saveBoard(D, t, P, T, fileName):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)[:, 0:len(D['geo'])]                  # board plane only (heat sink plane hidden)
    Q = (np.asarray(D['B'], dtype=float) @ np.atleast_2d(np.asarray(P, dtype=float)).T).T
    Ta = np.asarray(D['Ta'], dtype=float)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    out = {'X': Q, 'y': np.asarray(T, dtype=float), 'r': np.tile(Ta, (len(t), 1)), 't': t,
           'Ts': float(t[1] - t[0]) if len(t) > 1 else 1.0, 'geo': D['geo'], 'dx': D['dx'], 'dy': D['dy'],
           'Lx': D['Lx'], 'Ly': D['Ly'], 'k': D['k'], 'ky': D['ky'], 'rho': D['rho'], 'Cp': D['Cp'],
           'hv': D.get('hvEq', D['hv']),
           'hc': D['hc'], 'Ta': Ta, 'fl': D['fl']}

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    os.makedirs(os.path.dirname(os.path.abspath(fileName)), exist_ok=True)
    sio.savemat(fileName, out, do_compression=True)
    print("INFO: Board data saved to " + fileName)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return fileName


#######################################################################################################################
# References
#######################################################################################################################
