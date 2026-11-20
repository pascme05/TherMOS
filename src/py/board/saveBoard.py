#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         saveBoard
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.3
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function writes a simulated board profile in the 2D or 3D field format of TherMOS (as the FEM data in
data/tutorial), so that the field models (PO, PS, PF) can be trained and tested on it:
    X (Nt x N) heat generation (W/m³), y (Nt x N) temperature (°C), r (Nt x N) reference (ambient, °C), t, Ts,
    geo (N x 2 or N x 3), dx, dy(, dz), Lx, Ly(, Lz) (m), k, ky(, kz) (W/mK), rho (kg/m³), Cp (J/kgK), hv (W/m³K),
    hc = 0, Ta (°C), fl = 0
1) Lumped top view (D of genBoard with view 'xy', 2D): in contrast to the FEM data (F-P16) the heat sink is stored
   (hv: distributed sink of the lumped board), so that the pure Galerkin projection (POD protocol 2/3) has a heat
   sink, too. With a heat sink plane below the board only the board temperatures are stored and hv is the equivalent
   series sink hvEq (no spreading in the heat sink), the calibrated protocols correct the difference.
2) Layer-resolved model (view 'xz', 3D): every finite-volume node of the stack (layers below, board layers, die
   attach and die) with its position (x, y, z of the cell centre), material and heat generation Q = (Bp·P + q)/V.
   The cooling acts on the boundary cells and is part of the operator only (hv = 0).
The finite-volume operator of the model is stored as well (not for the lumped board with a heat sink plane, whose
second plane is hidden), so that the PF mode can evaluate the discrete heat equation (Mdl › pinf › phys = 2):

    Cn·dθ/dt + G·θ = Vn·Q,      θ = T − T_a

    Cn (N) capacitances (Ws/K), G (N x N) conductances incl. the cooling as triplets Gi, Gj (0-based), Gv (W/K),
    Vn (N) control volumes (m³; 2D: m² per metre depth)
Inputs:     1) D:       field data of genBoard
            2) t:       time vector (Nt) (sec)
            3) P:       chip losses (Nt x nChips) (W)
            4) T:       temperature (Nt x N) (°C)
            5) fileName: file name incl. path (.mat)
            6) q:       constant additional losses in W per node (N) that were used in the simulation, or None
            7) extra:   additional variables of the file (dict), e.g. {'kT': κ} (1/K), or None
Outputs:    1) fileName: written file

V1.1: 07.11.2026, Pascal Schirmer
- Board plane only and equivalent sink hvEq for boards with a heat sink plane

V1.2: 16.11.2026, Pascal Schirmer
- Layer-resolved 3D export, finite-volume operator (Cn, Gi, Gj, Gv, Vn), constant additional losses q in X

V1.3: 20.11.2026, Pascal Schirmer
- Additional variables (extra), e.g. the temperature coefficient of the conductivity kT
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.calcBoardMat import calcBoardMat

# ==============================================================================
# External
# ==============================================================================
import os
import numpy as np
import scipy.io as sio


#######################################################################################################################
# Main Function
#######################################################################################################################
def saveBoard(D, t, P, T, fileName, q=None, extra=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    t = np.asarray(t, dtype=float)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Ts = float(t[1] - t[0]) if len(t) > 1 else 1.0
    op = None

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Layer-Resolved Model (3D)
    # ==============================================================================
    if 'nod' in D:
        n = D['nod']
        N = len(n['x'])
        [C, G, Bp] = calcBoardMat(D)
        Ta = np.full(N, float(D['Ta0']))
        T = np.asarray(T, dtype=float)[:, 0:N]
        Q = P @ np.asarray(Bp, dtype=float).T
        if q is not None:
            Q = Q + np.asarray(q, dtype=float)[None, :]
        Q = Q / n['V'][None, :]
        zs = np.unique(np.round(D['zc'], 12))
        out = {'X': Q, 'y': T, 'r': np.tile(Ta, (len(t), 1)), 't': t, 'Ts': Ts,
               'geo': np.column_stack([n['x'], n['y'], n['z'] - zs[0]]), 'dx': D['dx'], 'dy': D['dx'],
               'dz': float(np.min(np.diff(zs))) if len(zs) > 1 else float(D['dzr'][0]),
               'Lx': float(D['xg'][-1]), 'Ly': float(D['yg'][-1]), 'Lz': float(zs[-1] - zs[0]),
               'k': n['kx'], 'ky': n['kx'], 'kz': n['kz'], 'rho': n['rho'], 'Cp': n['rc'] / n['rho'],
               'hv': np.zeros(N), 'hc': np.zeros(N), 'Ta': Ta, 'fl': np.zeros(N)}
        op = [C, G, n['V']]

    # ==============================================================================
    # Lumped Top View (2D)
    # ==============================================================================
    else:
        N = len(D['geo'])
        T = np.asarray(T, dtype=float)[:, 0:N]                          # board plane only (heat sink plane hidden)
        V = np.asarray(D.get('wA', np.ones(N)), dtype=float) * D['dx'] * D['dy']
        Q = (np.asarray(D['B'], dtype=float) @ P.T).T
        if q is not None:
            Q = Q + (np.asarray(q, dtype=float)[0:N] / V)[None, :]
        Ta = np.asarray(D['Ta'], dtype=float)
        out = {'X': Q, 'y': T, 'r': np.tile(Ta, (len(t), 1)), 't': t, 'Ts': Ts, 'geo': D['geo'], 'dx': D['dx'],
               'dy': D['dy'], 'Lx': D['Lx'], 'Ly': D['Ly'], 'k': D['k'], 'ky': D['ky'], 'rho': D['rho'],
               'Cp': D['Cp'], 'hv': D.get('hvEq', D['hv']), 'hc': D['hc'], 'Ta': Ta, 'fl': D['fl']}
        if 'k2' not in D:
            [C, G, _] = calcBoardMat(D)
            op = [C, G, V]

    # ==============================================================================
    # Finite-Volume Operator
    # ==============================================================================
    if op is not None:
        G = op[1].tocoo()
        out.update({'Cn': np.asarray(op[0], dtype=float), 'Gi': G.row.astype(float), 'Gj': G.col.astype(float),
                    'Gv': G.data.astype(float), 'Vn': np.asarray(op[2], dtype=float)})

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    out.update(extra or {})
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
