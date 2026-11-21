#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         krFit
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
This function "fits" the Krylov model (KR mode): a block Krylov reduction of the finite-volume operator of the
training board (calcKrylov). No temperature is used: the basis follows from the operator (C, G, V written by the
board designer) and the heat source footprints U of the training profiles alone. The model stores the settings and
the reduced model of the training board (for the model plots and the model size); the prediction (krSol) rebuilds
the reduction from the operator of the testing board, i.e. the Krylov model is exact per board but needs that
operator and one sparse factorisation for every new design.
Inputs:     1) data:    training data (X2: heat generation (W/m³) per profile, Data: geometry and operator)
            2) val:     validation data (not used)
            3) para:    all simulation parameters (Mdl.kr.q: moment blocks, Mdl.kr.hold, Mdl.pinf.srcTol)
            4) legacy:  not used
Outputs:    1) mdl:     model (q, hold, srcTol, U, r, tau, V, Cr, Gr, Br, tRed)

V1.0: 21.11.2026, Pascal Schirmer
- Krylov (moment matching) reduction of the board designer operator as an own mode
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.KR.calcKrylov import basis
from src.py.model.PF.pfData import encodeSrc, material

# ==============================================================================
# External
# ==============================================================================
import sys
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def krPara(para):
    """Settings of the KR mode with defaults: q (moment blocks), hold (0: zero-order, 1: linear), srcTol."""
    kr = dict(para.get('Mdl', {}).get('kr', {}) or {})
    pinf = para.get('Mdl', {}).get('pinf', {}) or {}
    return {'q': int(kr.get('q', 4) or 4), 'hold': int(kr.get('hold', 0) or 0),
            'srcTol': float(kr.get('srcTol', pinf.get('srcTol', 1e-6)) or 1e-6)}


def operator(data, mode='KR'):
    """Operator {'C', 'G', 'V'} of the data file, or exit with an explanation if the data file has none."""
    D = data['Data']
    d = np.asarray(D['geo']).shape[1]
    op = material(D, d)['op']
    if op is None:
        sys.exit("ERROR: %s mode needs the finite-volume operator of the board in the data file (Cn, Gi, Gj, Gv, Vn, "
                 "written by the board designer). Measured or FEM data without an operator cannot be reduced by "
                 "moment matching; use the PO/PS mode (snapshots) or the PF mode (physics 1) instead" % mode)
    return op


def reduce(data, q, srcTol):
    """Source footprints U (N x S), amplitudes p per profile (list of Nt x S) and the Krylov model of the data."""
    op = operator(data)
    [U, _] = encodeSrc(data['X2'], srcTol)
    if U.shape[1] == 0:
        sys.exit("ERROR: KR mode: the profiles have no heat sources")
    R = basis(op['C'], op['G'], op['V'][:, None] * U, q)
    p = [np.asarray(Q, dtype=float) @ U for Q in data['X2']]               # U orthonormal: p = Q·U
    return U, p, R


#######################################################################################################################
# Main Function
#######################################################################################################################
def krFit(data, val, para, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Reducing Krylov Model")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    kp = krPara(para)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    [U, _, R] = reduce(data, kp['q'], kp['srcTol'])

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    mdl = dict(kp, U=U, S=int(U.shape[1]), r=R['r'], tau=R['tau'], V=R['V'], Cr=R['C'], Gr=R['G'], Br=R['B'],
               tRed=R['tRed'], N=int(R['V'].shape[0]))
    print("INFO: %d nodes, %d sources, q = %d moment blocks -> r = %d states; slowest time constant %.3g s, "
          "fastest %.3g s; reduction %.3f s (one sparse LU of G, no temperature data)" %
          (mdl['N'], mdl['S'], kp['q'], R['r'], R['tau'][0], R['tau'][-1], R['tRed']))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Reducing Krylov Model")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
