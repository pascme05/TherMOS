#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         ssFit
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
This function identifies a discrete-time state-space model of the temperature response (MIMO, all inputs and outputs)

    x(k+1) = A·x(k) + B·u(k)
    y(k)   = C·x(k)

with the losses u, the temperatures y and n = para['Mdl']['ss']['K'] states (or 1..Kmax if opt = 1). The option
para['Mdl']['ss']['dis'] selects the method: (1) subspace identification (MATLAB n4sid, here fitN4sid), (2) subspace
identification refined by an output-error fit (MATLAB ssest, here fitPem). For dis = 2 the continuous matrices
(Ac = logm(A)/Ts, Bc) are added for information; the simulation always uses the discrete model.
Inputs:     1) data:    training data (idData)
            2) val:     validation data (not used)
            3) para:    all simulation parameters
Outputs:    1) mdl:     model {'sys': {A, B, C, D, Ts, ...}}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of ssFit.m; noise models (DisturbanceModel 'estimate') are not supported, the output-error model is used
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.numeric.fitN4sid import fitN4sid
from src.py.numeric.fitPem import fitPem

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from scipy.linalg import logm


#######################################################################################################################
# Main Function
#######################################################################################################################
def ssFit(data, val, para):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting State Space Model")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    ss = para['Mdl']['ss']
    K = int(ss['K'])
    Kmax = int(para['Mdl']['gen']['Kmax'])
    Ts = data['Ts']
    estX0 = 1 if ss['init'] == 2 else 0
    uList = data['idData']['u']
    yList = data['idData']['y']

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    if ss['noise'] == 2:
        print("WARN: Noise model not supported, output-error model is used")
    order = list(range(1, Kmax + 1)) if para['Mdl']['gen']['opt'] == 1 else K

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    sys = fitN4sid(uList, yList, order, estX0)
    if ss['dis'] != 1:
        sys = fitPem(sys, uList, yList, estX0, ss['sta'], int(para['Mdl']['gen']['iterMax']))

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    sys['Ts'] = Ts
    sys['dis'] = ss['dis']
    if ss['dis'] == 2:
        try:
            Ac = np.real(logm(sys['A'])) / Ts
            sys['Ac'] = Ac
            sys['Bc'] = np.linalg.solve(sys['A'] - np.eye(len(Ac)), Ac @ sys['B'])
        except (np.linalg.LinAlgError, ValueError):
            print("WARN: Continuous conversion not possible")
    print("INFO: State-Space Parameters A, B, C")
    print(np.array2string(sys['A'], precision=4))
    print(np.array2string(sys['B'], precision=4))
    print(np.array2string(sys['C'], precision=4))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting State Space Model")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'sys': sys}


#######################################################################################################################
# References
#######################################################################################################################
