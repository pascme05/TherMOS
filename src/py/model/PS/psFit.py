#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         psFit
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
This function fits the POD state-space model of a 2D or 3D temperature field. The POD modes φ are calculated from the
training snapshots (calcPod), the heat generation is projected onto the modes (q_i = ∫ Q·φ_i dΩ) and a discrete
state-space model with Ks states is identified between q and the modal coordinates a = Φᵀ·(T - T0):

    x(k+1) = A·x(k) + B·q(k),       a(k) = C·x(k),      T(k) = T0 + Φ·a(k)

The identification is subspace (fitN4sid) followed by an output-error fit (fitPem, MATLAB ssest with
DisturbanceModel 'none'). As in MATLAB, every profile is preceded by 2·Ks zero samples (zero initial state).
Inputs:     1) data:    training data (y: temperature at the nodes (°C), X: heat generation (W/m³), Data: geometry)
            2) val:     validation data (not used)
            3) para:    all simulation parameters
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
            5) th:      threshold of the relative element size (legacy only, 0.1 in 2D, 0.01 in 3D)
Outputs:    1) mdl:     model (rPhi, sPhi, theta, lam, E, K, Tavg, sys)

V1.0: 07.11.2026, Pascal Schirmer
- Python port of psFit.m and ps3Fit.m; projection without the factor α/k and the element size (F-P2, F-P8), modal
  coordinates per training profile (F-P14)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PO.calcPod import calcPod
from src.py.model.PO.calcPodQuad import calcPodQuad
from src.py.model.PO.calcPodProj import calcPodLoad
from src.py.numeric.fitN4sid import fitN4sid
from src.py.numeric.fitPem import fitPem

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Main Function
#######################################################################################################################
def psFit(data, val, para, legacy=0, th=0.1):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting Proper Orthogonal Model using SS")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    Ts = data['Ts']
    ss = para['Mdl']['ss']
    Ks = int(ss['K'])
    estX0 = 1 if ss['init'] == 2 else 0
    eps = max(para['Mdl']['gen']['eps'], 1e-10)

    # ==============================================================================
    # Variables
    # ==============================================================================
    T = np.asarray(data['y'], dtype=float)
    Q = np.asarray(data['X'], dtype=float)
    Nt = np.atleast_1d(data['Nt']) if legacy == 0 else np.array([T.shape[0]])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # POD
    # ==============================================================================
    [rPhi, lam, K, E, Tavg, theta, Tc] = calcPod(T, para, legacy)
    print('INFO: Number of used states (state space model) Ks: %i' % Ks)
    if np.linalg.norm(np.eye(K) - rPhi.T @ rPhi) > eps:
        print("ERROR: POD modes are not orthonormal")

    # ==============================================================================
    # Projection
    # ==============================================================================
    quad = calcPodQuad(data['Data'], legacy, 5, th)
    sPhi = (quad['Mg'] @ rPhi).reshape(quad['shape'] + (K,))
    phiQ = quad['Rq'] @ sPhi.reshape(-1, K)
    q = calcPodLoad(quad, phiQ, Q, legacy)

    # ==============================================================================
    # Identification Data
    # ==============================================================================
    uList, yList = [], []
    i0 = 0
    for n in Nt:
        s = slice(i0, i0 + int(n))
        a = theta[s] - theta[i0] if legacy == 0 else theta[s] - theta[0]
        uList.append(np.vstack([np.zeros((2 * Ks, K)), q[s]]))
        yList.append(np.vstack([np.zeros((2 * Ks, K)), a]))
        i0 = i0 + int(n)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    sys = fitN4sid(uList, yList, Ks, estX0)
    sys = fitPem(sys, uList, yList, estX0, ss['sta'], int(para['Mdl']['gen']['iterMax']))
    sys['Ts'] = Ts

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    mdl = {'rPhi': rPhi, 'sPhi': sPhi, 'theta': theta, 'lam': lam, 'E': E, 'K': K, 'Tavg': Tavg, 'sys': sys,
           'th': th}

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting Proper Orthogonal Model using SS")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
