#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         performance
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
This function evaluates the prediction for every output channel (node). The samples are split into transient and
steady-state samples using the temperature change per sample dT: a sample is transient if |dT| > th·max|dT|, with
th = para['Par']['gen']['th'], otherwise it is steady-state.
Inputs:     1) dataTest:    testing data (true values)
            2) dataPred:    predicted data
            3) para:        all simulation parameters
            4) mdl:         model (training time and size)
            5) legacy:      (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) out:         error metrics out['err'][tot|tr|ss][metric] (one value per channel), times and size

V1.0: 07.11.2026, Pascal Schirmer
- Python port of performance.m, transient detection uses the absolute temperature change so that cooling is
  transient as well (legacy: signed change, F-E2)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.calcError import calcError

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Main Function
#######################################################################################################################
def performance(dataTest, dataPred, para, mdl, legacy=0):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    yTrue = np.atleast_2d(np.asarray(dataTest['y'], dtype=float).T).T
    yPred = np.atleast_2d(np.asarray(dataPred['y'], dtype=float).T).T
    N = yTrue.shape[1]
    th = para['Par']['gen']['th']
    keys = ['MAE', 'RMSE', 'SAE', 'NAE', 'MAX', 'FIT', 'NRMSE']
    out = {'err': {s: {k: np.zeros(N) for k in keys} for s in ('tot', 'tr', 'ss')}}

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    dT = np.vstack([np.zeros((1, N)), np.diff(yTrue, axis=0)])
    if legacy == 1:
        maskTr = dT > th * np.max(dT, axis=0)
        maskSs = dT < th * np.max(dT, axis=0)
    else:
        maskTr = np.abs(dT) > th * np.max(np.abs(dT), axis=0)
        maskSs = ~maskTr

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for i in range(0, N):
        for (s, m) in (('tot', np.ones(len(yTrue), dtype=bool)), ('tr', maskTr[:, i]), ('ss', maskSs[:, i])):
            err = calcError(yTrue[m, i], yPred[m, i])
            for k in keys:
                out['err'][s][k][i] = err[k]

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    out['time'] = {'train': mdl.get('timeTrain', 0) if mdl else 0, 'test': dataPred.get('testTime', 0)}
    out['mem'] = mdl.get('size', 0) if mdl else 0

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
