#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcError
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
This function calculates the error metrics between the true and the predicted temperature of one channel. With the
error e = y - ŷ and N samples:

    MAE   = Σ|e| / N                                    mean absolute error (K)
    RMSE  = sqrt(Σe² / N)                               root mean square error (K)
    SAE   = (mean|y| - mean|ŷ|) / mean|y|               relative error of the mean absolute value (p.u.)
    NAE   = Σ|e| / Σ|y|                                 normalised absolute error (p.u.)
    MAX   = max|e|                                      maximum absolute error (K)
    FIT   = 100·(1 - ‖e‖ / ‖y - mean(y)‖)               fit percentage as in MATLAB compare (%)
    NRMSE = 100·RMSE / (max(y) - min(y))                root mean square error normalised to the range (%)

Inputs:     1) yTrue:   true temperature (Nt)
            2) yPred:   predicted temperature (Nt)
Outputs:    1) err:     dictionary with MAE, RMSE, SAE, NAE, MAX, FIT and NRMSE

V1.0: 07.11.2026, Pascal Schirmer
- Python port of calcerror.m. The MATLAB metric 'NRSME' is the fit percentage and is called FIT here. The real
  NRMSE is added. Undefined metrics (zero denominator, no samples) are NaN instead of arbitrary values (F-E1)
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
# Main Function
#######################################################################################################################
def calcError(yTrue, yPred):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    yTrue = np.asarray(yTrue, dtype=float).ravel()
    yPred = np.asarray(yPred, dtype=float).ravel()
    keys = ['MAE', 'RMSE', 'SAE', 'NAE', 'MAX', 'FIT', 'NRMSE']

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    if len(yTrue) == 0:
        return {k: np.nan for k in keys}
    e = yTrue - yPred

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    with np.errstate(divide='ignore', invalid='ignore'):
        rng = np.max(yTrue) - np.min(yTrue)
        cen = np.linalg.norm(yTrue - np.mean(yTrue))
        err = {
            'MAE': np.mean(np.abs(e)),
            'RMSE': np.sqrt(np.mean(e ** 2)),
            'SAE': (np.mean(np.abs(yTrue)) - np.mean(np.abs(yPred))) / np.mean(np.abs(yTrue)),
            'NAE': np.sum(np.abs(e)) / np.sum(np.abs(yTrue)),
            'MAX': np.max(np.abs(e)),
            'FIT': 100 * (1 - np.linalg.norm(e) / cen) if cen > 0 else np.nan,
            'NRMSE': 100 * np.sqrt(np.mean(e ** 2)) / rng if rng > 0 else np.nan,
        }

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {k: float(v) for k, v in err.items()}


#######################################################################################################################
# References
#######################################################################################################################
