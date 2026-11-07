#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         resmpData
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
This function resamples the temporal axis of the data to the target sampling period Ts of the configuration. The
interpolation method is selected with para['Exp']['gen']['samp']: (1) zero-order hold, (2) linear, (3) spline.
Inputs:     1) data:    loaded data (t, X, y, r)
            2) para:    all simulation parameters
            3) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) xOut:    resampled input data (Nt x ...)
            2) yOut:    resampled output data (Nt x ...)
            3) rOut:    resampled reference data (Nt x ...)
            4) tRef:    resampled time vector starting at 0 (sec)

V1.0: 07.11.2026, Pascal Schirmer
- Python port of resmpData.m, all spatial dimensions are resampled at once
- The interpolation option is used (MATLAB always used linear interpolation, F-D5), the last sample is kept and the
  time axis is shifted to start at 0 (legacy: linear, last sample dropped)
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
from scipy.interpolate import interp1d, CubicSpline


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _interp(tDat, val, tRef, kind):
    shape = val.shape
    flat = val.reshape(shape[0], -1)
    if kind == 'spline':
        res = CubicSpline(tDat, flat, axis=0, extrapolate=True)(tRef)
    else:
        res = interp1d(tDat, flat, kind=kind, axis=0, bounds_error=False, fill_value=(flat[0], flat[-1]),
                       assume_sorted=True)(tRef)
    return res.reshape((len(tRef),) + shape[1:])


#######################################################################################################################
# Main Function
#######################################################################################################################
def resmpData(data, para, legacy=0):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    Ts = para['Exp']['gen']['Ts']
    samp = para['Exp']['gen']['samp']

    # ==============================================================================
    # Variables
    # ==============================================================================
    tDat = np.asarray(data['t'], dtype=float).ravel()
    Nt = data['y'].shape[0]
    tDat = tDat[0:Nt]
    TsDat = np.mean(np.diff(tDat))

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Sampling Check
    # ==============================================================================
    if abs(TsDat - Ts) / Ts < 1e-3:
        print('INFO: Data sampling period equals target: %.2f sec' % Ts)
        tRef = np.arange(Nt) * Ts
        return [np.asarray(data['X']), np.asarray(data['y']), np.asarray(data['r']), tRef]
    print('INFO: Data sampling period: %.2f sec is not equal to target: %.2f sec' % (TsDat, Ts))

    # ==============================================================================
    # Interpolation Method
    # ==============================================================================
    if legacy == 1:
        kind = 'linear'
        print("INFO: Using linear interpolation (legacy)")
    elif samp == 1:
        kind = 'previous'
        print("INFO: Using ZoH for interpolation")
    elif samp == 2:
        kind = 'linear'
        print("INFO: Using linear interpolation")
    elif samp == 3:
        kind = 'spline'
        print("INFO: Using spline interpolation")
    else:
        kind = 'previous'
        print("WARN: Invalid interpolation method, ZOH is used")

    # ==============================================================================
    # Time Vector
    # ==============================================================================
    tDat = tDat - tDat[0]
    Tend = tDat[-1]
    if legacy == 1:
        tRef = np.arange(0, Tend - Ts + Ts * 1e-9, Ts)
    else:
        tRef = np.arange(0, Tend + Ts * 1e-9, Ts)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    yOut = _interp(tDat, np.asarray(data['y'], dtype=float), tRef, kind)
    xOut = _interp(tDat, np.asarray(data['X'], dtype=float), tRef, kind)
    rOut = _interp(tDat, np.asarray(data['r'], dtype=float), tRef, kind)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [xOut, yOut, rOut, tRef]


#######################################################################################################################
# References
#######################################################################################################################
