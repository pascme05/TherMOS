#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         normData
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
This function normalises the input (X) and output (y) data. The method is selected with para['Par']['gen']['normX']
and para['Par']['gen']['normY']: (1) none, (2) 0/1, (3) mean/range, (4) mean/std. The normalisation values are
calculated from the training data (sel=1) and applied to testing/validation data (sel=2). With sel=3 the inverse
normalisation is applied. The reference temperature of the loss model (Tref) is normalised like the output.

    0/1:        x_n = (x - min) / (max - min)
    mean/range: x_n = (x - avg) / (max - min)
    mean/std:   x_n = (x - avg) / std

Inputs:     1) data:    data of one split
            2) para:    all simulation parameters
            3) sel:     (1) calculate and apply, (2) apply, (3) inverse
            4) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) data:    (de-)normalised data
            2) para:    parameters including the normalisation values para['Dat']['normVal']

V1.0: 07.11.2026, Pascal Schirmer
- Python port of normData.m
- Mean/range normalisation of the structured output uses the same formula as the concatenated output (F-D3)
- Constant channels are handled per channel (range or std of zero is set to one, F-D4)
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
# Additional Functions
#######################################################################################################################
def _stats(x):
    return {'min': np.min(x, axis=0), 'max': np.max(x, axis=0), 'avg': np.mean(x, axis=0),
            'std': np.std(x, axis=0, ddof=1) if x.shape[0] > 1 else np.ones(x.shape[1:])}


def _fwd(x, s, method):
    rng = s['max'] - s['min']
    if method == 2:
        return (x - s['min']) / rng
    elif method == 3:
        return (x - s['avg']) / rng
    elif method == 4:
        return (x - s['avg']) / s['std']
    return x


def _inv(x, s, method):
    rng = s['max'] - s['min']
    if method == 2:
        return x * rng + s['min']
    elif method == 3:
        return x * rng + s['avg']
    elif method == 4:
        return x * s['std'] + s['avg']
    return x


#######################################################################################################################
# Main Function
#######################################################################################################################
def normData(data, para, sel, legacy=0):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    normX = para['Par']['gen']['normX']
    normY = para['Par']['gen']['normY']
    names = {1: 'No', 2: '0/1', 3: 'Mean/Range', 4: 'Mean/Std'}
    data = dict(data)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Normalisation Values
    # ==============================================================================
    if sel == 1:
        sX = _stats(np.asarray(data['X'], dtype=float))
        sY = _stats(np.asarray(data['y'], dtype=float))
        sR = _stats(np.asarray(data['r'], dtype=float))
        for s in (sX, sY, sR):
            if legacy == 1:
                if np.all(s['max'] == s['min']):
                    s['min'] = np.zeros_like(s['min'])
            else:
                const = (s['max'] - s['min']) == 0
                s['max'] = np.where(const, s['min'] + 1, s['max'])
                s['std'] = np.where(s['std'] == 0, 1, s['std'])
        para.setdefault('Dat', {})['normVal'] = {'X': sX, 'y': sY, 'r': sR}
    else:
        sX = para['Dat']['normVal']['X']
        sY = para['Dat']['normVal']['y']

    fnc = _inv if sel == 3 else _fwd

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Input
    # ==============================================================================
    if normX in (2, 3, 4):
        data['X'] = fnc(data['X'], sX, normX)
        if 'X2' in data:
            data['X2'] = [fnc(x, sX, normX) for x in data['X2']]
    print("INFO: " + names.get(normX, 'Invalid') + " normalisation applied to input")

    # ==============================================================================
    # Output
    # ==============================================================================
    if normY in (2, 3, 4):
        data['y'] = fnc(data['y'], sY, normY)
        if 'y2' in data:
            data['y2'] = [fnc(y, sY, normY) for y in data['y2']]
    print("INFO: " + names.get(normY, 'Invalid') + " normalisation applied to output")

    # ==============================================================================
    # Temperature Reference
    # ==============================================================================
    if sel == 1:
        para['Par']['loss']['Tref'] = fnc(para['Par']['loss']['Tref'], sY, normY)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [data, para]


#######################################################################################################################
# References
#######################################################################################################################
