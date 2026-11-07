#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         loadData
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
This function loads the training, testing and validation data from data/. Each split consists of one or more
profiles (files, sheets or IDs). Every profile is resampled to the target sampling period and afterwards all profiles
are concatenated (concatData).
Inputs:     1) para:    all simulation parameters
            2) setup:   includes all simulation variables
            3) path:    includes all path variables
Outputs:    1) data:    dictionary with the splits 'tr', 'te' and 'vl'

V1.0: 07.11.2026, Pascal Schirmer
- Python port of loadData.m, one loading routine for all three splits (F-D13)
- A warning is given if the profiles have different sampling periods
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.data.DataLoader import DataLoader
from src.py.data.resmpData import resmpData
from src.py.data.concatData import concatData

# ==============================================================================
# External
# ==============================================================================


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _toList(val):
    if isinstance(val, str):
        return [val]
    return list(val)


def _loadSplit(loader, files, sheets, stat, para, setup):
    # Selection of the profiles
    if setup['datSep'] == 1:
        items = [(f, sheets[0]) for f in files]
    elif setup['datSep'] == 2:
        items = [(files[0], s) for s in sheets]
    else:
        items = [(files[0], sheets[0])]

    # Loading and resampling
    out = None
    for (f, s) in items:
        temp = loader.loadData(f, s, stat, setup)
        [xOut, yOut, rOut, tOut] = resmpData(temp, para, setup['legacy'])
        if out is None:
            out = {k: v for k, v in temp.items() if k not in ('X', 'y', 'r', 't')}
            out.update({'X': [], 'y': [], 'r': [], 't': []})
        out['X'].append(xOut)
        out['y'].append(yOut)
        out['r'].append(rOut)
        out['t'].append(tOut)

    # Sampling time
    TsList = [t[1] - t[0] for t in out['t'] if len(t) > 1]
    if len(TsList) > 0 and (max(TsList) - min(TsList)) > 1e-6 * max(TsList):
        print('WARN: Profiles have different sampling periods ' + str(TsList))
    out['Ts'] = TsList[-1] if len(TsList) > 0 else para['Exp']['gen']['Ts']

    return out


#######################################################################################################################
# Main Function
#######################################################################################################################
def loadData(para, setup, path):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    loader = DataLoader(para['Exp']['gen']['Ts'], setup['format'], setup['datDim'], setup['datSep'],
                        para['Exp']['gen']['samp'], setup['inp'], setup['out'], path, setup['legacy'])
    data = {}
    names = {'tr': ('Training', 'trFile', 'trSheet', 1),
             'te': ('Testing', 'teFile', 'teSheet', 2),
             'vl': ('Validation', 'vlFile', 'vlSheet', 3)}

    ###################################################################################################################
    # Loading Data
    ###################################################################################################################
    for key, (txt, fKey, sKey, stat) in names.items():
        print("------------------------------------------")
        print("Loading " + txt + " Data")
        print("------------------------------------------")
        data[key] = _loadSplit(loader, _toList(setup[fKey]), _toList(setup[sKey]), stat, para, setup)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    print("------------------------------------------")
    print("Concatenating Data")
    print("------------------------------------------")
    for key in names:
        data[key] = concatData(data[key], setup, para)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Loading input data")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return data


#######################################################################################################################
# References
#######################################################################################################################
