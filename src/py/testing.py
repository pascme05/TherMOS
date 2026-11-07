#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         testing
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
This function predicts the testing data with all selected models. Models that were not trained in this run are
loaded from mdl/mdl_<tag>_<name>.pkl (.pt). For single-output models (RC, ML) the prediction and the ground truth are
reduced to the first channel.
Inputs:     1) mdls:    dictionary of the trained models {tag: mdl} (empty if not trained)
            2) data:    training, validation and testing data
            3) setup:   includes all simulation variables
            4) para:    all simulation parameters
            5) path:    includes all path variables
Outputs:    1) pred:    dictionary of the predictions {tag: data}
            2) grt:     dictionary of the ground truth {tag: data}
            3) mdls:    dictionary of the models {tag: mdl}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of testing.m; one prediction per selected model
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.mdlReg import mdlReg

# ==============================================================================
# External
# ==============================================================================
from os.path import join as pjoin
import pickle
import time
import numpy as np
import torch


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _first(d):
    out = dict(d)
    for k in ('y', 'X', 'r', 'off'):
        if k in out and np.ndim(out[k]) == 2:
            out[k] = np.asarray(out[k])[:, 0:1]
    for k in ('y2', 'X2', 'r2', 'off2'):
        if k in out and isinstance(out[k], list):
            out[k] = [np.atleast_2d(np.asarray(v).T).T[:, 0:1] for v in out[k]]
    return out


#######################################################################################################################
# Main Function
#######################################################################################################################
def testing(mdls, data, setup, para, path):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    pred, grt = {}, {}

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for m in mdlReg(setup):
        # ==============================================================================
        # Loading
        # ==============================================================================
        if m['tag'] not in mdls:
            name = pjoin(path['mdl'], 'mdl_' + m['tag'] + '_' + setup['name'])
            try:
                if m['tag'] in ('dl', 'pi'):
                    mdls[m['tag']] = torch.load(name + '.pt', weights_only=False)
                else:
                    with open(name + '.pkl', 'rb') as f:
                        mdls[m['tag']] = pickle.load(f)
                print("INFO: Model loaded successfully")
            except (OSError, pickle.UnpicklingError) as e:
                print("WARN: Failed to load the model " + name + " (" + str(e) + ")")
                continue

        # ==============================================================================
        # Prediction
        # ==============================================================================
        tStart = time.time()
        out = m['sol'](mdls[m['tag']], data['te'], para)
        out['testTime'] = time.time() - tStart
        out.pop('y2', None)

        # ==============================================================================
        # Output
        # ==============================================================================
        pred[m['tag']] = _first(out) if m['siso'] else out
        grt[m['tag']] = _first(data['te']) if m['siso'] else dict(data['te'])

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [pred, grt, mdls]


#######################################################################################################################
# References
#######################################################################################################################
