#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         printConsole
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
This function prints the error metrics of one model per output channel (1D) or for the selected node (2D/3D, node
closest to para['Dat']['gen']['inpX'/'inpY'/'inpZ']) and the average over all channels/nodes. The metrics are split
into all samples (Total), steady-state (Steady) and transient (Trans) samples (see performance):
MAE: mean absolute error (K), MSE: mean squared error (K²), MAX: maximum absolute error (K), FIT: fit percentage
100·(1 - ‖e‖/‖y - mean(y)‖) (%, called NRMSE in the MATLAB version), NRMSE: RMSE normalised to the range of y (%).
Inputs:     1) pred:    prediction of the model
            2) results: performance results of the model
            3) setup:   includes all simulation variables
            4) mdl:     model (timeTrain, size)
            5) para:    all simulation parameters
            6) name:    name of the model
Outputs:    None

V1.0: 07.11.2026, Pascal Schirmer
- Python port of printConsole.m; the fit percentage is labelled FIT and the real NRMSE is added (F-E1)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.findNode import findNode

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Main Function
#######################################################################################################################
def printConsole(pred, results, setup, mdl, para, name=''):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dim = setup['datDim']
    Nt, nXY = np.shape(pred['y'])
    err = results['err']
    keys = [('MAE', 1), ('RMSE', 2), ('MAX', 1), ('FIT', 1), ('NRMSE', 1)]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    if dim >= 2 and nXY > 1:
        g = para['Dat']['gen']
        pos = [g['inpX'], g['inpY']] + ([g['inpZ']] if dim == 3 else [])
        idx = [findNode(pred['Data']['geo'], pos, setup['legacy'])]
        names = ['Node %d' % idx[0]]
    else:
        idx = list(range(0, nXY))
        out = list(setup.get('out', []))
        names = [out[i] if i < len(out) else 'T%d' % (i + 1) for i in idx]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    def row(label, vals):
        return '| %-9s|' % label[0:9] + ''.join(' %7.2f %7.2f %7.2f |' % tuple(v) for v in vals)

    def vals(sel):
        out = []
        for (k, p) in keys:
            v = [np.asarray(err[s][k]) ** p for s in ('tot', 'ss', 'tr')]
            if sel is None:
                out.append([np.nanmax(x) if k == 'MAX' else np.nanmean(x) for x in v])
            else:
                out.append([x[sel] for x in v])
        return out

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    line = '=' * 131
    print(line)
    print("INFO: Results of the " + name + " model")
    print(line)
    print('|          |' + ''.join(' %-24s|' % h.center(24) for h in ('MAE (K)', 'MSE (K²)', 'MAX (K)', 'FIT (%)',
                                                                       'NRMSE (%)')))
    print('|   Node   |' + '   Total  Steady   Trans |' * 5)
    print('-' * 131)
    for (i, n) in zip(idx, names):
        print(row(n, vals(i)))
    print(row('Avg', vals(None)))
    print(line)
    print("INFO: Training time (sec): %5.2f" % (mdl.get('timeTrain', 0) if mdl else 0))
    print("INFO: Inference time (µs/sample): %5.2f" % (pred.get('testTime', 0) * 1e6 / Nt / nXY))
    print("INFO: Inference time (ms/time-step): %5.2f" % (pred.get('testTime', 0) * 1e3 / Nt))
    if 'odeTime' in pred:
        print("INFO: Inference time ode (ms/time-step): %5.2f" % (pred['odeTime'] * 1e3 / Nt))
    print("INFO: Number Model Parameters: %d" % (mdl.get('size', 0) if mdl else 0))


#######################################################################################################################
# References
#######################################################################################################################
