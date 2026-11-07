#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         main
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
This function runs a TherMOS simulation: loading of the parameters and data, pre-processing (loss update, offset
removal, dimension reduction, normalisation, padding), training and testing of all selected models and
post-processing (de-normalisation, offset correction, gradient/flux, performance, saving, printing and plotting).
Inputs:     1) setup:   includes all simulation variables
            2) path:    includes all path variables
Outputs:    1) out:     dictionary with data, pred, grt, results, mdls and para

V1.0: 07.11.2026, Pascal Schirmer
- Python port of main.m; several models can be selected at the same time, the identification data are rebuilt
  after normalisation and padding (legacy: MATLAB used the un-normalised data, F-M8), all selected models are kept
  (F-M2), setup['legacy'] selects the corrected (0) or the MATLAB (1) behaviour
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.loadPara import loadPara
from src.py.data.loadData import loadData
from src.py.data.updLoss import updLoss
from src.py.data.offRemove import offRemove
from src.py.data.offAdd import offAdd
from src.py.data.dimData import dimData
from src.py.data.normData import normData
from src.py.data.padData import padData
from src.py.data.concatData import calcIdData
from src.py.general.calcGrad import calcGrad
from src.py.general.calcGrad3D import calcGrad3D
from src.py.general.performance import performance
from src.py.general.saving import saving
from src.py.training import training
from src.py.testing import testing
from src.py.plotting.printConsole import printConsole
from src.py.plotting.plotMdl import plotMdl
from src.py.plotting.plotResults1D import plotResults1D
from src.py.plotting.plotResults2D import plotResults2D
from src.py.plotting.plotResults3D import plotResults3D

# ==============================================================================
# External
# ==============================================================================
import sys


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _banner(txt):
    print("=======================================================================")
    print(txt)
    print("=======================================================================")


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(setup, path):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Defaults
    # ==============================================================================
    setup.setdefault('legacy', 0)
    setup.setdefault('featRoll', 0)
    setup.setdefault('EWMA', [5, 50])
    setup.setdefault('EWMS', [5, 50])
    setup.setdefault('featRank', 0)
    for k in ('selRC', 'selSS', 'selSF', 'selPO', 'selPS', 'selML', 'selDL', 'selPI'):
        setup.setdefault(k, 0)
    lg = setup['legacy']

    # ==============================================================================
    # MSG IN
    # ==============================================================================
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")
    print("Welcome to the Thermal Model Order Reduction and Simulation Toolkit (TherMOS)!")
    print("Author:     Dr. Pascal Alexander Schirmer")
    print("Copyright:  Pascal Schirmer")
    print("Version:    v.1.0 (Python)")
    print("Date:       07.11.2026")
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")
    print("Setup File:   %s" % setup['name'])
    print("Config File:  %s" % setup['config'])
    print("Train Data:   %s" % str(setup.get('trFile')))
    print("Test Data:    %s" % str(setup.get('teFile')))
    print("Val Data:     %s" % str(setup.get('vlFile')))
    print("Models:       " + ', '.join(k[3:] for k in ('selRC', 'selSS', 'selSF', 'selPO', 'selPS', 'selML', 'selDL',
                                                       'selPI') if setup[k] != 0))
    print("Legacy:       %d (%s)" % (lg, 'MATLAB behaviour' if lg == 1 else 'corrected behaviour'))
    print("\n")

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    _banner("START: Loading Para and Data")
    try:
        para = loadPara(setup['config'], path)
    except (OSError, KeyError, ValueError) as e:
        sys.exit('ERROR: Configuration could not be loaded (' + str(e) + ')')
    data = loadData(para, setup, path)
    if data['tr']['Dim'] < 2 and (setup['selPO'] != 0 or setup['selPS'] != 0):
        sys.exit('ERROR: 1D data input and 2D/3D model')
    _banner("END: Loading Para and Data")
    print("\n")

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    _banner("START: Pre-Processing")

    # ==============================================================================
    # Losses
    # ==============================================================================
    if para['Par']['loss']['fw'] != 0 or para['Par']['loss']['bw'] != 0:
        [data, para] = updLoss(data, para, lg)
    else:
        print("INFO: Loss update deactivated")

    # ==============================================================================
    # Offset
    # ==============================================================================
    for s in ('tr', 'te', 'vl'):
        data[s] = offRemove(data[s], para, lg)

    # ==============================================================================
    # Dimension
    # ==============================================================================
    [data['tr'], _] = dimData(data['tr'], setup, para)
    [data['vl'], _] = dimData(data['vl'], setup, para)
    [data['te'], setup] = dimData(data['te'], setup, para)

    # ==============================================================================
    # Normalisation
    # ==============================================================================
    norm = para['Par']['gen']['normX'] != 1 or para['Par']['gen']['normY'] != 1
    if norm:
        [data['tr'], para] = normData(data['tr'], para, 1, lg)
        [data['te'], _] = normData(data['te'], para, 2, lg)
        [data['vl'], _] = normData(data['vl'], para, 2, lg)
    else:
        para['Dat'].setdefault('normVal', {})['y'] = {'max': 1}
        print("INFO: Data normalisation deactivated")

    # ==============================================================================
    # Padding
    # ==============================================================================
    if para['Exp']['gen']['pad'] != 0:
        data['tr'] = padData(data['tr'], para, lg)
        data['vl'] = padData(data['vl'], para, lg)
    else:
        print("INFO: Padding deactivated")

    # ==============================================================================
    # Identification Data
    # ==============================================================================
    if lg == 0:
        for s in ('tr', 'te', 'vl'):
            d = data[s]
            d['idData'] = calcIdData(d['X2'], d['y2'], d['Ts'], d.get('Data'), setup['datDim'], para, lg)
    _banner("END: Pre-Processing")
    print("\n")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    _banner("START: Calculation")
    mdls = training(data, setup, para, path) if setup.get('train', 1) == 1 else {}
    [pred, grt, mdls] = testing(mdls, data, setup, para, path)
    if len(pred) == 0:
        sys.exit('ERROR: No model was selected or could be loaded')
    _banner("END: Calculation")
    print("\n")

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    _banner("START: Post-Processing")

    # ==============================================================================
    # De-Normalisation
    # ==============================================================================
    if norm:
        for s in ('tr', 'te', 'vl'):
            [data[s], _] = normData(data[s], para, 3, lg)
        for tag in pred:
            [pred[tag], _] = normData(pred[tag], para, 3, lg)
            [grt[tag], _] = normData(grt[tag], para, 3, lg)

    # ==============================================================================
    # Offset
    # ==============================================================================
    if para['Exp']['gen']['init'] != 1:
        for s in ('tr', 'te', 'vl'):
            data[s] = offAdd(data[s], para)
        for tag in pred:
            pred[tag] = offAdd(pred[tag], para)
            grt[tag] = offAdd(grt[tag], para)

    # ==============================================================================
    # Gradient and Flux
    # ==============================================================================
    dOut = para['Dat']['gen']['dOut']
    for tag in pred:
        if dOut == 2 and setup['datDim'] == 2:
            grt[tag] = calcGrad(grt[tag], lg)
            pred[tag] = calcGrad(pred[tag], lg)
        elif dOut == 3 and setup['datDim'] == 3:
            grt[tag] = calcGrad3D(grt[tag], -1, lg)
            pred[tag] = calcGrad3D(pred[tag], -1, lg)

    # ==============================================================================
    # Performance
    # ==============================================================================
    results = {tag: performance(grt[tag], pred[tag], para, mdls.get(tag), lg) for tag in pred}
    _banner("END: Post-Processing")
    print("\n")

    ###################################################################################################################
    # Output
    ###################################################################################################################
    # ==============================================================================
    # Saving
    # ==============================================================================
    if setup.get('save', 0) == 1:
        saving({'data': data, 'pred': pred, 'grt': grt, 'results': results}, setup, para, path)
    else:
        print("INFO: Saving deactivated")

    # ==============================================================================
    # Printing
    # ==============================================================================
    for tag in pred:
        printConsole(pred[tag], results[tag], setup, mdls.get(tag), para, tag.upper())

    # ==============================================================================
    # Plotting
    # ==============================================================================
    if setup.get('plotMdl', 0) == 1:
        plotMdl(data, mdls, setup)
    if setup.get('plotOut', 0) != 0:
        if setup['datDim'] == 1:
            plotResults1D(data, pred, grt, results, setup)
        elif setup['datDim'] == 2:
            plotResults2D(data, pred, grt, results, setup, para)
        else:
            plotResults3D(data, pred, grt, results, setup, para)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("----------------------------------------------------------------------------------------------------------")
    print("Simulation Completed")
    print("----------------------------------------------------------------------------------------------------------")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'data': data, 'pred': pred, 'grt': grt, 'results': results, 'mdls': mdls, 'para': para}


#######################################################################################################################
# References
#######################################################################################################################
