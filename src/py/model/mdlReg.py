#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         mdlReg
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
This function returns the registry of all models: the setup switch that activates the model, the fit and the solve
function, the number of outputs (one channel for RC and ML) and the model size (number of parameters). training and
testing loop over this registry, so that several models can be selected at the same time.
Inputs:     1) setup:   includes all simulation variables
Outputs:    1) reg:     list of dictionaries (tag, name, fit, sol, siso, size) of the selected models
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.RC.rcFit import rcFit
from src.py.model.RC.rcSol import rcSol
from src.py.model.SS.ssFit import ssFit
from src.py.model.SS.ssSol import ssSol
from src.py.model.SF.sfFit import sfFit
from src.py.model.SF.sfSol import sfSol
from src.py.model.PO.poFit import poFit
from src.py.model.PO.poSol import poSol
from src.py.model.PS.psFit import psFit
from src.py.model.PS.psSol import psSol
from src.py.model.PS.ps3Fit import ps3Fit
from src.py.model.PS.ps3Sol import ps3Sol
from src.py.model.ML.mlFit import mlFit
from src.py.model.ML.mlSol import mlSol
from src.py.model.DL.dlFit import dlFit
from src.py.model.DL.dlSol import dlSol
from src.py.model.PINN.pinnFit import pinnFit
from src.py.model.PINN.pinnSol import pinnSol

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _nNet(m):
    return int(sum(p.numel() for p in m['sys'].parameters()))


def _nSys(s):
    return int(sum(np.size(s[k]) for k in ('A', 'B', 'C', 'D') if k in s))


#######################################################################################################################
# Main Function
#######################################################################################################################
def mdlReg(setup):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    lg = setup['legacy']
    reg = []

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    if setup.get('selRC', 0) == 1:
        reg.append({'tag': 'rc', 'name': 'Foster RC', 'siso': 1,
                    'fit': lambda d, v, p: rcFit(d, v, p, lg), 'sol': lambda m, d, p: rcSol(m, d, p, lg),
                    'size': lambda m: int(np.size(m['Rth']) + np.size(m['Cth']))})
    if setup.get('selSS', 0) == 1:
        reg.append({'tag': 'ss', 'name': 'State Space', 'siso': 0,
                    'fit': lambda d, v, p: ssFit(d, v, p), 'sol': lambda m, d, p: ssSol(m, d, p, lg),
                    'size': lambda m: _nSys(m['sys'])})
    if setup.get('selSF', 0) != 0:
        reg.append({'tag': 'sf', 'name': 'Nodal thermal network', 'siso': 0,
                    'fit': lambda d, v, p: sfFit(d, v, p, lg, int(setup['selSF'])),
                    'sol': lambda m, d, p: sfSol(m, d, p, lg),
                    'size': lambda m: int(np.size(m['p']) if 'p' in m else np.size(m['Rth']) + np.size(np.diag(m['Cth'])))})
    if setup.get('selPO', 0) == 1:
        reg.append({'tag': 'po', 'name': 'POD Galerkin', 'siso': 0,
                    'fit': lambda d, v, p: poFit(d, v, p, lg), 'sol': lambda m, d, p: poSol(m, d, p, lg),
                    'size': lambda m: int(np.size(m['rPhi']) + np.size(m['Cth']) + np.size(m['Gth']))})
    if setup.get('selPS', 0) >= 1:
        f3 = setup['selPS'] != 1
        reg.append({'tag': 'ps', 'name': 'POD State Space', 'siso': 0,
                    'fit': lambda d, v, p: (ps3Fit if f3 else psFit)(d, v, p, lg),
                    'sol': lambda m, d, p: (ps3Sol if f3 else psSol)(m, d, p, lg),
                    'size': lambda m: int(np.size(m['rPhi']) + _nSys(m['sys']))})
    if setup.get('selML', 0) != 0:
        reg.append({'tag': 'ml', 'name': 'Machine Learning', 'siso': 1,
                    'fit': lambda d, v, p: mlFit(d, v, p, setup, lg), 'sol': lambda m, d, p: mlSol(m, d, p, setup, lg),
                    'size': lambda m: 1})
    if setup.get('selDL', 0) != 0:
        reg.append({'tag': 'dl', 'name': 'Deep Learning', 'siso': 0,
                    'fit': lambda d, v, p: dlFit(d, v, p, setup, lg), 'sol': lambda m, d, p: dlSol(m, d, p, setup, lg),
                    'size': _nNet})
    if setup.get('selPI', 0) != 0:
        reg.append({'tag': 'pi', 'name': 'Physics-Informed NN', 'siso': 0,
                    'fit': lambda d, v, p: pinnFit(d, v, p, setup, lg),
                    'sol': lambda m, d, p: pinnSol(m, d, p, setup, lg), 'size': _nNet})

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return reg


#######################################################################################################################
# References
#######################################################################################################################
