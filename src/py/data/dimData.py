#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         dimData
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
This function adapts the dimension of the data to the dimension used by the model (para['Dat']['gen']['dOut']).
Possible reductions are 3D -> 1D and 2D -> 1D by selecting the input node (inpX/Y/Z) and output node (outX/Y/Z).
For the reduction the volumetric heat generation (W/m³) is multiplied with the cell size, i.e. dx·dy for 2D (W/m) and
dx·dy·dz for 3D (W).
Inputs:     1) data:    data of one split
            2) setup:   includes all simulation variables
            3) para:    all simulation parameters
Outputs:    1) out:     data with adapted dimension
            2) setup:   updated setup (datDim)

V1.0: 07.11.2026, Pascal Schirmer
- Python port of dimData.m
- idData is built from the reduced data (F-D8), 3D -> 1D uses dx·dy·dz (legacy: no scaling), 3D -> 2D is not
  implemented and stops with an error instead of silently returning the 3D data
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.findNode import findNode
from src.py.data.concatData import calcIdData

# ==============================================================================
# External
# ==============================================================================
import sys


#######################################################################################################################
# Main Function
#######################################################################################################################
def dimData(data, setup, para):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    dim = data['Dim']
    dOut = para['Dat']['gen']['dOut']
    gen = para['Dat']['gen']
    legacy = setup['legacy']
    setup = dict(setup)

    # ==============================================================================
    # Variables
    # ==============================================================================
    out = dict(data)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    if dim != setup['datDim']:
        print("WARN: Expected and actual data dimension is different")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Reduction 3D -> 2D
    # ==============================================================================
    if setup['datDim'] == 3 and dOut == 2:
        sys.exit("ERROR: Reduction 3D -> 2D is not implemented")

    # ==============================================================================
    # Reduction to 1D
    # ==============================================================================
    elif setup['datDim'] in (2, 3) and dOut == 1:
        # Positions
        D = data['Data']
        if setup['datDim'] == 2:
            idInp = findNode(D['geo'], [gen['inpX'], gen['inpY']], legacy)
            idOut = findNode(D['geo'], [gen['outX'], gen['outY']], legacy)
            scale = D['dx'] * D['dy']
        else:
            idInp = findNode(D['geo'], [gen['inpX'], gen['inpY'], gen['inpZ']], legacy)
            idOut = findNode(D['geo'], [gen['outX'], gen['outY'], gen['outZ']], legacy)
            scale = 1 if legacy == 1 else D['dx'] * D['dy'] * D['dz']

        # Unstructured
        out['X'] = data['X'][:, [idInp]] * scale
        out['y'] = data['y'][:, [idOut]]
        out['r'] = data['r'][:, [idOut]]
        out['off'] = data['off'][:, [idOut]]

        # Structured
        out['X2'] = [x[:, [idInp]] * scale for x in data['X2']]
        out['y2'] = [y[:, [idOut]] for y in data['y2']]
        out['r2'] = [r[:, [idOut]] for r in data['r2']]
        out['off2'] = [o[:, [idOut]] for o in data['off2']]

        # Dimension
        print("INFO: Data reduced from " + str(setup['datDim']) + "D -> 1D")
        setup['datDim'] = 1
        out['Dim'] = 1

    # ==============================================================================
    # Matching
    # ==============================================================================
    elif setup['datDim'] == dOut:
        print("INFO: Data dimension are matching")

    # ==============================================================================
    # Invalid
    # ==============================================================================
    else:
        sys.exit("ERROR: Reduction invalid, check data dimensions (datDim=" + str(setup['datDim']) + ", dOut=" +
                 str(dOut) + ")")

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    out['idData'] = calcIdData(out['X2'], out['y2'], out['Ts'], out.get('Data'), out['Dim'], para, legacy)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [out, setup]


#######################################################################################################################
# References
#######################################################################################################################
