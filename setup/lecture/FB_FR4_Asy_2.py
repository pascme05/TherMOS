#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         FB_FR4_Asy_2
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
This script starts a TherMOS simulation with the configuration defined below (Python version of FB_FR4_Asy_2.m).
The corrected numerics are used by default; setup['legacy'] = 1 reproduces the behaviour of the MATLAB version.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Path
# ==============================================================================
from os.path import dirname, abspath, isdir, join as pjoin
import sys

base = dirname(abspath(__file__))
while not isdir(pjoin(base, 'src', 'py')) and dirname(base) != base:
    base = dirname(base)
sys.path.insert(0, base)

# ==============================================================================
# Internal
# ==============================================================================
from src.py.main import main
from src.py.general.initPath import initPath

# ==============================================================================
# External
# ==============================================================================
import warnings

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Paths
#######################################################################################################################
path = initPath(base)

#######################################################################################################################
# Init
#######################################################################################################################
setup = {}

#######################################################################################################################
# Configuration
#######################################################################################################################

# ==============================================================================
# General
# ==============================================================================
setup['user'] = 'Pascal Schirmer'                                                                                        # name of the user (string)
setup['name'] = 'FR4'                                                                                                    # name of the simulation (string)

# ==============================================================================
# Experiment
# ==============================================================================

# ------------------------------------------
# Configuration
# ------------------------------------------
setup['config'] = '/lecture/lecture_TUM_Asy'                                                                             # name of the config file (string)

# ------------------------------------------
# Plotting and Saving
# ------------------------------------------
setup['plotOut'] = 1                                                                                                     # (0): model output is not plotted, (1): model output is plotted, (2): plotting one figure per output
setup['plotMdl'] = 0                                                                                                     # (0): model structure is not plotted, (1): model structure is plotted
setup['featRank'] = 0                                                                                                    # (0): No feature ranking, (1): feature ranking
setup['save'] = 0                                                                                                        # (0): output data not saved, (1): output data saved

# ------------------------------------------
# Training and Testing
# ------------------------------------------
setup['train'] = 1                                                                                                       # (0): no model training, (1): model training
setup['test'] = 1                                                                                                        # (0): no model testing, (1): model testing

# ==============================================================================
# Data
# ==============================================================================

# ------------------------------------------
# Format and Dimension
# ------------------------------------------
setup['format'] = 'mat'                                                                                                  # input data format: "xlsx" or "mat"
setup['datDim'] = 3                                                                                                      # input data dimension: (1): 1D data, (2): 2D data, (3): 3D data
setup['datSep'] = 1                                                                                                      # input data separation: (1): File based, (2): Sheet based, (3): ID based, (4): Ratio based

# ------------------------------------------
# Training and Testing Selection
# ------------------------------------------
# File based
setup['trFile'] = ['FB_FR4_10W_Asy_20per', 'FB_FR4_10W_Asy_30per']                                                       # list of training files (strings) at least one
setup['teFile'] = 'FB_FR4_20W_Asy_10per'                                                                                 # testing file (string) exactly one
setup['vlFile'] = ['FB_FR4_20W_Asy_10per']                                                                               # list of validation files (strings) at least one

# Sheet based
setup['trSheet'] = ['train-1', 'train-2', 'train-3']                                                                     # list of training sheets (strings) at least one
setup['teSheet'] = 'test'                                                                                                # testing sheet (string) exactly one
setup['vlSheet'] = ['val']                                                                                               # list of validation sheets (strings) at least one

# ID based
setup['trID'] = []                                                                                                       # list of training IDs (if empty all remaining ones are used)
setup['teID'] = []                                                                                                       # list of testing IDs
setup['vlID'] = []                                                                                                       # list of validation IDs

# Ratio based
setup['rTr'] = 0.7                                                                                                       # percentage of training data (testing data is 1 - rTr)
setup['rVl'] = 0.2                                                                                                       # percentage of validation data (this is taken from the training data)

# ------------------------------------------
# Inputs and Output Mapping
# ------------------------------------------
setup['inp'] = ['P1', 'P2', 'P3']                                                                                        # list of input feature (strings)
setup['out'] = ['T1', 'T2', 'T3']                                                                                        # list of temperature outputs (strings)
setup['ref'] = ['R1', 'R2', 'R3']                                                                                        # list of reference temperatures (strings)

# ==============================================================================
# Model
# ==============================================================================

# ------------------------------------------
# Analytical 1D Models
# ------------------------------------------
# Single-Input Single-Output (SISO)
setup['selRC'] = 0                                                                                                       # RC-Model (Foster) (0): de-activated, (1): activated

# Multiple-Input Multiple-Output (MIMO)
setup['selSS'] = 0                                                                                                       # SS-Model (State-Space) (0): de-activated, (1): activated
setup['selSF'] = 0                                                                                                       # SF-Model (Thermal Network) (0): de-activated, (1): nodal Foster (single-source experiments), (2): coupled Cauer network

# ------------------------------------------
# Analytical 2D Models
# ------------------------------------------
setup['selPO'] = 0                                                                                                       # PO-Model (Proper-Orthogonal) (0): de-activated, (1): activated
setup['selPS'] = 2                                                                                                       # PO-SS-Model (Proper-Orthogonal State Space) (0): de-activated, (1): 2D-activated, (2): 3D-activated

# ------------------------------------------
# Machine/Deep Learning (tbi)
# ------------------------------------------
setup['selML'] = 0                                                                                                       # ML-Model (Machine Learning) (0): de-activated, (1): KNN, (2): RF, (3): SVR
setup['selDL'] = 0                                                                                                       # DL-Model (Machine Learning) (0): de-activated, (1): activated
setup['selPI'] = 0                                                                                                       # Physics Informed DL-Model (tbd) (0): de-activated, (1): CNN, (2): LSTM, (3): CNN+LSTM

# ==============================================================================
# Python Port
# ==============================================================================
setup['legacy'] = 0                                                                                                      # (0): corrected numerics (default), (1): behaviour of the MATLAB version

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(setup, path)
