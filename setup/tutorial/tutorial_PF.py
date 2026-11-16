#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         tutorial_PF
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
This script starts the PF tutorial: a physics-informed field network learns the 3D temperature field of a board with
three chips from finite-volume data and predicts it for another load profile. The data are generated first with
    python setup/tutorial/genBoardPF.py
(data/board/tutorial_pf_train1 … 4, _val, _test; board designer model). setup['selPF'] = 1 trains the operator
network on the four training profiles and tests it on the designed profile; setup['selPF'] = 2 trains the coordinate
network on the sensors of the testing profile only (set Mdl › pinf › nSens in config/tutorial/tutorial_PF.xlsx, e.g. 50).
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
setup['name'] = 'tutorial_PF'                                                                                            # name of the simulation (string)

# ==============================================================================
# Experiment
# ==============================================================================

# ------------------------------------------
# Configuration
# ------------------------------------------
setup['config'] = '/tutorial/tutorial_PF'                                                                                # name of the config file (string)

# ------------------------------------------
# Plotting and Saving
# ------------------------------------------
setup['plotOut'] = 1                                                                                                     # (0): model output is not plotted, (1): model output is plotted, (2): plotting one figure per output
setup['plotMdl'] = 1                                                                                                     # (0): model structure is not plotted, (1): model structure is plotted
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
setup['trFile'] = ['tutorial_pf_train1', 'tutorial_pf_train2', 'tutorial_pf_train3', 'tutorial_pf_train4']               # list of training files (strings) at least one
setup['teFile'] = 'tutorial_pf_test'                                                                                     # testing file (string) exactly one
setup['vlFile'] = ['tutorial_pf_val']                                                                                    # list of validation files (strings) at least one

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
setup['selPS'] = 0                                                                                                       # PO-SS-Model (Proper-Orthogonal State Space) (0): de-activated, (1): activated

# ------------------------------------------
# Machine/Deep Learning (tbi)
# ------------------------------------------
setup['selML'] = 0                                                                                                       # ML-Model (Machine Learning) (0): de-activated, (1): KNN, (2): RF, (3): SVR
setup['selDL'] = 0                                                                                                       # DL-Model (Machine Learning) (0): de-activated, (1): activated
setup['selPI'] = 0                                                                                                       # Physics Informed DL-Model (tbd) (0): de-activated, (1): CNN, (2): LSTM, (3): CNN+LSTM
setup['selPF'] = 1                                                                                                       # Physics-Informed Field Network for 2D/3D fields (0): de-activated, (1): operator network (new load profiles), (2): coordinate network (field from sensors)

# ==============================================================================
# Python Port
# ==============================================================================
setup['legacy'] = 0                                                                                                      # (0): corrected numerics (default), (1): behaviour of the MATLAB version

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(setup, path)
