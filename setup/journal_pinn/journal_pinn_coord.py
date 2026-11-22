#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_coord
# Date:         22.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Coordinate network (setup['selPF'] = 2): one network T(x, t), trained on the testing profile only from
the 16 sensor temperatures (every fifth sensor is held out for the validation) and the heat equation; it
reconstructs the field at all other nodes.
Board: 3 chips (T1, D1, M1) on an IMS board of 30 mm x 20 mm, medium mesh (1 mm), layer-resolved 3D model, 16 sensor
nodes (Mdl > pinf > nSens in config/journal_pinn/pinn_coord.xlsx, chosen by farthest-point sampling, first sensor = hottest
node). The data sets are generated first (genBoardPF.py, once, files data/board/pinn_medium_*). Start with
    python setup/journal_pinn/journal_pinn_coord.py
Follow the code: src/py/main.py -> src/py/model/PF/pfFit.py (training) -> pfSol.py (testing), network in pfNet.py
(pfCoord: one network T(x, t)).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Path
# ==============================================================================
from os.path import dirname, abspath, isdir, isfile, join as pjoin
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
from setup.tutorial import genBoardPF

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
setup['name'] = 'journal_pinn_coord'                                                                                 # name of the simulation (string)

# ==============================================================================
# Experiment
# ==============================================================================

# ------------------------------------------
# Configuration
# ------------------------------------------
setup['config'] = '/journal_pinn/pinn_coord'                                                                                # name of the config file (string): medium mesh, 16 sensors (nSens), all other values as in the tutorial

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
setup['trFile'] = ['pinn_medium_train1', 'pinn_medium_train2', 'pinn_medium_train3', 'pinn_medium_train4']              # list of training files (strings) at least one
setup['teFile'] = 'pinn_medium_test'                                                                                     # testing file (string) exactly one
setup['vlFile'] = ['pinn_medium_val']                                                                                    # list of validation files (strings) at least one

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
setup['selPF'] = 2                                                                                                       # Physics-Informed Field Network for 2D/3D fields (0): de-activated, (1): operator network (new load profiles), (2): coordinate network (field from sensors)

# ==============================================================================
# Python Port
# ==============================================================================
setup['legacy'] = 0                                                                                                      # (0): corrected numerics (default), (1): behaviour of the MATLAB version

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    if not isfile(pjoin(base, 'data', 'board', 'pinn_medium_test.mat')):                                                 # data sets of the medium mesh (1 mm), generated once
        genBoardPF.main('pinn_medium', 4, 20.0, 'xz', layout=dict(genBoardPF.LAYOUT, dx=1.0))
    main(setup, path)
