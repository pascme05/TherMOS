#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Sec_IV_A
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This script reproduces Section IV-A (inside the training domain, test set T1): Foster, POD, the proposed network and the network without
physics on 3 meshes x 2 sensor versions (6 data sets). Every run is evaluated on all test sets T1-T4,
so that Section IV-B uses the same runs. Runtime: ~6 h on an RTX 3070.
Methods and data sets: config/journal_pinn/<name>.json; results: results/pfhb/<method>_<mesh>_<sensors>.json.
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
from setup.journal_pinn.mainPINN import main

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
setup['name'] = 'journal_pinn_Sec_IV_A'                                                                                  # name of the simulation (string)

# ==============================================================================
# Experiment
# ==============================================================================
setup['out'] = 'results/pfhb'                                                                                            # results folder relative to TherMOS (string)
setup['skip'] = 1                                                                                                        # (0): recompute everything, (1): skip runs whose results exist (restart after an interruption)
setup['epoch'] = 0                                                                                                       # (0): epochs of the configs, (N): at most N epochs (quick check of the chain, use another out)

# ==============================================================================
# Data
# ==============================================================================
setup['data'] = []                                                                                                       # meshes whose data sets are generated (data/board/pfhb/<mesh>): coarse, medium, fine (config data.json)

# ==============================================================================
# Runs
# ==============================================================================
setup['mesh'] = ['coarse', 'medium', 'fine']                                                                             # meshes of the runs: coarse (2 mm), medium (1 mm), fine (0.5 mm)
setup['sens'] = ['S', 'L']                                                                                               # sensor versions of the training labels: S (4 hot spots), L (16: hot spots, 4 NTC, 8 on the base)
setup['methods'] = ['foster', 'pod', 'pinn', 'nophys']                                                                   # methods (config <method>.json): foster, pod (full field, one run per mesh), pinn, nophys
setup['nTrain'] = [24]                                                                                                   # numbers of labelled training boards (first n of the training set, maximum 24)

# ==============================================================================
# Memory and Computational Load
# ==============================================================================
setup['scale'] = 0                                                                                                       # (0): off, (1): size, runtime and memory per mesh and profile length (scale_<mesh>.json)

# ==============================================================================
# Layout Optimisation
# ==============================================================================
setup['layout'] = ''                                                                                                     # network run for the layout optimisation ('': off, e.g. 'pinn_medium_L')
setup['nStart'] = 8                                                                                                      # random start layouts of the gradient descent
setup['nIter'] = 300                                                                                                     # gradient steps per start
setup['nRand'] = 100                                                                                                     # layouts of the FV random search (comparison)

# ==============================================================================
# Figures (scripts in docu/paper/figures/scripts)
# ==============================================================================
setup['figs'] = []                                                                                                       # figure data scripts in docu/paper/figures/scripts (e.g. fighb)
setup['build'] = 0                                                                                                       # (0): off, (1): compile the standalone figure PDFs (buildfigs.py, pdflatex)

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(setup)
