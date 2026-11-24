#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_all
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
This script runs all setups of the PINN journal paper in one go, in the order of setup['setups'] (the order of
setup/journal_pinn/readme.txt: data first, figures last). Every setup is run with its own values from its file, only
the results folder, skip and epoch are taken from here. Runtime: ~30 h (RTX 3070, sum of the estimates).
Lines marked CHANGE are the ones to adapt (name and results folder of a test, quick check, other computer); the
parameters of the methods are in config/journal_pinn/<method>.json (CHANGE there: GPU memory). Existing results in
setup['out'] are skipped, so an interrupted run restarts where it stopped. Linux in the background:
nohup python setup/journal_pinn/journal_pinn_all.py > journal_pinn_all.log 2>&1 &
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
setup['name'] = 'journal_pinn_all'                                                                                       # CHANGE: name of the simulation (string), e.g. a test name

# ==============================================================================
# Experiment
# ==============================================================================
setup['out'] = 'results/pfhb_v2'                                                                                         # CHANGE: results folder relative to TherMOS (string), used by all setups; another folder for a test (e.g. 'results/pfhb_test')
setup['skip'] = 1                                                                                                        # (0): recompute everything, (1): skip runs whose results exist (restart after an interruption)
setup['epoch'] = 0                                                                                                       # CHANGE: (0): epochs of the configs (paper), (N): at most N epochs (quick check of the chain, use another out)
setup['stop'] = 1                                                                                                        # (0): continue with the next setup after an error, (1): stop at the first failing setup

# ==============================================================================
# Setups (run order)
# ==============================================================================
setup['setups'] = ['journal_pinn_Tab_IV_V_data', 'journal_pinn_Tab_VII_nominal', 'journal_pinn_Tab_X_Fig_10_variable', 'journal_pinn_ablation', 'journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium', 'journal_pinn_Fig_11_samples', 'journal_pinn_Tab_XI_Fig_12_load', 'journal_pinn_Fig_13_layout', 'journal_pinn_figures']  # CHANGE: setups to run (setup/journal_pinn/<setup>.py), in this order; remove the ones not needed (data first, figures last)

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(setup)
