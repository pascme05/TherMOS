#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium
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
This script reproduces the results of the medium mesh in Section IV-A and IV-B: Table VIII (errors, runtime and
parameters on N1, sensors L), Table IX (errors per chip), Fig. 6 (field map of a board of N1: FV, POD, network S
and L), Fig. 7 (spread of the error over the boards of N1, T2, T1, T3), Fig. 8 (step response of the network for
four h) and Fig. 9 (mission profile T4). All runs of the medium mesh; they are skipped if Tables VII and X were
run before. Runtime: ~6 h (all networks of the medium mesh), ~10 min if they exist.
Lines marked CHANGE are the ones to adapt (name and results folder of a test, quick check, other computer); the
parameters of the methods are in config/journal_pinn/<method>.json (CHANGE there: GPU memory). Existing results in
setup['out'] are skipped, so the setups share the network runs (e.g. pinn_medium_L is trained once).
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
setup['name'] = 'journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium'                                                            # CHANGE: name of the simulation (string), e.g. a test name

# ==============================================================================
# Experiment
# ==============================================================================
setup['out'] = 'results/pfhb_v2'                                                                                         # CHANGE: results folder relative to TherMOS (string); another folder for a test (e.g. 'results/pfhb_test')
setup['skip'] = 1                                                                                                        # (0): recompute everything, (1): skip runs whose results exist (restart after an interruption)
setup['epoch'] = 0                                                                                                       # CHANGE: (0): epochs of the configs (paper), (N): at most N epochs (quick check of the chain, use another out)

# ==============================================================================
# Data
# ==============================================================================
setup['data'] = []                                                                                                       # meshes whose data sets are generated (data/board/pfhb/<mesh>): coarse, medium, fine (config data.json)

# ==============================================================================
# Runs
# ==============================================================================
setup['mesh'] = ['medium']                                                                                               # CHANGE: meshes of the runs: coarse (2 mm), medium (1 mm), fine (0.5 mm; networks ~3-5 h per run)
setup['sens'] = ['S', 'L']                                                                                               # sensor versions of the training labels: S (4 hot spots), L (16: hot spots, 4 NTC, 8 on the base)
setup['methods'] = ['fosterNom', 'podNom', 'foster', 'pinn', 'nophys', 'deeponet']                                       # methods (config <method>.json): foster/fosterNom, pod/podNom (Nom: nominal layout; pod full field), pinn, nophys, deeponet
setup['nTrain'] = [24]                                                                                                   # numbers of labelled training boards (first n of the training set, maximum 24)
setup['fieldmap'] = ['medium']                                                                                           # meshes of the field map (copper layer at the peak of a testing board of N1: FV, errors of POD and the networks S, L)

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
setup['figs'] = []                                                                                                       # CHANGE: figure data scripts in docu/paper/figures/scripts (fighb: tables, figoverview: Fig. 1, figsub: Figs. 2, 3, 8; needs docu/paper)
setup['build'] = 0                                                                                                       # CHANGE: (0): off, (1): compile the standalone figure PDFs (buildfigs.py, pdflatex)

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(setup)
