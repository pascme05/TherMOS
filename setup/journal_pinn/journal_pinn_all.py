#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_all
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
PINN journal paper (parametric modal operator network), all results.
Runs every setup of the paper in the order of dependence (data, IV-A, IV-B, IV-C, IV-D, VI, figures).
Existing results are skipped, so the script can be restarted after an interruption.
Config:     config/journal_pinn/journal_pinn_*.json (hyperparameters: journal_pinn_net.json)
Runtime:    ≈ 11 h on an RTX 3070 (data ≈ 30 min, 27 trainings of ≈ 17 min)
Usage:      python setup/journal_pinn/journal_pinn_all.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
            (existing results are skipped; options: see setup/journal_pinn/runPaper.py)
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
from setup.journal_pinn.runPaper import execute

#######################################################################################################################
# Configuration
#######################################################################################################################
CONFIGS = ['journal_pinn_data', 'journal_pinn_Sec_IV_A', 'journal_pinn_Sec_IV_B', 'journal_pinn_Sec_IV_C',
           'journal_pinn_Sec_IV_D', 'journal_pinn_Sec_VI', 'journal_pinn_figures']       # in the order of dependence

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    for c in CONFIGS:
        execute(c)

