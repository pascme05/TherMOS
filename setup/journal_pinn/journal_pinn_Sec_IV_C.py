#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Sec_IV_C
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
PINN journal paper (parametric modal operator network), Section IV-C (Sensitivity to the Number of Training Samples).
Number of labelled boards with and without physics (n3 … n12np), simulated time per board (tL10, tL20, tP1), layouts
only (pos1 … pos24) and conditions only (mu1 … mu24). Fig. samples, Table samples.
Config:     config/journal_pinn/journal_pinn_Sec_IV_C.json
Runtime:    ≈ 4.5 h (17 trainings)
Usage:      python setup/journal_pinn/journal_pinn_Sec_IV_C.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
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
CONFIG = 'journal_pinn_Sec_IV_C'                                                     # config/journal_pinn/<CONFIG>.json

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    execute(CONFIG)

