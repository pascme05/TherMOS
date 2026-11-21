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
PINN journal paper (parametric modal operator network), Section IV-A (Comparative Results inside the Training Domain).
Proposed network, second seed, no physics, K = 8/16, no relative coordinates, no steady-state loss, and the
baselines FV, pMOR, pMOR with positions and Krylov. Tables: models, ablation; Fig.: fields, T4.
Config:     config/journal_pinn/journal_pinn_Sec_IV_A.json
Runtime:    ≈ 2.5 h (8 trainings of ≈ 17 min on an RTX 3070, baselines ≈ 10 min)
Usage:      python setup/journal_pinn/journal_pinn_Sec_IV_A.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
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
CONFIG = 'journal_pinn_Sec_IV_A'                                                     # config/journal_pinn/<CONFIG>.json

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    execute(CONFIG)

