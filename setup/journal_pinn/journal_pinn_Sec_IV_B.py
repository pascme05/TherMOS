#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Sec_IV_B
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
PINN journal paper (parametric modal operator network), Section IV-B (Extrapolation).
Networks with physics-only boards anywhere (physAll) and over an extended range (physExt), and the one-at-a-time
sensitivity of Fig. sens. Requires base and nophys of Section IV-A.
Config:     config/journal_pinn/journal_pinn_Sec_IV_B.json
Runtime:    ≈ 1 h (2 trainings, sensitivity ≈ 20 min)
Usage:      python setup/journal_pinn/journal_pinn_Sec_IV_B.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
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
CONFIG = 'journal_pinn_Sec_IV_B'                                                     # config/journal_pinn/<CONFIG>.json

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    execute(CONFIG)

