#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Sec_IV_D
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
PINN journal paper (parametric modal operator network), Section IV-D (Memory Requirements and Computational Load).
Runtime and memory of FV, Krylov and the network over the grid, the profile length and the sampling time.
Requires base of Section IV-A. Fig. scale, Table load.
Config:     config/journal_pinn/journal_pinn_Sec_IV_D.json
Runtime:    ≈ 30 min (the finest grid needs ≈ 2 GB RAM)
Usage:      python setup/journal_pinn/journal_pinn_Sec_IV_D.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
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
CONFIG = 'journal_pinn_Sec_IV_D'                                                     # config/journal_pinn/<CONFIG>.json

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    execute(CONFIG)

