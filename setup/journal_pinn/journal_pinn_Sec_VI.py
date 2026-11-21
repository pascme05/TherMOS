#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         journal_pinn_Sec_VI
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
PINN journal paper (parametric modal operator network), Section VI (Discussion: layout optimization, single-board extensions).
Layout optimisation through the network physAll (VI-C, requires Section IV-B) and the single-board runs of
Table single (VI-E, src/py/main.py in the PF mode). The convergence of VI-D uses the histories of IV-A/IV-B.
Config:     config/journal_pinn/journal_pinn_Sec_VI.json
Runtime:    ≈ 2 h (layout ≈ 10 min, 12 single-board runs)
Usage:      python setup/journal_pinn/journal_pinn_Sec_VI.py [--force] [--dry] [--only a,b] [--epoch N] [--out DIR]
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
CONFIG = 'journal_pinn_Sec_VI'                                                     # config/journal_pinn/<CONFIG>.json

#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    execute(CONFIG)

