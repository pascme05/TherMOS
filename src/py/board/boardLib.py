#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         boardLib
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This file holds the technology library of the board designer: board stack-ups (layers with in-plane and
through-plane conductivity), chip technologies (die material, thickness, size, die attach, typical loss and maximum
junction temperature) and cooling conditions (heat transfer coefficient on the bottom of the board). Material values
are taken from src/py/tools/toolData.py (OPTELIA library, Incropera 2007, Wintrich 2015).
Layer:  (name, material, thickness (mm), k_xy (W/mK), k_z (W/mK), rho (kg/m³), cp (J/kgK))
Chip:   dict(mat, t (mm), w (mm), l (mm), attach (material), t_att (mm), P (W), Tjmax (°C), note)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.tools.toolData import MATERIALS, COOLING, via_k

# ==============================================================================
# External
# ==============================================================================


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _lay(name, mat, t, kxy=None, kz=None):
    k, rho, cp = MATERIALS[mat][0:3]
    return (name, mat, t, k if kxy is None else kxy, k if kz is None else kz, rho, cp)


def _fr4(name, t, kz=None):
    return (name, 'FR4', t, MATERIALS['FR4 (in-plane)'][0], MATERIALS['FR4 (through-plane)'][0] if kz is None else kz,
            MATERIALS['FR4 (in-plane)'][1], MATERIALS['FR4 (in-plane)'][2])


#######################################################################################################################
# Board technologies
#######################################################################################################################
_VIA = round(via_k(1.0, 0.3, 25), 2)
BOARDS = {
    'FR4 2-layer, 35 µm Cu': [_lay('Cu top', 'Cu', 0.035), _fr4('FR4 core', 1.5), _lay('Cu bottom', 'Cu', 0.035)],
    'FR4 2-layer, 70 µm Cu': [_lay('Cu top', 'Cu', 0.070), _fr4('FR4 core', 1.5), _lay('Cu bottom', 'Cu', 0.070)],
    'FR4 4-layer, 35 µm Cu planes': [_lay('Cu top', 'Cu', 0.035), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu inner 1', 'Cu', 0.035), _fr4('FR4 core', 0.71),
                                     _lay('Cu inner 2', 'Cu', 0.035), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu bottom', 'Cu', 0.035)],
    'FR4 2-layer with thermal vias': [_lay('Cu top', 'Cu', 0.070), _fr4('FR4 with vias', 1.5, kz=_VIA),
                                      _lay('Cu bottom', 'Cu', 0.070)],
    'IMS, Al base, 100 µm dielectric': [_lay('Cu foil', 'Cu', 0.070), _lay('Dielectric', 'IMS dielectric', 0.100),
                                        _lay('Al base', 'Al alloy (6063)', 1.5)],
    'DBC Al₂O₃ (0.38 mm)': [_lay('Cu top', 'Cu', 0.3), _lay('Al₂O₃', 'Al2O3 (96 %)', 0.38), _lay('Cu bottom', 'Cu', 0.3)],
    'DBC AlN (0.63 mm)': [_lay('Cu top', 'Cu', 0.3), _lay('AlN', 'AlN', 0.63), _lay('Cu bottom', 'Cu', 0.3)],
    'AMB Si₃N₄ (0.32 mm)': [_lay('Cu top', 'Cu', 0.3), _lay('Si₃N₄', 'Si3N4', 0.32), _lay('Cu bottom', 'Cu', 0.3)],
    'DBC Al₂O₃ on 3 mm Cu baseplate': [_lay('Cu top', 'Cu', 0.3), _lay('Al₂O₃', 'Al2O3 (96 %)', 0.38),
                                       _lay('Cu bottom', 'Cu', 0.3), _lay('Solder', 'SAC305 solder', 0.1),
                                       _lay('Cu baseplate', 'Cu', 3.0)],
}

#######################################################################################################################
# Chip technologies
#######################################################################################################################
CHIPS = {
    'Si IGBT': dict(mat='Si', t=0.12, w=10.0, l=10.0, attach='SAC305 solder', t_att=0.10, P=60.0, Tjmax=175.0,
                    note="1200 V trench-field-stop IGBT die, soldered"),
    'Si diode': dict(mat='Si', t=0.15, w=6.0, l=8.0, attach='SAC305 solder', t_att=0.10, P=20.0, Tjmax=175.0,
                     note="free-wheeling diode die, soldered"),
    'Si MOSFET': dict(mat='Si', t=0.20, w=5.0, l=6.0, attach='SAC305 solder', t_att=0.08, P=5.0, Tjmax=175.0,
                      note="low-voltage trench MOSFET (SuperSO8-like footprint)"),
    'SiC MOSFET': dict(mat='SiC (4H)', t=0.18, w=4.0, l=5.0, attach='Sintered Ag', t_att=0.05, P=25.0, Tjmax=175.0,
                       note="1200 V SiC MOSFET die, silver sintered"),
    'GaN HEMT': dict(mat='Si', t=0.20, w=3.0, l=5.0, attach='SAC305 solder', t_att=0.05, P=6.0, Tjmax=150.0,
                     note="GaN-on-Si HEMT, the heat flows through the Si substrate"),
    'Controller IC': dict(mat='Si', t=0.30, w=5.0, l=5.0, attach='SAC305 solder', t_att=0.05, P=0.5, Tjmax=125.0,
                          note="gate driver / controller in a QFN package (estimate)"),
}

#######################################################################################################################
# Cooling (h on the bottom of the board, W/m²K)
#######################################################################################################################
COOL = {k: v for k, v in COOLING.items() if v is not None and v > 0}
H_TOP = 10.0                                                                                     # natural convection top


#######################################################################################################################
# References
#######################################################################################################################
# [1] F. P. Incropera et al., Fundamentals of Heat and Mass Transfer, 6th ed., Wiley, 2007.
# [2] U. Scheuermann, A. Wintrich et al., Application Manual Power Semiconductors, SEMIKRON, 2015.
