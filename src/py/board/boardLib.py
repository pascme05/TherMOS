#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         boardLib
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This file holds the technology library of the board designer: board stack-ups (layers with material, thickness and
copper coverage), chip technologies (die material, thickness, size, die attach, typical loss and maximum junction
temperature), cooling conditions (heat transfer coefficient on the bottom of the board) and the averaged thermal vias.
Material values are taken from src/py/tools/toolData.py (OPTELIA library, Incropera 2007, Wintrich 2015).
Layer:  dict(layer (name), mat (material or 'FR4'), t (mm), cov (copper coverage of a Cu layer, %), kxy/kz (W/mK,
        optional overrides))
Chip:   dict(mat, t (mm), w (mm), l (mm), attach (material), t_att (mm), P (W), Tjmax (°C), note)

A patterned copper layer (coverage c < 100 %, rest filled with laminate) is homogenised [3]:

    k_xy = c·k_Cu + (1 − c)·k_FR4,xy        (parallel, in-plane)
    k_z  = 1 / (c/k_Cu + (1 − c)/k_FR4,z)    (series, through-plane)

Thermal vias are averaged over their array (pitch p, drill d, plating t_pl): the FR4 layers in the via region get the
through-plane conductivity k_z,eff = φ_Cu·k_Cu + φ_hole·k_fill + (1 − φ_Cu − φ_hole)·k_z,FR4 with the area fractions
φ_Cu = π(r² − (r − t_pl)²)/p² of the copper barrel and φ_hole = π(r − t_pl)²/p² of the hole (toolData.via_k); k_xy is
not changed.

V1.1: 07.11.2026, Pascal Schirmer
- Layers as dicts with copper coverage, 4-layer 70 µm and 6-layer 35/70 µm FR4 boards, averaged thermal vias
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
# Settings
#######################################################################################################################
FR4 = {'kxy': MATERIALS['FR4 (in-plane)'][0], 'kz': MATERIALS['FR4 (through-plane)'][0],
       'rho': MATERIALS['FR4 (in-plane)'][1], 'cp': MATERIALS['FR4 (in-plane)'][2]}
LAYER_MATS = ['FR4'] + [m for m, v in MATERIALS.items() if v[3] in ('Metal', 'Ceramic', 'Joining', 'Board', 'TIM')
                        and not m.startswith('FR4')]
VIA_FILL = {'open (air)': 0.026, 'plugged (epoxy)': 0.3, 'filled (Cu paste)': 15.0, 'filled (solid Cu)': 390.0}


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _lay(name, mat, t, cov=100.0, kz=None):
    out = {'layer': name, 'mat': mat, 't': t, 'cov': cov}
    if kz is not None:
        out['kz'] = kz
    return out


def _fr4(name, t, kz=None):
    return _lay(name, 'FR4', t, kz=kz)


def _six(tCu):
    """6-layer FR4 board of 1.6 mm: signal / plane / signal / signal / plane / signal (coverage 40 / 90 %)."""
    tc = tCu * 1e-3
    f = (1.6 - 6 * tc) / 1.38
    return [_lay('L1 signal (top)', 'Cu', tc, 40.0), _fr4('prepreg', round(0.21 * f, 3)),
            _lay('L2 plane (GND)', 'Cu', tc, 90.0), _fr4('core', round(0.36 * f, 3)),
            _lay('L3 signal', 'Cu', tc, 40.0), _fr4('prepreg', round(0.24 * f, 3)),
            _lay('L4 signal', 'Cu', tc, 40.0), _fr4('core', round(0.36 * f, 3)),
            _lay('L5 plane (PWR)', 'Cu', tc, 90.0), _fr4('prepreg', round(0.21 * f, 3)),
            _lay('L6 signal (bottom)', 'Cu', tc, 40.0)]


def _num(v):
    """Float or None for empty table cells."""
    try:
        v = float(v)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def layerProps(l):
    """
    Properties of one layer (dict, see above): returns (kxy, kz, rho, cp) with the copper coverage and the optional
    overrides kxy/kz applied.
    """
    if l['mat'] == 'FR4':
        kxy, kz, rho, cp = FR4['kxy'], FR4['kz'], FR4['rho'], FR4['cp']
    else:
        k, rho, cp = MATERIALS[l['mat']][0:3]
        kxy, kz = k, k
        c = _num(l.get('cov'))
        c = 1.0 if c is None else min(max(c, 0.0), 100.0) / 100.0
        if l['mat'] == 'Cu' and c < 1.0:
            kxy = c * k + (1 - c) * FR4['kxy']
            kz = 1.0 / (c / k + (1 - c) / FR4['kz']) if c > 0 else FR4['kz']
            rc = c * rho * cp + (1 - c) * FR4['rho'] * FR4['cp']
            rho = c * rho + (1 - c) * FR4['rho']
            cp = rc / rho
    kxy = kxy if _num(l.get('kxy')) is None else _num(l['kxy'])
    kz = kz if _num(l.get('kz')) is None else _num(l['kz'])
    return kxy, kz, rho, cp


def viaK(vias, kz):
    """Through-plane conductivity of an FR4 layer (kz, W/mK) in the via region; vias: dict(d (mm), tpl (µm), pitch
    (mm), fill)."""
    kFill = VIA_FILL.get(vias.get('fill', 'open (air)'), 0.026)
    return via_k(float(vias['pitch']), float(vias['d']), float(vias['tpl']), filled=True, k_board=kz, k_fill=kFill)


#######################################################################################################################
# Board technologies (top → bottom)
#######################################################################################################################
_VIA = round(via_k(1.0, 0.3, 25), 2)
BOARDS = {
    'FR4 2-layer, 35 µm Cu': [_lay('Cu top', 'Cu', 0.035), _fr4('FR4 core', 1.5), _lay('Cu bottom', 'Cu', 0.035)],
    'FR4 2-layer, 70 µm Cu': [_lay('Cu top', 'Cu', 0.070), _fr4('FR4 core', 1.5), _lay('Cu bottom', 'Cu', 0.070)],
    'FR4 4-layer, 35 µm Cu planes': [_lay('Cu top', 'Cu', 0.035), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu inner 1', 'Cu', 0.035), _fr4('FR4 core', 0.71),
                                     _lay('Cu inner 2', 'Cu', 0.035), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu bottom', 'Cu', 0.035)],
    'FR4 4-layer, 70 µm Cu planes': [_lay('Cu top', 'Cu', 0.070), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu inner 1', 'Cu', 0.070), _fr4('FR4 core', 0.57),
                                     _lay('Cu inner 2', 'Cu', 0.070), _fr4('FR4 prepreg', 0.36),
                                     _lay('Cu bottom', 'Cu', 0.070)],
    'FR4 6-layer, 35 µm Cu': _six(35),
    'FR4 6-layer, 70 µm Cu': _six(70),
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
# [3] Y. Shabany, Heat Transfer: Thermal Management of Electronics, CRC Press, 2010.
