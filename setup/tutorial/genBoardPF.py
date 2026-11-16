#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoardPF
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This script generates the field data of the PF tutorial (physics-informed field network) with the finite-volume model
of the board designer: three chips (T1, D1, M1) on an IMS board of 30 mm x 20 mm, layer-resolved 3D model (view 'xz')
or lumped top view (view 'xy', 2D).
1) training data: random load levels of every chip (independent excitation), files <name>_train1 … _trainN
2) validation data: another random profile, file <name>_val
3) testing data: the designed profiles of the chips (pulse, pulse, sine), file <name>_test
The files are written to data/board/ in the field format of TherMOS (saveBoard) and include the finite-volume
operator for the discrete physics loss (Mdl › pinf › phys = 2).
The time constants of the board are short (τ_max ≈ 0.7 s): Ts = 0.05 s and load periods of 4 s (random profiles hold
a level for 1 s) give transients in every profile.
Usage:      python setup/tutorial/genBoardPF.py [name] [number of training profiles] [duration per profile (s)] [view]
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
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.saveBoard import saveBoard
from src.py.board.genProfile import genProfile

# ==============================================================================
# External
# ==============================================================================
import json
import warnings
import numpy as np

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Layout
#######################################################################################################################
LAYOUT = {'view': 'xz', 'W': 30, 'L': 20, 'board': 'IMS, Al base, 100 µm dielectric', 'h': 1500, 'Ta': 25, 'dx': 1.0,
          'yCut': 10,
          'chips': [{'name': 'T1', 'tech': 'Si IGBT', 'x': 8, 'y': 10, 'w': 6, 'l': 6, 'P': 40, 'profile': 'pulse'},
                    {'name': 'D1', 'tech': 'Si diode', 'x': 17, 'y': 13, 'w': 4, 'l': 5, 'P': 15, 'profile': 'pulse'},
                    {'name': 'M1', 'tech': 'SiC MOSFET', 'x': 24, 'y': 7, 'w': 4, 'l': 5, 'P': 20, 'profile': 'sine'}]}


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def profiles(D, t, period, kind=None, seed=0):
    """Loss profiles (Nt x nChips) (W): the designed profile of every chip or random levels (kind = 'random')."""
    par = {'period': period, 'duty': 0.5, 'amp': 0.5, 'hold': max(period / 4, t[1] - t[0])}
    return np.column_stack([genProfile(kind or c.get('profile', 'step'), c['P'], t, par, seed=seed + i)
                            for i, c in enumerate(D['chips'])])


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(name='tutorial_pf', nTr=4, tTr=20.0, view='xz', Ts=0.05, period=4.0, layout=None):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Generating the field data of the PF tutorial")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    lay = dict(layout or LAYOUT, view=view)
    folder = pjoin(base, 'data', 'board')

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    D = genBoard(lay)
    [C, G, Bp] = calcBoardMat(D)
    N = len(D['nod']['x']) if 'nod' in D else len(D['geo'])
    print("INFO: %s model with %d nodes and %d chips (%s)" % ('Layer-resolved 3D' if 'nod' in D else 'Lumped 2D', N,
                                                              len(D['chips']), ', '.join(c['name'] for c in D['chips'])))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    t = np.arange(0.0, tTr + Ts * 1e-9, Ts)
    jobs = [("train%d" % (k + 1), t, profiles(D, t, period, 'random', 11 + 17 * k)) for k in range(0, nTr)]
    jobs.append(('val', t, profiles(D, t, period, 'random', 997)))
    tTe = np.arange(0.0, 2 * tTr + Ts * 1e-9, Ts)
    jobs.append(('test', tTe, profiles(D, tTe, period)))
    files = []
    for (tag, tt, P) in jobs:
        [T, _] = simBoard(C, G, Bp, P, Ts, lay['Ta'], 4, steady=False)
        files.append(saveBoard(D, tt, P, T, pjoin(folder, name + '_' + tag + '.mat')))
        print("INFO: %s: %d samples, T max %.1f °C" % (tag, len(tt), float(np.max(T))))

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    with open(pjoin(folder, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(lay, f, indent=1)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: %d files written to data/board/ (%s_train1 … _train%d, _val, _test)" % (len(files), name, nTr))

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return files


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else 'tutorial_pf', int(a[1]) if len(a) > 1 else 4, float(a[2]) if len(a) > 2 else 20.0,
         a[3] if len(a) > 3 else 'xz')
