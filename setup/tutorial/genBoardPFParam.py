#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoardPFParam
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
This script generates the board families of the parametric PF study (src/py/model/PF/pfParam.py): IMS boards with three
chips whose cooling h, dielectric conductivity k_diel, die attach factor s_att, Al base heat capacity factor s_rc and
chip positions vary. Training positions keep every chip centre outside the band x_c ∈ [12, 16] mm, so that the test set
T3 can check unseen positions. Files (data/board/pfparam/, npz, see pfParam.saveFam):
    train   24 boards, LHS in μ, 2 random loss profiles of 40 s each with finite-volume temperatures (labelled)
    phys    200 boards, LHS in μ, positions outside the band, operator only (physics loss)
    physAll 200 boards as phys, but positions anywhere (also inside the band)
    physExt 200 boards as phys, but μ over an extended range (h 250–8000 W/m²K, k_diel 0.5–12 W/mK; set 'physExt')
    val     6 boards, 1 random profile of 40 s
    t1      10 boards, interpolation in μ, positions outside the band, designed profile (pulse/pulse/sine, 40 s)
    t1f     10 boards as t1 but the nominal chip positions of the PF tutorial (pMOR comparison)
    pmor    8 boards with the nominal positions, 1 random profile of 40 s (snapshots of the pMOR baseline)
    t2      6 boards, extrapolation in h and k_diel (nominal positions), designed profile
    t3      6 boards with one chip inside the band (2 per chip) plus one board with 4 chips, designed profile
    t4      1 board, mission profile of 30 min from cold (segments 1–20 s, ramps 0–1 s, levels 0–100 %); temperatures
            of all nodes every 1 s and of the die nodes every sample
    t4slow  as t4 with slow segments (10–120 s, ramps 0–5 s, levels 10–100 %; first version of the study)
    trainPos  24 labelled boards as train, but nominal condition μ (only the chip positions vary; set 'trainPos')
    trainMu   24 labelled boards as train, but nominal chip positions (only μ varies; set 'trainMu')
Usage:      python setup/tutorial/genBoardPFParam.py [set ...]       (default: all sets)
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
from src.py.model.PF.pfParam import (CHIPS, NOM, MU, sampleMu, samplePos, randProfile, buildBoard, simulate, saveFam)
from src.py.board.genProfile import genProfile
from src.py.board.genMission import genMission

# ==============================================================================
# External
# ==============================================================================
import os
import time
import warnings
import numpy as np

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
TS = 0.05                                                               # sampling time (s)
T_END = 40.0                                                            # length of the profiles (s)
BAND = (12.0, 16.0)                                                     # x_c band without training chips (mm)
HOLDS = (0.5, 8.0)                                                      # hold times of the random profiles (s)
T4_END = 1800.0                                                         # length of the mission profile (s)
T4_DEC = 20                                                             # full field every T4_DEC samples
T4_FAST = ((1.0, 20.0), (0.0, 1.0), (0.0, 1.0))                         # t4: segments, ramps (s), levels (of P)


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def designed(t, chips):
    """Designed profile of the PF tutorial: pulses (period 4 s, duty 0.5) for T1 and D1, sine for M1 (and M2)."""
    par = {'period': 4.0, 'duty': 0.5, 'amp': 0.5}
    kind = {'Si IGBT': 'pulse', 'Si diode': 'pulse', 'SiC MOSFET': 'sine'}
    return np.column_stack([genProfile(kind[c['tech']], c['P'], t, par) for c in chips])


def mission(t, rng, chips, dur=(10.0, 120.0), ramp=(0.0, 5.0), lev=(0.1, 1.0)):
    """Mission profile: segments of dur s with ramps of ramp s and random levels (lev of P) per chip."""
    seq, tt = [], 0.0
    while tt < t[-1]:
        s = {'segment': 's', 'duration [s]': float(rng.uniform(*dur)), 'ramp [s]': float(rng.uniform(*ramp))}
        for c in chips:
            s[c['name'] + ' [W]'] = float(c['P'] * rng.uniform(*lev))
        seq.append(s)
        tt += s['duration [s]']
    return genMission(seq, chips, t)[0]


def nominal():
    return [(float(c['x']), float(c['y'])) for c in CHIPS]


def labelled(boards, profs):
    """Adds the profiles and the finite-volume temperature rises to every board."""
    for B, P in zip(boards, profs):
        B['P'] = P
        B['th'] = [simulate(B, p, TS) for p in P]
    return boards


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(sets=None):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Generating the board families of the parametric PF study")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    folder = pjoin(base, 'data', 'board', 'pfparam')
    os.makedirs(folder, exist_ok=True)
    sets = sets or ['train', 'phys', 'physAll', 'val', 't1', 't1f', 'pmor', 't2', 't3', 't4']
    t = np.arange(0.0, T_END + TS * 1e-9, TS)
    seeds = {'train': 1, 'phys': 2, 'physAll': 3, 'val': 4, 't1': 5, 't1f': 6, 'pmor': 7, 't2': 8, 't3': 9, 't4': 10,
             'physExt': 11, 't4slow': 10, 'trainPos': 12, 'trainMu': 13}

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for s in sets:
        t0 = time.time()
        rng = np.random.default_rng(seeds[s])
        if s == 'train':
            B = [buildBoard(m, samplePos(rng, BAND)) for m in sampleMu(24, rng)]
            B = labelled(B, [[randProfile(t, rng, HOLDS) for _ in range(0, 2)] for _ in B])
        elif s == 'trainPos':
            B = [buildBoard(dict(NOM), samplePos(rng, BAND)) for _ in range(0, 24)]
            B = labelled(B, [[randProfile(t, rng, HOLDS) for _ in range(0, 2)] for _ in B])
        elif s == 'trainMu':
            B = [buildBoard(m, nominal()) for m in sampleMu(24, rng)]
            B = labelled(B, [[randProfile(t, rng, HOLDS) for _ in range(0, 2)] for _ in B])
        elif s in ('phys', 'physAll'):
            B = [buildBoard(m, samplePos(rng, BAND if s == 'phys' else None)) for m in sampleMu(200, rng)]
        elif s == 'physExt':
            lim = dict(MU, h=(250.0, 8000.0, True), kDiel=(0.5, 12.0, True))
            B = [buildBoard(m, samplePos(rng, BAND)) for m in sampleMu(200, rng, lim)]
        elif s == 'val':
            B = [buildBoard(m, samplePos(rng, BAND)) for m in sampleMu(6, rng)]
            B = labelled(B, [[randProfile(t, rng, HOLDS)] for _ in B])
        elif s == 't1':
            B = [buildBoard(m, samplePos(rng, BAND)) for m in sampleMu(10, rng)]
            B = labelled(B, [[designed(t, CHIPS)] for _ in B])
        elif s == 't1f':
            B = [buildBoard(m, nominal()) for m in sampleMu(10, rng)]
            B = labelled(B, [[designed(t, CHIPS)] for _ in B])
        elif s == 'pmor':
            B = [buildBoard(m, nominal()) for m in sampleMu(8, rng)]
            B = labelled(B, [[randProfile(t, rng, HOLDS)] for _ in B])
        elif s == 't2':
            mus = [dict(NOM, h=250.0), dict(NOM, h=8000.0), dict(NOM, kDiel=0.5), dict(NOM, kDiel=12.0),
                   dict(NOM, h=250.0, kDiel=0.5), dict(NOM, h=8000.0, kDiel=12.0)]
            B = [buildBoard(m, nominal()) for m in mus]
            B = labelled(B, [[designed(t, CHIPS)] for _ in B])
        elif s == 't3':
            mus = sampleMu(7, rng)
            B = [buildBoard(mus[i], samplePos(rng, BAND, i % 3)) for i in range(0, 6)]
            ch4 = CHIPS + [dict(CHIPS[2], name='M2')]
            B.append(buildBoard(mus[6], samplePos(rng, BAND, None, ch4), ch4))
            B = labelled(B, [[designed(t, Bi['chips'])] for Bi in B])
        elif s in ('t4', 't4slow'):
            tt = np.arange(0.0, T4_END + TS * 1e-9, TS)
            Bi = buildBoard(sampleMu(1, rng)[0], nominal())
            P = mission(tt, rng, CHIPS) if s == 't4slow' else mission(tt, rng, CHIPS, *T4_FAST)
            th = simulate(Bi, P, TS)
            die = np.concatenate(Bi['die'])
            Bi['P'], Bi['th'] = [P], [th[::T4_DEC]]
            saveFam(pjoin(folder, s + '.npz'), [Bi])
            np.savez_compressed(pjoin(folder, s + 'die.npz'), th=th[:, die].astype(np.float32), die=die, dec=T4_DEC)
            print("INFO: %s: 1 board, %d samples (%.0f s), T max %.1f K, %.1f s" % (s, len(tt), tt[-1],
                                                                              float(th.max()), time.time() - t0))
            continue
        saveFam(pjoin(folder, s + '.npz'), B)
        thMax = max([float(np.max(x)) for Bi in B for x in Bi.get('th', [])] or [0.0])
        print("INFO: %s: %d boards (N = %d … %d nodes), rise max %.1f K, %.1f s" %
              (s, len(B), min(len(Bi['C']) for Bi in B), max(len(Bi['C']) for Bi in B), thMax, time.time() - t0))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Board families written to data/board/pfparam/ (training range of μ: %s)" %
          ', '.join('%s %g–%g' % (k, v[0], v[1]) for k, v in MU.items()))


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:] or None)


#######################################################################################################################
# References
#######################################################################################################################
