#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoardPFStudy
# Date:         20.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This script generates the field data of the PF studies on the board of the PF tutorial (genBoardPF: three chips on an
IMS board, layer-resolved 3D finite-volume model, same load profiles):
1) 'nl' — temperature-dependent conductivity: every conductance g of the model (also to the ambient) is scaled with
   k(θ̄)/k₀ = 1 − κ·θ̄ at the mean rise θ̄ of its two nodes (ambient: θ = 0), κ = kT (1/K). The heat balance

       C·dθ/dt = −G·u(θ) + Bp·P,      u = θ − κ·θ²/2   (Kirchhoff variable)

   is integrated with the implicit Euler method (nSub sub-steps per sample, Newton iterations with the Jacobian
   C/h + G·diag(1 − κ·θ) of the start of the sample). Files <name>nl_* store kT, files <name>nl0_* hold the same
   temperatures without kT (a model that assumes a constant conductivity).
2) 'mm' — model mismatch: the 'measured' board has the conduction L and the cooling H = diag(G·1) of the design model
   scaled with sK and sH (e.g. optimistic material data and a degraded heat sink),

       C·dθ/dt = −(sK·L + sH·H)·θ + Bp·P,

   the files <name>mm_* store these temperatures but the nominal operator of the design (what the designer knows).
Usage:      python setup/tutorial/genBoardPFStudy.py [nl | mm | all]
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
sys.path.insert(0, dirname(abspath(__file__)))

# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.saveBoard import saveBoard
from genBoardPF import LAYOUT, profiles

# ==============================================================================
# External
# ==============================================================================
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
KT = 0.003                                                              # temperature coefficient of k (1/K)
SK, SH = 0.85, 0.6                                                      # measured board: conduction and cooling factors


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def simNonlin(C, G, Bp, P, Ts, Ta, kT, nSub=4, nNewton=3):
    """Temperatures (Nt x N) (°C) of C·dθ/dt = −G·u(θ) + Bp·P, u = θ − kT·θ²/2, θ(0) = 0 (implicit Euler)."""
    N, h = len(C), Ts / nSub
    th, T = np.zeros(N), np.zeros((len(P), N))
    Ch = sp.diags(C / h)
    for k in range(1, len(P)):
        b = Bp @ P[k]
        lu = spl.splu((Ch + G @ sp.diags(1.0 - kT * th)).tocsc())       # Jacobian at the start of the sample
        for _ in range(0, nSub):
            x = th.copy()
            for _ in range(0, nNewton):
                F = C / h * (x - th) + G @ (x - 0.5 * kT * x ** 2) - b
                x = x - lu.solve(F)
            th = x
        T[k] = th
    return T + Ta


def split(G):
    """Conduction part L (zero row sums) and cooling H = diag(G·1) of the conductance matrix."""
    H = np.asarray(G.sum(axis=1)).ravel()
    return [(G - sp.diags(H)).tocsr(), sp.diags(H).tocsr()]


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(case='all', name='tutorial_pf', nTr=4, tTr=20.0, Ts=0.05, period=4.0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Generating the field data of the PF studies")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    lay = dict(LAYOUT, view='xz')
    folder = pjoin(base, 'data', 'board')
    D = genBoard(lay)
    [C, G, Bp] = calcBoardMat(D)
    [L, H] = split(G)
    t = np.arange(0.0, tTr + Ts * 1e-9, Ts)
    tTe = np.arange(0.0, 2 * tTr + Ts * 1e-9, Ts)
    jobs = [("train%d" % (k + 1), t, profiles(D, t, period, 'random', 11 + 17 * k)) for k in range(0, nTr)]
    jobs += [('val', t, profiles(D, t, period, 'random', 997)), ('test', tTe, profiles(D, tTe, period))]
    print("INFO: Layer-resolved 3D model with %d nodes, cooling %.3g W/K in total" % (len(C), float(np.sum(H.diagonal()))))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Temperature-Dependent Conductivity
    # ==============================================================================
    if case in ('nl', 'all'):
        for (tag, tt, P) in jobs:
            T = simNonlin(C, G, Bp, P, Ts, lay['Ta'], KT)
            saveBoard(D, tt, P, T, pjoin(folder, name + 'nl_' + tag + '.mat'), extra={'kT': KT})
            saveBoard(D, tt, P, T, pjoin(folder, name + 'nl0_' + tag + '.mat'))
            print("INFO: nl %s: T max %.1f °C (κ = %g 1/K)" % (tag, float(np.max(T)), KT))

    # ==============================================================================
    # Model Mismatch
    # ==============================================================================
    if case in ('mm', 'all'):
        Gm = (SK * L + SH * H).tocsr()
        for (tag, tt, P) in jobs:
            [T, _] = simBoard(C, Gm, Bp, P, Ts, lay['Ta'], 4, steady=False)
            saveBoard(D, tt, P, T, pjoin(folder, name + 'mm_' + tag + '.mat'))
            print("INFO: mm %s: T max %.1f °C (conduction x %g, cooling x %g)" % (tag, float(np.max(T)), SK, SH))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Field data of the PF studies written to data/board/")


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')


#######################################################################################################################
# References
#######################################################################################################################
