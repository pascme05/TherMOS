#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoardPINN
# Date:         08.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This script generates the multi-output PINN example from the board of the board designer (data/board/tutorial.json,
three chips T1, D1, M1 on an IMS board), as the button 'Export PINN data' of the board designer does:
1) finite-volume model of the board (genBoard, calcBoardMat), step responses of every chip (1 W) and the coupled Cauer
   network of the chips fitted to them (2 stages per chip, die-average junction temperature)
2) training data: random load levels of every chip (sheets train1 … trainN), validation: another random profile,
   testing: the designed profiles of the board (pulse, pulse, sine) with the constant board losses
3) data/board/tutorial_pinn.xlsx, the network data/board/tutorial_pinn.json and the config
   config/gui/tutorial_PI_board.json (PI mode, LSTM, Cauer physics, T_ref = Ta)
The time constants of the board are short (τ_max ≈ 0.7 s): Ts = 0.05 s and load periods of 4 s (random profiles hold
a level for 1 s) give transients in every profile.
Usage:      python setup/tutorial/genBoardPINN.py [name] [number of training profiles] [duration per profile (s)]
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from os.path import dirname, abspath, isdir, join as pjoin
import json
import sys
import warnings

base = dirname(abspath(__file__))
while not isdir(pjoin(base, 'src', 'py')) and dirname(base) != base:
    base = dirname(base)
sys.path.insert(0, base)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from gui.views.tools import board as B
from gui.core import config as conf


#######################################################################################################################
# Main
#######################################################################################################################
def main(name='tutorial', nTr=4, tTr=40.0, Ts=0.05, Nr=2, period=4.0):
    lay = json.load(open(pjoin(base, 'data', 'board', 'tutorial.json'), encoding='utf-8'))
    lay['view'] = 'xz'                                                  # layer-resolved model (as the designer)
    for c in lay['chips']:
        c['period'] = period                                            # load period (random profiles: hold period/4)
    D = genBoard(lay)
    C, G, Bp = calcBoardMat(D)
    names = [c['name'] for c in D['chips']]
    M = len(names)
    q = B._q(D, len(C))
    print("INFO: Board with %d nodes and %d chips (%s)" % (len(C), M, ', '.join(names)))

    # Step responses and Cauer network (die average)
    import scipy.sparse.linalg as spl
    from scipy.sparse.linalg import eigsh
    import scipy.sparse as sp
    TH = np.column_stack([spl.splu(G.tocsc()).solve(Bp[:, j]) for j in range(M)])
    Ravg = Bp.T @ TH
    lam = float(eigsh(G.tocsc(), k=1, M=sp.diags(C).tocsc(), sigma=0, which='LM', return_eigenvectors=False)[0])
    tStep = 5 / lam
    Za = None
    for j in range(M):
        ts, T = B._logSteps(C, G, Bp, j, tStep, lambda x: None)
        Za = np.zeros((len(ts), M, M)) if Za is None else Za
        Za[:, :, j] = T @ Bp
    net = B._nodal(Za[1:], ts[1:], Ravg, Nr)
    print("INFO: R_th matrix (die average, K/W):\n" + np.array2string(Ravg, precision=4))
    print("INFO: Self-heating time constants (63 %%): %s s, tau_max = %.1f s" % (net['tau'], 1 / lam))

    # Profiles
    t = np.arange(0.0, tTr + Ts * 1e-9, Ts)
    jobs = [(f"train{k + 1}", t, B._profiles(lay, D, t, kind='random', seed=11 + 17 * k)) for k in range(nTr)]
    jobs += [('val', t, B._profiles(lay, D, t, kind='random', seed=997)),
             ('test', np.arange(0.0, 2 * tTr + Ts * 1e-9, Ts), None)]
    sheets = {}
    for k, (sh, tt, P) in enumerate(jobs):
        if P is None:
            P = B._profiles(lay, D, tt)
        T, _ = simBoard(C, G, Bp, P, Ts, lay['Ta'], 4, q=q)
        Tj = (T - lay['Ta']) @ Bp + lay['Ta']
        df = pd.DataFrame({'time': tt, 'time_id': tt, 'id': k + 1})
        for i, n in enumerate(names):
            df[f"P_{n}"] = P[:, i]
        df['Ta'] = lay['Ta']
        for i, n in enumerate(names):
            df[f"T_{n}"] = Tj[:, i]
        sheets[sh] = df
        print("INFO: %s: %d samples, T_j max %s °C" % (sh, len(tt), np.array2string(Tj.max(0), precision=1)))
    with pd.ExcelWriter(pjoin(base, 'data', 'board', f"{name}_pinn.xlsx")) as xw:
        for sh, df in sheets.items():
            df.to_excel(xw, sheet_name=sh, index=False)

    # Network and config
    from src.py.model.SF.sfNet import sfNet
    [Gm, Cm, E] = sfNet(net['p'], M, Nr)
    struct = {'type': 'cauer', 'src': [f"x{i + 1}" for i in range(M)],
              'cauer': {'G': np.asarray(Gm).tolist(), 'C': np.asarray(Cm).ravel().tolist(),
                        'E': np.asarray(E).tolist(),
                        'out': [i * Nr + 1 for i in range(M)],
                        'labels': [f"{n}·{k + 1}" for n in names for k in range(Nr)]},
              'origin': f"board designer: coupled Cauer network ({Nr} stage(s) per chip)"}
    json.dump(struct, open(pjoin(base, 'data', 'board', f"{name}_pinn.json"), 'w', encoding='utf-8'), indent=1)
    W = int(min(max(3 * float(np.max(net['tau'])) / Ts, 50), 400))
    cfg = conf.read_config('tutorial_PI')
    cfg['Run'].update({'name': f"{name}_PI_board", 'config': '/gui/_gui_run', 'mode': 'PI', 'format': 'xlsx',
                       'datDim': 1, 'datSep': 2, 'trFile': [f"{name}_pinn"], 'vlFile': [f"{name}_pinn"],
                       'teFile': f"{name}_pinn", 'trSheet': [f"train{k + 1}" for k in range(nTr)], 'vlSheet': ['val'],
                       'teSheet': 'test', 'inp': [f"P_{n}" for n in names] + ['Ta'], 'out': [f"T_{n}" for n in names],
                       'ref': ['Ta'], 'selPI': 2, 'archPI': None, 'physPI': struct, 'featRoll': 0})
    cfg['Exp']['gen'].update({'Ts': Ts, 'init': 1})
    cfg['Mdl']['dl'].update({'W': W, 'stride': max(W // 10, 1), 'epoch': 60, 'batch': 32, 'initLr': 0.002,
                             'valPat': 12, 'gradTh': 1, 'lrDropPr': 20, 'lrDropFa': 0.5, 'shu': 3})
    cfg['Mdl']['pinn'].update({'phys': 2, 'alpha': 1, 'beta': 1, 'gamma': 0, 'warm': max(W // 20, 1), 'x0': 1,
                               'col': 3, 'pred': 1, 'TrefSrc': M + 1, 'node': 1})
    cfg['Mdl']['gen']['dev'] = 1
    conf.save_config(cfg, f"{name}_PI_board")
    print("DONE: data/board/%s_pinn.xlsx, data/board/%s_pinn.json, config/gui/%s_PI_board.json (W = %d)" %
          (name, name, name, W))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else 'tutorial', int(a[1]) if len(a) > 1 else 4, float(a[2]) if len(a) > 2 else 40.0)
