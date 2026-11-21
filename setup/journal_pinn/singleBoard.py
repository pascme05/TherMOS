#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         singleBoard
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
Worker of the single-board extensions (Section VI-E of the PINN journal paper): one run of src/py/main.py in the PF
mode on the board of the PF tutorial (data/board/<data>_train1…4, _val, _test). The parameters are those of
config/other/FR4 with the overrides of the run (json string as the only argument):
    data    name of the data set (tutorial_pf, tutorial_pfnl, tutorial_pfnl0, tutorial_pfmm)
    Ts      sampling time (s)
    setup   model selection (e.g. {"selPF": 1} operator network, {"selPF": 2} coordinate PINN, {"selPS": 2} POD SS)
    dl      training parameters (Mdl › dl), pinf: parameters of the PF mode (Mdl › pinf), ss/pod: POD state space
Output: one line 'RESULT {json}' per model with MAE, RMSE, MAX (K), the maximum rise (K) and the times (s).
It is started as a separate process by runPaper.py, so that every run starts from a clean state.
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
import src.py.main as M
from src.py.general.initPath import initPath
from src.py.general.loadPara import loadPara

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
# Main Function
#######################################################################################################################
def main(a):
    name = a['data']

    def para(cfg, pth):
        p = loadPara('/other/FR4', pth)
        p['Exp']['gen'].update({'Ts': a.get('Ts', 0.05), 'init': 1})
        p['Dat']['gen'].update({'dOut': 3, 'inpX': 0.008, 'inpY': 0.010, 'inpZ': 0.00161, 'outX': 0.008, 'outY': 0.010,
                                'outZ': 0.00161})
        p['Mdl']['gen']['dev'] = 1
        p['Mdl']['dl'].update(a.get('dl', {}))
        p['Mdl']['pinf'] = dict(a.get('pinf', {}))
        p['Mdl']['ss'].update(a.get('ss', {}))
        p['Mdl'].setdefault('pod', {}).update(a.get('pod', {}))
        return p

    M.loadPara = para
    setup = {'user': 'journal_pinn', 'name': 'journal_pinn_single', 'config': '/other/FR4', 'plotOut': 0, 'plotMdl': 0,
             'save': 0, 'train': 1, 'test': 1, 'format': 'mat', 'datDim': 3, 'datSep': 1,
             'trFile': [name + '_train%d' % i for i in (1, 2, 3, 4)], 'teFile': name + '_test', 'vlFile': [name + '_val'],
             'trSheet': ['a'], 'teSheet': 'a', 'vlSheet': ['a'], 'inp': [], 'out': [], 'ref': [], 'legacy': 0}
    setup.update(a.get('setup', {'selPF': 1}))
    out = M.main(setup, initPath(base))
    for tag, r in out['results'].items():
        y, yp = out['grt'][tag]['y'], out['pred'][tag]['y']
        res = {'model': tag, 'mae': float(np.mean(np.abs(y - yp))), 'rmse': float(np.sqrt(np.mean((y - yp) ** 2))),
               'max': float(np.max(np.abs(y - yp))), 'rise': float(np.max(y) - np.min(y)),
               'timeTrain': float(r['time']['train']), 'timeTest': float(r['time']['test'])}
        print("RESULT " + json.dumps(res), flush=True)


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
