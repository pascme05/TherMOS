#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         mainPINN
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
Main function of the setups of the PINN journal paper (setup/journal_pinn/journal_pinn_*.py, half-bridge module). It
executes the parts of a setup in this order: data, runs (every method on every mesh, sensor version and number of
training boards), memory and computational load, layout optimisation, figure scripts. The parameters are read from
config/journal_pinn/<name>.json (sections as in config/gui/start.json, '_<Variable>' keys are comments):
data.json (meshes, data sets, operating points) and one file per method (foster, pod, pinn, nophys, deeponet); the
methods fosterNom and podNom are the baselines fitted on the nominal layout (trainNom) with the parameters of foster and pod.
The section Mesh of a method config (e.g. "fine": {"nR": 2048}) replaces values on that mesh.
Runs whose results exist are skipped (setup['skip'] = 1); missing test sets of existing network runs are added.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Path
# ==============================================================================
from os.path import dirname, abspath, isdir, isfile, join as pjoin
import sys

base = dirname(abspath(__file__))
while not isdir(pjoin(base, 'src', 'py')) and dirname(base) != base:
    base = dirname(base)
sys.path.insert(0, base)

# ==============================================================================
# External
# ==============================================================================
import os
import json
import time
import subprocess
import warnings

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
DATA = pjoin(base, 'data', 'board', 'pfhb')                             # data sets of the paper
FIGS = pjoin(base, 'docu', 'paper', 'figures')                          # paper figures (not in the repository)
NET = ('K', 'width', 'depth', 'nF', 'tauCap', 'alpha', 'beta', 'gamma', 'epoch', 'pat', 'nStep', 'lr', 'clip', 'seed',
       'Ts', 'nbD', 'nbP', 'nR', 'nbS', 'tP', 'nCache', 'nNew', 'nDense')


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def readCfg(name):
    """Config config/journal_pinn/<name>.json as it is (sections as in config/gui/start.json)."""
    with open(pjoin(base, 'config', 'journal_pinn', name + '.json'), encoding='utf-8') as f:
        return json.load(f)


def loadCfg(name):
    """
    Parameters of config/journal_pinn/<name>.json as {Variable: Value}: the sections (Net, Loss, ...) are merged, keys
    starting with '_' are comments; the section Mesh gives '<Variable>_<mesh>' (e.g. nR_fine), which replaces
    <Variable> on that mesh (runMethod).
    """
    out = {}
    for sec, d in readCfg(name).items():
        if sec == 'Info' or sec.startswith('_') or not isinstance(d, dict):
            continue
        if sec == 'Mesh':
            out.update({'%s_%s' % (k, m): v for m, dm in d.items() if isinstance(dm, dict)
                        for k, v in dm.items() if not k.startswith('_')})
        else:
            out.update({k: v for k, v in d.items() if not k.startswith('_')})
    return out


def applyData():
    """Sets the meshes, set sizes and operating points of config/journal_pinn/data.json in the toolkit modules."""
    from src.py.model.PF import pfParam
    from setup.tutorial import genBoardPFHB as g
    c = loadCfg('data')
    for m, d in readCfg('data')['Mesh'].items():
        if not m.startswith('_'):
            pfParam.MESH[m] = {'dx': d['dx'], 'dzMax': d['dz'] * 1e-3, 'rowMax': int(d['row'])}
    g.SIZES.update({'train': c['nTrain'], 'val': c['nVal'], 'phys': c['nPhys'], 't1': c['nT1'], 'nProf': c['nProf']})
    g.TS, g.T_END, g.T4_END, g.DEC = c['Ts'], c['tEnd'], c['t4End'], int(c['dec'])
    g.OP.update({'Vdc': c['Vdc'], 'fs': c['fs'], 'Tj': c['Tj']})
    g.SEG.update({'I': (0.0, c['Imax']), 'fo': (c['foMin'], c['foMax']), 'hold': (c['holdMin'], c['holdMax'])})
    return c


def _script(name, args):
    p = pjoin(FIGS, 'scripts', name + '.py')
    if not isfile(p):
        print("WARN: %s not found (paper folder docu/paper is not part of the repository), skipped" % p)
        return
    if subprocess.run([sys.executable, p] + args, cwd=base).returncode:
        raise RuntimeError("%s failed" % name)


#######################################################################################################################
# Parts
#######################################################################################################################
def runMethod(method, mesh, sens, nTr, out, epoch=0):
    """
    One run of a method (foster, fosterNom, pod, podNom, pinn, nophys, deeponet) on a mesh with a sensor version and
    nTr training boards.
    """
    from setup.tutorial import studyPFHB as st
    nom = method.endswith('Nom')
    c = loadCfg(method[:-3] if nom else method)
    c.update({k[:-len(mesh) - 1]: v for k, v in c.items() if k.endswith('_' + mesh)})      # mesh-specific values
    if method.startswith('foster'):
        st.runFoster(mesh, sens, int(c['K']), nTr, out, nom)
    elif method.startswith('pod'):
        st.runPOD(mesh, c['energy'], nTr, out, int(c['rMax']), nom)
    else:
        cfg = {k: c[k] for k in NET}
        cfg.update({k: c[k] for k in ('arch', 'hid', 'layers') if k in c})                   # deeponet
        cfg.update(rel=bool(c['rel']), holds=(c['holdMin'], c['holdMax']))
        st.runNet(method, mesh, sens, cfg, nTr, out, epoch, bool(c['phys']))


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(setup):
    from setup.tutorial import studyPFHB as st
    out = pjoin(base, setup['out'])
    skip, epoch = setup['skip'], setup['epoch']
    os.makedirs(out, exist_ok=True)
    print("------------------------------------------")
    print("START: %s (results in %s)" % (setup['name'], setup['out']))
    print("------------------------------------------")
    t0 = time.time()
    applyData()

    # data
    if setup['data']:
        from setup.tutorial import genBoardPFHB
        genBoardPFHB.main(setup['data'], None, bool(skip))

    # runs: method x mesh x sensor version x number of training boards (POD needs the full field: one run per mesh)
    for mesh in setup['mesh']:
        for nTr in setup['nTrain']:
            for method in setup['methods']:
                for sens in (['F'] if method.startswith('pod') else setup['sens']):
                    name = st.runName(method, mesh, sens, nTr)
                    if skip and isfile(pjoin(out, name + '.json')):
                        print("INFO: run %s exists, skipped" % name)
                        if method in ('pinn', 'nophys', 'deeponet'):
                            st.addTests(name, out)
                        continue
                    runMethod(method, mesh, sens, nTr, out, epoch)

    # field map of a nominal testing board (FV and errors of POD and the networks)
    for mesh in setup.get('fieldmap', []):
        if not (skip and isfile(pjoin(out, 'fieldmap_%s.json' % mesh))):
            st.fieldMap(mesh, out)

    # memory and computational load
    if setup['scale']:
        for mesh in setup['mesh']:
            if not (skip and isfile(pjoin(out, 'scale_%s.json' % mesh))):
                st.scale(mesh, out)

    # layout optimisation
    if setup['layout'] and not (skip and isfile(pjoin(out, 'layout_%s.json' % setup['layout']))):
        st.optLayout(setup['layout'], setup['layout'].split('_')[1], out, setup['nStart'], setup['nIter'],
                     setup['nRand'])

    # figure data and figures
    for f in setup['figs']:
        _script(f, [out])
    if setup['build']:
        _script('buildfigs', [])
    print("DONE: %s in %.0f s" % (setup['name'], time.time() - t0))
