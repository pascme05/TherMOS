#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         runPaper
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
Engine of the setups of the PINN journal paper (setup/journal_pinn/journal_pinn_*.py). Every setup names one json file
of config/journal_pinn/ with a list of steps, which are executed in order:
    data      board families of setup/tutorial/genBoardPFParam.py ('sets'), or the single-board data ('single': true)
    run       training and evaluation of one network (setup/tutorial/studyPFParam.py): 'train' labelled boards,
              physics family 'phys' (null: no physics loss), optional 'data' = [labelled family, profiles per board,
              length (s)], 'cfg' = overrides of the hyperparameters of config/journal_pinn/journal_pinn_net.json
    baseline  fv, pmor, pmorPos or krylov (studyPFParam)
    layout    layout optimisation through the network 'model' (studyPFParam)
    single    one single-board run of src/py/main.py (singleBoard.py, separate process), results in <out>/single.json
    script    a Python script (figure data of docu/paper/figures/scripts); {out} and {folder} in 'args' are replaced
A step whose result exists is skipped (run: <out>/<name>.json, baseline: <out>/<name>.json, layout:
<out>/layout_<model>.json, data: <folder>/<set>.npz, single: entry in <out>/single.json, script: its 'check' file),
so an interrupted setup continues where it stopped.
Options:    --force         recompute existing results
            --dry           list the steps and whether they would run
            --only a,b      only the steps with these names (run, baseline, single) or types
            --epoch N       maximum number of epochs of the runs (quick check, e.g. --epoch 2)
            --out DIR       results folder (default results/pfparam)
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

#######################################################################################################################
# Settings
#######################################################################################################################
CFG = pjoin(base, 'config', 'journal_pinn')
FOLDER = pjoin(base, 'data', 'board', 'pfparam')
BOARD = pjoin(base, 'data', 'board')


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def loadCfg(name):
    """Json config of config/journal_pinn/ without the comment keys (_comment, _help)."""
    with open(pjoin(CFG, name + '.json'), encoding='utf-8') as f:
        return _clean(json.load(f))


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if not k.startswith('_')}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return d


def _single(out):
    p = pjoin(out, 'single.json')
    if isfile(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {}


def _done(s, out):
    """True if the result of step s exists."""
    t = s['type']
    if t in ('run', 'baseline'):
        return isfile(pjoin(out, s['name'] + '.json'))
    if t == 'layout':
        return isfile(pjoin(out, 'layout_%s.json' % s['model']))
    if t == 'data':
        if s.get('single'):
            return all(isfile(pjoin(BOARD, n + '_test.mat')) for n in ('tutorial_pf', 'tutorial_pfnl', 'tutorial_pfnl0',
                                                                       'tutorial_pfmm'))
        return all(isfile(pjoin(FOLDER, n + '.npz')) for n in s['sets'])
    if t == 'single':
        return s['name'] in _single(out)
    if t == 'script':
        return bool(s.get('check')) and isfile(pjoin(base, s['check']))
    return False


def _label(s):
    return s.get('name') or s.get('model') or s.get('script') or ('single-board data' if s.get('single') else
                                                                   ','.join(s.get('sets', [])))


#######################################################################################################################
# Steps
#######################################################################################################################
def _data(s):
    if s.get('single'):
        from setup.tutorial import genBoardPF, genBoardPFStudy
        genBoardPF.main('tutorial_pf', 4, 20.0, 'xz')
        genBoardPFStudy.main('all')
    else:
        from setup.tutorial import genBoardPFParam
        genBoardPFParam.main([n for n in s['sets'] if not isfile(pjoin(FOLDER, n + '.npz'))] or s['sets'])


def _run(s, out, epoch):
    from setup.tutorial import studyPFParam as study
    cfg = dict(loadCfg('journal_pinn_net'), **s.get('cfg', {}))
    cfg['holds'] = tuple(cfg['holds'])
    sel = tuple(s['data']) if s.get('data') else ('train', None, None)
    study.RUNS[s['name']] = (cfg, s['train'], s.get('phys'), sel)
    study.run(s['name'], FOLDER, out, epoch)


def _baseline(s, out):
    from setup.tutorial import studyPFParam as study
    {'fv': study.baseFV, 'pmor': study.basePMOR, 'pmorPos': study.basePMORPos,
     'krylov': study.baseKrylov}[s['name']](FOLDER, out)


def _singleRun(s, out, par, epoch=None):
    net = par.get(s.get('net') or '', {})                                  # operator / coordinate / none (POD)
    a = {'data': s['data'], 'Ts': par.get('Ts', 0.05),
         'dl': {**par.get('dl', {}), **net.get('dl', {}), **s.get('dl', {}), **({'epoch': epoch} if epoch else {})},
         'pinf': dict(net.get('pinf', {}), **s.get('pinf', {})),
         'setup': s.get('setup') or net.get('setup') or {'selPF': 1}, 'ss': s.get('ss', {}), 'pod': s.get('pod', {})}
    r = subprocess.run([sys.executable, pjoin(base, 'setup', 'journal_pinn', 'singleBoard.py'), json.dumps(a)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=base)
    res = [json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith('RESULT ')]
    if r.returncode or not res:
        raise RuntimeError("single-board run %s failed:\n%s" % (s['name'], r.stderr[-2000:]))
    for x in res:
        print("INFO: %s %s: MAE %.2f K, RMSE %.2f K, MAX %.1f K (rise %.1f K), training %.0f s" %
              (s['name'], x['model'], x['mae'], x['rmse'], x['max'], x['rise'], x['timeTrain']))
    rep = _single(out)
    rep[s['name']] = {'spec': a, 'results': res}
    with open(pjoin(out, 'single.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=1)


def _script(s, out):
    p = pjoin(base, s['script'])
    if not isfile(p):
        print("WARN: %s not found (the paper folder docu/paper is not part of the repository), step skipped" % s['script'])
        return
    args = [a.replace('{out}', out).replace('{folder}', FOLDER) for a in s.get('args', [])]
    r = subprocess.run([sys.executable, p] + args, cwd=base)
    if r.returncode:
        raise RuntimeError("%s failed (exit code %d)" % (s['script'], r.returncode))


#######################################################################################################################
# Main Function
#######################################################################################################################
def execute(name, argv=None):
    """Executes the steps of config/journal_pinn/<name>.json (options: see the module description)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out = argv[argv.index('--out') + 1] if '--out' in argv else pjoin(base, 'results', 'pfparam')
    out = abspath(out)
    epoch = int(argv[argv.index('--epoch') + 1]) if '--epoch' in argv else None
    only = argv[argv.index('--only') + 1].split(',') if '--only' in argv else None
    force, dry = '--force' in argv, '--dry' in argv
    os.makedirs(out, exist_ok=True)
    c = loadCfg(name)
    steps = [s for s in c['steps'] if not only or s['type'] in only or s.get('name') in only or s.get('model') in only]

    print("------------------------------------------")
    print("START: %s (%s), %d steps, results in %s" % (c['name'], c['section'], len(steps), out))
    print("------------------------------------------")
    t0 = time.time()
    for i, s in enumerate(steps):
        done = _done(s, out) and not force
        print("STEP %d/%d: %s %s%s" % (i + 1, len(steps), s['type'], _label(s), ' (exists, skipped)' if done else ''))
        if done or dry:
            continue
        ts = time.time()
        if s['type'] == 'data':
            _data(s)
        elif s['type'] == 'run':
            _run(s, out, epoch)
        elif s['type'] == 'baseline':
            _baseline(s, out)
        elif s['type'] == 'layout':
            from setup.tutorial import studyPFParam as study
            study.optLayout(FOLDER, out, s['model'])
        elif s['type'] == 'single':
            _singleRun(s, out, c.get('single', {}), epoch)
        elif s['type'] == 'script':
            _script(s, out)
        print("INFO: step %s finished in %.0f s" % (_label(s), time.time() - ts))
    print("DONE: %s in %.0f s" % (c['name'], time.time() - t0))
