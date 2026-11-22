#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         journals.py
# Date:         22.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Reproduction of the journal papers from the GUI (pages Journals › TherMOS journal and Journals › PINN journal).
TherMOS journal: the setups setup/journal/*.py have a GUI config (config/gui/<setup>.json) that is loaded like a
tutorial and run in its operating mode. PINN journal: the setups setup/journal_pinn/*.py call mainPINN.main(setup)
(no operating mode of the GUI); their setup values are read from the script (lines setup['x'] = value  # comment),
can be changed and are run as a background process (log in <out>/logs/<setup>.log). The method configs
config/journal_pinn/*.json are shown and edited as tables (value and comment of every parameter).
Command line of a run with changed values: python -m gui.core.journals <setup script> '<json of the changes>'.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PINN_DIR = 'setup/journal_pinn'
PINN_CFG = 'config/journal_pinn'
END_OK, END_ERR = 'DONE:', 'ERROR:'                                   # last line of a finished run (mainPINN)

#######################################################################################################################
# Journals
#######################################################################################################################
# TherMOS journal: setup -> section, title, operating mode of the GUI, description (settings of the setup script)
THERMOS = [
    dict(id='journal_Sec_IV_B_1sec', sec='Sec. IV-B', title="Step response, Ts = 1 s", mode='PO',
         text="POD-Galerkin model of the 2D board, trained on the loss step of 5 W and tested on the step of 20 W, "
              "sampling time 1 s (data step_Pv_5W_Ts1sec, step_Pv_20W_Ts1sec, config journal/TherMOS_1sec)."),
    dict(id='journal_Sec_IV_B_01sec', sec='Sec. IV-B', title="Step response, Ts = 0.1 s", mode='PO',
         text="As above with the sampling time 0.1 s (data step_Pv_5W_Ts01sec, step_Pv_20W_Ts01sec, config "
              "journal/TherMOS_01sec)."),
    dict(id='journal_Sec_IV_C_01sec', sec='Sec. IV-C', title="Load sequence 1, Ts = 0.1 s", mode='PO',
         text="POD-Galerkin model trained on the 5 W step and tested on the load sequence journal_Seq1, sampling "
              "time 0.1 s (config journal/TherMOS_01sec)."),
    dict(id='journal_Sec_IV_D_1sec', sec='Sec. IV-D', title="Load sequence 2, Ts = 1 s", mode='PO',
         text="POD-Galerkin model trained on the 5 W step and tested on the load sequence journal_Seq2, sampling "
              "time 1 s (config journal/TherMOS_1sec)."),
    dict(id='IMS3D_Total', sec='3D', title="3D IMS board", mode='PS',
         text="POD state-space model of the 3D field of an IMS board (3D variant), trained on 10 W and tested on "
              "20 W (data IMS_10W, IMS_20W, config other/IMS3D_Total)."),
]
THERMOS_DIR = 'setup/journal'

# PINN journal: setup (run order) -> results of the paper, runtime (RTX 3070, estimates of setup/journal_pinn/readme)
PINN = [
    dict(id='journal_pinn_Tab_IV_V_data', res="Tables IV, V; Figs. 4, 5", what="Data sets on 3 meshes", time="~2 h"),
    dict(id='journal_pinn_Tab_VII_nominal', res="Table VII", what="Nominal layout N1, T2, T4: Foster, POD, networks",
         time="~20 h"),
    dict(id='journal_pinn_Tab_X_Fig_10_variable', res="Table X, Fig. 10", what="Variable layouts T1, T3",
         time="~1 h (shared runs)"),
    dict(id='journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium', res="Tables VIII, IX; Figs. 6-9",
         what="Medium mesh details, field map", time="~10 min (after 2, 3)"),
    dict(id='journal_pinn_Fig_11_samples', res="Fig. 11", what="Number of training boards", time="~5 h"),
    dict(id='journal_pinn_Tab_XI_Fig_12_load', res="Table XI, Fig. 12", what="Memory and runtime (idle computer)",
         time="~30 min"),
    dict(id='journal_pinn_Fig_13_layout', res="Fig. 13", what="Layout optimisation", time="~10 min"),
    dict(id='journal_pinn_figures', res="all figures and tables", what="Figure data and PDFs (docu/paper)",
         time="~5 min"),
]
PINN_METHODS = ['foster', 'fosterNom', 'podNom', 'pinn', 'nophys', 'deeponet']
MESHES = ['coarse', 'medium', 'fine']


#######################################################################################################################
# Setup scripts
#######################################################################################################################
def script(sid, folder=PINN_DIR):
    return os.path.join(ROOT, folder, sid + '.py')


_LINE = re.compile(r"^setup\['(\w+)'\]\s*=\s*(.*?)\s*(?:#\s?(.*))?$")


def parse_setup(path):
    """
    Setup values of a setup script without executing it: [{'key', 'value', 'text' (value as written), 'comment',
    'group' (heading of the block), 'change' (comment starts with CHANGE)}] and the description (docstring).
    """
    with open(path, encoding='utf-8') as f:
        src = f.read()
    rows, group, lines = [], '', src.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('# ====') and i + 1 < len(lines) and lines[i + 1].startswith('# ') and \
                not lines[i + 1].startswith('# ===='):
            group = lines[i + 1][2:].strip()
        m = _LINE.match(ln)
        if not m:
            continue
        try:
            v = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        c = (m.group(3) or '').strip()
        rows.append(dict(key=m.group(1), value=v, text=m.group(2), comment=c, group=group,
                         change=c.startswith('CHANGE')))
    return rows, (ast.get_docstring(ast.parse(src)) or '').strip()


def setup_dict(path):
    return {r['key']: r['value'] for r in parse_setup(path)[0]}


def write_setup(path, values):
    """Writes changed values back into the setup script (value replaced, comment kept in its column)."""
    with open(path, encoding='utf-8') as f:
        lines = f.read().split('\n')
    for i, ln in enumerate(lines):
        m = _LINE.match(ln)
        if m and m.group(1) in values:
            s = "setup['%s'] = %s" % (m.group(1), repr(values[m.group(1)]))
            lines[i] = (s.ljust(120) + ' # ' + m.group(3)) if m.group(3) is not None else s
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


#######################################################################################################################
# Results of a PINN setup
#######################################################################################################################
def expected(setup):
    """Result files (relative to setup['out']) that a PINN setup writes (as mainPINN.main), in run order."""
    out = []
    for mesh in setup.get('mesh', []):
        for n in setup.get('nTrain', [24]):
            for m in setup.get('methods', []):
                for s in (['F'] if m.startswith('pod') else setup.get('sens', [])):
                    out.append('%s_%s_%s%s.json' % (m, mesh, s, '' if n == 24 else '_n%d' % n))
    out += ['fieldmap_%s.json' % m for m in setup.get('fieldmap', [])]
    if setup.get('scale'):
        out += ['scale_%s.json' % m for m in setup.get('mesh', [])]
    if setup.get('layout'):
        out.append('layout_%s.json' % setup['layout'])
    return out


def data_sets(setup):
    """Data folders of the meshes generated by a setup (data/board/pfhb/<mesh>)."""
    return ['data/board/pfhb/%s' % m for m in setup.get('data', [])]


def status(setup):
    """[(file, exists)] of the expected results (data folders: exists if not empty)."""
    out = os.path.join(ROOT, setup.get('out', 'results/pfhb'))
    st = [(f, os.path.isdir(os.path.join(ROOT, f)) and bool(os.listdir(os.path.join(ROOT, f))))
          for f in data_sets(setup)]
    return st + [(f, os.path.isfile(os.path.join(out, f))) for f in expected(setup)]


TESTS = ['n1', 't1', 't2', 't3', 't4']                                # testing sets of the paper (Table V)


def summary(name, out='results/pfhb'):
    """Hot-spot RMSE (K) per testing set and training time (s) of a run result, {} if not a run result."""
    try:
        with open(os.path.join(ROOT, out, name), encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    res = {t.upper(): d[t]['hot']['rmse'] for t in TESTS if isinstance(d.get(t), dict) and 'hot' in d[t]}
    if res and 'timeTrain' in d:
        res['train (s)'] = d['timeTrain']
    return res


#######################################################################################################################
# Runs (background process)
#######################################################################################################################
def log_file(setup):
    return os.path.join(ROOT, setup.get('out', 'results/pfhb'), 'logs', '%s.log' % setup.get('name', 'run'))


def command(path, changes=None):
    """Command line of a run: the setup script itself, or this module with the changed values."""
    if not changes:
        return [sys.executable, '-u', path]
    return [sys.executable, '-u', '-m', 'gui.core.journals', path, json.dumps(changes)]


def start(path, changes=None):
    """Starts a setup in the background; console output to the log file. Returns (process, log file)."""
    s = setup_dict(path)
    s.update(changes or {})
    log = log_file(s)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    fh = open(log, 'w', encoding='utf-8')
    flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    p = subprocess.Popen(command(path, changes), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
                         creationflags=flags)
    fh.close()
    return p, log


def read_log(log, n=200):
    """Last n lines of a log and its state: 'none', 'running' (no end line yet), 'done' or 'error'."""
    if not os.path.isfile(log):
        return '', 'none'
    with open(log, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    tail = '\n'.join(lines[-n:])
    last = next((ln for ln in reversed(lines) if ln.strip()), '')
    if last.startswith(END_OK):
        return tail, 'done'
    if 'Traceback' in tail or last.startswith(END_ERR):
        return tail, 'error'
    return tail, 'running'


def _run(path, changes):
    """Entry of a background run: setup of the script + changes, then mainPINN.main."""
    import runpy
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    s = runpy.run_path(path, run_name='gui')['setup']
    s.update(changes)
    from setup.journal_pinn.mainPINN import main
    main(s)


#######################################################################################################################
# Method configs (config/journal_pinn/<name>.json)
#######################################################################################################################
def config_names():
    d = os.path.join(ROOT, PINN_CFG)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith('.json'))


def read_config(name):
    with open(os.path.join(ROOT, PINN_CFG, name + '.json'), encoding='utf-8') as f:
        return json.load(f)


def config_rows(cfg):
    """Parameters of a config as rows {'parameter' (Section.Variable or Mesh.fine.Variable), 'value', 'comment',
    'change'}; the section Info is left out."""
    rows = []

    def walk(d, path):
        for k, v in d.items():
            if k.startswith('_'):
                continue
            if isinstance(v, dict):
                walk(v, path + [k])
            else:
                c = str(d.get('_' + k, ''))
                rows.append({'parameter': '.'.join(path + [k]), 'value': v, 'comment': c, 'change': 'CHANGE' in c})
    walk({k: v for k, v in cfg.items() if k != 'Info'}, [])
    return rows


def set_values(cfg, values):
    """Config with the values {'Section.Variable': value} replaced (comments and order kept)."""
    for p, v in values.items():
        *path, last = p.split('.')
        d = cfg
        for k in path:
            d = d[k]
        d[last] = v
    return cfg


def write_config(name, cfg):
    with open(os.path.join(ROOT, PINN_CFG, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write('\n')


def parse_value(text, old):
    """Value of an edited table cell (Python/JSON literal; plain text for string parameters)."""
    if not isinstance(text, str):
        return text
    try:
        v = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        if isinstance(old, str):
            return text.strip()
        raise ValueError("'%s' is not a valid value" % text)
    if isinstance(old, float) and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    return v


#######################################################################################################################
# Command line
#######################################################################################################################
if __name__ == '__main__':
    _run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
