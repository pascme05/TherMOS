#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         adapter.py
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
Connection of the GUI to TherMOS. This is the only file (together with gui/core/help.py) that is specific to the
toolkit: names, config workbook sheets, run settings of start.py, operating modes (one per model family), input
folders and the call of the simulation. All other GUI modules are generic and read their settings from here.

TherMOS specifics:
    - setup is flat (setup['name'], setup['trFile'], setup['selPO'], ...); the GUI keeps it in cfg['Run'].
    - the config workbook (Exp, Dat, Mdl, Par) is read by loadPara inside main → FEED = 'xlsx'.
    - one mode per model: the mode sets its own sel flag (with its sub-mode value) and all others to 0.
    - main saves and closes its figures (saveFigs); during a GUI run the figures are kept open for the gallery.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import copy
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#######################################################################################################################
# Names
#######################################################################################################################
NAME = 'TherMOS'                                                      # short name (page title, JSON meta)
TITLE = 'Thermal Model Order Reduction and Simulation'                # full name (Setup page)
TAGLINE = "Set up the data here, then open a model in the sidebar. Every model order reduction is one mode; the " \
          "Theory & Tools section has the loss and thermal stack calculators, the MOR principles and the board " \
          "designer. Parameters are stored as JSON config (sidebar), the data are read from data/."
SRC_PATHS = ['']                                                      # added to sys.path (relative to ROOT)
LOGO = None                                                           # e.g. 'docu/logo.png' (optional)
DOC = 'docu/GUI.md'                                                   # user guide shown on the Documentation page

#######################################################################################################################
# Config workbook (Name | Category | Description | Variable | Type | Value | Unit)
#######################################################################################################################
CONFIG_DIR = 'config/gui'                                             # JSON configs of the GUI and _gui_run.xlsx
CONFIG_WORKBOOK = 'config/default/default.xlsx'                       # source of schema and defaults (= start.py)
SHEETS = {
    'Exp': ('Exp', True),
    'Dat': ('Dat', True),
    'Mdl': ('Mdl', True),
    'Par': ('Par', True),
}
def _ints():
    """Parameters with Type 'integer'/'integer+' in the template workbook (TherMOS workbooks have a Type column)."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(os.path.join(ROOT, CONFIG_WORKBOOK), read_only=True, data_only=True)
        out = set()
        for sheet, (section, _) in SHEETS.items():
            for r in wb[sheet].iter_rows(min_row=2, values_only=True):
                if len(r) > 4 and r[3] and str(r[4]).startswith('integer'):
                    out.add((section, str(r[1] or section), str(r[3])))
        return out
    except Exception:
        return set()


INTS = _ints()                                                        # {(section, cat, var)} forced integer
STRS = set()                                                          # {(section, cat, var)} forced text
NO_OPTIONS = {('Mdl', 'dl', 'seq')}                                   # descriptions enumerate cases, value continuous
OPTIONAL = set()                                                      # empty value allowed
OPTIONS = {('Dat', 'gen', 'dOut'): {3: '3-dimensional'}}                         # options the code accepts beyond the workbook
LABELS = {}                                                           # {(section, cat, var): name} if Name is empty
PERCENT_AS_FRACTION = True                                            # unit '%': stored 0.05, shown 5 %; False: 5

#######################################################################################################################
# Run settings (formerly start.py)
#######################################################################################################################
RUN_SECTION = 'Run'                                                   # section of the GUI config holding setup
CONFIG_VAR = 'config'                                                 # run setting naming the config workbook
MODE_VAR = 'mode'                                                     # run setting of the operating mode
SEL = {'RC': 'selRC', 'SS': 'selSS', 'SF': 'selSF', 'PO': 'selPO', 'PS': 'selPS', 'ML': 'selML', 'DL': 'selDL',
       'PI': 'selPI'}
RUN = {'user': 'Pascal Schirmer', 'name': 'default', 'config': '/default/default', 'mode': 'SS',
       'plotOut': 1, 'plotMdl': 0, 'featRank': 0, 'save': 0, 'train': 1, 'test': 1,
       'format': 'mat', 'datDim': 1, 'datSep': 1,
       'trFile': ['default'], 'teFile': 'default', 'vlFile': ['default'],
       'trSheet': ['train-1', 'train-2', 'train-3'], 'teSheet': 'test', 'vlSheet': ['val'],
       'trID': [], 'teID': [], 'vlID': [], 'rTr': 0.7, 'rVl': 0.2,
       'inp': ['P1', 'P2', 'P3'], 'out': ['T1', 'T2', 'T3'], 'ref': ['R1', 'R2', 'R3'],
       'selRC': 1, 'selSS': 1, 'selSF': 1, 'selPO': 1, 'selPS': 1, 'selML': 1, 'selDL': 1, 'selPI': 3,
       'featRoll': 0, 'EWMA': [5, 50], 'EWMS': [5, 50], 'legacy': 0}
# (var, kind, label, help): kind 'text', 'toggle', 'int', 'select:a|b|c', 'expert:<kind>'; the data settings
# (files, sheets, IDs, ratios, channels) have their own form on the Setup page (gui/views/project.py)
RUN_FIELDS = [
    ('name', 'text', "Simulation name", "Name of the simulation; figures and models are stored under results/<name>/ "
                                        "and mdl/mdl_<model>_<name>.pkl."),
    ('user', 'text', "User", "Name of the user (only printed)."),
    ('train', 'toggle', "Train the model", "Off: the model mdl/mdl_<model>_<name>.pkl of an earlier run is loaded."),
    ('test', 'toggle', "Test the model", "Predict the testing data and calculate the error metrics."),
    ('save', 'toggle', "Save output data", "Write result_*.pkl into results/<name>/."),
    ('plotMdl', 'toggle', "Plot model structure", "Figures of the model itself (modes, poles, Foster pairs, ...)."),
    ('plotOut', 'select:0|1|2', "Plot model output", "0: no output figures, 1: all outputs in one figure, "
                                                     "2: one figure per output."),
    ('legacy', 'select:0|1', "Numerics", "0: corrected numerics of the Python port (default), 1: behaviour of the "
                                         "MATLAB version (see docu/pythonPort.md)."),
    ('featRank', 'expert:toggle', "Feature ranking", "Box plot, correlation and mutual information of the inputs."),
    ('featRoll', 'expert:toggle', "Rolling features (ML/DL)", "Adds EWMA/EWMS features of the inputs (ML, DL)."),
]
DATA_FIELDS = ('format', 'datDim', 'datSep', 'trFile', 'teFile', 'vlFile', 'trSheet', 'teSheet', 'vlSheet', 'trID',
               'teID', 'vlID', 'rTr', 'rVl', 'inp', 'out', 'ref')

#######################################################################################################################
# Operating modes
#######################################################################################################################
# mode -> title, config section of its parameters, extra sections, icon, description, model variant (run setting,
# options) 'pick',
# Mdl categories shown in its Parameters view, default setup script (run settings + config of the mode)
MODES = {
    'RC': dict(title='RC network (Foster)', section='Mdl', extra=['Par'], icon=':material/electrical_services:',
               text="Thermal RC network: K Foster pairs (Rth, τ) per output fitted to the measured temperature "
                    "response. Fast, physical, but only for the operating point it was fitted on.",
               cats=['gen', 'rc'], script='setup/tutorial/tutorial_RC.py'),
    'SS': dict(title='State space', section='Mdl', extra=['Par'], icon=':material/linear_scale:',
               text="Linear state-space model x' = A·x + B·u, T = C·x + D·u identified from the data (subspace "
                    "N4SID/MOESP + output-error fit). Multi-input multi-output, cross-coupling included.",
               cats=['gen', 'ss'], script='setup/tutorial/tutorial_SS.py'),
    'SF': dict(title='Nodal Foster (SF)', section='Mdl', extra=['Par'], icon=':material/hub:',
               text="Nodal Foster model of M coupled nodes C·dT/dt = P - G·T ('structure function' model of the "
                    "MATLAB version): Foster pairs from the self-heating curves, coupling from the steady-state Rth "
                    "matrix. Needs one single-source experiment per node. Experimental (F-F1).",
               cats=['gen', 'rc'], script='startRth.py'),
    'PO': dict(title='POD-Galerkin (2D)', section='Mdl', extra=['Dat', 'Exp'], icon=':material/grid_on:',
               text="Proper orthogonal decomposition of 2D temperature fields: K spatial modes from the snapshots, "
                    "heat equation projected onto them (Galerkin) and calibrated with the training data.",
               cats=['gen', 'pod'], script='setup/tutorial/tutorial_PO.py'),
    'PS': dict(title='POD state space (2D/3D)', section='Mdl', extra=['Dat', 'Exp'], icon=':material/view_in_ar:',
               text="POD modes as states of an identified state-space model: the modal coordinates are fitted with "
                    "the SS identification. Works for 2D and 3D fields.",
               pick=('selPS', {1: '2D field', 2: '3D field'}), cats=['gen', 'pod', 'ss'], script='start3D.py'),
    'ML': dict(title='Machine learning', section='Mdl', extra=['Par'], icon=':material/scatter_plot:',
               text="Regression of the temperatures from the (rolling) input features: linear, tree, SVR, GPR, "
                    "ensemble or MLP. No dynamics unless rolling features are used.",
               pick=('selML', {1: 'Linear regression', 2: 'Regression tree', 3: 'Support vector regression',
                              4: 'Gaussian process', 5: 'Ensemble', 6: 'Neural network (MLP)'}),
               cats=['gen', 'knn', 'dt', 'svm', 'gpr', 'en', 'nn'], script='setup/tutorial/tutorial_ML.py'),
    'DL': dict(title='Deep learning', section='Mdl', extra=['Par'], icon=':material/neurology:',
               text="Sequence networks (CNN, LSTM, CNN+LSTM) on windows of the inputs, seq2seq or seq2point. "
                    "Training on CPU can take long.",
               pick=('selDL', {1: 'CNN', 2: 'LSTM', 3: 'CNN + LSTM'}), cats=['gen', 'dl'],
               script='setup/tutorial/tutorial_DL.py'),
    'PI': dict(title='Physics-informed DL', section='Mdl', extra=['Par'], icon=':material/science:',
               text="CNN-LSTM with a physics residual of a lumped thermal model in the loss "
                    "(α·data + β·physics). β = 0 trains on the data only.",
               pick=('selPI', {1: 'CNN', 2: 'LSTM', 3: 'CNN + LSTM'}), cats=['gen', 'dl', 'pinn'],
               script='setup/tutorial/tutorial_PI.py'),
}

#######################################################################################################################
# Files
#######################################################################################################################
FOLDERS = {'data': 'data'}                                            # input data (searched recursively)
FILE_EXT = ('.mat', '.xlsx')
FILE_GROUPS = []
GROUP_SELECTOR = None
RESULTS_DIR = 'results'
RESULT_PROJECTS = True                                                # results/<name>/gui_<mode>


def _safe(name):
    return ''.join('_' if c in '<>:"/\|?*' else c for c in str(name).strip()) or 'default'


def results_root(cfg):
    """Results folder of the simulation (results/<name>, the toolkit stores its figures there)."""
    return os.path.join(ROOT, RESULTS_DIR, _safe(cfg[RUN_SECTION].get('name', 'default')))


def run_folders(cfg, mode):
    """Folder of a GUI run: results/<name>/gui_<mode> (one per model, the latest run is kept)."""
    f = os.path.join(results_root(cfg), 'gui_' + str(mode))
    os.makedirs(f, exist_ok=True)
    return [f]


def data_files():
    """Data sets under data/ as names relative to data/ without extension (as accepted by the DataLoader)."""
    out = []
    base = os.path.join(ROOT, 'data')
    for r, _, fs in os.walk(base):
        for f in fs:
            n, e = os.path.splitext(f)
            if e.lower() in FILE_EXT and not f.startswith('~$'):
                out.append((os.path.relpath(os.path.join(r, n), base).replace(os.sep, '/'), e.lower()[1:]))
    return sorted(out)


#######################################################################################################################
# Ready-made configs from the setup scripts
#######################################################################################################################
def script_setup(script):
    """setup dict of a start/setup script (executed without calling main)."""
    import runpy
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        g = runpy.run_path(os.path.join(ROOT, script), run_name='gui')
    finally:
        os.chdir(cwd)
    return copy.deepcopy(g['setup'])


def _scripts():
    out = {}
    for d in ['', 'setup/tutorial', 'setup/journal', 'setup/lecture', 'setup/other']:
        folder = os.path.join(ROOT, d)
        for f in sorted(os.listdir(folder)):
            if f.endswith('.py') and (d or f.startswith('start')) and f != 'start_gui.py':
                out[os.path.splitext(f)[0]] = os.path.join(d, f).replace(os.sep, '/')
    return out


SCRIPTS = _scripts()                                                  # name -> setup script


def mode_of(setup):
    """Mode of a setup script: first active model."""
    return next((m for m, s in SEL.items() if setup.get(s, 0)), 'SS')


def run_settings(setup):
    """Run settings of the GUI from a setup dict (sel values kept as sub-mode of the active model)."""
    G = {k: copy.deepcopy(v) for k, v in setup.items() if k in RUN}
    G['mode'] = mode_of(setup)
    for m, s in SEL.items():
        G[s] = setup.get(s) or RUN[s]
    return G


def _extra():
    out = {}
    for name, script in SCRIPTS.items():
        try:
            s = script_setup(script)
        except Exception as e:                                        # scripts with missing data still give a setup
            print(f"WARN: {script} skipped ({e})")
            continue
        wb = 'config/' + str(s['config']).strip('/').replace('\\', '/') + '.xlsx'
        if os.path.isfile(os.path.join(ROOT, wb)):
            out[name] = (wb, run_settings(s))
    return out


class _Lazy(dict):
    """EXTRA_CONFIGS is only evaluated by build_schema (executing all scripts is slow)."""
    def items(self):
        if not dict.__len__(self):
            self.update(_extra())
        return dict.items(self)


EXTRA_CONFIGS = _Lazy()

#######################################################################################################################
# Toolkit
#######################################################################################################################
FEED = 'xlsx'


def init_setup():
    """Empty setup of the toolkit (start.py starts from {})."""
    return {RUN_SECTION: {}}


def paths():
    """Path variable of the toolkit (repo root, initPath uses os.getcwd() without argument)."""
    from src.py.general.initPath import initPath
    return initPath(ROOT)


def cast(setup):
    return setup


def flat_setup(setup, mode):
    """Flat TherMOS setup of a GUI setup: only the model of the mode is active."""
    G = copy.deepcopy(setup[RUN_SECTION])
    for m, s in SEL.items():
        G[s] = (int(G.get(s) or 1) if m == mode else 0)
    if str(G.get('config', '')).strip('/') == '_gui_run':
        G['config'] = '/gui/_gui_run'
    G.pop('mode', None)
    return G


def run(setup, path, mode):
    """Runs main() with the model of the mode; the figures are kept open for the GUI gallery."""
    import matplotlib.pyplot as plt
    import src.py.main as M
    save = M.saveFigs

    def keep(s, p, dpi=150):
        close = plt.close
        plt.close = lambda *a, **k: None
        try:
            return save(s, p, dpi)
        finally:
            plt.close = close

    M.saveFigs = keep
    try:
        return M.main(flat_setup(setup, mode), path)
    finally:
        M.saveFigs = save


def _thin(a, n=2000):
    a = np.asarray(a)
    if a.ndim == 0 or len(a) <= n:
        return a
    return a[::int(np.ceil(len(a) / n))]


def compact(data, mode):
    """Part of the results stored with a run (gui_run.pkl): metrics, thinned predictions, small model data."""
    if not isinstance(data, dict):
        return data
    out = {'results': copy.deepcopy(data.get('results')), 'pred': {}, 'grt': {}, 'mdls': {}}
    for tag, p in (data.get('pred') or {}).items():
        g = data['grt'][tag]
        y = np.asarray(g['y'])
        nodes = y.reshape(len(y), -1).shape[1]
        n = 2000 if nodes <= 50 else max(10, min(200, int(2e7 // (8 * 2 * max(nodes, 1)))))   # fields: ≤ 160 MB
        step = max(1, int(np.ceil(len(y) / n)))
        out['pred'][tag] = {'y': np.asarray(p['y'])[::step], 'Ts': g.get('Ts', 1) * step}
        D = g.get('Data')
        D = {k: v for k, v in D.items() if k in ('geo', 'k', 'rho', 'Cp', 'hc', 'dx', 'dy', 'dz', 'Lx', 'Ly', 'Lz')} \
            if isinstance(D, dict) else None
        out['grt'][tag] = {'y': y[::step], 'Ts': g.get('Ts', 1) * step, 'Data': D}
    for tag, m in (data.get('mdls') or {}).items():
        out['mdls'][tag] = {k: v for k, v in m.items() if isinstance(v, (int, float, str, list, np.ndarray, dict,
                                                                         np.integer, np.floating))}
    return out


@contextlib.contextmanager
def session():
    """Context of a run (warnings off, as in the start scripts)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
