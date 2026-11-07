#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         study.py
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
Engine of the Study page (no Streamlit): parameter studies on top of the toolkit, i.e. repeated runs of main() with
changed configurations. The toolkit itself has no grid search or optimiser.
- tunable(): numeric model parameters of a mode (schema, Mdl categories of the mode)
- study_cfg(): config of one evaluation (train + test, nothing saved, no plots; optionally the validation data as
  testing data so that the objective is not measured on the test set)
- evaluate(): one run → objective and all metrics
- grid(): all combinations of value lists
- RandomLocal: built-in optimiser (random search, then coordinate search around the best point, seed 42)
- OptunaOpt: TPE sampler of Optuna (optional dependency, same ask/tell interface)
- forward(): greedy forward selection of input channels
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import copy
import itertools
import math
import time

import numpy as np

from gui import adapter as A
from gui.core.config import slot

#######################################################################################################################
# Settings
#######################################################################################################################
METRICS = {'MAE': 'min', 'RMSE': 'min', 'MAX': 'min', 'FIT': 'max'}
NO_TUNE = {('gen', 'sol'), ('gen', 'opt'), ('ss', 'sol'), ('ss', 'init'), ('ss', 'noise'), ('dl', 'loss'),
           ('dl', 'valFreq'), ('dl', 'shu'), ('gen', 'nSub')}


#######################################################################################################################
# Parameters
#######################################################################################################################
def tunable(schema, mode):
    """Numeric parameters of the mode's Mdl categories: [(key 'Mdl.cat.var', schema entry)]."""
    cats = A.MODES[mode].get('cats') or []
    out = []
    for p in schema.get('Mdl', []):
        if p['cat'] in cats and p['type'] in ('int', 'float') and (p['cat'], p['var']) not in NO_TUNE:
            out.append((f"Mdl.{p['cat']}.{p['var']}", p))
    return out


def get_value(cfg, key, schema):
    sec, cat, var = key.split('.')
    p = next(q for q in schema[sec] if q['cat'] == cat and q['var'] == var)
    return slot(cfg, sec, p).get(var, p['default'])


def set_values(cfg, changes, schema):
    """Copy of cfg with the changes {'Mdl.cat.var' | 'Run.var': value}."""
    cfg = copy.deepcopy(cfg)
    for key, v in changes.items():
        parts = key.split('.')
        if parts[0] == A.RUN_SECTION:
            cfg[A.RUN_SECTION][parts[1]] = v
            continue
        sec, cat, var = parts
        p = next(q for q in schema[sec] if q['cat'] == cat and q['var'] == var)
        slot(cfg, sec, p)[var] = int(round(v)) if p['type'] == 'int' else float(v)
    return cfg


#######################################################################################################################
# Evaluation
#######################################################################################################################
def study_cfg(cfg, mode, use_val=True):
    """
    Config of one evaluation; returns (cfg, note). With use_val the testing set is replaced by the (first) validation
    set: file → teFile = vlFile[0], sheet → teSheet = vlSheet[0], ID → teID = vlID; not possible for a ratio split.
    """
    cfg = copy.deepcopy(cfg)
    G = cfg[A.RUN_SECTION]
    G.update({'train': 1, 'test': 1, 'save': 0, 'plotOut': 0, 'plotMdl': 0, 'featRank': 0,
              'name': f"study_{mode}_{G.get('name', '')}"})
    note = 'testing data'
    if use_val:
        sep = int(G.get('datSep', 1))
        if sep == 1 and G.get('vlFile'):
            G['teFile'] = list(G['vlFile'])[0]
            note = f"validation data (file {G['teFile']})"
        elif sep == 2 and G.get('vlSheet'):
            G['teSheet'] = list(G['vlSheet'])[0]
            note = f"validation data (sheet {G['teSheet']})"
        elif sep == 3 and G.get('vlID'):
            G['teID'] = list(G['vlID'])
            note = f"validation data (IDs {', '.join(map(str, G['teID']))})"
        else:
            note = 'testing data (a ratio split has no separate validation set)'
    return cfg, note


def evaluate(cfg, mode, schema, metric='MAE'):
    """One run of the mode: dict(obj, metrics {MAE, RMSE, MAX, FIT}, time, error)."""
    from gui.core.runner import run_mode
    t0 = time.perf_counter()
    run = run_mode(cfg, mode, schema)
    out = {'time': time.perf_counter() - t0, 'error': run.get('error'), 'obj': math.nan, 'metrics': {}}
    if run.get('error') or not isinstance(run.get('data'), dict):
        out['error'] = (run.get('error') or 'no results')[0:400]
        return out
    try:
        res = run['data']['results']
        err = res[next(iter(res))]['err']['tot']
        out['metrics'] = {k: float(np.nanmean(np.asarray(err[k], dtype=float))) for k in METRICS if k in err}
        out['obj'] = out['metrics'].get(metric, math.nan)
    except Exception as e:                                                       # result without error metrics
        out['error'] = f"no metrics: {e}"
    return out


def better(a, b, metric):
    """True if objective a is better than b."""
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return not (isinstance(a, float) and math.isnan(a))
    if isinstance(a, float) and math.isnan(a):
        return False
    return a > b if METRICS.get(metric, 'min') == 'max' else a < b


#######################################################################################################################
# Grid search
#######################################################################################################################
def parse_values(txt, kind):
    """'1, 2, 4' or '1:8' (int range) or '1e-4:1e-2:5' (5 values, log spaced if both > 0 and ratio ≥ 10)."""
    txt = str(txt).strip()
    if ':' in txt:
        a = [float(x) for x in txt.split(':')]
        if len(a) == 2 and kind == 'int':
            return list(range(int(a[0]), int(a[1]) + 1))
        n = int(a[2]) if len(a) > 2 else 5
        if a[0] > 0 and a[1] / a[0] >= 10:
            v = np.geomspace(a[0], a[1], n)
        else:
            v = np.linspace(a[0], a[1], n)
        return sorted({int(round(x)) for x in v}) if kind == 'int' else [float(x) for x in v]
    v = [float(x) for x in txt.replace(';', ',').split(',') if x.strip()]
    return [int(round(x)) for x in v] if kind == 'int' else v


def grid(specs):
    """specs {key: [values]} → list of change dicts (all combinations)."""
    keys = list(specs)
    return [dict(zip(keys, c)) for c in itertools.product(*[specs[k] for k in keys])]


#######################################################################################################################
# Optimisers (ask / tell)
#######################################################################################################################
class RandomLocal:
    ###################################################################################################################
    # Constructor
    ###################################################################################################################
    def __init__(self, space, metric, budget, seed=42):
        """space {key: (lo, hi, 'int'|'float', log)}"""
        self.space, self.metric, self.budget = space, metric, int(budget)
        self.rng = np.random.default_rng(seed)
        self.nRand = max(3, self.budget // 2)
        self.hist = []
        self.best = None
        self.step = {k: 0.25 for k in space}                                   # fraction of the (log) range
        self.queue = []

    ###################################################################################################################
    # Helpers
    ###################################################################################################################
    def _from_u(self, k, u):
        lo, hi, kind, log = self.space[k]
        u = min(max(u, 0.0), 1.0)
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if log else lo + u * (hi - lo)
        return int(round(v)) if kind == 'int' else float(v)

    def _to_u(self, k, v):
        lo, hi, kind, log = self.space[k]
        if log:
            return (math.log(v) - math.log(lo)) / max(math.log(hi) - math.log(lo), 1e-12)
        return (v - lo) / max(hi - lo, 1e-12)

    def _seen(self, x):
        return any(h[0] == x for h in self.hist)

    ###################################################################################################################
    # Ask / Tell
    ###################################################################################################################
    def ask(self):
        if len(self.hist) < self.nRand or self.best is None:
            for _ in range(50):
                x = {k: self._from_u(k, self.rng.uniform()) for k in self.space}
                if not self._seen(x):
                    return x
            return x
        while not self.queue:
            if all(s < 1e-3 for s in self.step.values()):
                self.step = {k: 0.25 for k in self.space}
            for k in self.space:
                for sgn in (1, -1):
                    x = dict(self.best[0])
                    x[k] = self._from_u(k, self._to_u(k, x[k]) + sgn * self.step[k])
                    if not self._seen(x) and x not in self.queue:
                        self.queue.append(x)
            if not self.queue:
                self.step = {k: s / 2 for k, s in self.step.items()}
        return self.queue.pop(0)

    def tell(self, x, obj):
        self.hist.append((x, obj))
        if self.best is None or better(obj, self.best[1], self.metric):
            self.best = (x, obj)
            self.queue = []                                                 # restart the local search at the best
        elif len(self.hist) >= self.nRand and not self.queue:
            self.step = {k: s / 2 for k, s in self.step.items()}


class OptunaOpt:
    ###################################################################################################################
    # Constructor
    ###################################################################################################################
    def __init__(self, space, metric, budget, seed=42):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.space, self.metric = space, metric
        self.study = optuna.create_study(direction='maximize' if METRICS.get(metric) == 'max' else 'minimize',
                                         sampler=optuna.samplers.TPESampler(seed=seed))
        self.trial = None
        self.best = None

    ###################################################################################################################
    # Ask / Tell
    ###################################################################################################################
    def ask(self):
        self.trial = self.study.ask()
        x = {}
        for k, (lo, hi, kind, log) in self.space.items():
            if kind == 'int':
                x[k] = self.trial.suggest_int(k, int(lo), int(hi), log=bool(log) and lo > 0)
            else:
                x[k] = self.trial.suggest_float(k, float(lo), float(hi), log=bool(log) and lo > 0)
        return x

    def tell(self, x, obj):
        import optuna
        if isinstance(obj, float) and math.isnan(obj):
            self.study.tell(self.trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self.trial, obj)
        if self.best is None or better(obj, self.best[1], self.metric):
            self.best = (x, obj)


def optuna_available():
    try:
        import optuna                                                                   # noqa: F401
        return True
    except ImportError:
        return False


#######################################################################################################################
# Feature selection
#######################################################################################################################
def forward_next(candidates, chosen):
    """Next feature sets of a greedy forward selection: chosen + one of the remaining candidates."""
    return [chosen + [c] for c in candidates if c not in chosen]
