#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_gui_pf
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
Tests of the PF mode in the GUI: every parameter of Mdl › pinf is explained (workbook description, schema and extended
help), the mode is registered everywhere, the plain-language findings of a run, and the pages of the mode (workflow,
parameters, theory, run, interactive results of a short training) render without an error (Streamlit AppTest).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfData import DEFAULTS
from src.py.model.PF.pfFit import pfFit
from src.py.model.PF.pfSol import pfSol
from src.py.general.performance import performance
from tests.test_pf import _split, _para

# ==============================================================================
# External
# ==============================================================================
import json
import os
import pickle
import numpy as np
import pytest

st = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _mode_page(view):
    import streamlit as st
    from gui.core import config as conf
    from gui.core import forms
    from gui.views import mode
    if 'cfg' not in st.session_state:
        forms.set_config(conf.read_config('tutorial_PF'), 'tutorial_PF')
        st.session_state.setdefault('runs', {})
    st.session_state['view|PF'] = view
    mode.make_page('PF')()


def _result_page(file):
    import pickle
    from gui.core.viz_thermos import RENDER
    with open(file, 'rb') as f:
        data = pickle.load(f)
    RENDER['PF'](data, {'Dat': {'gen': {'inpZ': 0.0}}, 'Run': {}}, 'res|PF')


#######################################################################################################################
# Tests
#######################################################################################################################
def test_parameters_explained():
    from gui.core import help as H
    schema = json.load(open(os.path.join(ROOT, 'gui', 'schema.json'), encoding='utf-8'))
    ps = {p['var']: p for p in schema['Mdl'] if p['cat'] == 'pinf'}
    assert set(ps) == set(DEFAULTS)                                     # workbook, schema and code agree
    for var, p in ps.items():
        assert len(p['desc']) > 30 and p['name'] != var, var           # name and description from the workbook
        assert len(H.HELP['Mdl.pinf.' + var]) > 60, var                 # extended help of the GUI
        assert p['default'] == DEFAULTS[var], var
    assert ps['phys']['options'] and ps['branch']['options'] and ps['col']['options']


def test_mode_registered():
    from gui import adapter as A
    from gui.core import help as H, theory, workflow, viz_thermos, viz_inputs
    assert A.SEL['PF'] == 'selPF' and A.RUN['selPF'] == 1 and 'pinf' in A.MODES['PF']['cats']
    assert all('PF' in d for d in (theory.THEORY, workflow.WORKFLOWS, viz_thermos.RENDER, viz_inputs.INPUTS))
    cfg = {'Run': {'mode': 'PF', 'selPF': 2}, 'Mdl': {'pinf': {'phys': 1, 'beta': 1}}}
    assert H.RELEVANT['Mdl']['pinf'](cfg) and H.RELEVANT['Mdl']['dl'](cfg) and not H.RELEVANT['Mdl']['pinn'](cfg)
    assert not H.ACTIVE['Mdl.pinf.K'](cfg) and H.ACTIVE['Mdl.pinf.steps'](cfg)      # coordinate network
    assert os.path.isfile(os.path.join(ROOT, A.MODES['PF']['script']))


def test_findings():
    from gui.core.viz_thermos import pf_findings
    info = {'dT': 100.0, 'alpha': 1.0, 'beta': 1.0, 'lossData': [1e-2, 1e-4], 'lossPhys': [1e-1, 2e-4],
            'valLoss': [4.0, 1.0]}
    m = {'var': 1, 'geo': np.zeros((10, 2))}
    f = dict(pf_findings(info, m, {'MAE': np.array([0.5, 1.5]), 'MAX': np.array([3.0, 9.0])}))
    assert '1 K = 1.0 % of the temperature scale' in f['Verdict'] and '9 K' in f['Verdict']
    assert 'reproduced well' in f['Verdict'] and 'balanced' in f['Physics'] and 'more epochs' in f['Training']
    f = dict(pf_findings(dict(info, beta=0.0), dict(m, var=2, sens=np.arange(4), sensVal=np.arange(1)),
                         {'MAE': np.array([8.0]), 'MAX': np.array([20.0])}))
    assert 'does not describe' in f['Verdict'] and 'No physics loss' in f['Physics'] and '5 sensor' in f['Sensors']
    f = dict(pf_findings(info, m, {'MAE': np.array([4.0]), 'MAX': np.array([9.0])}))
    assert 'collocation' in f['Action']                                 # test error far above the validation error


@pytest.mark.parametrize('view', ['Workflow', 'Parameters', 'Theory', 'Run'])
def test_mode_pages(view):
    at = AppTest.from_function(_mode_page, kwargs={'view': view}, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error or view == 'Run', [e.value for e in at.error]   # Run: the tutorial data may be missing


def test_results_page(tmp_path):
    from gui import adapter as A
    tr = _split(tmp_path, 'xz', 'tr', 'random', 3)
    te = _split(tmp_path, 'xz', 'te', 'pulse', 0)
    para = _para({'phys': 2, 'K': 4, 'width': 16, 'depth': 2, 'warm': 20, 'nCol': 64, 'nDat': 256}, epoch=3)
    para['Par']['gen']['th'] = 0.01
    for var, d, pinf in ((1, tr, {}), (2, te, {'nSens': 20, 'steps': 5})):
        para['Mdl']['pinf'].update(pinf)
        mdl = pfFit(dict(d, _te=te), tr, para, {'selPF': var})
        out = pfSol(mdl, te, para, {'selPF': var})
        res = performance(te, out, para, mdl)
        data = A.compact({'results': {'pf': res}, 'pred': {'pf': out}, 'grt': {'pf': te}, 'mdls': {'pf': mdl}}, 'PF')
        file = str(tmp_path / ('run%d.pkl' % var))
        with open(file, 'wb') as f:
            pickle.dump(data, f)
        at = AppTest.from_function(_result_page, kwargs={'file': file}, default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any('Verdict' in m.value for m in at.markdown)
        assert len(at.metric) >= 6                                      # key figures and model facts


#######################################################################################################################
# References
#######################################################################################################################
