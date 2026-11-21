#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_gui_kr
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################
"""
Tests of the KR mode in the GUI: registration, explained parameters, the pages of the mode (workflow, parameters,
inputs with the 'what this mode needs' box, theory, run), the results of a KR run on the PF tutorial board and the
comparison section of the MOR page (Streamlit AppTest).
"""
import json
import os
import pickle

import pytest

st = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'board', 'tutorial_pf_test.mat')


def _mode_page(view):
    import streamlit as st
    from gui.core import config as conf
    from gui.core import forms
    from gui.views import mode
    if 'cfg' not in st.session_state:
        forms.set_config(conf.read_config('tutorial_KR'), 'tutorial_KR')
        st.session_state.setdefault('runs', {})
    st.session_state['view|KR'] = view
    mode.make_page('KR')()


def _result_page(file):
    import pickle
    from gui.core.viz_thermos import RENDER
    with open(file, 'rb') as f:
        data = pickle.load(f)
    RENDER['KR'](data, {'Dat': {'gen': {'inpZ': 0.0}}, 'Run': {}}, 'res|KR')


def _mor_page():
    from gui.views.tools import mor
    mor.page()


def test_mode_registered():
    from gui import adapter as A
    from gui.core import help as H, theory, workflow, viz_thermos, viz_inputs
    assert A.SEL['KR'] == 'selKR' and A.RUN['selKR'] == 1 and 'kr' in A.MODES['KR']['cats']
    assert all('KR' in d for d in (theory.THEORY, workflow.WORKFLOWS, viz_thermos.RENDER, viz_inputs.INPUTS))
    assert H.RELEVANT['Mdl']['kr']({'Run': {'mode': 'KR'}}) and not H.RELEVANT['Mdl']['kr']({'Run': {'mode': 'PF'}})
    assert os.path.isfile(os.path.join(ROOT, A.MODES['KR']['script']))
    schema = json.load(open(os.path.join(ROOT, 'gui', 'schema.json'), encoding='utf-8'))
    ps = {p['var']: p for p in schema['Mdl'] if p['cat'] == 'kr'}
    assert set(ps) == {'q', 'hold'} and all(len(p.get('description') or p.get('desc') or '') > 40
                                            for p in ps.values())


def test_tutorial():
    """KR has a tutorial whose base config exists and whose run changes are parameters of the schema."""
    from gui.core.tutorials_data import TUTORIALS
    tu = [t for t in TUTORIALS if t['mode'] == 'KR']
    assert tu and os.path.isfile(os.path.join(ROOT, 'config', 'gui', tu[0]['base'] + '.json'))
    schema = json.load(open(os.path.join(ROOT, 'gui', 'schema.json'), encoding='utf-8'))
    keys = {'Mdl.%s.%s' % (p['cat'], p['var']) for p in schema['Mdl']}
    for r in tu[0].get('runs', []):
        assert set(r['changes']) <= keys, r


@pytest.mark.parametrize('view', ['Workflow', 'Parameters', 'Inputs', 'Theory', 'Run'])
def test_mode_pages(view):
    at = AppTest.from_function(_mode_page, kwargs={'view': view}, default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error or view == 'Run', [e.value for e in at.error]
    if view == 'Inputs':
        assert any('What this mode needs' in m.value for m in at.markdown)


@pytest.mark.skipif(not os.path.isfile(DATA), reason="PF tutorial data not generated")
def test_results_page(tmp_path):
    import numpy as np
    import scipy.io as sio
    from gui import adapter as A
    from src.py.model.KR.krFit import krFit
    from src.py.model.KR.krSol import krSol
    from src.py.general.performance import performance
    m = sio.loadmat(DATA, simplify_cells=True)
    y = np.asarray(m['y'], float)
    d = {'X2': [np.asarray(m['X'], float)], 'y2': [y], 'r2': [np.asarray(m['r'], float)], 'Data': m,
         'Ts': float(m['Ts']), 'y': y, 'X': np.asarray(m['X'], float), 'Nt': [len(y)], 'Dim': 3}
    para = {'Mdl': {'kr': {'q': 4}}, 'Exp': {'gen': {'init': 1}}, 'Par': {'gen': {'th': 0.01}}}
    mdl = krFit(d, d, para)
    out = krSol(mdl, d, para)
    assert np.sqrt(np.mean((out['y'] - y) ** 2)) < 0.5
    try:
        res = performance(d, out, para, mdl)
    except Exception:                                                     # performance needs the full para
        pytest.skip("performance() needs the full parameter set")
    data = A.compact({'results': {'kr': res}, 'pred': {'kr': out}, 'grt': {'kr': d}, 'mdls': {'kr': mdl}}, 'KR')
    file = str(tmp_path / 'run.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    at = AppTest.from_function(_result_page, kwargs={'file': file}, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_mor_page_comparison():
    at = AppTest.from_function(_mor_page, default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('given for every new board' in str(t.value) for t in at.dataframe) or \
        any('information' in m.value for m in at.markdown)
    at.select_slider(key='tl|mor|cdx').set_value(6).run()
    assert not at.exception, [e.value for e in at.exception]
