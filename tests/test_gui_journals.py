#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_gui_journals
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
Tests of the journal pages (gui/core/journals.py, gui/views/journals.py): every setup of both papers is found and
parsed, the expected result names are those of mainPINN, setup values and configs are written back without losing
the comments, a background run of a setup finishes, and both pages render (Streamlit AppTest).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import journals as J
from setup.tutorial.studyPFHB import runName

import json
import os
import shutil
import pytest

st = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest                                # noqa: E402


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _thermos():
    from gui.views import journals
    journals.thermos_page()


def _pinn():
    from gui.views import journals
    journals.pinn_page()


#######################################################################################################################
# Tests
#######################################################################################################################
def test_setups_found():
    for t in J.THERMOS:
        assert os.path.isfile(J.script(t['id'], J.THERMOS_DIR)), t['id']
    for p in J.PINN:
        rows, doc = J.parse_setup(J.script(p['id']))
        keys = [r['key'] for r in rows]
        assert doc and {'name', 'out', 'mesh', 'methods', 'figs'} <= set(keys), p['id']
        assert all(r['comment'] for r in rows), p['id']
        assert {'name', 'out', 'epoch'} <= {r['key'] for r in rows if r['change']}


def test_expected_names():
    s = dict(mesh=['medium'], nTrain=[3, 24], methods=['podNom', 'pinn'], sens=['S', 'L'], scale=1,
             layout='pinn_medium_L', fieldmap=['medium'])
    want = [runName(m, 'medium', x, n) + '.json' for n in (3, 24) for m in ('podNom', 'pinn')
            for x in (['F'] if m == 'podNom' else ['S', 'L'])]
    assert J.expected(s) == want + ['fieldmap_medium.json', 'scale_medium.json', 'layout_pinn_medium_L.json']


def test_write_setup(tmp_path):
    f = tmp_path / 's.py'
    shutil.copy(J.script('journal_pinn_figures'), f)
    J.write_setup(str(f), {'epoch': 2, 'mesh': ['coarse'], 'name': 'test'})
    rows, _ = J.parse_setup(str(f))
    d = {r['key']: r for r in rows}
    assert d['epoch']['value'] == 2 and d['mesh']['value'] == ['coarse'] and d['name']['value'] == 'test'
    assert d['epoch']['change'] and len(rows) == len(J.parse_setup(J.script('journal_pinn_figures'))[0])


def test_config_roundtrip():
    for n in J.config_names():
        cfg = J.read_config(n)
        rows = J.config_rows(cfg)
        assert rows and all(r['comment'] for r in rows), n
        new = J.set_values(json.loads(json.dumps(cfg)), {r['parameter']: r['value'] for r in rows})
        assert new == cfg
    assert J.parse_value('2048', 1.0) == 2048.0 and J.parse_value("['a']", []) == ['a']
    with pytest.raises(ValueError):
        J.parse_value('[1,', [])


def test_background_run():
    out = 'results/_test_journals'
    try:
        p, log = J.start(J.script('journal_pinn_figures'), {'out': out, 'figs': [], 'build': 0, 'name': 'jtest'})
        assert p.wait(timeout=300) == 0
        txt, state = J.read_log(log)
        assert state == 'done', txt
    finally:
        shutil.rmtree(os.path.join(J.ROOT, out), ignore_errors=True)


@pytest.mark.parametrize('page', [_thermos, _pinn])
def test_pages(page):
    at = AppTest.from_function(page, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
