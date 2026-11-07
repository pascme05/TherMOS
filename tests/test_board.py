#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         test_board
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
Unit tests of the board designer: analytic lumped board, energy balance in both views, transient converging to the
steady state, POD-Galerkin reduced model and the Galerkin heat sink of the PO projection.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.genBoard import genBoard
from src.py.board.calcBoardMat import calcBoardMat
from src.py.board.simBoard import simBoard
from src.py.board.romBoard import romBoard, simRom
from src.py.board.genProfile import genProfile

# ==============================================================================
# External
# ==============================================================================
import numpy as np

#######################################################################################################################
# Settings
#######################################################################################################################
LAY = {'view': 'xy', 'W': 40, 'L': 30, 'board': 'FR4 2-layer, 35 µm Cu', 'h': 1500, 'Ta': 25, 'dx': 1.0, 'yCut': 10,
       'chips': [{'tech': 'Si IGBT', 'x': 10, 'y': 10, 'P': 30}, {'tech': 'SiC MOSFET', 'x': 30, 'y': 20, 'P': 15}]}


#######################################################################################################################
# Tests
#######################################################################################################################
def test_uniform_analytic():
    lay = dict(LAY, chips=[{'tech': 'Si MOSFET', 'x': 20, 'y': 15, 'w': 50, 'l': 40, 'P': 2.0}])
    D = genBoard(lay)
    C, G, Bp = calcBoardMat(D)
    _, Tss = simBoard(C, G, Bp, np.full((2, 1), 2.0), 1.0, 25)
    assert np.allclose(Tss - 25, 2.0 / (D['hEff'] * 0.040 * 0.030), rtol=1e-9)


def test_energy_balance():
    for view, depth in (('xy', None), ('xz', 1)):
        D = genBoard(dict(LAY, view=view))
        C, G, Bp = calcBoardMat(D)
        P = np.array([[30.0, 15.0][0:len(D['chips'])]] * 2)
        _, Tss = simBoard(C, G, Bp, P, 1.0, 25)
        d = D['tb'] if view == 'xy' else D['chips'][0]['l'] * 1e-3
        Qout = np.sum(D['wA'] * D['hv'] * (Tss - 25)) * D['dx'] * D['dy'] * d
        assert np.isclose(Qout, P[-1].sum(), rtol=1e-8)


def test_transient_and_rom():
    D = genBoard(LAY)
    C, G, Bp = calcBoardMat(D)
    t = np.arange(0, 300.0, 1.0)
    P = np.column_stack([genProfile('random', c['P'], t, {'hold': 10}, seed=i) for i, c in enumerate(D['chips'])])
    T, _ = simBoard(C, G, Bp, P, 1.0, 25, nSub=2)
    _, Tss = simBoard(C, G, Bp, np.full((2, 2), 10.0), 1.0, 25)
    Tl, _ = simBoard(C, G, Bp, np.full((2000, 2), 10.0), 5.0, 25)
    assert np.abs(Tl[-1] - Tss).max() < 1e-6
    rom = romBoard(C, G, Bp, T, 25, 10)
    Tr = simRom(rom, P, 1.0, 25, nSub=2)
    assert np.abs(Tr - T).max() < 0.02 * (T.max() - 25)


def test_coverage_and_six_layer():
    from src.py.board.boardLib import BOARDS, layerProps
    assert np.allclose(layerProps({'mat': 'Cu', 't': 0.035, 'cov': 100})[0:2], (390.0, 390.0))
    for name in ('FR4 6-layer, 35 µm Cu', 'FR4 6-layer, 70 µm Cu'):
        assert abs(sum(l['t'] for l in BOARDS[name]) - 1.6) < 5e-3
        test_lay = dict(LAY, board=name, chips=[{'tech': 'Si MOSFET', 'x': 20, 'y': 15, 'w': 50, 'l': 40, 'P': 2.0}])
        D = genBoard(test_lay)
        _, Tss = simBoard(*calcBoardMat(D), np.full((2, 1), 2.0), 1.0, 25)
        assert np.allclose(Tss - 25, 2.0 / (D['hEff'] * 0.040 * 0.030), rtol=1e-9)


def test_vias():
    base = dict(LAY, board='FR4 4-layer, 35 µm Cu planes')
    vias = {'mode': 'chips', 'd': 0.3, 'tpl': 25, 'fill': 'open (air)', 'margin': 1.0}
    for view in ('xy', 'xz'):
        D0 = genBoard(dict(base, view=view))
        D1 = genBoard(dict(base, view=view, vias=dict(vias, mode='none', pitch=1.0)))
        assert np.array_equal(D0['hv'], D1['hv']) and np.array_equal(D0['ky'], D1['ky'])
        tj = []
        for pitch in (3.0, 1.5, 1.0, 0.7):
            D = genBoard(dict(base, view=view, vias=dict(vias, pitch=pitch)))
            C, G, Bp = calcBoardMat(D)
            P = np.array([[30.0, 15.0][0:len(D['chips'])]] * 2)
            _, Tss = simBoard(C, G, Bp, P, 1.0, 25)
            d = D['tb'] if view == 'xy' else D['chips'][0]['l'] * 1e-3
            assert np.isclose(np.sum(D['wA'] * D['hv'] * (Tss - 25)) * D['dx'] * D['dy'] * d, P[-1].sum(), rtol=1e-8)
            tj.append(Tss.max())
        assert np.all(np.diff(tj) < 0)


def test_custom_stack_and_chip():
    lay = dict(LAY, layers=[{'layer': 'Cu', 'mat': 'Cu', 't': 0.5}], chipLib={'myChip': dict(
        mat='Si', t=0.2, w=4.0, l=4.0, attach='Sintered Ag', t_att=0.05, P=5.0, Tjmax=150.0, note='')},
        chips=[{'tech': 'myChip', 'x': 20, 'y': 15, 'P': 5.0}])
    D = genBoard(lay)
    assert np.isclose(D['kEff'], 390.0) and D['chips'][0]['Tjmax'] == 150.0
