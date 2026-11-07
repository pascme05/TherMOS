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
    for view in ('xy', 'xz'):
        D = genBoard(dict(LAY, view=view))
        C, G, Bp = calcBoardMat(D)
        P = np.array([[30.0, 15.0][0:len(D['chips'])]] * 2)
        _, Tss = simBoard(C, G, Bp, P, 1.0, 25)
        if view == 'xy':
            Qout = np.sum(D['wA'] * D['hv'] * (Tss - 25)) * D['dx'] * D['dy'] * D['tb']
        else:
            Qout = np.sum(D['gOut'] * (Tss - 25))                         # layer-resolved model, all chips
        assert np.isclose(Qout, P[-1].sum(), rtol=1e-8)


def test_layer_resolved_uniform():
    # uniform heating of a board-sized die: the 3D model reduces to the 1D series path through all rows
    from src.py.board.boardLib import BELOW
    lay = dict(LAY, chips=[{'tech': 'Si MOSFET', 'x': 20, 'y': 15, 'w': 50, 'l': 40, 'P': 2.0}], hTop=0.0,
               below=BELOW['Gap filler 2 mm + Al heat sink 5 mm'], view='xz')
    D = genBoard(lay)
    C, G, Bp = calcBoardMat(D)
    _, Tss = simBoard(C, G, Bp, np.full((2, 1), 2.0), 1.0, 25)
    Tb = Tss[D['idx'][0].ravel()]                                           # bottom row of the heat sink
    k = BELOW['Gap filler 2 mm + Al heat sink 5 mm'][-1]
    from src.py.board.boardLib import layerProps
    dz0 = float(k['t']) * 1e-3 / int(min(max(np.ceil(float(k['t']) / 0.5 - 1e-9), 1), 4))
    assert np.allclose(Tb - 25, 2.0 / 1.2e-3 * (dz0 / (2 * layerProps(k)[1]) + 1 / 1500), rtol=1e-6)


def test_mission():
    from src.py.board.genMission import genMission
    seq = [{'segment': 'a', 'duration [s]': 10, 'ramp [s]': 0, 'scale [-]': 1.0},
           {'segment': 'b', 'duration [s]': 10, 'ramp [s]': 5, 'scale [-]': 0.5, 'T1 [W]': 40}]
    P, tEnd = genMission(seq, [{'name': 'T1', 'P': 20}, {'name': 'D1', 'P': 10}], np.arange(0, 21.0), 1)
    assert tEnd == 20 and np.allclose(P[5], [20, 10]) and np.allclose(P[15], [40, 5])
    assert np.allclose(P[12], [20 + 20 * 2 / 5, 10 - 5 * 2 / 5])


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
        if view == 'xy':
            assert np.array_equal(D0['hv'], D1['hv']) and np.array_equal(D0['ky'], D1['ky'])
        else:
            assert (calcBoardMat(D0)[1] != calcBoardMat(D1)[1]).nnz == 0
        tj = []
        for pitch in (3.0, 1.5, 1.0, 0.7):
            D = genBoard(dict(base, view=view, vias=dict(vias, pitch=pitch)))
            C, G, Bp = calcBoardMat(D)
            P = np.array([[30.0, 15.0][0:len(D['chips'])]] * 2)
            _, Tss = simBoard(C, G, Bp, P, 1.0, 25)
            Q = np.sum(D['wA'] * D['hv'] * (Tss - 25)) * D['dx'] * D['dy'] * D['tb'] if view == 'xy' else                 np.sum(D['gOut'] * (Tss - 25))
            assert np.isclose(Q, P[-1].sum(), rtol=1e-8)
            tj.append(Tss.max())
        assert np.all(np.diff(tj) < 0)


def test_custom_stack_and_chip():
    lay = dict(LAY, layers=[{'layer': 'Cu', 'mat': 'Cu', 't': 0.5}], chipLib={'myChip': dict(
        mat='Si', t=0.2, w=4.0, l=4.0, attach='Sintered Ag', t_att=0.05, P=5.0, Tjmax=150.0, note='')},
        chips=[{'tech': 'myChip', 'x': 20, 'y': 15, 'P': 5.0}])
    D = genBoard(lay)
    assert np.isclose(D['kEff'], 390.0) and D['chips'][0]['Tjmax'] == 150.0


def test_layers_below():
    # uniform heating: plane 2 stays uniform, the series path board → interface → heat sink → coolant is exact
    from src.py.board.boardLib import BELOW
    lay = dict(LAY, chips=[{'tech': 'Si MOSFET', 'x': 20, 'y': 15, 'w': 50, 'l': 40, 'P': 2.0}], hTop=0.0,
               below=BELOW['Gap filler 2 mm + Al heat sink 5 mm'])
    D = genBoard(lay)
    C, G, Bp = calcBoardMat(D)
    assert len(C) == 2 * D['geo'].shape[0]
    _, Tss = simBoard(C, G, Bp, np.full((2, 1), 2.0), 1.0, 25)
    N = D['geo'].shape[0]
    assert np.allclose(Tss[:N] - 25, 2.0 * (D['R12'] + D['R2a']) / (0.040 * 0.030), rtol=1e-9)
    assert np.allclose(Tss[N:] - 25, 2.0 * D['R2a'] / (0.040 * 0.030), rtol=1e-9)

    # energy balance with chips: the heat leaves through the bottom of the heat sink (plane 2) and the top
    D = genBoard(dict(LAY, below=BELOW['Gap filler 2 mm + Al heat sink 5 mm']))
    C, G, Bp = calcBoardMat(D)
    _, Tss = simBoard(C, G, Bp, np.array([[30.0, 15.0]] * 2), 1.0, 25)
    V = D['wA'] * D['dx'] * D['dy'] * D['tb']
    Qout = np.sum(V * D['h2'] * (Tss[N:] - 25)) + np.sum(V * D['hv'] * (Tss[:N] - 25))
    assert np.isclose(Qout, 45.0, rtol=1e-8)

    # cross-section: the layers below are resolved
    D = genBoard(dict(LAY, view='xz', below=BELOW['Gap filler 2 mm + Al heat sink 5 mm']))
    assert D['nBelow'] > 0 and D['Ly'] > 7e-3 and D['rowName'][D['nBelow'] - 1].startswith('Gap')


def test_electrothermal_coupling():
    # steady state with linear temperature coefficients is the exact fixed point; the transient converges to it
    import scipy.sparse.linalg as spl
    D = genBoard(dict(LAY, view='xz', board='DBC AlN (0.63 mm)'))
    C, G, Bp = calcBoardMat(D)
    P = np.array([[30.0, 15.0]] * 2000)
    cpl = {'W': Bp, 'alpha': np.array([0.003, 0.005]), 'Tref': 25.0}
    T, Tss = simBoard(C, G, Bp, P, 2.0, 25, 1, cpl=cpl)
    Tj = Bp.T @ (Tss - 25) + 25
    Pe = P[-1] * (1 + cpl['alpha'] * (Tj - 25))
    assert np.allclose(spl.spsolve(G.tocsc(), Bp @ Pe) + 25, Tss, atol=1e-9)
    assert np.allclose(T[-1], Tss, atol=1e-3)
    _, Tss0 = simBoard(C, G, Bp, P, 2.0, 25, 1)
    assert np.all(Tss >= Tss0)
