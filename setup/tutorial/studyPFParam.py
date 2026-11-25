#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         studyPFParam
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
This script runs the studies of the parametric PF network on the board families of genBoardPFParam:
    runs (training + evaluation on val, t1, t1f, t2, t3, t4):
        base        24 labelled boards + 200 physics-only boards (positions outside the band)
        nophys      24 labelled boards, no physics loss
        physAll     24 labelled boards + 200 physics-only boards with positions anywhere (also inside the band)
        n3/n6/n12   3/6/12 labelled boards + 200 physics-only boards; n3np/n6np/n12np without physics
        K8/K16      8/16 space functions (default 32); norel: without the coordinates relative to the chips;
                    noSS: without the steady-state loss
        lin         base + log-linear condition heads (rates and amplitudes follow a power law of the condition)
        linBoth     lin + physics-only boards phys and physExt (400 boards)
        v2          network V1.2 of the PINN journal: lin + phys and physExt + moment loss (delta 1, 2 moments);
                    v2phys: the same with phys only (linBoth, v2 need the set physExt: python
                    setup/tutorial/genBoardPFParam.py physExt)
        tL10/tL20   24 labelled boards with one profile of 10/20 s; tP1: one profile of 40 s (base: two of 40 s)
        pos<n>      n = 1/3/6/24 labelled boards of trainPos (nominal μ, only the positions vary)
        mu<n>       n = 1/3/6/24 labelled boards of trainMu (nominal positions, only μ varies)
    eval:<run>[,<run>...]  re-evaluates the mission profiles t4 and t4slow with the saved network of the run
    layout[:run]    layout optimisation through the network <run> (default physAll): gradient descent on the chip
                    centres, check with the FV model, FV random search for comparison
    baselines:
        fv          runtime of the finite-volume model per new board (assembly, factorisation, time stepping)
        pmor        parametric model order reduction: global POD basis (r = 16, 32) of the snapshots of the pmor
                    boards, Galerkin projection of C, G, Bp of every test board with the nominal positions (t1f, t2)
        pmorPos     pMOR for varying positions: global POD basis (r = 16 … 128) of the labelled training boards on the
                    superset grid of all layouts (t1, t1f, t3)
        krylov      per-board block Krylov reduction from the operator of the new board (no data, r = 3·q)
Results: <out>/<run>.json (metrics per test set and board), <out>/<run>.pt (network), log on the console.
Usage:      python setup/tutorial/studyPFParam.py <run|fv|pmor> [...] [--out DIR] [--epoch N]
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
# Internal
# ==============================================================================
from src.py.model.PF.pfParam import (loadFam, buildBoard, simulate, layout, samplePos, chipCtx, muHat, _ok, CHIPS, NOM,
                                     P_REF, W_MM, L_MM, MARGIN, GAP, Z_MAX)
from src.py.model.PF.pfParamFit import fitParam, evalBoard, prepBoard, predict, buildNet
from src.py.board.genBoard import genBoard

# ==============================================================================
# External
# ==============================================================================
import os
import json
import time
import warnings
import numpy as np
import torch
import scipy.sparse.linalg as spl

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
NP = {'beta': 0.0, 'gamma': 0.0}                                       # no operator information (data only)
RUNS = {'base': ({}, 24, 'phys'), 'nophys': (NP, 24, None), 'physAll': ({}, 24, 'physAll'),
        'n3': ({}, 3, 'phys'), 'n6': ({}, 6, 'phys'), 'n12': ({}, 12, 'phys'),
        'n3np': (NP, 3, None), 'n6np': (NP, 6, None), 'n12np': (NP, 12, None),
        'physExt': ({}, 24, 'physExt'), 'base2': ({'seed': 7}, 24, 'phys'), 'nophys2': (dict(NP, seed=7), 24, None),
        'noSS': ({'gamma': 0.0}, 24, 'phys'), 'K8': ({'K': 8}, 24, 'phys'), 'K16': ({'K': 16}, 24, 'phys'),
        'norel': ({'rel': False}, 24, 'phys')}
# network V1.2 (journal_pinn): log-linear condition heads (lin), moment loss (delta, nMom), both physics sets
V2 = {'lin': True, 'delta': 1.0, 'nMom': 2}
RUNS.update({'lin': ({'lin': True}, 24, 'phys'), 'linBoth': ({'lin': True}, 24, 'phys+physExt'),
             'v2': (V2, 24, 'phys+physExt'), 'v2phys': (V2, 24, 'phys')})
# training-sample study: (cfg, labelled boards, physics family, (labelled family, profiles per board, length (s)))
RUNS.update({'tL10': ({}, 24, 'phys', ('train', 1, 10.0)), 'tL20': ({}, 24, 'phys', ('train', 1, 20.0)),
             'tP1': ({}, 24, 'phys', ('train', 1, 40.0))})
RUNS.update({'pos%d' % n: ({}, n, 'phys', ('trainPos', 2, 40.0)) for n in (1, 3, 6, 24)})
RUNS.update({'mu%d' % n: ({}, n, 'phys', ('trainMu', 2, 40.0)) for n in (1, 3, 6, 24)})
TESTS = ['val', 't1', 't1f', 't2', 't3']
T4S = ['t4', 't4slow']                                                  # mission profiles (fast, slow segments)
ZOOM = 60.0                                                             # junction traces at every sample up to (s)
TS = 0.05


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _fam(folder, name):
    return loadFam(pjoin(folder, name + '.npz'))


def _evalT4(net, dT, folder, dev, name='t4'):
    """Errors on the 30 min mission profile: full field every 1 s and the die nodes at every sample."""
    B = _fam(folder, name)[0]
    z = np.load(pjoin(folder, name + 'die.npz'))
    dec, die, thDie = int(z['dec']), z['die'], z['th']
    P = B['P'][0]
    with torch.no_grad():
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        bt = prepBoard(B, dev)
        y = predict(net, bt, P, dT, TS, tIdx=torch.arange(0, len(P), dec, device=dev)).cpu().numpy()
        yd = predict(net, bt, P, dT, TS, rows=torch.as_tensor(die, device=dev)).cpu().numpy()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        tm = time.time() - t0
    e, ed = y - B['th'][0], yd - thDie
    nz = min(len(P), int(round(ZOOM / TS)) + 1)
    nd = np.cumsum([0] + [len(d) for d in B['die']])
    tj = [(yd[:, nd[c]:nd[c + 1]].max(axis=1), thDie[:, nd[c]:nd[c + 1]].max(axis=1)) for c in range(0, len(B['die']))]
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'max': float(np.max(np.abs(e))),
            'rmseDie': float(np.sqrt(np.mean(ed ** 2))), 'maxDie': float(np.max(np.abs(ed))),
            'rmseTj': [float(np.sqrt(np.mean((a - b) ** 2))) for a, b in tj], 'time': tm, 'nSamples': len(P),
            'trace': {'t': (np.arange(len(P)) * TS)[::dec].tolist(), 'ref': tj[0][1][::dec].tolist(),
                      'pred': tj[0][0][::dec].tolist()},
            'zoom': {'t': (np.arange(nz) * TS).tolist(), 'ref': [tj[c][1][:nz].tolist() for c in range(len(tj))],
                     'pred': [tj[c][0][:nz].tolist() for c in range(len(tj))]}}


def _summary(rows):
    """Pooled metrics of a test set (RMSE over all boards weighted with the number of values)."""
    return {'rmse': float(np.sqrt(np.mean([r['rmse'] ** 2 for r in rows]))),
            'mae': float(np.mean([r['mae'] for r in rows])), 'max': float(np.max([r['max'] for r in rows])),
            'rmseTj': float(np.sqrt(np.mean([np.mean(np.square(r['rmseTj'])) for r in rows]))),
            'time': float(np.mean([r['time'] for r in rows]))}


#######################################################################################################################
# Runs
#######################################################################################################################
def run(name, folder, out, epoch=None):
    cfg, nTr, ph, sel = (RUNS[name] + (('train', None, None),))[0:4]
    cfg = dict(cfg, **({'epoch': epoch} if epoch else {}))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = _fam(folder, sel[0])[0:nTr]
    if sel[1]:                                                            # fewer / shorter labelled profiles
        L = int(round(sel[2] / TS)) + 1
        for B in train:
            B['P'], B['th'] = [P[0:L] for P in B['P'][0:sel[1]]], [th[0:L] for th in B['th'][0:sel[1]]]
    phys = [B for f in ph.split('+') for B in _fam(folder, f)] if ph else []       # e.g. 'phys+physExt'
    val = _fam(folder, 'val')
    print("START: run %s (%d labelled, %s physics boards) on %s" % (name, nTr, len(phys), dev))
    res = fitParam(train, phys, val, cfg, dev)
    net, dT = res['net'], res['dT']
    torch.save({'state': net.state_dict(), 'dT': dT, 'cfg': res['cfg']}, pjoin(out, name + '.pt'))
    rep = {'run': name, 'train': len(train), 'phys': len(phys), 'time': res['time'], 'epochs': len(res['hist']['val']),
           'best': res['best'], 'dT': dT, 'hist': res['hist'], 'cfg': {k: v for k, v in res['cfg'].items()}}
    for s in TESTS:
        fam = _fam(folder, s)
        rows = []
        for B in fam:
            e = evalBoard(net, B, dT, TS, dev)
            rows.append({'mu': B['mu'], 'pos': B['pos'], 'nChips': len(B['chips']), 'rmse': float(e['rmse']),
                         'mae': float(e['mae']), 'max': float(e['max']), 'rmseTj': [float(v) for v in e['rmseTj']],
                         'time': float(e['time'])})
        rep[s] = {'boards': rows, 'sum': _summary(rows)}
        print("INFO: %s %s: RMSE %.2f K, MAE %.2f K, MAX %.1f K, T_j RMSE %.2f K" %
              (name, s, rep[s]['sum']['rmse'], rep[s]['sum']['mae'], rep[s]['sum']['max'], rep[s]['sum']['rmseTj']))
    rep['data'] = {'fam': sel[0], 'boards': len(train), 'profiles': int(sum(len(B['P']) for B in train)),
                   'samples': int(sum(len(P) for B in train for P in B['P']))}
    _t4(rep, net, dT, folder, dev)
    with open(pjoin(out, name + '.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


def _t4(rep, net, dT, folder, dev):
    for s in T4S:
        if isfile(pjoin(folder, s + '.npz')):
            rep[s] = _evalT4(net, dT, folder, dev, s)
            print("INFO: %s %s (30 min): RMSE %.2f K, MAE %.2f K, MAX %.1f K, die RMSE %.2f K, %.2f s" %
                  (rep['run'], s, rep[s]['rmse'], rep[s]['mae'], rep[s]['max'], rep[s]['rmseDie'], rep[s]['time']))


def evalT4(name, folder, out):
    """Re-evaluates the mission profiles (t4, t4slow) with a saved network; the other entries of the json are kept."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dT = _loadNet(pjoin(out, name + '.pt'), dev)
    with open(pjoin(out, name + '.json')) as f:
        rep = json.load(f)
    _t4(rep, net, dT, folder, dev)
    with open(pjoin(out, name + '.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# Baselines
#######################################################################################################################
def baseFV(folder, out):
    """Runtime of the finite-volume model for a new board: assembly (genBoard), and simulation of the 40 s profile."""
    rep = {}
    for s in ('t1', 't4'):
        rows = []
        for B in _fam(folder, s):
            t0 = time.time()
            D = genBoard(layout(B['mu'], B['pos'], B['chips']))
            t1 = time.time()
            simulate(buildBoard(B['mu'], B['pos'], B['chips']), B['P'][0], TS)
            t2 = time.time()
            rows.append({'assembly': t1 - t0, 'sim': t2 - t1, 'nSamples': len(B['P'][0]), 'N': len(D['nod']['x'])})
        rep[s] = rows
        print("INFO: FV %s: assembly %.3f s, simulation %.2f s per board (%d samples)" %
              (s, np.mean([r['assembly'] for r in rows]), np.mean([r['sim'] for r in rows]), rows[0]['nSamples']))
    with open(pjoin(out, 'fv.json'), 'w') as f:
        json.dump(rep, f, indent=1)


def basePMOR(folder, out, ranks=(8, 16, 32)):
    """Global POD basis of the snapshots of the pmor boards, Galerkin projection per test board (t1f, t2)."""
    snap = np.concatenate([th for B in _fam(folder, 'pmor') for th in B['th']])
    [_, sv, Vt] = np.linalg.svd(snap, full_matrices=False)
    rep = {'sv': (sv[0:40] / sv[0]).tolist()}
    for r in ranks:
        V = Vt[0:r].T                                                    # N x r
        rep[r] = {}
        for s in ('t1f', 't2'):
            rows = []
            for B in _fam(folder, s):
                t0 = time.time()
                Cr, Gr, Br = V.T @ (B['C'][:, None] * V), V.T @ (B['G'] @ V), V.T @ B['Bp']
                P, h, nSub = B['P'][0], TS / 4, 4
                A = np.linalg.inv(Cr / h + Gr)                          # implicit Euler as the reference
                q, Q = np.zeros(r), np.zeros((len(P), r))
                for k in range(1, len(P)):
                    for _ in range(0, nSub):
                        q = A @ (Cr / h @ q + Br @ P[k])
                    Q[k] = q
                y = Q @ V.T
                tm = time.time() - t0
                th = B['th'][0]
                e = y - th
                tj = [float(np.sqrt(np.mean((y[:, d].max(axis=1) - th[:, d].max(axis=1)) ** 2))) for d in B['die']]
                rows.append({'mu': B['mu'], 'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
                             'max': float(np.max(np.abs(e))), 'rmseTj': tj, 'time': tm})
            rep[r][s] = {'boards': rows, 'sum': _summary(rows)}
            print("INFO: pMOR r = %d %s: RMSE %.2f K, MAX %.1f K, T_j RMSE %.2f K, %.3f s" %
                  (r, s, rep[r][s]['sum']['rmse'], rep[r][s]['sum']['max'], rep[r][s]['sum']['rmseTj'],
                   rep[r][s]['sum']['time']))
    with open(pjoin(out, 'pmor.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


def _slots(B):
    """Index of every node in the superset grid (rows x 21 x 31) of all layouts: row = rank of the node height."""
    g = np.asarray(B['geo']) * 1e3
    zr = np.unique(np.round(g[:, 2], 6))
    row = np.searchsorted(zr, np.round(g[:, 2], 6))
    return (row * 21 + np.round(g[:, 1]).astype(int)) * 31 + np.round(g[:, 0]).astype(int)


def _rom(B, V, P):
    """Galerkin projection of C, G, Bp of board B on V (N x r), implicit Euler with 4 sub-steps; field (L x N)."""
    r, h = V.shape[1], TS / 4
    Cr, Gr, Br = V.T @ (B['C'][:, None] * V), V.T @ (B['G'] @ V), V.T @ B['Bp']
    A = np.linalg.inv(Cr / h + Gr)
    Ah, q, Q = A @ (Cr / h), np.zeros(r), np.zeros((len(P), r))
    U = (A @ Br) @ P.T
    for k in range(1, len(P)):
        for _ in range(0, 4):
            q = Ah @ q + U[:, k]
        Q[k] = q
    return Q @ V.T


def _err(y, B):
    th = B['th'][0]
    e = y - th
    tj = [float(np.sqrt(np.mean((y[:, d].max(axis=1) - th[:, d].max(axis=1)) ** 2))) for d in B['die']]
    return {'mu': B['mu'], 'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'max': float(np.max(np.abs(e))), 'rmseTj': tj}


def basePMORPos(folder, out, ranks=(16, 32, 64, 128)):
    """
    pMOR for varying positions: global POD basis of the snapshots of the labelled training boards (random positions)
    on the superset grid of all layouts; for a test board the rows of its nodes are taken from the basis.
    """
    tr = _fam(folder, 'train')
    nS = 7 * 21 * 31
    X = []
    for B in tr:
        s = _slots(B)
        for th in B['th']:
            Z = np.zeros((len(th[::2]), nS), dtype=np.float32)
            Z[:, s] = th[::2]
            X.append(Z)
    X = np.concatenate(X)
    [w, E] = np.linalg.eigh((X.T @ X).astype(np.float64))
    E = E[:, ::-1]
    rep = {'sv': np.sqrt(np.maximum(w[::-1][0:150], 0) / w[-1]).tolist()}
    for r in ranks:
        rep[r] = {}
        for s in ('t1', 't1f', 't3'):
            rows = []
            for B in _fam(folder, s):
                if len(B['chips']) != 3:
                    continue
                t0 = time.time()
                V, _ = np.linalg.qr(E[_slots(B), 0:r])
                y = _rom(B, V, B['P'][0])
                rows.append(dict(_err(y, B), time=time.time() - t0))
            rep[r][s] = {'boards': rows, 'sum': _summary(rows)}
            print("INFO: pMOR (positions) r = %d %s: RMSE %.2f K, MAX %.1f K, T_j RMSE %.2f K, %.3f s" %
                  (r, s, rep[r][s]['sum']['rmse'], rep[r][s]['sum']['max'], rep[r][s]['sum']['rmseTj'],
                   rep[r][s]['sum']['time']))
    with open(pjoin(out, 'pmorPos.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


def baseKrylov(folder, out, blocks=(2, 4, 8)):
    """
    Per-board reduction without training data: block Krylov basis of the moments at s = 0,
    V = orth[G⁻¹·Bp, (G⁻¹·C)·G⁻¹·Bp, …] (q blocks, r = q·nChips), from the operator of the new board.
    """
    rep = {}
    for q in blocks:
        rep[q] = {}
        for s in ('t1', 't1f', 't2', 't3'):
            rows = []
            for B in _fam(folder, s):
                t0 = time.time()
                lu = spl.splu(B['G'].tocsc())
                Kb = [lu.solve(B['Bp'])]
                for _ in range(1, q):
                    Kb.append(lu.solve(B['C'][:, None] * Kb[-1]))
                V, _ = np.linalg.qr(np.column_stack(Kb))
                y = _rom(B, V, B['P'][0])
                rows.append(dict(_err(y, B), time=time.time() - t0))
            rep[q][s] = {'boards': rows, 'sum': _summary(rows)}
            print("INFO: Krylov q = %d (r = %d) %s: RMSE %.2f K, MAX %.1f K, T_j RMSE %.2f K, %.3f s" %
                  (q, 3 * q, s, rep[q][s]['sum']['rmse'], rep[q][s]['sum']['max'], rep[q][s]['sum']['rmseTj'],
                   rep[q][s]['sum']['time']))
        # 30 min mission profile (field every T4_DEC samples)
        B = _fam(folder, 't4')[0]
        t0 = time.time()
        lu = spl.splu(B['G'].tocsc())
        Kb = [lu.solve(B['Bp'])]
        for _ in range(1, q):
            Kb.append(lu.solve(B['C'][:, None] * Kb[-1]))
        V, _ = np.linalg.qr(np.column_stack(Kb))
        y = _rom(B, V, B['P'][0])
        tm = time.time() - t0
        dec = len(B['P'][0]) // (len(B['th'][0]) - 1) if len(B['th'][0]) > 1 else 1
        e = y[::dec][0:len(B['th'][0])] - B['th'][0]
        rep[q]['t4'] = {'rmse': float(np.sqrt(np.mean(e ** 2))), 'max': float(np.max(np.abs(e))), 'time': tm}
        print("INFO: Krylov q = %d t4 (30 min): RMSE %.3f K, %.2f s" % (q, rep[q]['t4']['rmse'], tm))
    with open(pjoin(out, 'krylov.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# Layout Optimisation
#######################################################################################################################
def _loadNet(path, dev):
    ck = torch.load(path, map_location=dev)
    net = buildNet(ck['cfg']).to(dev)                                 # with the log-linear heads if cfg['lin']
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ck['dT']


def _peakNet(net, dT, pos, off, ctxC, muH, p):
    """
    Junction temperatures T_j,c(t) (nc x L) (K) of the network for the chip centres pos (nc x 2) (mm, tensor, may
    require grad): die nodes of chip c at pos_c + off_c (off: list of n_c x 3, mm in x, y, m in z), context from the
    centres, the constant chip part ctxC (nc x 5) and μ̂ (4).
    """
    nc = pos.shape[0]
    ctx = torch.cat([2 * pos[:, 0:1] / W_MM - 1, 2 * pos[:, 1:2] / L_MM - 1, ctxC, muH.expand(nc, -1)], dim=1)
    b = net.coeff(p, net.rates(ctx), TS)                                       # nc x L x K
    tj = []
    for c in range(0, nc):
        gx, gy, gz = off[c][:, 0] + pos[c, 0], off[c][:, 1] + pos[c, 1], off[c][:, 2]
        xh = torch.stack([2 * gx / W_MM - 1, 2 * gy / L_MM - 1, 2 * gz / Z_MAX - 1], dim=-1)            # n x 3
        xs = torch.stack([torch.cat([xh, torch.stack([2 * (gx - pos[j, 0]) / W_MM, 2 * (gy - pos[j, 1]) / L_MM],
                                                     dim=-1)], dim=-1) for j in range(0, nc)])        # nc x n x 5
        th = torch.einsum('clk,cnk->ln', b, net.basis(xs, ctx)) * dT
        tj.append(th.max(dim=1).values)
    return torch.stack(tj)


def _penalty(pos, chips):
    """Squared violation (mm²) of the edge margin and of the chip spacing (GAP) for the centres pos (nc x 2)."""
    w = torch.tensor([c['w'] for c in chips], dtype=pos.dtype, device=pos.device)
    l = torch.tensor([c['l'] for c in chips], dtype=pos.dtype, device=pos.device)
    r = torch.relu
    pen = torch.sum(r(MARGIN + w / 2 - pos[:, 0]) ** 2 + r(pos[:, 0] + w / 2 - W_MM + MARGIN) ** 2 +
                    r(MARGIN + l / 2 - pos[:, 1]) ** 2 + r(pos[:, 1] + l / 2 - L_MM + MARGIN) ** 2)
    for i in range(0, len(chips)):
        for j in range(i + 1, len(chips)):
            dx = GAP + (w[i] + w[j]) / 2 - torch.abs(pos[i, 0] - pos[j, 0])
            dy = GAP + (l[i] + l[j]) / 2 - torch.abs(pos[i, 1] - pos[j, 1])
            pen = pen + torch.minimum(r(dx), r(dy)) ** 2
    return pen


def _peakFV(mu, pos, P):
    B = buildBoard(mu, pos)
    th = simulate(B, P, TS)
    return np.array([float(th[:, d].max()) for d in B['die']])


def optLayout(folder, out, model='physAll', nStart=8, nIter=300, nRand=100, tauS=2.0, wPen=50.0, seed=0):
    """
    Layout optimisation through the network: the chip centres minimise the smooth maximum of the peak junction
    temperatures J = τ·log Σ_c exp(max_t T_j,c / τ) + w·penalty (gradient descent, Adam, nStart random starts) for the
    designed 40 s profile at the nominal condition. The best layout (rounded to the 1 mm grid of the FV model) is
    checked with the FV model and compared with an FV random search over nRand layouts; the network is also evaluated
    on the random layouts (ranking and error of the peak T_j).
    """
    from setup.tutorial.genBoardPFParam import designed
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dT = _loadNet(pjoin(out, model + '.pt'), dev)
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 40.0 + TS * 1e-9, TS)
    P = designed(t, CHIPS)
    p = torch.as_tensor(P.T / P_REF, dtype=torch.float32, device=dev)
    B0 = buildBoard(NOM, [(c['x'], c['y']) for c in CHIPS])
    off = [torch.as_tensor(np.column_stack([B0['geo'][d, 0] * 1e3 - c['x'], B0['geo'][d, 1] * 1e3 - c['y'],
                                            B0['geo'][d, 2]]), dtype=torch.float32, device=dev)
           for d, c in zip(B0['die'], CHIPS)]
    ctxC = torch.as_tensor(np.stack([chipCtx(B0, c)[2:] for c in range(0, len(CHIPS))]), device=dev)
    muH = torch.as_tensor(muHat(NOM), device=dev)[None]

    def peaks(pos):
        with torch.no_grad():
            return _peakNet(net, dT, torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=dev), off, ctxC, muH,
                            p).max(dim=1).values.cpu().numpy()

    # gradient descent from nStart random layouts
    t0 = time.time()
    runs = []
    for s in range(0, nStart):
        x = torch.tensor(samplePos(rng), dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nIter)
        path = []
        for it in range(0, nIter):
            opt.zero_grad()
            pk = _peakNet(net, dT, x, off, ctxC, muH, p).max(dim=1).values
            J = tauS * torch.logsumexp(pk / tauS, dim=0) + wPen * _penalty(x, CHIPS)
            J.backward()
            opt.step()
            sch.step()
            path.append(x.detach().cpu().numpy().copy())
        pos = [tuple(map(float, np.round(v))) for v in x.detach().cpu().numpy()]
        runs.append({'start': path[0].tolist(), 'end': path[-1].tolist(), 'pos': pos, 'ok': _ok(pos, CHIPS),
                     'peakNet': peaks(pos).tolist(), 'path': np.asarray(path)[::10].tolist()})
    tOpt = time.time() - t0
    good = [r for r in runs if r['ok']]
    best = min(good, key=lambda r: max(r['peakNet']))
    t0 = time.time()
    best['peakFV'] = _peakFV(NOM, best['pos'], P).tolist()
    tCheck = time.time() - t0
    nom = [(float(c['x']), float(c['y'])) for c in CHIPS]
    rep = {'model': model, 'timeOpt': tOpt, 'timeCheck': tCheck, 'runs': runs, 'best': best,
           'nominal': {'pos': nom, 'peakNet': peaks(nom).tolist(), 'peakFV': _peakFV(NOM, nom, P).tolist()}}
    print("INFO: network optimisation (%d starts x %d steps): %.1f s, %d/%d valid after rounding" %
          (nStart, nIter, tOpt, len(good), nStart))
    print("INFO: nominal layout: peak T_j FV %s K, network %s K" %
          (np.round(rep['nominal']['peakFV'], 1), np.round(rep['nominal']['peakNet'], 1)))
    print("INFO: optimised layout %s: peak T_j FV %s K, network %s K" %
          (best['pos'], np.round(best['peakFV'], 1), np.round(best['peakNet'], 1)))

    # FV random search and network on the same layouts
    t0 = time.time()
    rnd = [samplePos(rng) for _ in range(0, nRand)]
    pkF = np.array([_peakFV(NOM, q, P) for q in rnd])
    tRand = time.time() - t0
    t0 = time.time()
    pkN = np.array([peaks(q) for q in rnd])
    tRandNet = time.time() - t0
    jF, jN = pkF.max(axis=1), pkN.max(axis=1)
    rk = lambda v: np.argsort(np.argsort(v))
    rep['random'] = {'pos': rnd, 'peakFV': pkF.tolist(), 'peakNet': pkN.tolist(), 'timeFV': tRand,
                     'timeNet': tRandNet, 'best': rnd[int(np.argmin(jF))], 'bestPeakFV': float(jF.min()),
                     'rmsePeak': float(np.sqrt(np.mean((pkN - pkF) ** 2))),
                     'spearman': float(np.corrcoef(rk(jF), rk(jN))[0, 1])}
    print("INFO: FV random search (%d layouts): best max peak T_j %.1f K, %.0f s; network on the same layouts %.1f s, "
          "peak T_j RMSE %.2f K, rank correlation %.3f" % (nRand, jF.min(), tRand, tRandNet,
                                                           rep['random']['rmsePeak'], rep['random']['spearman']))
    with open(pjoin(out, 'layout_%s.json' % model), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(argv):
    folder = pjoin(base, 'data', 'board', 'pfparam')
    out = argv[argv.index('--out') + 1] if '--out' in argv else pjoin(base, 'results', 'pfparam')
    epoch = int(argv[argv.index('--epoch') + 1]) if '--epoch' in argv else None
    os.makedirs(out, exist_ok=True)
    names = [a for i, a in enumerate(argv) if not a.startswith('--') and (i == 0 or argv[i - 1] not in ('--out',
                                                                                                        '--epoch'))]
    for n in names:
        if n == 'fv':
            baseFV(folder, out)
        elif n == 'pmor':
            basePMOR(folder, out)
        elif n == 'pmorPos':
            basePMORPos(folder, out)
        elif n == 'krylov':
            baseKrylov(folder, out)
        elif n.startswith('eval:'):
            for r in n.split(':')[1].split(','):
                evalT4(r, folder, out)
        elif n.startswith('layout'):
            optLayout(folder, out, n.split(':')[1] if ':' in n else 'physAll')
        else:
            run(n, folder, out, epoch)


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])


#######################################################################################################################
# References
#######################################################################################################################
