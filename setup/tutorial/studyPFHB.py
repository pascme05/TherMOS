#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         studyPFHB
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
Half-bridge study of the PINN journal paper (data: genBoardPFHB.py). Three methods are trained on the same labelled
boards (train, the first nTr of them) of one mesh and evaluated on the same test sets n1, t1–t4:
    foster  coupled Foster model (pfFoster) fitted on the sensor labels, regressed over the condition μ
    pod     global POD basis (rank r) of the full-field snapshots of the labelled boards on the superset grid of all
            layouts, Galerkin projection of C, G, Bp of every test board (needs the full field, not only sensors)
    pinn    parametric operator network (pfParamFit) trained on the sensor labels plus the physics loss of the
            physics-only boards; 'nophys' is the same network without physics (β = γ = 0)
Nominal layout: the baselines can also be fitted on the boards with the chips at the nominal positions (trainNom,
run names fosterNom / podNom) and are then evaluated on the nominal test sets n1, t2, t4 only; POD requires a fixed
layout, since the field of a moved chip is not contained in the snapshots. The network is always trained on the
variable layouts (train) and evaluated on all test sets (addTests completes the test sets of existing runs).
Sensor versions: S = 4 hot spots, L = 16 points (hot spots, 4 NTC positions, 8 points on the Al base).
Errors (K) against the finite-volume model of the same mesh: hot spots (all methods, every sample) and the whole
field (pod, pinn; every DEC-th sample): MAE, RMSE, MAX, pooled over the boards of a test set, plus the runtime per board.
Results: results/pfhb/<run>.json (+ .pt of the networks), run = <method>_<mesh>_<sens>[_n<nTr>].
Functions:  runNet, addTests, runFoster, podBasis, runPOD, scale, optLayout
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Path
# ==============================================================================
from os.path import dirname, abspath, isdir, join as pjoin
import sys

base = dirname(abspath(__file__))
while not isdir(pjoin(base, 'src', 'py')) and dirname(base) != base:
    base = dirname(base)
sys.path.insert(0, base)

# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfParam import (loadFam, buildBoard, simulate, samplePos, chipCtx, muHat, _ok, CHIPS_HB, NOM,
                                     P_REF, MESH)
from src.py.model.PF.pfParamFit import fitParam, prepBoard, predict, buildNet
from src.py.model.PF import pfFoster
from setup.tutorial.studyPFParam import _peakNet, _penalty

# ==============================================================================
# External
# ==============================================================================
import os
import json
import time
import warnings
import numpy as np
import torch
import scipy.sparse as sp

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
TS = 0.05
TESTS = ['n1', 't1', 't2', 't3', 't4']
TESTS_NOM = ['n1', 't2', 't4']                                          # test sets with the nominal layout
NSENS = {'S': 4, 'L': 16}
NHOT = 4
ZOOM = 60.0                                                             # traces at every sample up to (s)
_CACHE = {}


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def fam(mesh, s):
    """Boards of the set s on the mesh (cached)."""
    if (mesh, s) not in _CACHE:
        _CACHE[(mesh, s)] = loadFam(pjoin(base, 'data', 'board', 'pfhb', mesh, s + '.npz'))
    return _CACHE[(mesh, s)]


def _acc():
    return {'se': 0.0, 'ae': 0.0, 'mx': 0.0, 'n': 0}


def _add(a, e):
    a['se'] += float(np.sum(e ** 2))
    a['ae'] += float(np.sum(np.abs(e)))
    a['mx'] = max(a['mx'], float(np.max(np.abs(e))))
    a['n'] += e.size


def _met(a):
    return {'mae': a['ae'] / a['n'], 'rmse': float(np.sqrt(a['se'] / a['n'])), 'max': a['mx']} if a['n'] else None


def evaluate(pred, mesh, rep, tests=None):
    """
    Errors of a method on the test sets: pred(B, P) -> (hot-spot rises (L x 4), field every dec samples (L/dec x N) or
    None). Writes rep[set] = {'hot', 'field' (MAE, RMSE, MAX in K), 'time' (s per board), 'boards'} and for t4 the
    hot-spot traces (every 1 s and every sample of the first ZOOM s).
    """
    for s in tests or TESTS:
        hot, fld, rows, tm, chip = _acc(), _acc(), [], [], [_acc() for _ in range(0, NHOT)]
        for B in fam(mesh, s):
            P = B['P'][0]
            t0 = time.time()
            yh, yf = pred(B, P)
            tm.append(time.time() - t0)
            e = yh - B['ths'][0][:, 0:NHOT]
            _add(hot, e)
            for c in range(0, NHOT):
                _add(chip[c], e[:, c])
            b = {'mu': B['mu'], 'pos': B['pos'], 'hotRmse': float(np.sqrt(np.mean(e ** 2))),
                 'hotMax': float(np.max(np.abs(e))), 'chipRmse': np.sqrt(np.mean(e ** 2, axis=0)).tolist(),
                 'time': tm[-1]}
            if yf is not None:
                ef = yf - B['th'][0]
                _add(fld, ef)
                b['fieldRmse'] = float(np.sqrt(np.mean(ef ** 2)))
            rows.append(b)
            if s == 't4':
                dec, nz = B['dec'], int(round(ZOOM / TS)) + 1
                rep['trace'] = {'t': (np.arange(len(P)) * TS)[::dec].tolist(),
                                'ref': B['ths'][0][::dec, 0:NHOT].T.tolist(), 'pred': yh[::dec].T.tolist(),
                                'zoomRef': B['ths'][0][0:nz, 0:NHOT].T.tolist(), 'zoomPred': yh[0:nz].T.tolist()}
        rep[s] = {'hot': _met(hot), 'field': _met(fld), 'chip': [_met(a) for a in chip], 'time': float(np.mean(tm)),
                  'boards': rows}
        print("INFO: %s %s: hot spots MAE %.2f K, RMSE %.2f K, MAX %.1f K%s, %.3f s per board" %
              (rep['run'], s, rep[s]['hot']['mae'], rep[s]['hot']['rmse'], rep[s]['hot']['max'],
               '' if rep[s]['field'] is None else '; field RMSE %.2f K' % rep[s]['field']['rmse'], rep[s]['time']))
    return rep


def _save(rep, out):
    with open(pjoin(out, rep['run'] + '.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


def runName(method, mesh, sens, nTr=24):
    return '%s_%s_%s' % (method, mesh, sens) + ('' if nTr == 24 else '_n%d' % nTr)


#######################################################################################################################
# Network
#######################################################################################################################
def loadNet(path, dev):
    ck = torch.load(path, map_location=dev)
    c = ck['cfg']
    net = buildNet(c).to(dev)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ck['dT']


def netPred(net, dT, dev):
    def pred(B, P):
        with torch.no_grad():
            bt = prepBoard(B, dev)
            yh = predict(net, bt, P, dT, TS, rows=torch.as_tensor(np.asarray(B['sens'][0:NHOT]), device=dev))
            yf = predict(net, bt, P, dT, TS, tIdx=torch.arange(0, len(P), B['dec'], device=dev))
            if dev.type == 'cuda':
                torch.cuda.synchronize()
        return yh.cpu().numpy(), yf.cpu().numpy()
    return pred


def runNet(method, mesh, sens, cfg, nTr, out, epoch=0, usePhys=True):
    """Trains the network on the first nTr labelled boards (labels at the sensors) and evaluates it."""
    name = runName(method, mesh, sens, nTr)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(cfg, sens=NSENS[sens], **({'epoch': epoch} if epoch else {}))
    if not usePhys:
        cfg.update(beta=0.0, gamma=0.0)
    train, val = fam(mesh, 'train')[0:nTr], fam(mesh, 'val')
    phys = fam(mesh, 'phys') if usePhys else []
    print("START: run %s (%d labelled, %d physics-only boards, %s labels) on %s" %
          (name, len(train), len(phys), sens, dev))
    res = fitParam(train, phys, val, cfg, dev)
    net, dT = res['net'], res['dT']
    torch.save({'state': net.state_dict(), 'dT': dT, 'cfg': res['cfg']}, pjoin(out, name + '.pt'))
    rep = {'run': name, 'method': method, 'mesh': mesh, 'sens': sens, 'train': nTr, 'phys': len(phys),
           'timeTrain': res['time'], 'epochs': len(res['hist']['val']), 'best': res['best'], 'dT': dT,
           'hist': res['hist'], 'cfg': res['cfg'], 'params': int(sum(q.numel() for q in net.parameters()))}
    _save(evaluate(netPred(net, dT, dev), mesh, rep), out)


def addTests(run, out, tests=None):
    """Evaluates the stored network of run on the test sets that its results do not contain yet (default TESTS)."""
    p = pjoin(out, run + '.json')
    with open(p) as f:
        rep = json.load(f)
    miss = [s for s in (tests or TESTS) if s not in rep]
    if not miss:
        return
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dT = loadNet(pjoin(out, run + '.pt'), dev)
    print("START: run %s, additional test sets %s" % (run, miss))
    _save(evaluate(netPred(net, dT, dev), rep['mesh'], rep, miss), out)


#######################################################################################################################
# Foster
#######################################################################################################################
def runFoster(mesh, sens, K, nTr, out, nom=False):
    """
    Coupled Foster model on the sensor labels of the first nTr labelled boards, regressed over μ; nom: fitted on the
    nominal layout (trainNom, run fosterNom) and evaluated on the nominal test sets.
    """
    method = 'fosterNom' if nom else 'foster'
    name = runName(method, mesh, sens, nTr)
    nS = NSENS[sens]
    t0 = time.time()
    mdl = pfFoster.select(fam(mesh, 'trainNom' if nom else 'train')[0:nTr], fam(mesh, 'valNom' if nom else 'val'), nS,
                          K, TS)
    rep = {'run': name, 'method': method, 'mesh': mesh, 'sens': sens, 'train': nTr, 'K': K,
           'timeTrain': time.time() - t0, 'fits': mdl['fits'], 'params': int(mdl['W'].size),
           'variant': mdl['variant'], 'valRmse': mdl['valRmse']}
    print("START: run %s, K = %d, fit %.1f s, fit RMSE per board %.2f–%.2f K, variant %s (validation RMSE %s)" %
          (name, K, rep['timeTrain'], min(f['rmse'] for f in mdl['fits']), max(f['rmse'] for f in mdl['fits']),
           mdl['variant'], {k: round(v, 2) for k, v in mdl['valRmse'].items()}))
    _save(evaluate(lambda B, P: (pfFoster.predict(mdl, B['mu'], P)[:, 0:NHOT], None), mesh, rep,
                   TESTS_NOM if nom else None), out)


#######################################################################################################################
# POD
#######################################################################################################################
def _keys(B, dx):
    """Key of every node in the superset grid of all layouts: (height in µm, y and x index of the dx grid)."""
    g = np.asarray(B['geo'])
    zu = np.round(g[:, 2] * 1e6).astype(np.int64)
    iy, ix = np.round(g[:, 1] * 1e3 / dx * 2).astype(np.int64), np.round(g[:, 0] * 1e3 / dx * 2).astype(np.int64)
    return (zu * 1000 + iy) * 1000 + ix


def _rom(B, V, P):
    """Galerkin projection of C, G, Bp of board B on V (N x r), implicit Euler with 4 sub-steps; coefficients (L x r)."""
    r, h = V.shape[1], TS / 4
    Cr, Gr, Br = V.T @ (B['C'][:, None] * V), V.T @ (B['G'] @ V), V.T @ B['Bp']
    A = np.linalg.inv(Cr / h + Gr)
    Ah, q, Q = A @ (Cr / h), np.zeros(r), np.zeros((len(P), r))
    U = (A @ Br) @ P.T
    for k in range(1, len(P)):
        for _ in range(0, 4):
            q = Ah @ q + U[:, k]
        Q[k] = q
    return Q


def podBasis(mesh, energy, nTr, rMax=256, nom=True):
    """
    Global POD basis of the full-field snapshots of the first nTr labelled boards (trainNom if nom, else train) on the
    superset grid of all their layouts (method of snapshots). Rank r: smallest r whose singular values hold the
    fraction energy of Σσ² (e.g. 0.9999), at most rMax. Returns (keys, E (superset nodes x r), info).
    """
    dx = MESH[mesh]['dx']
    tr = fam(mesh, 'trainNom' if nom else 'train')[0:nTr]
    t0 = time.time()
    keys = np.unique(np.concatenate([_keys(B, dx) for B in tr]))
    X = []
    for B in tr:
        s = np.searchsorted(keys, _keys(B, dx))
        for th in B['th']:
            Z = np.zeros((len(th), len(keys)), dtype=np.float32)
            Z[:, s] = th
            X.append(Z)
    X = np.concatenate(X).astype(np.float64)
    [w, U] = np.linalg.eigh(X @ X.T)
    w, U = w[::-1], U[:, ::-1]
    r = int(min(np.searchsorted(np.cumsum(np.maximum(w, 0)) / np.sum(np.maximum(w, 0)), energy) + 1, rMax))
    E = (X.T @ U[:, 0:r]) / np.sqrt(np.maximum(w[0:r], 1e-30))                    # superset nodes x r
    info = {'r': r, 'energy': energy, 'timeTrain': time.time() - t0, 'snapshots': int(X.shape[0]),
            'sv': np.sqrt(np.maximum(w[0:100], 0) / w[0]).tolist(), 'nodes': int(len(keys)), 'params': int(E.size),
            'energyKept': float(np.sum(w[0:r]) / np.sum(w))}
    return keys, E, info


def podV(keys, E, B, dx):
    """Basis of board B (N x r): rows of its nodes in the superset basis E (zero for unknown nodes), orthonormalised."""
    k = _keys(B, dx)
    i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    return np.linalg.qr(np.where((keys[i] == k)[:, None], E[i], 0.0))[0]


def podPred(keys, E, dx):
    """Prediction of a board (hot spots, field every dec samples): rows of its nodes, orthonormalised, Galerkin."""
    def pred(B, P):
        V = podV(keys, E, B, dx)
        Q = _rom(B, V, P)
        return Q @ V[B['sens'][0:NHOT]].T, Q[::B['dec']] @ V.T
    return pred


def runPOD(mesh, energy, nTr, out, rMax=256, nom=False):
    """
    POD (podBasis) of the first nTr labelled boards; for a test board the rows of its nodes are taken (zero for nodes
    that no training board has, e.g. die cells in the band), orthonormalised and used for a Galerkin projection. nom:
    basis of the nominal layout (trainNom, run podNom), evaluated on the nominal test sets.
    """
    method = 'podNom' if nom else 'pod'
    name = runName(method, mesh, 'F', nTr)
    keys, E, info = podBasis(mesh, energy, nTr, rMax, nom)
    rep = dict({'run': name, 'method': method, 'mesh': mesh, 'sens': 'F', 'train': nTr}, **info)
    print("START: run %s, r = %d, %d snapshots on %d superset nodes, basis %.1f s, energy %.6f" %
          (name, info['r'], info['snapshots'], info['nodes'], info['timeTrain'], info['energyKept']))
    _save(evaluate(podPred(keys, E, MESH[mesh]['dx']), mesh, rep, TESTS_NOM if nom else None), out)


#######################################################################################################################
# Field map
#######################################################################################################################
def fieldMap(mesh, out, board=0, energy=0.9999, rMax=256):
    """
    Field of the nominal testing board n1[board] on the top of the copper layer at the peak of the hot spot of T_H:
    FV, and the errors of POD (nominal layout) and of the networks with the sensor versions S and L. Writes
    fieldmap_<mesh>.json (x, y in mm, FV rise and errors in K, time of the peak in s).
    """
    from src.py.model.PF.pfParam import MATERIALS
    B = fam(mesh, 'n1')[board]
    P = B['P'][0]
    th = simulate(B, P, TS)
    k = int(np.argmax(th[:, B['sens'][0]]))
    geo = np.asarray(B['geo'])
    rho = np.asarray(buildBoard(B['mu'], B['pos'], CHIPS_HB, mesh)['rho'])            # material of the nodes
    cu = np.flatnonzero(np.abs(rho - MATERIALS['Cu'][1]) < 1.0)
    cu = cu[geo[cu, 2] >= geo[cu, 2].max() - 1e-9]
    rep = {'mesh': mesh, 'board': board, 'mu': B['mu'], 't': k * TS, 'x': (geo[cu, 0] * 1e3).tolist(),
           'y': (geo[cu, 1] * 1e3).tolist(), 'fv': th[k, cu].tolist()}
    keys, E, _ = podBasis(mesh, energy, 24, rMax, True)
    V = podV(keys, E, B, MESH[mesh]['dx'])
    rep['pod'] = (_rom(B, V, P[0:k + 1])[k] @ V[cu].T - th[k, cu]).tolist()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for sens in ('S', 'L'):
        net, dT = loadNet(pjoin(out, runName('pinn', mesh, sens) + '.pt'), dev)
        with torch.no_grad():
            f = predict(net, prepBoard(B, dev), P, dT, TS, rows=torch.as_tensor(cu, device=dev),
                        tIdx=torch.as_tensor([k], device=dev)).cpu().numpy()[0]
        rep['pinn' + sens] = (f - th[k, cu]).tolist()
    print("INFO: field map %s, board %d, t = %.2f s: RMSE POD %.2f K, proposed S %.2f K, L %.2f K" %
          (mesh, board, rep['t'], *[float(np.sqrt(np.mean(np.square(rep[m])))) for m in ('pod', 'pinnS', 'pinnL')]))
    with open(pjoin(out, 'fieldmap_%s.json' % mesh), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# Memory and computational load
#######################################################################################################################
def _best(fn, dev, n=3):
    """Shortest wall time (s) of n calls of fn after one warm-up call (GPU synchronised)."""
    ts = []
    for i in range(0, n + 1):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if i > 0:
            ts.append(time.time() - t0)
    return min(ts)


def scale(mesh, out, lengths=(40.0, 300.0, 1800.0, 14400.0), nFV=300.0):
    """
    Size and cost per board on the mesh: N, nnz(G), nnz of the LU factors, assembly and FV time (measured up to nFV s,
    linear beyond), and the time of Foster, POD and network for the profile lengths (s) (t4 profile repeated).
    """
    from scipy.sparse.linalg import splu
    rep = {'mesh': mesh}
    B4 = fam(mesh, 't4')[0]
    t0 = time.time()
    B = buildBoard(B4['mu'], B4['pos'], CHIPS_HB, mesh)
    rep['assembly'] = time.time() - t0
    N = len(B['C'])
    lu = splu((sp.diags(B['C'] / (TS / 4)) + B['G']).tocsc(), permc_spec='MMD_AT_PLUS_A')
    rep.update({'N': N, 'nnzG': int(B['G'].nnz), 'nnzLU': int(lu.L.nnz + lu.U.nnz),
                'memFV': float((lu.L.nnz + lu.U.nnz) * 12 + B['G'].nnz * 12 + N * 8 * 3)})
    P0 = B4['P'][0]
    rows = []
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = mdlF = None
    pt = pjoin(out, runName('pinn', mesh, 'L') + '.pt')
    if os.path.isfile(pt):
        net, dT = loadNet(pt, dev)
    if os.path.isfile(pjoin(out, runName('foster', mesh, 'L') + '.json')):
        mdlF = pfFoster.fitFam(fam(mesh, 'train'), NSENS['L'], 4, TS)
    pp = pjoin(out, runName('podNom', mesh, 'F') + '.json')
    rPod = int(json.load(open(pp))['r']) if os.path.isfile(pp) else 32            # rank of the POD run of this mesh
    rep['rPod'] = rPod
    for Lsec in lengths:
        L = int(round(Lsec / TS)) + 1
        P = np.tile(P0, (int(np.ceil(L / len(P0))), 1))[0:L]
        r = {'L': Lsec, 'samples': L}
        if Lsec <= nFV:
            t0 = time.time()
            simulate(B, P, TS)
            r['fv'] = time.time() - t0
        if mdlF is not None:
            r['foster'] = _best(lambda: pfFoster.predict(mdlF, B['mu'], P), dev)
        V = np.linalg.qr(np.random.default_rng(0).standard_normal((N, rPod)))[0]

        def runPod():
            Cr, Gr, Br = V.T @ (B['C'][:, None] * V), V.T @ (B['G'] @ V), V.T @ B['Bp']
            A = np.linalg.inv(Cr / (TS / 4) + Gr)
            Ah, U, q = A @ (Cr / (TS / 4)), (A @ Br) @ P.T, np.zeros(rPod)
            Q = np.zeros((L, rPod))
            for k in range(1, L):
                for _ in range(0, 4):
                    q = Ah @ q + U[:, k]
                Q[k] = q
            return Q @ V[B['sens'][0:NHOT]].T
        r['pod'] = _best(runPod, dev)
        if net is not None:
            with torch.no_grad():
                bt = prepBoard(B, dev)
                rows_ = torch.as_tensor(np.asarray(B['sens'][0:NHOT]), device=dev)
                r['pinn'] = _best(lambda: predict(net, bt, P, dT, TS, rows=rows_), dev)
                r['pinnField'] = _best(lambda: predict(net, bt, P[0:min(L, 801)], dT, TS), dev)
        rows.append(r)
        print("INFO: scale %s N = %d, L = %.0f s: %s" % (mesh, N, Lsec, {k: round(v, 3) for k, v in r.items()}))
    fvs = [(r['samples'], r['fv']) for r in rows if 'fv' in r]
    for r in rows:
        if 'fv' not in r:
            r['fv'] = fvs[-1][1] * r['samples'] / fvs[-1][0]
            r['fvExtrap'] = True
    rep['rows'] = rows
    with open(pjoin(out, 'scale_%s.json' % mesh), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# Layout Optimisation
#######################################################################################################################
def _peakFV(mu, pos, P, mesh):
    B = buildBoard(mu, pos, CHIPS_HB, mesh)
    th = simulate(B, P, TS)
    return np.array([float(th[:, d].max()) for d in B['die']])


def optLayout(run, mesh, out, nStart=8, nIter=300, nRand=100, tauS=2.0, wPen=50.0, seed=0):
    """
    Layout optimisation through the network of run: the chip centres minimise the smooth maximum of the peak junction
    temperatures J = τ·log Σ_c exp(max_t T_j,c / τ) + w·penalty (Adam, nStart random starts) for the designed profile
    at the nominal condition; the best layout (rounded to 1 mm) is checked with FV and compared with an FV random
    search over nRand layouts, on which the network is evaluated as well (error and rank correlation of the peaks).
    """
    from setup.tutorial.genBoardPFHB import designed
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dT = loadNet(pjoin(out, run + '.pt'), dev)
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 40.0 + TS * 1e-9, TS)
    P = designed(t)
    p = torch.as_tensor(P.T / P_REF, dtype=torch.float32, device=dev)
    ch = CHIPS_HB
    B0 = buildBoard(NOM, [(c['x'], c['y']) for c in ch], ch, mesh)
    off = [torch.as_tensor(np.column_stack([B0['geo'][d, 0] * 1e3 - c['x'], B0['geo'][d, 1] * 1e3 - c['y'],
                                            B0['geo'][d, 2]]), dtype=torch.float32, device=dev)
           for d, c in zip(B0['die'], ch)]
    ctxC = torch.as_tensor(np.stack([chipCtx(B0, c)[2:] for c in range(0, len(ch))]), device=dev)
    muH = torch.as_tensor(muHat(NOM), device=dev)[None]

    def peaks(pos):
        with torch.no_grad():
            return _peakNet(net, dT, torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=dev), off, ctxC, muH,
                            p).max(dim=1).values.cpu().numpy()

    t0 = time.time()
    runs = []
    for _ in range(0, nStart):
        x = torch.tensor(samplePos(rng, chips=ch), dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nIter)
        path = []
        for _ in range(0, nIter):
            opt.zero_grad()
            pk = _peakNet(net, dT, x, off, ctxC, muH, p).max(dim=1).values
            J = tauS * torch.logsumexp(pk / tauS, dim=0) + wPen * _penalty(x, ch)
            J.backward()
            opt.step()
            sch.step()
            path.append(x.detach().cpu().numpy().copy())
        pos = [tuple(map(float, np.round(v))) for v in x.detach().cpu().numpy()]
        runs.append({'start': path[0].tolist(), 'end': path[-1].tolist(), 'pos': pos, 'ok': _ok(pos, ch),
                     'peakNet': peaks(pos).tolist(), 'path': np.asarray(path)[::10].tolist()})
    tOpt = time.time() - t0
    good = [r for r in runs if r['ok']] or runs
    best = min(good, key=lambda r: max(r['peakNet']))
    t0 = time.time()
    best['peakFV'] = _peakFV(NOM, best['pos'], P, mesh).tolist()
    tCheck = time.time() - t0
    nom = [(float(c['x']), float(c['y'])) for c in ch]
    rep = {'model': run, 'timeOpt': tOpt, 'timeCheck': tCheck, 'runs': runs, 'best': best,
           'nominal': {'pos': nom, 'peakNet': peaks(nom).tolist(), 'peakFV': _peakFV(NOM, nom, P, mesh).tolist()}}
    print("INFO: layout: nominal FV %s K, optimised %s FV %s K, network %.1f s" %
          (np.round(rep['nominal']['peakFV'], 1), best['pos'], np.round(best['peakFV'], 1), tOpt))
    t0 = time.time()
    rnd = [samplePos(rng, chips=ch) for _ in range(0, nRand)]
    pkF = np.array([_peakFV(NOM, q, P, mesh) for q in rnd])
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
    print("INFO: FV random search (%d layouts): best %.1f K, %.0f s; network %.1f s, peak RMSE %.2f K, rank corr. %.3f"
          % (nRand, jF.min(), tRand, tRandNet, rep['random']['rmsePeak'], rep['random']['spearman']))
    with open(pjoin(out, 'layout_%s.json' % run), 'w') as f:
        json.dump(rep, f, indent=1, default=float)


#######################################################################################################################
# References
#######################################################################################################################
