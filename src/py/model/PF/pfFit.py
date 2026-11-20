#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfFit
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function trains the physics-informed field network (PF mode) for 2D or 3D temperature fields T(x, t) with heat
sources Q(x, t) that vary in space and time. The heat generation is encoded as Q(x, t) ≈ Σ_j U_j(x)·p_j(t) (pfData),
the network predicts the temperature rise θ = T − T_ref. Two variants (setup['selPF']):

1) Operator network: θ̂(x, t) = ΔT·Σ_k b_k(t)·φ_k(x) with a sequence network (branch, dlNet) from the source
   amplitudes p(t) to the coefficients b(t) and a fully connected network (trunk) from the position x to the basis
   functions φ(x) (pfNet). It is trained on windows of W samples of all training profiles and predicts profiles with
   other loads. Every profile is preceded by 'warm' samples without sources (thermal equilibrium, θ = 0); the first
   'warm' samples of every window are excluded from the data and the physics loss (the memory of the branch settles).
2) Coordinate network: θ̂(x, t) = ΔT·f(x, t) with one fully connected network. It is trained on ONE profile, the
   testing profile, from the temperatures at the sensor nodes only (every fifth sensor is held out for the
   validation) and reconstructs the field at all other nodes with the heat equation (the method of [1]).

The loss has three dimensionless parts (ΔT: temperature scale, 99 % quantile of the measured rise):

    L = α·L_data + β·L_phys + γ·L_init
    L_data = mean(((θ̂ − θ)/ΔT)²)                        measured temperatures at the sensor nodes (nDat per step)
    L_phys = mean(r²)                                    residual of the heat equation (pfPhys), phys = 1: at nCol
                                                         collocation points (random nodes moved by up to half a cell),
                                                         phys = 2: at nCol nodes with the finite-volume operator;
                                                         variant 1, col = 1: also for the sources of the validation
                                                         and testing profiles (collocation profiles: only their
                                                         inputs are used, the physics needs no temperatures)
    L_init = mean((θ̂/ΔT)²) for t ≤ 0                    thermal equilibrium at the start of every profile

Time discretisation of the residual: the heat generation of sample k is taken to act during the step from sample
k − 1 to k (sample and hold, the convention of the board designer and of sampled measurements). Variant 1 evaluates
the residual in the middle of every step (Crank-Nicolson): dθ/dt = (θ_k − θ_k−1)/Ts, θ = (θ_k + θ_k−1)/2, Q = Q_k.
Variant 2 differentiates the network in time and uses Q of the step that contains the collocation time.

Training: Adam, learning rate schedule and gradient clipping as in pinnFit of the PI mode (Mdl › dl: initLr, lrAdapt,
lrDropPr, lrDropFa, gradTh), early stopping after valPat epochs without improvement of the validation error; the
network with the lowest validation error is kept. Variant 1: validation profiles predicted as in testing; one epoch
is one pass over all windows. Variant 2: held-out sensors; one epoch is 'steps' optimiser steps.
Inputs:     1) data:    training data (variant 2: the testing data in data['_te'])
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables (selPF, archPF)
            5) legacy:  not used (the PF mode has no MATLAB counterpart)
Outputs:    1) mdl:     model {'sys': network, 'var', 'info': training history, 'sc': coordinate scaling, 'U': source
                        basis, 'geo', 'dT', 'sens': sensor nodes, ...}

V1.0: 16.11.2026, Pascal Schirmer
- Initial version

V1.1: 20.11.2026, Pascal Schirmer
- Physics 2 with temperature-dependent conductivity (kT of the data file), physics 3: inverse finite-volume operator
  with learned factors of conduction and cooling (info['sK'], info['sH'])
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlNet import dlNet
from src.py.model.PF.pfNet import pfOperator, pfCoord
from src.py.model.PF.pfData import pfPara, checkNorm, geoScale, toHat, rise, encodeSrc, projectSrc, mapNodes, \
    selSensors, material, VARIANTS, PHYS
from src.py.model.PF.pfPhys import basisLap, resOperator, resOperatorFV, fvTerm, coordDeriv, coordRate, solveG
from src.py.model.PINN.pinnFit import lrSchedule
from src.py.general.getDevice import getDevice

# ==============================================================================
# External
# ==============================================================================
import copy
import sys
import time
import numpy as np
import torch


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _t(a, dev):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def _scale(th):
    """Temperature scale ΔT (K): 99 % quantile of the measured rises th (list of Nt x n)."""
    return float(max(np.percentile(np.concatenate([np.abs(x).ravel() for x in th]), 99), 1e-6))


def _physics(mat, U, sc, phys, dev):
    """
    Tensors of the physics residual: ρc, κ = k·s², h_v, U, conductance scale g = Σ 2·k_i/ℓ_i² + h_v (ℓ_i = 1/s_i:
    half the extent of the domain), the jitter of the collocation points (half a cell) and for phys = 2
    the operator: C, the function x → G⁻¹·x and the steady-state fields Θ = G⁻¹·V·U of the source footprints.
    """
    h = np.where(mat['h'] > 0, mat['h'], (2.0 / sc['s']) / max(len(mat['rc']) ** (1.0 / sc['d']), 1.0))
    ph = {'rc': _t(mat['rc'], dev), 'kap': _t(mat['k'] * sc['s'][None, :] ** 2, dev), 'hv': _t(mat['hv'], dev),
          'U': _t(U, dev), 'jit': _t(0.5 * h * sc['s'], dev),
          'g': _t(np.sum(2 * mat['k'] * sc['s'][None, :] ** 2, axis=1) + mat['hv'], dev)}
    kr = contrast(mat)
    if phys == 1 and kr > KMAX:
        print("WARN: Physics 1 on a domain with material interfaces (conductivity ratio %.0f): the residual with "
              "automatic differentiation neglects the jump of k between the materials and can pull the network away "
              "from the data; use physics 2 (board designer data) or physics 0" % kr)
    ph['kT'], ph['ls'], ph['H'] = 0.0, None, None
    if phys >= 2:
        if mat['op'] is None:
            sys.exit("ERROR: PF mode: physics %d needs the finite-volume operator in the data file (Cn, Gi, Gj, Gv, "
                     "Vn, written by the board designer); use physics 1 (automatic differentiation) for other data"
                     % phys)
        import scipy.sparse.linalg as spl
        lu = spl.splu(mat['op']['G'].tocsc())
        ph['C'] = _t(mat['op']['C'], dev)
        ph['solve'] = solveG(lu)
        ph['Tss'] = _t(lu.solve(mat['op']['V'][:, None] * U), dev)
        ph['kT'] = float(mat.get('kT', 0.0) or 0.0)
        if ph['kT']:
            print("INFO: Temperature-dependent conductivity of the data file: k = k0·(1 − %.4g·θ), residual in the "
                  "Kirchhoff variable u = θ − κ·θ²/2" % ph['kT'])
        if phys == 3:
            ph['H'] = _t(np.asarray(mat['op']['G'].sum(axis=1)).ravel(), dev)     # cooling: row sums of G
            ph['ls'] = torch.zeros(2, device=dev, requires_grad=True)             # log s_k, log s_h
    return ph


def factors(ph):
    """Learned factors (s_k, s_h) of conduction and cooling (phys = 3) as a tensor, or None."""
    return None if ph['ls'] is None else torch.exp(ph['ls'])


def _optimiser(net, ph, lr):
    """Adam for the network and, for phys = 3, the log factors of the operator with SLR times the learning rate."""
    groups = [{'params': list(net.parameters()), 'fac': 1.0}]
    if ph['ls'] is not None:
        groups.append({'params': [ph['ls']], 'lr': lr * SLR, 'fac': SLR})
    return torch.optim.Adam(groups, lr=lr)


def _factors(info, s):
    """Stores and prints the learned factors of conduction and cooling (phys = 3) of the kept network."""
    if s is None:
        return
    info['sK'], info['sH'] = float(s[0]), float(s[1])
    print("INFO: Learned factors of the finite-volume operator: conduction s_k = %.3f, cooling s_h = %.3f (1 = data "
          "file)" % (s[0], s[1]))


def contrast(mat):
    """Ratio of the largest to the smallest thermal conductivity of the nodes (all directions)."""
    k = np.asarray(mat['k'], dtype=float)
    return float(np.max(k) / max(np.min(k), 1e-300))


def _colloc(xh, ph, n):
    """Collocation points: n random nodes moved by up to half a cell (inside the domain) and their node indices."""
    idc = torch.randint(0, xh.shape[0], (n,), device=xh.device)
    xc = xh[idc] + (2 * torch.rand(n, xh.shape[1], device=xh.device) - 1) * ph['jit']
    return idc, torch.clamp(xc, -1.0, 1.0)


def _history():
    return {'loss': [], 'lossData': [], 'lossPhys': [], 'lossInit': [], 'valLoss': [], 'learnRate': [], 'time': []}


def _log(info, ep, epoch, lrEp, m, lv, t0):
    for k, v in zip(('loss', 'lossData', 'lossPhys', 'lossInit'), m):
        info[k].append(float(v))
    info['valLoss'].append(float(lv))
    info['learnRate'].append(float(lrEp))
    info['time'].append(time.time() - t0)
    print("INFO: Epoch %d/%d, lr %.3e, loss %.4e (data %.3e, physics %.3e, init %.3e), validation RMSE %.3f K" %
          (ep + 1, epoch, lrEp, m[0], m[1], m[2], m[3], np.sqrt(lv)))


#######################################################################################################################
# Operator Network
#######################################################################################################################
def _fitOperator(data, val, para, setup, pp, dl, dev):
    # ==============================================================================
    # Parameters
    # ==============================================================================
    init = int(para['Exp']['gen'].get('init', 1) or 1)
    Ts = float(data['Ts'])
    W, stride = int(dl['W']), max(int(dl['stride']), 1)
    epoch, batch, lr = int(dl['epoch']), int(dl['batch']), float(dl['initLr'])
    valPat, gradTh = int(dl.get('valPat', 10) or 10), float(dl.get('gradTh', 0) or 0)
    K, phys, warm = pp['K'], pp['phys'], max(pp['warm'], 0)
    wD, wP, wI = pp['alpha'], pp['beta'] if phys > 0 else 0.0, pp['gamma']

    # ==============================================================================
    # Geometry, Sources and Sensors
    # ==============================================================================
    D = data['Data']
    geo = np.asarray(D['geo'], dtype=float)
    sc = geoScale(geo)
    d, N = sc['d'], len(geo)
    xh = toHat(geo, sc)
    th2 = rise(data, init)
    [U, sv] = encodeSrc(data['X2'], pp['srcTol'])
    S = U.shape[1]
    if S == 0:
        sys.exit("ERROR: PF mode: the training data have no heat sources (X = 0)")
    p2 = [projectSrc(q, U)[0] for q in data['X2']]
    idxS = selSensors(xh, pp['nSens'], pp['seed'], int(np.argmax(np.max(np.abs(th2[0]), axis=0))))
    dT = _scale([x[:, idxS] for x in th2])
    mat = material(D, d)
    print("INFO: %dD field with %d nodes, %d training profile(s), Ts = %g s" % (d, N, len(th2), Ts))
    print("INFO: Source encoding: %d amplitude(s) p_j(t) (singular values above %g·σ_1; σ_j/σ_1 = %s)" %
          (S, pp['srcTol'], np.array2string(sv[0:min(len(sv), S + 2)], precision=2)))
    print("INFO: %d sensor node(s) (%s), temperature scale ΔT = %.4g K" %
          (len(idxS), 'all nodes' if len(idxS) == N else 'farthest-point sampling', dT))
    print("INFO: Physics: %s; alpha = %g, beta = %g, gamma = %g" % (PHYS.get(phys, '?'), wD, wP, wI))

    # ==============================================================================
    # Windows (every profile is preceded by 'warm' samples of equilibrium)
    # ==============================================================================
    W = int(min(W, min(len(x) for x in th2) + warm))
    Pa, Ta, pad, st, o = [], [], [], [], 0
    for p, th in zip(p2, th2):
        L = len(p) + warm
        Pa += [np.zeros((warm, S)), p]
        Ta += [np.zeros((warm, len(idxS))), th[:, idxS]]
        pad += [np.ones(warm), np.zeros(len(p))]
        st += list(o + np.arange(0, L - W + 1, stride))
        o = o + L
    Pa, Ta, pad = _t(np.concatenate(Pa), dev), _t(np.concatenate(Ta) / dT, dev), _t(np.concatenate(pad), dev)
    st = torch.as_tensor(np.asarray(st), dtype=torch.long, device=dev)
    win = torch.arange(W, device=dev)
    nD, nC = min(pp['nDat'], len(idxS)), max(pp['nCol'], 1)
    print("INFO: %d training windows (W = %d samples = %g s, stride %d, warm-up %d), %d sensor(s) and %d physics "
          "point(s) per step" % (len(st), W, W * Ts, stride, warm, nD, nC if phys else 0))
    if W <= warm + 2:
        sys.exit("ERROR: PF mode: the window W = %d must be longer than the warm-up of %d samples" % (W, warm))

    # ==============================================================================
    # Validation Profiles
    # ==============================================================================
    Dv = val.get('Data') or D
    mp = mapNodes(geo, np.asarray(Dv['geo'], dtype=float))
    Uv = U[mp]
    iv = idxS[np.linspace(0, len(idxS) - 1, min(len(idxS), 2048)).astype(int)]
    iv = iv[iv < len(mp)]
    xv = _t(toHat(np.asarray(Dv['geo'], dtype=float)[iv], sc), dev)
    vl = [(_t(np.vstack([np.zeros((warm, S)), projectSrc(q, Uv)[0]]), dev), _t(th[:, iv], dev))
          for q, th in zip(val['X2'], rise(val, init))]

    # ==============================================================================
    # Collocation Profiles (sources without temperatures, physics loss only)
    # ==============================================================================
    Pc, stC = None, None
    if wP > 0 and pp['col'] == 1:
        pc2 = [projectSrc(q, Uv)[0] for q in val['X2']]
        te = data.get('_te')
        if te is not None:
            Ue = U[mapNodes(geo, np.asarray((te.get('Data') or D)['geo'], dtype=float))]
            pc2 += [projectSrc(q, Ue)[0] for q in te['X2']]
        tempP, tempS, o = [], [], 0
        for p in pc2:
            L = len(p) + warm
            if L < W:
                continue
            tempP += [np.zeros((warm, S)), p]
            tempS += list(o + np.arange(0, L - W + 1, stride))
            o = o + L
        if tempS:
            Pc, stC = _t(np.concatenate(tempP), dev), torch.as_tensor(np.asarray(tempS), dtype=torch.long, device=dev)
            print("INFO: %d collocation windows (sources of the validation and testing profiles, physics loss only)"
                  % len(stC))

    # ==============================================================================
    # Network
    # ==============================================================================
    pAll = np.concatenate(p2)
    sd = pAll.std(axis=0)
    sd[sd < 1e-12 * max(np.max(np.abs(pAll)), 1e-300)] = 1.0
    branch = dlNet(pp['branch'], S, K, 0, pAll.mean(axis=0), sd, arch=setup.get('archPF') or None)
    net = pfOperator(branch, d, K, pp['width'], pp['depth'], pp['fourier']).to(dev)
    xhT, iS = _t(xh, dev), torch.as_tensor(idxS, dtype=torch.long, device=dev)
    ph = _physics(mat, U, sc, phys, dev)
    opt = _optimiser(net, ph, lr)
    [lrStep, _, _, _] = lrSchedule(dl, lr)
    print("INFO: Branch %s (%d inputs → %d coefficients), trunk %d x %d tanh, %d Fourier frequencies (%d → %d basis "
          "functions), %d parameters" % ({1: 'CNN', 2: 'LSTM', 3: 'CNN-LSTM'}.get(pp['branch'], 'user-defined')
                                         if not setup.get('archPF') else 'user-defined', S, K, pp['depth'],
                                         pp['width'], pp['fourier'], d, K, sum(q.numel() for q in net.parameters())))

    def lossFnc(idx):
        ii = st[idx][:, None] + win[None, :]                            # batch x W
        P, m0 = Pa[ii], pad[ii]
        cols = torch.randperm(len(iS), device=dev)[0:nD]
        b = net.coeff(P)
        pred = b @ net.basis(xhT[iS[cols]]).T
        tgt = Ta[ii.reshape(-1)][:, cols].reshape(len(idx), W, nD)
        lD = torch.mean((pred - tgt)[:, warm:] ** 2)
        lI = torch.sum(m0[..., None] * pred ** 2) / torch.clamp(torch.sum(m0) * nD, min=1.0)
        lP = torch.zeros((), device=dev)
        if wP > 0:
            if Pc is not None:                                           # collocation windows
                Pq = Pc[stC[torch.randint(0, len(stC), (len(idx),), device=dev)][:, None] + win[None, :]]
                P, b = torch.cat([P, Pq]), torch.cat([b, net.coeff(Pq)])
            db, bm, Pk = (b[:, 1:] - b[:, :-1]) / Ts, 0.5 * (b[:, 1:] + b[:, :-1]), P[:, 1:]    # step k − 1 → k
            if phys == 1:
                [idc, xc] = _colloc(xhT, ph, nC)
                r = resOperator(bm, db, net.basis(xc), basisLap(net.trunk, xc, ph['kap'][idc]), Pk, ph['rc'][idc],
                                ph['hv'][idc], ph['U'][idc], ph['g'][idc], dT)
            else:
                phi = net.basis(xhT)
                idc = torch.randperm(N, device=dev)[0:min(nC, N)]
                Hphi = ph['solve'](ph['H'][:, None] * phi)[idc] if phys == 3 else None
                r = resOperatorFV(bm, db, phi[idc], ph['solve'](ph['C'][:, None] * phi)[idc], Pk, ph['Tss'][idc], dT,
                                  ph['kT'], Hphi, factors(ph))
            lP = torch.mean(r[:, warm:] ** 2)
        return wD * lD + wP * lP + wI * lI, lD, lP, lI

    def valFnc():
        se, n = 0.0, 0
        with torch.no_grad():
            phi = net.basis(xv)
            for P, th in vl:
                e = dT * (net.coeff(P[None])[0, warm:] @ phi.T) - th
                se, n = se + float(torch.sum(e ** 2)), n + e.numel()
        return se / max(n, 1)

    # ==============================================================================
    # Training
    # ==============================================================================
    best, bestNet, wait, bestS = np.inf, copy.deepcopy(net.state_dict()), 0, None
    info = _history()
    t0 = time.time()
    for ep in range(0, epoch):
        net.train()
        order = torch.randperm(len(st), device=dev)
        sums, nB = np.zeros(4), 0
        for i in range(0, len(st), batch):
            idx = order[i:i + batch]
            if len(idx) < 2 and len(st) > 1:                            # BatchNorm needs two samples
                continue
            opt.zero_grad()
            out = lossFnc(idx)
            out[0].backward()
            if gradTh > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradTh)
            opt.step()
            sums, nB = sums + np.array([float(v.detach()) for v in out]), nB + 1
        lrEp = opt.param_groups[0]['lr']
        net.eval()
        lv = valFnc()
        _log(info, ep, epoch, lrEp, sums / max(nB, 1), lv, t0)
        new = lrStep(lv)
        if new is not None:
            for g in opt.param_groups:
                g['lr'] = new * g['fac']
        if lv < best:
            best, bestNet, wait = lv, copy.deepcopy(net.state_dict()), 0
            bestS = None if ph['ls'] is None else factors(ph).detach().cpu().numpy()
        else:
            wait = wait + 1
            if wait >= valPat:
                print("INFO: Early stopping (validation patience %d)" % valPat)
                break

    # ==============================================================================
    # Model
    # ==============================================================================
    net.load_state_dict(bestNet)
    info.update({'best': best, 'nWin': len(st), 'valMode': 'validation profiles predicted as in testing'})
    _factors(info, bestS if ph['ls'] is not None else None)
    with torch.no_grad():
        phi = net.eval().basis(xhT).cpu().numpy()                       # basis functions at the training nodes
    return {'sys': net.to('cpu').eval(), 'info': info, 'U': U, 'phi': phi, 'geo': geo.astype(np.float32), 'W': W, 'S': S, 'K': K,
            'warm': warm, 'sens': idxS, 'sv': sv[0:SVN], 'dT': dT, 'sc': sc, 'Ts': Ts}


SVN = 20                                                                # stored singular values of the sources
SLR = 10.0                                                              # learning rate factor of the operator factors
KMAX = 10.0                                                             # conductivity ratio above which physics 1 warns


#######################################################################################################################
# Coordinate Network
#######################################################################################################################
def _fitCoord(data, para, pp, dl, dev):
    # ==============================================================================
    # Parameters
    # ==============================================================================
    te = data.get('_te')
    if te is None:
        sys.exit("ERROR: PF mode, coordinate network: the testing data are needed for the training")
    init = int(para['Exp']['gen'].get('init', 1) or 1)
    Ts = float(te['Ts'])
    epoch, lr, steps = int(dl['epoch']), float(dl['initLr']), max(pp['steps'], 1)
    valPat, gradTh = int(dl.get('valPat', 10) or 10), float(dl.get('gradTh', 0) or 0)
    phys = pp['phys']
    wD, wP, wI = pp['alpha'], pp['beta'] if phys > 0 else 0.0, pp['gamma']
    if len(te['y2']) > 1:
        print("WARN: The coordinate network describes one profile, only the first testing profile is used")

    # ==============================================================================
    # Geometry, Sources and Sensors
    # ==============================================================================
    D = te['Data']
    geo = np.asarray(D['geo'], dtype=float)
    sc = geoScale(geo)
    d, N = sc['d'], len(geo)
    xh = toHat(geo, sc)
    th = rise(te, init)[0]
    Nt = len(th)
    tEnd = max((Nt - 1) * Ts, Ts)
    [U, sv] = encodeSrc([te['X2'][0]], pp['srcTol'])
    S = U.shape[1]
    p = projectSrc(te['X2'][0], U)[0] if S else np.zeros((Nt, 0))
    idxS = selSensors(xh, pp['nSens'], pp['seed'], int(np.argmax(np.max(np.abs(th), axis=0))))
    if len(idxS) == N:
        print("WARN: nSens = 0: all nodes are sensors, the network only fits the data; set the number of sensors "
              "(Mdl › pinf › nSens) to reconstruct the field from few measurements")
    hold = np.arange(len(idxS)) % 5 == 4 if len(idxS) >= 5 else np.zeros(len(idxS), dtype=bool)
    idxT, idxV = idxS[~hold], idxS[hold] if hold.any() else idxS
    dT = _scale([th[:, idxT]])
    mat = material(D, d)
    print("INFO: %dD field with %d nodes, one profile with %d samples (Ts = %g s, %g s)" % (d, N, Nt, Ts, tEnd))
    print("INFO: %d sensor node(s) for the training, %d held out for the validation; the other %d nodes are only "
          "used to evaluate the error" % (len(idxT), len(idxV) if hold.any() else 0, N - len(idxS)))
    print("INFO: Source encoding: %d amplitude(s); temperature scale ΔT = %.4g K" % (S, dT))
    print("INFO: Physics: %s; alpha = %g, beta = %g, gamma = %g" % (PHYS.get(phys, '?'), wD, wP, wI))

    # ==============================================================================
    # Network
    # ==============================================================================
    net = pfCoord(d, pp['width'], pp['depth'], pp['fourier']).to(dev)
    ph = _physics(mat, U, sc, phys, dev)
    opt = _optimiser(net, ph, lr)
    [lrStep, _, _, _] = lrSchedule(dl, lr)
    xhT, pT = _t(xh, dev), _t(p, dev)
    tH = _t(2 * np.arange(Nt) * Ts / tEnd - 1, dev)                     # scaled time of every sample
    sT = 2.0 / tEnd                                                     # ∂t̂/∂t (1/s)
    ThT, ThV = _t(th[:, idxT] / dT, dev), _t(th[:, idxV], dev)
    iT = torch.as_tensor(idxT, dtype=torch.long, device=dev)
    xV = xhT[torch.as_tensor(idxV, dtype=torch.long, device=dev)]
    nD, nC = max(pp['nDat'], 1), max(pp['nCol'], 1)
    nT = int(min(max(nC // N, 1), 8))                                   # phys = 2: time samples per step
    print("INFO: Network %d x %d tanh, %d Fourier frequencies, %d parameters; %d data and %d physics point(s) per "
          "step, %d steps per epoch" % (pp['depth'], pp['width'], pp['fourier'],
                                        sum(q.numel() for q in net.parameters()), nD,
                                        (nC if phys == 1 else nT * N) if phys else 0, steps))

    def lossFnc():
        it = torch.randint(0, Nt, (nD,), device=dev)
        js = torch.randint(0, len(iT), (nD,), device=dev)
        lD = torch.mean((net(torch.cat([xhT[iT[js]], tH[it][:, None]], dim=1)) - ThT[it, js]) ** 2)
        i0 = torch.randint(0, N, (max(nD // 4, 1),), device=dev)
        lI = torch.mean(net(torch.cat([xhT[i0], -torch.ones(len(i0), 1, device=dev)], dim=1)) ** 2)
        lP = torch.zeros((), device=dev)
        if wP > 0 and phys == 1:
            [idc, xc] = _colloc(xhT, ph, nC)
            tc = torch.rand(nC, device=dev) * (Nt - 1)                  # time in samples
            Q = torch.sum(ph['U'][idc] * pT[torch.clamp(tc.ceil().long(), min=1, max=Nt - 1)], dim=1)   # hold
            [f, g, h2] = coordDeriv(net, torch.cat([xc, (2 * tc / max(Nt - 1, 1) - 1)[:, None]], dim=1))
            r = (ph['rc'][idc] * sT * g[:, d] - torch.sum(ph['kap'][idc] * h2[:, 0:d], dim=1) + ph['hv'][idc] * f
                 - Q / dT) / ph['g'][idc]
            lP = torch.mean(r ** 2)
        elif wP > 0:
            k = torch.randint(1, Nt, (nT,), device=dev)                 # middle of the step k − 1 → k
            [f, ft] = coordRate(net, xhT, tH[k] - 1.0 / (Nt - 1))
            u = fvTerm(f, None, ph['kT'], dT, None)
            Hu = ph['solve'](ph['H'][:, None] * u.T).T if phys == 3 else None
            r = sT * ph['solve'](ph['C'][:, None] * ft.T).T + fvTerm(f, Hu, ph['kT'], dT, factors(ph)) \
                - pT[k] @ ph['Tss'].T / dT
            lP = torch.mean(r ** 2)
        return wD * lD + wP * lP + wI * lI, lD, lP, lI

    def valFnc():
        se = 0.0
        with torch.no_grad():
            for i in range(0, Nt, 64):
                tt = tH[i:i + 64]
                z = torch.cat([xV[None].expand(len(tt), -1, -1), tt[:, None, None].expand(-1, len(xV), 1)], dim=-1)
                se = se + float(torch.sum((dT * net(z) - ThV[i:i + 64]) ** 2))
        return se / max(ThV.numel(), 1)

    # ==============================================================================
    # Training
    # ==============================================================================
    best, bestNet, wait, bestS = np.inf, copy.deepcopy(net.state_dict()), 0, None
    info = _history()
    t0 = time.time()
    for ep in range(0, epoch):
        net.train()
        sums = np.zeros(4)
        for _ in range(0, steps):
            opt.zero_grad()
            out = lossFnc()
            out[0].backward()
            if gradTh > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradTh)
            opt.step()
            sums = sums + np.array([float(v.detach()) for v in out])
        lrEp = opt.param_groups[0]['lr']
        net.eval()
        lv = valFnc()
        _log(info, ep, epoch, lrEp, sums / steps, lv, t0)
        new = lrStep(lv)
        if new is not None:
            for g in opt.param_groups:
                g['lr'] = new * g['fac']
        if lv < best:
            best, bestNet, wait = lv, copy.deepcopy(net.state_dict()), 0
            bestS = None if ph['ls'] is None else factors(ph).detach().cpu().numpy()
        else:
            wait = wait + 1
            if wait >= valPat:
                print("INFO: Early stopping (validation patience %d)" % valPat)
                break

    # ==============================================================================
    # Model
    # ==============================================================================
    net.load_state_dict(bestNet)
    info.update({'best': best, 'valMode': 'held-out sensors' if hold.any() else 'training sensors'})
    _factors(info, bestS if ph['ls'] is not None else None)
    return {'sys': net.to('cpu').eval(), 'info': info, 'U': U, 'geo': geo.astype(np.float32), 'S': S, 'Nt': Nt,
            'tEnd': tEnd, 'sens': idxT, 'sensVal': idxV if hold.any() else np.zeros(0, dtype=int), 'sv': sv[0:SVN],
            'dT': dT, 'sc': sc, 'Ts': Ts}


#######################################################################################################################
# Main Function
#######################################################################################################################
def pfFit(data, val, para, setup, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Training Physics-Informed Field Network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    checkNorm(para)
    var = int(setup.get('selPF', 1) or 1)
    pp = pfPara(para)
    dl = para['Mdl']['dl']
    dev = getDevice(para)
    torch.manual_seed(pp['seed'])
    np.random.seed(pp['seed'])
    print("INFO: Variant %d: %s" % (var, VARIANTS.get(var, VARIANTS[1])))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    if var == 2:
        mdl = _fitCoord(data, para, pp, dl, dev)
    else:
        mdl = _fitOperator(data, val, para, setup, pp, dl, dev)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    mdl.update({'var': 2 if var == 2 else 1, 'phys': pp['phys'], 'seq': 0, 'para': pp})
    mdl['info'].update({'alpha': pp['alpha'], 'beta': pp['beta'] if pp['phys'] > 0 else 0.0, 'gamma': pp['gamma'],
                        'dT': mdl['dT'], 'device': str(dev)})

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("INFO: Best validation RMSE %.3f K (%s)" % (np.sqrt(mdl['info']['best']), mdl['info']['valMode']))
    print("DONE: Training Physics-Informed Field Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return mdl


#######################################################################################################################
# References
#######################################################################################################################
# [1] S. Cai et al., Physics-informed neural networks for heat transfer problems, J. Heat Transfer 143 (6), 2021.
# [2] S. Wang et al., Learning the solution operator of parametric partial differential equations with
#     physics-informed DeepONets, Science Advances 7 (40), 2021.
