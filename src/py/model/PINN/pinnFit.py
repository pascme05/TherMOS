#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnFit
# Date:         08.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.2.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function trains a physics-informed neural network for the temperatures. A sequence network (dlNet: setup
selPI = (1) CNN, (2) LSTM, (3) CNN-LSTM, or the user-defined architecture setup['archPI']) predicts the states x of a
linear thermal network (pinnPhys: Foster terms or Cauer nodes) on windows of W samples; the outputs are
T̂ = T_ref + O·x. The loss has three dimensionless parts (ΔT: temperature scale of the training data):

    L = α·L_data + β·L_phys + γ·L_init
    L_data = mean(((T̂ - T)/ΔT)²)                                            measured outputs
    L_phys = mean(((C·dx/dt + G·x - E·P)/(diag(G)·ΔT))²)                    every physics state, central differences
    L_init = Σ w·((T̂ - T(0))/ΔT)² / Σ w,   w = exp(-t/τ_min)            start of every profile (t: time in profile)

The first 'warm' samples of every window are excluded from all losses (the recurrent state of the network settles).
Initial state (x0 = 1): every training window gets the measured temperature at its own start (one per stride), the
validation and testing profiles only the measured temperature of their very first sample (pinnSol).
Training: Adam, learning rate schedule (lrAdapt = 0: factor lrDropFa every lrDropPr epochs; lrAdapt = 1: factor
lrDropFa after lrDropPr epochs without improvement of the validation error, at least 1e-3·initLr), gradient clipping
(gradTh > 0). Validation once per epoch exactly as in testing: every validation profile is predicted from its first
measured sample (Mdl › pinn › pred), so that early stopping (valPat epochs without improvement) and the kept network
(lowest validation error) refer to the error of the testing procedure. The network is trained on the device of
para['Mdl']['gen']['dev'] and returned on the CPU.
The normalisation of the data (normX, normY) is undone inside the network (pinnNet), the physics uses raw units.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables (selPI, archPI, physPI)
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model {'sys': pinnNet, 'info': training history, 'seq': 0, 'W', 'phys': physics definition}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of pinnFit.m; windows per profile, physics constants configurable (F-I1), the time derivative uses the
  sampling time of the data instead of Ts = 1 sec (F-I2)

V1.1: 07.11.2026, Pascal Schirmer
- The training history stores the data loss and the physics loss separately

V1.2: 07.11.2026, Pascal Schirmer
- General physics: losses from a feature or an equation, reference temperature from a feature or a constant

V2.0: 08.11.2026, Pascal Schirmer
- Physics as a nodal network (Foster, several outputs, Cauer with hidden nodes), the network predicts its states;
  dimensionless losses (β no longer depends on the units), central differences, warm-up, initial-condition loss,
  gradient clipping, user-defined architecture (archPI), raw units under normalisation, prediction over the whole
  profile (pinnSol)

V2.1: 08.11.2026, Pascal Schirmer
- Validation as in testing (whole profiles from the first measured sample), adaptive learning rate (lrAdapt), the
  learning rate is printed every epoch
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlNet import dlNet
from src.py.model.PINN.pinnNet import pinnNet
from src.py.model.PINN.pinnPhys import physDef, physPara, sources, jsonable, split, reference, simPhys
from src.py.model.PINN.pinnSol import predictNet
from src.py.general.getDevice import getDevice

# ==============================================================================
# External
# ==============================================================================
import copy
import sys
import time
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def affine(para, F, M):
    """Affine maps of the normalisation (x_norm = (x_raw - b)/a) of the inputs and outputs."""
    g = para.get('Par', {}).get('gen', {})
    nv = para.get('Dat', {}).get('normVal') or {}

    def ab(s, method, n):
        if s is None or method not in (2, 3, 4):
            return np.ones(n), np.zeros(n)
        rng = np.asarray(s['max'], dtype=float) - np.asarray(s['min'], dtype=float)
        if method == 2:
            return rng, np.asarray(s['min'], dtype=float)
        if method == 3:
            return rng, np.asarray(s['avg'], dtype=float)
        return np.asarray(s['std'], dtype=float), np.asarray(s['avg'], dtype=float)
    aX, bX = ab(nv.get('X'), int(g.get('normX', 1) or 1), F)
    ay, by = ab(nv.get('y'), int(g.get('normY', 1) or 1), M)
    return {'aX': np.broadcast_to(aX, (F,)).copy(), 'bX': np.broadcast_to(bX, (F,)).copy(),
            'ay': np.broadcast_to(ay, (M,)).copy(), 'by': np.broadcast_to(by, (M,)).copy()}


def windows(d, W, stride, Ts):
    """
    Windows of every profile: X, y, r (B x W x .), time in the profile (B x W), output at the start of the profile and
    at the start of the window (B x M).
    """
    Xs, ys, rs, ts, y0, yw = [], [], [], [], [], []
    [X2, y2, r2, _] = split(d)
    for X, y, r in zip(X2, y2, r2):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        r = np.zeros_like(y) if r is None else np.asarray(r, dtype=np.float32).reshape(len(y), -1)
        if len(X) < W:
            continue
        st = np.arange(0, len(X) - W + 1, stride)
        view = lambda a: sliding_window_view(a, W, axis=0).transpose(0, 2, 1)[st]
        Xs.append(view(X))
        ys.append(view(y))
        rs.append(view(r))
        ts.append((st[:, None] + np.arange(W)[None, :]).astype(np.float32) * Ts)
        y0.append(np.repeat(y[0:1], len(st), 0))
        yw.append(y[st])
    if not Xs:
        return None
    return [torch.from_numpy(np.ascontiguousarray(np.concatenate(a))) for a in (Xs, ys, rs, ts, y0, yw)]


def colWindows(profiles, W, stride, Ts, ph, aff, init):
    """
    Collocation windows: input profiles without measured temperatures (only in the physics loss). The initial state
    of every window (input x0) is the physics network simulated from the profile start (equilibrium with T_ref).
    profiles: list of (X, r) in the units of the training data (normalised inputs, raw reference channels).
    """
    X2, y2, r2 = [], [], []
    for X, r in profiles:
        X = np.asarray(X, dtype=float)
        if len(X) < W:
            continue
        r = None if r is None else np.asarray(r, dtype=float).reshape(len(X), -1)
        Xr = X * aff['aX'] + aff['bX']
        ref = np.asarray(reference(Xr, r, ph, init), dtype=float)
        P = np.asarray(sources(Xr, ref, ph), dtype=float) if ph['S'] else np.zeros((len(X), 0))
        _, Tp = simPhys(ph, P, Ts) if ph['nP'] else (None, np.zeros_like(ref))
        X2.append(X)
        y2.append((ref + Tp - aff['by']) / aff['ay'])
        r2.append(np.zeros_like(ref) if r is None else r)
    if not X2:
        return None
    return windows({'X2': X2, 'y2': y2, 'r2': r2}, W, stride, Ts)


def lrSchedule(dl, lr0):
    """Learning-rate schedule: step(lv) after every epoch with the validation error lv returns the new rate or None."""
    mode, per, fac = int(dl.get('lrAdapt', 0) or 0), max(int(dl['lrDropPr']), 1), float(dl['lrDropFa'])
    st = {'lr': float(lr0), 'best': np.inf, 'wait': 0, 'ep': 0}

    def step(lv):
        st['ep'] = st['ep'] + 1
        if mode == 1:                                                   # adaptive: on a plateau of the validation
            if lv < st['best'] * (1 - 1e-3):
                st['best'], st['wait'] = lv, 0
            else:
                st['wait'] = st['wait'] + 1
            if st['wait'] >= per and st['lr'] * fac >= 1e-3 * lr0:
                st['lr'], st['wait'] = st['lr'] * fac, 0
                return st['lr']
        elif st['ep'] % per == 0:                                       # fixed steps
            st['lr'] = st['lr'] * fac
            return st['lr']
        return None
    return step, mode, per, fac


def valProfiles(net, d, W, pred, dev, ay):
    """Mean squared error (K²) of the validation profiles predicted as in testing (from their first sample)."""
    [X2, y2, r2, _] = split(d)
    se, n = 0.0, 0
    for X, y, r in zip(X2, y2, r2):
        y = np.asarray(y, dtype=float)
        if len(y) < 2:
            continue
        yp, _ = predictNet(net, X, r, W, pred, dev, y[0])
        se, n = se + float(np.sum(((yp[0:len(y)] - y) * ay) ** 2)), n + y.size
    return se / max(n, 1)


def deriv(x, Ts):
    """Time derivative along the window (central differences, one-sided at the ends)."""
    return torch.gradient(x, spacing=Ts, dim=1)[0]


#######################################################################################################################
# Main Function
#######################################################################################################################
def pinnFit(data, val, para, setup, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Training Physics-Informed Neural Network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    dl = para['Mdl']['dl']
    W, stride = int(dl['W']), max(int(dl['stride']), 1)
    epoch, batch, lr = int(dl['epoch']), int(dl['batch']), dl['initLr']
    valPat = int(dl.get('valPat', 10) or 10)
    gradTh = float(dl.get('gradTh', 0) or 0)
    sel = 3 if legacy == 1 else int(setup.get('selPI', 3) or 3)
    arch = setup.get('archPI') or None
    Ts = 1.0 if legacy == 1 else float(data['Ts'])
    init = int(para['Exp']['gen'].get('init', 1) or 1)
    dev = getDevice(para)
    torch.manual_seed(42)

    # ==============================================================================
    # Physics
    # ==============================================================================
    F, M = np.shape(data['X2'][0])[1], np.shape(data['y2'][0])[1]
    try:
        ph = physDef(para, setup, M, F)
    except Exception as e:
        sys.exit("ERROR: PINN physics: %s" % e)
    wD, wP, wI = ph['alpha'], ph['beta'], ph['gamma']
    warm = min(max(int(ph['warm']), 0), W - 2)
    if ph['nP'] == 0:
        wP, wI = 0.0, wI
    aff = affine(para, F, M)

    # ==============================================================================
    # Windows
    # ==============================================================================
    tr = windows(data, W, stride, Ts)
    vl = windows(val, W, max(stride, W // 2), Ts)
    print("INFO: %d training and %d validation profiles (cut at every session ID)" % (len(split(data)[0]),
                                                                                      len(split(val)[0])))
    if tr is None:
        sys.exit("ERROR: PINN: every training profile is shorter than the window W = %d" % W)
    if vl is None:
        print("WARN: Validation profiles shorter than W, the training data is used for validation")
        vl = [a[0:max(len(a) // 10, 1)] for a in tr]
    [Xtr, ytr, rtr, ttr, y0tr, ywtr] = tr
    x0 = int(ph['x0'])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Temperature scale and check of the physics
    # ==============================================================================
    Xr = Xtr[0:min(len(Xtr), 256)] * torch.as_tensor(aff['aX'], dtype=torch.float32) + \
        torch.as_tensor(aff['bX'], dtype=torch.float32)
    yr = ytr[0:len(Xr)] * torch.as_tensor(aff['ay'], dtype=torch.float32) + torch.as_tensor(aff['by'],
                                                                                             dtype=torch.float32)
    ref = {'TrefSrc': ph['TrefSrc'], 'Tref': ph['Tref'], 'M': M, 'init': init}
    probe = pinnNet(torch.nn.Identity(), ph['O'], 1.0, aff, ref, 0)
    Tr = probe.reference(Xr, rtr[0:len(Xr)])
    dT = float(max(np.percentile(np.abs((yr - Tr).numpy()), 99), 1.0))
    try:
        P0 = sources(Xr, yr, ph)
        if ph['S'] and not bool(torch.all(torch.isfinite(P0))):
            raise ValueError("the loss sources must give finite values")
    except Exception as e:
        sys.exit("ERROR: PINN physics (loss sources): %s" % e)
    nHid = int(np.sum(~np.any(ph['O'][:, 0:ph['nP']] != 0, axis=0)))
    print("INFO: Physics %s (%s): %d physics states (%d not measured), %d free output(s), %d loss source(s): %s" %
          (ph['type'], ph['origin'], ph['nP'], nHid, len(ph['free']), ph['S'], ', '.join(ph['src']) or '-'))
    if ph['nP']:
        print("INFO: Time constants of the physics: %s sec" % np.array2string(ph['tau'], precision=4))
        if ph['S']:
            print("INFO: Mean loss sources: %s W" % np.array2string(P0.reshape(-1, ph['S']).mean(0).numpy(),
                                                                       precision=4))
    print("INFO: Temperature scale dT = %.4g K, %d training windows (W = %d, stride %d), warm-up %d samples, "
          "alpha = %g, beta = %g, gamma = %g" % (dT, len(Xtr), W, stride, warm, wD, wP, wI))

    # ==============================================================================
    # Network
    # ==============================================================================
    mu = Xtr.reshape(-1, F).mean(0).numpy()
    sd = Xtr.reshape(-1, F).std(0).numpy()
    sd[sd < 1e-12] = 1
    if arch:
        print("INFO: User-defined architecture: " + ' -> '.join(f"{l['type']} {l['units']}" for l in arch['body'])
              + f" -> head {arch.get('head', [])}")
    if x0:
        mu, sd = np.concatenate([mu, np.zeros(M)]), np.concatenate([sd, np.ones(M)])
        print("INFO: Initial state as input: measured temperature rise at the start of every window (M = %d)" % M)
    body = dlNet(sel, F + (M if x0 else 0), ph['n'], 0, mu, sd, arch=arch)
    net = pinnNet(body, ph['O'], dT, aff, ref, x0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lrStep, lrMode, lrPer, lrFac = lrSchedule(dl, lr)
    pred = int(ph['pred'])
    if lrMode == 1:
        print("INFO: Adaptive learning rate: factor %g after %d epoch(s) without improvement of the validation error "
              "(initial %g, minimum %g)" % (lrFac, lrPer, lr, 1e-3 * lr))
        if valPat <= lrPer:
            print("WARN: valPat = %d <= lrDropPr = %d: early stopping ends the training before the learning rate is "
                  "reduced" % (valPat, lrPer))
    else:
        print("INFO: Learning rate: initial %g, factor %g every %d epoch(s)" % (lr, lrFac, lrPer))
    print("INFO: Validation as in testing: %d profile(s) predicted %s from their first measured sample" %
          (len(split(val)[0]), 'in one pass' if pred == 1 else 'with consecutive windows'))
    G = torch.as_tensor(ph['G'], dtype=torch.float32, device=dev)
    C = torch.as_tensor(ph['C'], dtype=torch.float32, device=dev)
    E = torch.as_tensor(ph['E'], dtype=torch.float32, device=dev)
    gd = torch.diag(G) * dT if ph['nP'] else None
    tauMin = float(ph['tau'][0]) if ph['nP'] else max(W * Ts / 10, Ts)
    nP = ph['nP']

    # ==============================================================================
    # Collocation profiles (inputs only, physics loss)
    # ==============================================================================
    colSrc = int(ph.get('col', 0) or 0)
    prof = list(data.get('col') or [])                                   # (X, r) given directly (API, examples)
    for k, d in ((1, val), (2, data.get('_te'))):
        if colSrc & k and d is not None and wP > 0:
            [Xc, _, rc, _] = split(d)
            prof += list(zip(Xc, rc))
    co = colWindows(prof, W, stride, Ts, ph, aff, init) if prof and wP > 0 and nP else None
    if co is not None:
        print("INFO: %d collocation windows (inputs without temperatures, only in the physics loss)" % len(co[0]))

    def lossFnc(X, y, r, t, y0, yw, phys=True):
        x, T, Xraw = net.outputs(X, r, yw)
        yraw = net.fromData(y)
        sl = slice(warm, None)
        lD = torch.mean(((T - yraw)[:, sl] / dT) ** 2)
        lP = torch.zeros((), device=X.device)
        lI = torch.zeros((), device=X.device)
        if phys and nP:
            xp = x[..., 0:nP]
            P = sources(Xraw, T, ph)
            res = (C * deriv(xp, Ts) + xp @ G.T - (P @ E.T if ph['S'] else 0)) / gd
            lP = torch.mean(res[:, sl] ** 2)
        if phys and wI > 0:
            w = torch.exp(-t[:, sl] / tauMin)[..., None]
            lI = torch.sum(w * ((T - y0[:, None, :])[:, sl] / dT) ** 2) / torch.clamp(torch.sum(w) * M, min=1e-9)
        return wD * lD + wP * lP + wI * lI, lD, lP, lI

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    best, bestNet, wait = np.inf, copy.deepcopy(net.state_dict()), 0
    vlProf = bool(len(split(val)[0])) and any(len(x) >= 2 for x in split(val)[0])
    info = {'loss': [], 'valLoss': [], 'valWin': [], 'learnRate': [], 'lossData': [], 'lossPhys': [], 'lossInit': [],
            'valPhys': [], 'lrAdapt': lrMode, 'valMode': 'profiles' if vlProf else 'windows', 'alpha': wD,
            'beta': wP, 'gamma': wI, 'dT': dT, 'warm': warm, 'device': str(dev),
            'nWin': len(Xtr), 'nCol': 0 if co is None else len(co[0]), 'time': []}
    order = torch.randperm(len(Xtr)) if dl['shu'] == 2 else torch.arange(len(Xtr))
    t0 = time.time()
    for ep in range(0, epoch):
        if dl['shu'] not in (1, 2):
            order = torch.randperm(len(Xtr))
        net.train()
        sums, nB = np.zeros(4), 0
        for b in range(0, len(Xtr), batch):
            idx = order[b:b + batch]
            if len(idx) < 2:                                                 # BatchNorm needs two samples
                continue
            opt.zero_grad()
            out = lossFnc(*[a[idx].to(dev) for a in tr])
            if co is not None:                                               # physics on the collocation windows
                ic = torch.randint(0, len(co[0]), (len(idx),))
                oc = lossFnc(*[a[ic].to(dev) for a in co])
                out = (out[0] + wP * oc[2], out[1], 0.5 * (out[2] + oc[2]), out[3])
            out[0].backward()
            if gradTh > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradTh)
            opt.step()
            sums, nB = sums + np.array([float(v.detach()) for v in out]), nB + 1
        lrEp = opt.param_groups[0]['lr']

        # Validation: profiles as in testing (MSE in K²), physics loss and data loss of the validation windows
        net.eval()
        lw, pv, nV = 0.0, 0.0, 0
        with torch.no_grad():
            for b in range(0, len(vl[0]), 4 * batch):
                o = lossFnc(*[a[b:b + 4 * batch].to(dev) for a in vl])
                lw, pv, nV = lw + float(o[1]) * len(vl[0][b:b + 4 * batch]), pv + float(o[2]) * \
                    len(vl[0][b:b + 4 * batch]), nV + len(vl[0][b:b + 4 * batch])
            lv = valProfiles(net, val, W, pred, dev, aff['ay']) if vlProf else None
        lw, pv = lw / max(nV, 1) * dT ** 2, pv / max(nV, 1)
        lv = lw if lv is None or not np.isfinite(lv) else lv
        m = sums / max(nB, 1)
        info['loss'].append(float(m[0]))
        info['lossData'].append(float(m[1]))
        info['lossPhys'].append(float(m[2]))
        info['lossInit'].append(float(m[3]))
        info['valLoss'].append(lv)
        info['valWin'].append(lw)
        info['valPhys'].append(pv)
        info['learnRate'].append(lrEp)
        info['time'].append(time.time() - t0)
        print("INFO: Epoch %d/%d, lr %.3e, loss %.4e (data %.3e, physics %.3e, init %.3e), validation RMSE %.3f K "
              "(as testing; windows with measured start %.3f K), validation physics %.3e" %
              (ep + 1, epoch, lrEp, m[0], m[1], m[2], m[3], np.sqrt(lv), np.sqrt(lw), pv))
        new = lrStep(lv)
        if new is not None:
            for g in opt.param_groups:
                g['lr'] = new
            print("INFO: Learning rate reduced to %.3e" % new)
        if lv < best:
            best, bestNet, wait = lv, copy.deepcopy(net.state_dict()), 0
        else:
            wait = wait + 1
            if wait >= valPat:
                print("INFO: Early stopping (validation patience %d)" % valPat)
                break

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    net.load_state_dict(bestNet)
    net = net.to('cpu')
    net.eval()
    info['best'] = best

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("INFO: Best validation RMSE %.3f K (validation profiles predicted as in testing)" % np.sqrt(best))
    print("DONE: Training Physics-Informed Neural Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    phs = jsonable({k: v for k, v in ph.items() if k != 'struct'})
    phs['struct'] = ph['struct']
    return {'sys': net, 'info': info, 'seq': 0, 'W': W, 'phys': phs, 'pred': int(ph['pred']), 'Ts': Ts}


#######################################################################################################################
# References
#######################################################################################################################
