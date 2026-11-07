#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnFit
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function trains a physics-informed neural network (seq2seq, see dlNet: CNN, LSTM or CNN-LSTM) for the
temperatures. The physics is a lumped thermal model of one output node (para['Mdl']['pinn']['node']):

    L = α·mean((T̂ - T)²) + β·mean(r²),     r = Cth·dT̂/dt - ((T_ref - T̂)/Rth + Pv)

with the thermal resistance Rth (K/W) and capacitance Cth (Ws/K) from the node to the reference temperature T_ref.
The losses Pv (W) are either an input feature (PvSrc = k > 0) or an equation (PvSrc = 0, PvEq, see calcEq) of the
input features x1, x2, ... (raw units), the predicted temperature T and the constants Rth, Cth, Rs and those of
'const' ('k=0.01; p0=50'). T_ref is an input feature (TrefSrc = k > 0) or the constant Tref (TrefSrc = 0). The default
(PvEq = 3*Rs*x4**2, TrefSrc = 2) is the stator model of the MATLAB version (motorTemp data: ambient temperature in
feature 2, stator current in feature 4). The physics weight β is zero by default (as in MATLAB), i.e. the network is
trained on the data only and the physics loss is monitored. Validation once per epoch, early stopping after valPat
epochs without improvement. The network is trained on the device of para['Mdl']['gen']['dev'] (getDevice) and
returned on the CPU.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables (selPI: (1) CNN, (2) LSTM, (3) CNN-LSTM)
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model {'sys': network, 'info': training history, 'seq': 0, 'W', 'phys': physics settings}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of pinnFit.m; windows per profile, physics constants configurable (F-I1), the time derivative uses the
  sampling time of the data instead of Ts = 1 sec (F-I2)

V1.1: 07.11.2026, Pascal Schirmer
- The training history stores the data loss and the physics loss separately (info['lossData'], info['lossPhys'],
  unweighted means per epoch)

V1.2: 07.11.2026, Pascal Schirmer
- General physics: losses from a feature or an equation, reference temperature from a feature or a constant,
  residual node selectable; the network follows selPI (CNN, LSTM, CNN-LSTM; legacy = 1: CNN-LSTM as MATLAB);
  CPU/GPU device
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlNet import dlNet
from src.py.model.DL.dlWin import dlWin
from src.py.general.calcEq import calcEq
from src.py.general.getDevice import getDevice

# ==============================================================================
# External
# ==============================================================================
import copy
import sys
import numpy as np
import torch


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _windows(d, W, stride):
    Xs, ys = [], []
    for X, y in zip(d['X2'], d['y2']):
        Xw, yw = dlWin(X, y, W, 0, stride, 0)
        if len(Xw) > 0:
            Xs.append(Xw)
            ys.append(yw)
    return [torch.from_numpy(np.concatenate(Xs)), torch.from_numpy(np.concatenate(ys))]


def physPara(para):
    """Physics settings of the PINN with defaults (also used by the GUI)."""
    phy = para['Mdl'].get('pinn', {}) or {}
    const = {}
    txt = phy.get('const', 0)
    if isinstance(txt, str) and txt.strip() not in ('', '0', '-'):
        for item in txt.replace(',', ';').split(';'):
            if item.strip():
                k, v = item.split('=')
                const[k.strip()] = float(v)
    Rth, Cth, Rs = float(phy.get('Rth', 0.0265)), float(phy.get('Cth', 2.08e4)), float(phy.get('Rs', 14.1e-3))
    return {'Rth': Rth, 'Cth': Cth, 'alpha': float(phy.get('alpha', 1.0)), 'beta': float(phy.get('beta', 0.0)),
            'PvSrc': int(phy.get('PvSrc', 0) or 0), 'PvEq': str(phy.get('PvEq', '3*Rs*x4**2') or '3*Rs*x4**2'),
            'TrefSrc': int(phy.get('TrefSrc', 2) if phy.get('TrefSrc') is not None else 2),
            'Tref': float(phy.get('Tref', 25.0) or 0.0), 'node': max(int(phy.get('node', 1) or 1), 1),
            'const': {'Rth': Rth, 'Cth': Cth, 'Rs': Rs, **const}}


def physTerms(X, T, ph):
    """Losses Pv and reference temperature T_ref of the residual for the windows X (B x W x F) and T (B x W)."""
    if ph['PvSrc'] > 0:
        Pv = X[:, :, ph['PvSrc'] - 1]
    else:
        var = {f"x{i + 1}": X[:, :, i] for i in range(0, X.shape[2])}
        var['T'] = T
        Pv = calcEq(ph['PvEq'], var, ph['const']) + 0 * T
    Tref = X[:, :, ph['TrefSrc'] - 1] if ph['TrefSrc'] > 0 else torch.full_like(T, ph['Tref'])
    return [Pv, Tref]


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
    ph = physPara(para)
    W, stride = int(dl['W']), max(int(dl['stride']), 1)
    epoch, batch, lr = int(dl['epoch']), int(dl['batch']), dl['initLr']
    valPat = int(dl.get('valPat', 10) or 10)
    Rth, Cth = ph['Rth'], ph['Cth']
    wD, wP = ph['alpha'], ph['beta']
    sel = 3 if legacy == 1 else int(setup.get('selPI', 3) or 3)
    Ts = 1.0 if legacy == 1 else data['Ts']
    dev = getDevice(para)
    torch.manual_seed(42)

    # ==============================================================================
    # Variables
    # ==============================================================================
    [Xtr, ytr] = _windows(data, W, stride)
    [Xvl, yvl] = _windows(val, W, stride)
    F, M = Xtr.shape[2], ytr.shape[2]
    j = min(ph['node'], M) - 1

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Check Physics
    # ==============================================================================
    try:
        if ph['PvSrc'] > F or ph['TrefSrc'] > F:
            raise ValueError("feature %d does not exist (%d features)" % (max(ph['PvSrc'], ph['TrefSrc']), F))
        [Pv0, Tr0] = physTerms(Xtr[0:2], ytr[0:2, :, j], ph)
        if tuple(Pv0.shape) != tuple(ytr[0:2, :, j].shape) or not bool(torch.all(torch.isfinite(Pv0))):
            raise ValueError("the losses must give one finite value per sample")
    except Exception as e:
        sys.exit("ERROR: PINN physics (losses / reference temperature): %s" % e)
    print("INFO: Physics residual of output %d: Pv = %s, T_ref = %s" %
          (j + 1, ('x%d' % ph['PvSrc']) if ph['PvSrc'] > 0 else ph['PvEq'],
           ('x%d' % ph['TrefSrc']) if ph['TrefSrc'] > 0 else ('%g °C' % ph['Tref'])))

    # ==============================================================================
    # Network
    # ==============================================================================
    mu = Xtr.reshape(-1, F).mean(0).numpy()
    sd = Xtr.reshape(-1, F).std(0).numpy()
    sd[sd < 1e-12] = 1
    net = dlNet(sel, F, M, 0, mu, sd).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xvl, yvl = Xvl.to(dev), yvl.to(dev)

    def lossFnc(X, y):
        yp = net(X)
        lD = torch.mean((yp - y) ** 2)
        loss = wD * lD
        lP = torch.zeros((), device=X.device)
        if wP > 0:
            T = yp[:, :, j]
            dT = torch.diff(T, dim=1, append=T[:, -1:] + (T[:, -1:] - T[:, -2:-1])) / Ts
            [Pv, Tref] = physTerms(X, T, ph)
            r = Cth * dT - ((Tref - T) / Rth + Pv)
            lP = torch.mean(r ** 2)
            loss = loss + wP * lP
        return loss, lD, lP

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    best, bestNet, wait = np.inf, copy.deepcopy(net.state_dict()), 0
    info = {'loss': [], 'valLoss': [], 'learnRate': [], 'lossData': [], 'lossPhys': [], 'alpha': wD, 'beta': wP,
            'device': str(dev)}
    order = torch.randperm(len(Xtr)) if dl['shu'] == 2 else torch.arange(len(Xtr))
    for ep in range(0, epoch):
        if dl['shu'] not in (1, 2):
            order = torch.randperm(len(Xtr))
        net.train()
        lossEp, lossD, lossP, nB = 0.0, 0.0, 0.0, 0
        for b in range(0, len(Xtr), batch):
            idx = order[b:b + batch]
            if len(idx) < 2 and sel == 1:
                continue
            opt.zero_grad()
            [loss, lD, lP] = lossFnc(Xtr[idx].to(dev), ytr[idx].to(dev))
            loss.backward()
            opt.step()
            lossEp = lossEp + float(loss.detach())
            lossD, lossP, nB = lossD + float(lD.detach()), lossP + float(lP.detach()), nB + 1
        if (ep + 1) % max(int(dl['lrDropPr']), 1) == 0:
            for g in opt.param_groups:
                g['lr'] = g['lr'] * dl['lrDropFa']

        # Validation (β = 0: the physics loss is monitored on the validation data)
        net.eval()
        with torch.no_grad():
            yp = net(Xvl)
            lv = float(torch.mean((yp - yvl) ** 2))
            lPhys = lossP / max(nB, 1)
            if wP == 0:
                T = yp[:, :, j]
                dT = torch.diff(T, dim=1, append=T[:, -1:] + (T[:, -1:] - T[:, -2:-1])) / Ts
                [Pv, Tref] = physTerms(Xvl, T, ph)
                lPhys = float(torch.mean((Cth * dT - ((Tref - T) / Rth + Pv)) ** 2))
        info['loss'].append(lossEp)
        info['valLoss'].append(lv)
        info['learnRate'].append(opt.param_groups[0]['lr'])
        info['lossData'].append(lossD / max(nB, 1))
        info['lossPhys'].append(lPhys)
        print("INFO: Epoch %d/%d, training loss %.4e, validation loss %.4e" % (ep + 1, epoch, lossEp, lv))
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

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Training Physics-Informed Neural Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'sys': net, 'info': info, 'seq': 0, 'W': W, 'phys': {k: v for k, v in ph.items()}}


#######################################################################################################################
# References
#######################################################################################################################
