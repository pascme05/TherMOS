#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnFit
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
This function trains a physics-informed CNN-LSTM network (seq2seq, see dlNet with sel = 3) for the temperature of a
lumped stator model. The loss is

    L = α·mean((T̂ - T)²) + β·mean(r²),     r = Cth·dT̂/dt - ((T_amb - T̂)/Rth + 3·Rs·Is²)

with the thermal resistance Rth (K/W), the capacitance Cth (Ws/K), the stator resistance Rs (Ohm), the stator
current Is (feature 4, A) and the ambient temperature T_amb (feature 2, °C) as in the MATLAB version (motorTemp
data). The physics weight β is zero by default (as in MATLAB), i.e. the network is trained on the data only. The
parameters are taken from para['Mdl']['pinn'] (Rth, Cth, Rs, alpha, beta) if available. Validation once per epoch,
early stopping after valPat epochs without improvement.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model {'sys': network, 'info': training history, 'seq': 0, 'W'}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of pinnFit.m; windows per profile, physics constants configurable (F-I1), the time derivative uses the
  sampling time of the data instead of Ts = 1 sec (F-I2)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlNet import dlNet
from src.py.model.DL.dlWin import dlWin

# ==============================================================================
# External
# ==============================================================================
import copy
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
    phy = para['Mdl'].get('pinn', {})
    W, stride = int(dl['W']), max(int(dl['stride']), 1)
    epoch, batch, lr = int(dl['epoch']), int(dl['batch']), dl['initLr']
    valPat = int(dl.get('valPat', 10) or 10)
    Rth, Cth, Rs = phy.get('Rth', 0.0265), phy.get('Cth', 2.08e4), phy.get('Rs', 14.1e-3)
    wD, wP = phy.get('alpha', 1.0), phy.get('beta', 0.0)
    Ts = 1.0 if legacy == 1 else data['Ts']
    torch.manual_seed(42)

    # ==============================================================================
    # Variables
    # ==============================================================================
    [Xtr, ytr] = _windows(data, W, stride)
    [Xvl, yvl] = _windows(val, W, stride)
    F, M = Xtr.shape[2], ytr.shape[2]
    if wP > 0 and F < 4:
        raise ValueError('The physics loss needs the ambient temperature (feature 2) and the current (feature 4)')

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    mu = Xtr.reshape(-1, F).mean(0).numpy()
    sd = Xtr.reshape(-1, F).std(0).numpy()
    sd[sd < 1e-12] = 1
    net = dlNet(3, F, M, 0, mu, sd)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def lossFnc(X, y):
        yp = net(X)
        loss = wD * torch.mean((yp - y) ** 2)
        if wP > 0:
            T = yp[:, :, 0]
            dT = torch.diff(T, dim=1, append=T[:, -1:] + (T[:, -1:] - T[:, -2:-1])) / Ts
            r = Cth * dT - ((X[:, :, 1] - T) / Rth + 3 * Rs * X[:, :, 3] ** 2)
            loss = loss + wP * torch.mean(r ** 2)
        return loss

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    best, bestNet, wait = np.inf, copy.deepcopy(net.state_dict()), 0
    info = {'loss': [], 'valLoss': [], 'learnRate': []}
    order = torch.randperm(len(Xtr)) if dl['shu'] == 2 else torch.arange(len(Xtr))
    for ep in range(0, epoch):
        if dl['shu'] not in (1, 2):
            order = torch.randperm(len(Xtr))
        net.train()
        lossEp = 0.0
        for b in range(0, len(Xtr), batch):
            idx = order[b:b + batch]
            opt.zero_grad()
            loss = lossFnc(Xtr[idx], ytr[idx])
            loss.backward()
            opt.step()
            lossEp = lossEp + float(loss)
        if (ep + 1) % max(int(dl['lrDropPr']), 1) == 0:
            for g in opt.param_groups:
                g['lr'] = g['lr'] * dl['lrDropFa']
        net.eval()
        with torch.no_grad():
            lv = float(torch.mean((net(Xvl) - yvl) ** 2))
        info['loss'].append(lossEp)
        info['valLoss'].append(lv)
        info['learnRate'].append(opt.param_groups[0]['lr'])
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
    net.eval()

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Training Physics-Informed Neural Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'sys': net, 'info': info, 'seq': 0, 'W': W}


#######################################################################################################################
# References
#######################################################################################################################
