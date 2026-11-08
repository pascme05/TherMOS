#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         dlFit
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function trains a deep learning model (PyTorch) that maps a window of the input features (losses, optionally
extended with EWMA/EWMS features) to the temperatures of all nodes (see dlNet). The windows are cut per profile, so
no window spans two profiles. Training uses Adam with a learning rate schedule (lrAdapt = 0: factor lrDropFa every
lrDropPr epochs; lrAdapt = 1: factor lrDropFa after lrDropPr epochs without improvement of the validation loss, at
least 1e-3·initLr), gradient norm clipping (gradTh), validation every valFreq iterations and early stopping after
valPat validations without improvement; the network with the lowest validation loss is returned.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables (selDL, featRoll, EWMA, EWMS, archDL: user-defined
                        architecture of dlNet, optional, replaces selDL)
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model {'sys': network, 'info': training history, 'seq', 'W'}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of dlFit.m; windows per profile, the seq2point target is the seq-th sample in training and inference
  (F-L5), CNN seq2seq network (MATLAB had no layers for this case, F-L7)

V1.1: 07.11.2026, Pascal Schirmer
- Training on the CPU or the GPU (para['Mdl']['gen']['dev'], getDevice); the network is returned on the CPU

V1.2: 08.11.2026, Pascal Schirmer
- User-defined architecture (setup['archDL'], see dlNet)

V1.3: 08.11.2026, Pascal Schirmer
- Adaptive learning rate (lrAdapt), the learning rate is printed every epoch and stored in the history
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlNet import dlNet
from src.py.model.DL.dlWin import dlWin
from src.py.general.calcFeat import calcFeat
from src.py.general.getDevice import getDevice

# ==============================================================================
# External
# ==============================================================================
import copy
import numpy as np
import torch
import torch.nn as nn


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _windows(d, W, seq, stride, setup, legacy):
    Xs, ys = [], []
    for X, y in zip(d['X2'], d['y2']):
        if setup.get('featRoll', 0) != 0:
            X = calcFeat(X, setup['EWMA'], setup['EWMS'], None, legacy)
        Xw, yw = dlWin(X, y, W, seq, stride, 0)
        if len(Xw) > 0:
            Xs.append(Xw)
            ys.append(yw)
    if len(Xs) == 0:
        raise ValueError('Profiles are shorter than the window length W')
    return [np.concatenate(Xs), np.concatenate(ys)]


def _loss(sel, legacy):
    if sel == 2:
        return nn.L1Loss()
    elif sel == 3:
        return nn.HuberLoss()
    elif sel == 4:
        l1 = nn.L1Loss()
        return (lambda a, b: 2 * l1(a, b)) if legacy == 1 else l1
    else:
        return nn.MSELoss()


#######################################################################################################################
# Main Function
#######################################################################################################################
def dlFit(data, val, para, setup, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Training Deep Learning model")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    dl = para['Mdl']['dl']
    W, stride, seq = int(dl['W']), max(int(dl['stride']), 1), int(dl['seq'])
    seq = min(seq, W)
    epoch, batch = int(dl['epoch']), int(dl['batch'])
    valFreq = max(int(dl['valFreq']), 1)
    valPat = int(dl.get('valPat', 10) or 10)
    dev = getDevice(para)
    torch.manual_seed(42)
    np.random.seed(42)

    # ==============================================================================
    # Variables
    # ==============================================================================
    [Xtr, ytr] = _windows(data, W, seq, stride, setup, legacy)
    [Xvl, yvl] = _windows(val, W, seq, stride, setup, legacy)
    F, M = Xtr.shape[2], ytr.shape[-1]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    mu = Xtr.reshape(-1, F).mean(axis=0)
    sd = Xtr.reshape(-1, F).std(axis=0)
    sd[sd < 1e-12] = 1
    arch = setup.get('archDL') or None
    if arch:
        print("INFO: User-defined architecture: " + ' -> '.join(f"{l['type']} {l['units']}" for l in arch['body'])
              + f" -> head {arch.get('head', [])}")
    net = dlNet(int(setup['selDL']), F, M, seq, mu, sd, arch=arch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=dl['initLr'])
    lrAd = int(dl.get('lrAdapt', 0) or 0)
    if lrAd == 1:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=dl['lrDropFa'], threshold=1e-3,
                                                         patience=max(int(dl['lrDropPr']) - 1, 0),
                                                         min_lr=1e-3 * dl['initLr'])
        print("INFO: Adaptive learning rate: factor %g after %d epoch(s) without improvement of the validation loss"
              % (dl['lrDropFa'], max(int(dl['lrDropPr']), 1)))
    else:
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(int(dl['lrDropPr']), 1), gamma=dl['lrDropFa'])
    lossFnc = _loss(int(dl['loss']), legacy)
    Xtr, ytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xvl, yvl = torch.from_numpy(Xvl).to(dev), torch.from_numpy(yvl).to(dev)
    order = torch.arange(len(Xtr))
    if dl['shu'] == 2:
        order = torch.randperm(len(Xtr))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    best, bestNet, wait, it, stop = np.inf, copy.deepcopy(net.state_dict()), 0, 0, False
    info = {'iter': [], 'loss': [], 'valIter': [], 'valLoss': [], 'lrIter': [], 'learnRate': [], 'lrAdapt': lrAd,
            'device': str(dev)}
    for ep in range(0, epoch):
        if dl['shu'] not in (1, 2):
            order = torch.randperm(len(Xtr))
        net.train()
        for b in range(0, len(Xtr), batch):
            idx = order[b:b + batch]
            if len(idx) < 2:                                            # BatchNorm needs two samples
                continue
            opt.zero_grad()
            loss = lossFnc(net(Xtr[idx].to(dev)), ytr[idx].to(dev))
            loss.backward()
            if dl['gradTh'] > 0 and np.isfinite(dl['gradTh']):
                nn.utils.clip_grad_norm_(net.parameters(), dl['gradTh'])
            opt.step()
            it = it + 1
            info['iter'].append(it)
            info["loss"].append(float(loss.detach()))

            # Validation
            if it % valFreq == 0:
                net.eval()
                with torch.no_grad():
                    lv = float(lossFnc(net(Xvl), yvl))
                net.train()
                info['valIter'].append(it)
                info['valLoss'].append(lv)
                if lv < best:
                    best, bestNet, wait = lv, copy.deepcopy(net.state_dict()), 0
                else:
                    wait = wait + 1
                    stop = wait >= valPat
                if stop:
                    break
        lrEp = opt.param_groups[0]['lr']
        info['lrIter'].append(it)
        info['learnRate'].append(lrEp)
        if lrAd == 1:
            sch.step(info['valLoss'][-1] if info['valLoss'] else float(loss))
        else:
            sch.step()
        print("INFO: Epoch %d/%d, lr %.3e, training loss %.4e, best validation loss %.4e" % (ep + 1, epoch, lrEp,
                                                                                            float(loss), best))
        if opt.param_groups[0]['lr'] < lrEp:
            print("INFO: Learning rate reduced to %.3e" % opt.param_groups[0]['lr'])
        if stop:
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
    print("DONE: Training Deep Learning model")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'sys': net, 'info': info, 'seq': seq, 'W': W}


#######################################################################################################################
# References
#######################################################################################################################
