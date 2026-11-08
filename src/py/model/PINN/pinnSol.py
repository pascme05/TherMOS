#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnSol
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
This function predicts the temperatures with the trained physics-informed network. Prediction mode (Mdl › pinn ›
pred): (1) the network runs over every testing profile in one pass (the recurrent state is carried from the start of
the profile, no block boundaries; networks with self-attention fall back to windows), (2) consecutive windows of W
samples (as dlSol). If the losses depend on temperature (Rohm ≠ 0) the losses are corrected iteratively as in dlSol.
The predicted states of the physics network are stored in out['xPhys'] (list per profile, K).
Inputs:     1) mdl:     trained model
            2) data:    testing data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) out:     prediction

V1.0: 07.11.2026, Pascal Schirmer
- Prediction as dlSol (seq2seq windows)

V2.0: 08.11.2026, Pascal Schirmer
- Prediction over the whole profile, reference channels, models of V1.x (dlNet) are still predicted with dlSol
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.DL.dlSol import dlSol
from src.py.model.DL.dlWin import dlWin
from src.py.general.calcTheta import calcTheta
from src.py.general.getDevice import getDevice
from src.py.model.PINN.pinnPhys import split

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import torch


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _attention(net):
    return any(type(m).__name__ == 'TransformerEncoderLayer' for m in net.modules())


def predictNet(net, X, r, W, pred, dev, y0=None):
    """
    Outputs (Nt x M, data units) and states (Nt x n, K) of one profile. y0: measured outputs at the start of the
    profile (data units, needed for x0 = 1). Windows (pred = 2) run one after the other; with x0 = 1 every window
    starts from the predicted output at the end of the previous one.
    """
    X = np.asarray(X, dtype=np.float32)
    r = np.zeros((len(X), 1), dtype=np.float32) if r is None else np.asarray(r, dtype=np.float32).reshape(len(X), -1)
    M = net.O.shape[0]
    y0 = None if y0 is None else torch.as_tensor(np.asarray(y0, dtype=np.float32).reshape(1, M), device=dev)
    with torch.no_grad():
        if pred == 1 and not _attention(net):
            x, T, _ = net.outputs(torch.from_numpy(X)[None].to(dev), torch.from_numpy(r)[None].to(dev), y0)
            return net.toData(T)[0].cpu().numpy(), x[0].cpu().numpy()
        Xw = torch.from_numpy(dlWin(X, None, W, 0, 1, 1)[0])
        rw = torch.from_numpy(dlWin(r, None, W, 0, 1, 1)[0])
        if getattr(net, 'x0', 0):
            ys, xs, yk = [], [], y0
            for i in range(0, len(Xw)):
                x, T, _ = net.outputs(Xw[i:i + 1].to(dev), rw[i:i + 1].to(dev), yk)
                yd = net.toData(T)
                ys.append(yd.cpu().numpy())
                xs.append(x.cpu().numpy())
                yk = yd[:, -1]
        else:
            ys, xs = [], []
            for i in range(0, len(Xw), 1024):
                x, T, _ = net.outputs(Xw[i:i + 1024].to(dev), rw[i:i + 1024].to(dev))
                ys.append(net.toData(T).cpu().numpy())
                xs.append(x.cpu().numpy())
        y, x = np.concatenate(ys), np.concatenate(xs)
        return y.reshape(-1, y.shape[-1])[0:len(X)], x.reshape(-1, x.shape[-1])[0:len(X)]


#######################################################################################################################
# Main Function
#######################################################################################################################
def pinnSol(mdl, data, para, setup, legacy=0):
    ###################################################################################################################
    # Models of V1.x
    ###################################################################################################################
    if not hasattr(mdl['sys'], 'outputs'):
        return dlSol(mdl, data, para, dict(setup, featRoll=0), legacy)

    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Predicting Physics-Informed Neural Network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    X2 = [np.asarray(x, dtype=float) for x in data['X2']]
    [Xp, y2m, r2, sl] = split(data)                                     # profiles cut at every session ID
    Toff2 = data['r2']
    errMax = para['Mdl']['gen']['err']
    nSub = int(para['Mdl']['gen']['nSub'])
    pred = int(mdl.get('pred', 1) or 1)
    out = dict(data)
    dev = getDevice(para)
    net = mdl['sys'].to(dev).eval()
    if pred == 1 and _attention(net):
        print("INFO: Self-attention layers: prediction with windows of W = %d samples" % mdl['W'])
    print("INFO: Prediction %s" % ('over the whole profile' if pred == 1 else 'with windows of W = %d samples'
                                   % mdl['W']))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    def run(Xs):
        Xc = np.concatenate([np.asarray(x, dtype=float) for x in Xs])
        res = [predictNet(net, Xc[k], r, mdl['W'], pred, dev, None if y is None else np.asarray(y)[0])
               for k, r, y in zip(sl, r2, y2m)]
        y, x = np.concatenate([a[0] for a in res]), np.concatenate([a[1] for a in res])
        o = np.cumsum([0] + [len(v) for v in Xs])
        return [y[o[i]:o[i + 1]] for i in range(len(Xs))], [x[o[i]:o[i + 1]] for i in range(len(Xs))]
    print("INFO: %d testing profile(s)" % len(sl))
    y2, x2 = run(X2)

    # ==============================================================================
    # Loss iteration
    # ==============================================================================
    if para['Par']['loss']['Rohm'] != 0:
        Pv2 = X2
        for it in range(0, nSub):
            Pv2 = [x * calcTheta(y[:, 0:1] + r[:, 0:1], para, legacy) for x, y, r in zip(X2, y2, Toff2)]
            yNew, x2 = run(Pv2)
            err = np.mean(np.abs(np.concatenate(yNew) - np.concatenate(y2)))
            y2 = yNew
            if err < errMax:
                break
        X2 = Pv2
        print("INFO: Loss iteration finished after %d iterations" % (it + 1))

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    out['y'] = np.concatenate(y2)
    out['X'] = np.concatenate(X2)
    out['xPhys'] = x2
    mdl['sys'].to('cpu')

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Predicting Physics-Informed Neural Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
