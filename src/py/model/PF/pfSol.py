#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfSol
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function predicts the temperature field of the testing data with the trained physics-informed field network.
1) Operator network: the heat generation of every testing profile is projected onto the source basis of the training
   (p = Q·U⁺, the relative error of this encoding is printed), the branch runs once over the whole profile (preceded
   by the 'warm' samples of equilibrium, as in the training) and the trunk is evaluated at all nodes of the testing
   mesh: T̂(x, t) = T_ref + ΔT·Σ_k b_k(t)·φ_k(x). The testing mesh may differ from the training mesh (the footprints
   U are taken from the nearest training node).
2) Coordinate network: T̂(x, t) = T_ref + ΔT·f(x, t) at all nodes and samples of the profile it was trained on.
The coefficients b (operator network) are stored in out['theta_hat'], the basis functions at the nodes in
out['phi'].
Inputs:     1) mdl:     trained model
            2) data:    testing data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables
            5) legacy:  not used
Outputs:    1) out:     prediction (y: temperature at the nodes (Nt x N), odeTime)

V1.0: 16.11.2026, Pascal Schirmer
- Initial version
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfData import toHat, offset, projectSrc, mapNodes
from src.py.general.getDevice import getDevice

# ==============================================================================
# External
# ==============================================================================
import sys
import time
import numpy as np
import torch


#######################################################################################################################
# Main Function
#######################################################################################################################
def pfSol(mdl, data, para, setup, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Predicting Physics-Informed Field Network")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    init = int(para['Exp']['gen'].get('init', 1) or 1)
    geo = np.asarray(data['Data']['geo'], dtype=float)
    N = len(geo)
    dT = mdl['dT']
    off = offset(data, init)
    out = dict(data)
    dev = getDevice(para)
    net = mdl['sys'].to(dev).eval()
    xh = torch.as_tensor(toHat(geo, mdl['sc']), device=dev)
    y2 = []

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    tStart = time.time()
    with torch.no_grad():
        # ==============================================================================
        # Operator Network
        # ==============================================================================
        if mdl['var'] == 1:
            U = mdl['U'][mapNodes(mdl['geo'], geo.astype(np.float32))]
            warm, S = int(mdl['warm']), mdl['U'].shape[1]
            phi = torch.cat([net.basis(xh[i:i + 65536]) for i in range(0, N, 65536)])
            b2 = []
            for Q, o in zip(data['X2'], off):
                [p, err] = projectSrc(Q, U)
                print("INFO: Profile with %d samples, relative error of the source encoding %.2e" % (len(p), err))
                if err > 0.05:
                    print("WARN: The heat sources of this profile are not covered by the sources of the training "
                          "data (error %.1f %%): train with profiles that drive these sources" % (100 * err))
                P = torch.as_tensor(np.vstack([np.zeros((warm, S)), p]), dtype=torch.float32, device=dev)
                b = net.coeff(P[None])[0, warm:]
                y2.append((dT * (b @ phi.T)).cpu().numpy().astype(float) + o)
                b2.append(b.cpu().numpy())
            out['theta_hat'] = np.concatenate(b2)
            out['phi'] = phi.cpu().numpy()

        # ==============================================================================
        # Coordinate Network
        # ==============================================================================
        else:
            Nt = len(data['y2'][0])
            if Nt != mdl['Nt'] or N != len(mdl['geo']):
                sys.exit("ERROR: PF mode, coordinate network: the model was trained on a profile with %d samples "
                         "and %d nodes, the testing profile has %d samples and %d nodes; this variant reconstructs "
                         "the profile it was trained on (train the model again)" % (mdl['Nt'], len(mdl['geo']), Nt, N))
            tH = torch.as_tensor(2 * np.arange(Nt) * mdl['Ts'] / mdl['tEnd'] - 1, dtype=torch.float32, device=dev)
            th = np.zeros((Nt, N))
            step = max(int(2e6 // N), 1)
            for i in range(0, Nt, step):
                tt = tH[i:i + step]
                z = torch.cat([xh[None].expand(len(tt), -1, -1), tt[:, None, None].expand(-1, N, 1)], dim=-1)
                th[i:i + step] = (dT * net(z)).cpu().numpy()
            y2.append(th + off[0])
            for k in range(1, len(data['y2'])):                         # further profiles: not modelled
                y2.append(np.full(np.shape(data['y2'][k]), np.nan))

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    out['odeTime'] = time.time() - tStart
    out['y'] = np.concatenate(y2)
    mdl['sys'].to('cpu')

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Predicting Physics-Informed Field Network")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
