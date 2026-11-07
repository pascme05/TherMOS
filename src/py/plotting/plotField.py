#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         plotField
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
This function plots the results of the 2D/3D models for one model. 3D fields are shown in the x-y plane at the
height z = inpZ. Figures:
1) Material: thermal conductivity k (W/mK), specific heat capacity Cp (J/kgK), density ρ (kg/m³).
2) Temperature field at the last time step: measured, predicted and error T - T̂ (K), with the selected node.
3) Time response at the selected node (inpX, inpY(, inpZ)): measured and predicted temperature and the error, and
   the mean absolute error over all nodes per time step.
4) If calculated: magnitude of the temperature gradient |∇T| (K/m) and heat flux |q| = k·|∇T| (W/m²).
Inputs:     1) pred:    prediction of the model
            2) grt:     ground truth
            3) setup:   includes all simulation variables
            4) para:    all simulation parameters
            5) tag:     model tag
Outputs:    None

V1.0: 07.11.2026, Pascal Schirmer
- Python port of the field plots of plotResults2D.m/plotResults3D.m; common colour limits for measured and predicted
  fields, units on all colour bars, error plotted with a symmetric diverging colour map, gradient and flux titles
  swapped back (F-PL1, F-PL2, F-PL6)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.map2D import map2D
from src.py.general.map3D import map3D
from src.py.general.findNode import findNode

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _grid(v, D, d, z0):
    g = np.asarray(D['geo'], dtype=float)
    g = g - g.min(axis=0)
    x = np.arange(0, D['Lx'] + D['dx'] * 1e-6, D['dx'])
    y = np.arange(0, D['Ly'] + D['dy'] * 1e-6, D['dy'])
    if d == 2:
        return [map2D(v, g[:, 0], g[:, 1], x, y, 2), x, y]
    z = np.arange(0, D['Lz'] + D['dz'] * 1e-6, D['dz'])
    iz = int(np.argmin(np.abs(z - z0)))
    return [map3D(v, g[:, 0], g[:, 1], g[:, 2], x, y, z, 2)[..., iz], x, y]


def _img(ax, fig, F, x, y, title, unit, cmap='jet', lim=None, node=None):
    im = ax.pcolormesh(x * 1e3, y * 1e3, F, shading='auto', cmap=cmap,
                       vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
    if node is not None:
        ax.plot(node[0] * 1e3, node[1] * 1e3, 'kx', ms=8)
    ax.set_title(title)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    r = (y[-1] - y[0]) / max(x[-1] - x[0], 1e-12)
    ax.set_aspect('equal' if 0.2 < r < 5 else 'auto')  # thin layers (e.g. 50 x 2 mm) are stretched
    fig.colorbar(im, ax=ax, label=unit)


#######################################################################################################################
# Main Function
#######################################################################################################################
def plotField(pred, grt, setup, para, tag):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    D = grt['Data']
    d = 3 if np.asarray(D['geo']).shape[1] == 3 and setup['datDim'] == 3 else 2
    g = para['Dat']['gen']
    z0 = g.get('inpZ', 0) if d == 3 else 0
    pos = [g['inpX'], g['inpY']] + ([z0] if d == 3 else [])
    node = findNode(D['geo'], pos, setup['legacy'])
    yT, yP = np.asarray(grt['y']), np.asarray(pred['y'])
    tt = np.arange(len(yT)) * grt['Ts']
    sub = ' (z = %.1f mm)' % (z0 * 1e3) if d == 3 else ''

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Material
    # ==============================================================================
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Material Properties' + sub)
    for a, (k, t, u) in zip(ax, (('k', 'Thermal Conductivity k', 'W/mK'), ('Cp', 'Specific Heat Capacity Cp', 'J/kgK'),
                                 ('rho', 'Mass Density ρ', 'kg/m³'))):
        [F, x, y] = _grid(np.asarray(D[k], dtype=float).ravel(), D, d, z0)
        _img(a, fig, F[0], x, y, t, u)
    fig.tight_layout()

    # ==============================================================================
    # Temperature Field
    # ==============================================================================
    [F, x, y] = _grid(np.vstack([yT[-1], yP[-1]]), D, d, z0)
    lim = [np.nanmin(F), np.nanmax(F)]
    E = F[0] - F[1]
    eMax = max(np.nanmax(np.abs(E)), 1e-12)
    gN = np.asarray(D['geo'], dtype=float)[node] - np.asarray(D['geo'], dtype=float).min(axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Temperature at t = %.0f sec, %s model%s' % (tt[-1], tag.upper(), sub))
    _img(ax[0], fig, F[0], x, y, 'Measured', 'Temperature (°C)', lim=lim, node=gN)
    _img(ax[1], fig, F[1], x, y, 'Predicted', 'Temperature (°C)', lim=lim, node=gN)
    _img(ax[2], fig, E, x, y, 'Error T - T̂', 'Error (K)', 'RdBu_r', [-eMax, eMax], gN)
    fig.tight_layout()

    # ==============================================================================
    # Time Response
    # ==============================================================================
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle('Time Response at node %d (x = %.1f mm, y = %.1f mm), %s model' % (node, gN[0] * 1e3, gN[1] * 1e3,
                                                                                    tag.upper()))
    ax[0].plot(tt, yT[:, node], 'k', label='Measured')
    ax[0].plot(tt, yP[:, node], 'r--', label='Predicted')
    ax[0].set_ylabel('Temperature (°C)')
    ax[0].legend()
    ax[1].plot(tt, yT[:, node] - yP[:, node], 'b')
    ax[1].set_ylabel('Error T - T̂ (K)')
    ax[2].plot(tt, np.mean(np.abs(yT - yP), axis=1), 'b')
    ax[2].set_ylabel('MAE all nodes (K)')
    ax[2].set_xlabel('Time (sec)')
    for a in ax:
        a.grid(True)
    fig.tight_layout()

    # ==============================================================================
    # Gradient and Flux
    # ==============================================================================
    if 'q' in pred and 'q' in grt:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temperature Gradient and Heat Flux at t = %.0f sec, %s model%s' % (tt[-1], tag.upper(), sub))
        for j, (k, t, u) in enumerate((('qk', 'Temperature Gradient |∇T|', 'K/m'), ('q', 'Heat Flux |q|', 'W/m²'))):
            A, B = np.asarray(grt[k]), np.asarray(pred[k])
            A, B = (A[-1], B[-1]) if A.ndim == d + 1 else (A, B)
            if d == 3:
                iz = int(np.argmin(np.abs(np.arange(A.shape[-1]) * D['dz'] - z0)))
                A, B = A[..., iz], B[..., iz]
            lim = [np.nanmin([A, B]), np.nanmax([A, B])]
            _img(ax[j, 0], fig, A, x, y, t + ' (measured)', u, 'hot', lim)
            _img(ax[j, 1], fig, B, x, y, t + ' (predicted)', u, 'hot', lim)
        fig.tight_layout()


#######################################################################################################################
# References
#######################################################################################################################
