#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoard
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
This function turns a board layout (board technology, cooling, chips with technology, position and size) into the
2D field description of TherMOS on a regular grid of nodes from edge to edge (spacing dx = dy, vertex-centred finite
volumes: the control volume of a node on an edge is a half cell, of a corner node a quarter cell, so that the control
volumes cover the board area exactly and match the node-to-node quadrature of the POD projection). Two views are
available:

1) Top view (x-y): the board stack-up is lumped over its thickness t_b = Σ t_l into effective properties

       k_eff = Σ k_xy,l·t_l / t_b,   (ρc)_eff = Σ ρ_l·c_l·t_l / t_b
       h_eff = 1 / (Σ t_l/k_z,l + 1/h) + h_top,   h_v = h_eff / t_b   (distributed heat sink, W/m³K)

   so that (ρc)_eff·∂T/∂t = ∇·(k_eff∇T) − h_v·(T − T_a) + Q with Q = P_chip/(A_chip·t_b) under every chip. T is the
   temperature of the top copper; the die and the die attach add their heat capacity under the chip and the
   junction temperature follows from T_j = T + P·R_jb with R_jb = (t_die/k_die + t_att/k_att)/A_chip.
2) Cross-section (x-z) at y = yCut: every layer of the stack-up, the die attach and the die are resolved (layers get
   at least one cell, dz is chosen from the thinnest layer), the space above the board is air. The coolant acts on
   the bottom row (h_v = 2h/dz, half cells), natural convection on the top row (2h_top/dz). The conductivity is anisotropic
   (k: x direction, ky: z direction). Only chips cut by y = yCut are included; their heat Q = P/(w·t_die·l) assumes
   no spreading along y (2D cross-section).

Inputs:     1) layout:  board layout {'view', 'W', 'L' (mm), 'board', 'h' (W/m²K), 'hTop' (W/m²K), 'Ta' (°C),
                        'dx' (mm), 'yCut' (mm), 'chips': [{'name', 'tech', 'x', 'y', 'w', 'l' (mm), ...}]}
Outputs:    1) D:       field data (TherMOS format, SI units): geo, k, ky, rho, Cp, hv, hc, Ta, fl, dx, dy, Lx, Ly,
                        B (N x nChips, W/m³ per W of every chip), Rjb (K/W per chip), shape (ny, nx), wA (N, control
                        volume per cell volume: 1, 1/2 on edges, 1/4 in corners), chips (used)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.boardLib import BOARDS, CHIPS, H_TOP
from src.py.tools.toolData import MATERIALS

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _chip(c):
    """Chip with the defaults of its technology."""
    t = CHIPS[c.get('tech', 'Si MOSFET')]
    out = {**t, **{k: v for k, v in c.items() if v is not None}}
    out['w'], out['l'] = float(out['w']), float(out['l'])
    return out


def _w(n):
    """Control volume weights along one axis (half cells at both ends)."""
    w = np.ones(n)
    if n > 1:
        w[0], w[-1] = 0.5, 0.5
    return w


def _mat(name):
    k, rho, cp = MATERIALS[name][0:3]
    return k, rho, cp


#######################################################################################################################
# Main Function
#######################################################################################################################
def genBoard(layout):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    view = layout.get('view', 'xy')
    W, L = float(layout['W']) * 1e-3, float(layout['L']) * 1e-3
    dx = float(layout.get('dx', 1.0)) * 1e-3
    h, hTop = float(layout.get('h', 1500.0)), float(layout.get('hTop', H_TOP))
    Ta = float(layout.get('Ta', 25.0))
    layers = BOARDS[layout['board']]
    chips = [_chip(c) for c in layout.get('chips', [])]

    # ==============================================================================
    # Variables
    # ==============================================================================
    D = {'view': view, 'Ta0': Ta}

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Top View
    # ==============================================================================
    if view == 'xy':
        # Grid
        nx, ny = max(int(round(W / dx)), 1) + 1, max(int(round(L / dx)), 1) + 1
        x, y = np.arange(nx) * dx, np.arange(ny) * dx
        wA = np.outer(_w(ny), _w(nx))
        X, Y = np.meshgrid(x, y)                                                                     # (ny, nx)

        # Effective board
        t = np.array([l[2] for l in layers]) * 1e-3
        tb = t.sum()
        kxy = np.sum([l[3] * ti for l, ti in zip(layers, t)]) / tb
        rc = np.sum([l[5] * l[6] * ti for l, ti in zip(layers, t)]) / tb
        rho = np.sum([l[5] * ti for l, ti in zip(layers, t)]) / tb
        Rz = np.sum([ti / l[4] for l, ti in zip(layers, t)])
        hEff = 1.0 / (Rz + 1.0 / max(h, 1e-9)) + hTop
        K = np.full((ny, nx), kxy)
        RC = np.full((ny, nx), rc)
        B, Rjb, used = [], [], []

        # Chips
        for c in chips:
            m = (np.abs(X - c['x'] * 1e-3) <= c['w'] * 5e-4 + 1e-9) & (np.abs(Y - c['y'] * 1e-3) <= c['l'] * 5e-4 + 1e-9)
            if not m.any():                                                                  # chip smaller than a cell
                m = np.zeros_like(m)
                m[int(np.argmin(np.abs(y - c['y'] * 1e-3))), int(np.argmin(np.abs(x - c['x'] * 1e-3)))] = True
            kd, rd, cd = _mat(c['mat'])
            ka, ra, ca = _mat(c['attach'])
            RC[m] += (rd * cd * c['t'] + ra * ca * c['t_att']) * 1e-3 / tb
            b = np.zeros((ny, nx))
            b[m] = 1.0 / (wA[m].sum() * dx * dx * tb)
            B.append(b.ravel())
            Rjb.append((c['t'] * 1e-3 / kd + c['t_att'] * 1e-3 / ka) / (c['w'] * c['l'] * 1e-6))
            used.append(c)
        D.update({'k': K.ravel(), 'ky': K.ravel(), 'rho': np.full(nx * ny, rho), 'Cp': RC.ravel() / rho,
                  'hv': np.full(nx * ny, hEff / tb), 'tb': tb, 'kEff': kxy, 'hEff': hEff, 'Rz': Rz})

    # ==============================================================================
    # Cross-Section
    # ==============================================================================
    else:
        yCut = float(layout.get('yCut', layout['L'] / 2)) * 1e-3
        used = [c for c in chips if abs(c['y'] * 1e-3 - yCut) <= c['l'] * 5e-4]
        stack = list(reversed(layers))                                                           # bottom → top
        tMin = min([l[2] for l in stack] + [c['t'] for c in used] + [c['t_att'] for c in used]) * 1e-3
        tTot = sum(l[2] for l in stack) * 1e-3 + max([(c['t'] + c['t_att']) * 1e-3 for c in used] + [0.0])
        dz = max(tMin, tTot / 150.0)
        nx = max(int(round(W / dx)), 1) + 1
        x = np.arange(nx) * dx

        # Rows of the board
        rows = []
        for l in stack:
            rows += [(l[3], l[4], l[5] * l[6], l[5])] * max(1, int(round(l[2] * 1e-3 / dz)))
        nb = len(rows)
        nTop = max([max(1, int(round(c['t_att'] * 1e-3 / dz))) + max(1, int(round(c['t'] * 1e-3 / dz)))
                    for c in used] + [0]) + 1
        nz = nb + nTop
        wA = np.outer(_w(nz), _w(nx))
        ka, ra, ca = _mat('Air')
        Kx = np.full((nz, nx), ka)
        Kz = np.full((nz, nx), ka)
        RC = np.full((nz, nx), ra * ca)
        RHO = np.full((nz, nx), ra)
        for j, (kxy, kz, rc, rho) in enumerate(rows):
            Kx[j], Kz[j], RC[j], RHO[j] = kxy, kz, rc, rho

        # Chips
        B, Rjb = [], []
        for c in used:
            cols = np.abs(x - c['x'] * 1e-3) <= c['w'] * 5e-4 + 1e-9
            if not cols.any():
                cols[int(np.argmin(np.abs(x - c['x'] * 1e-3)))] = True
            na, nd = max(1, int(round(c['t_att'] * 1e-3 / dz))), max(1, int(round(c['t'] * 1e-3 / dz)))
            kA, rA, cA = _mat(c['attach'])
            kD, rD, cD = _mat(c['mat'])
            for j in range(nb, nb + na):
                Kx[j, cols], Kz[j, cols], RC[j, cols], RHO[j, cols] = kA, kA, rA * cA, rA
            b = np.zeros((nz, nx))
            for j in range(nb + na, nb + na + nd):
                Kx[j, cols], Kz[j, cols], RC[j, cols], RHO[j, cols] = kD, kD, rD * cD, rD
                b[j, cols] = 1.0
            b = b / ((b * wA).sum() * dx * dz * c['l'] * 1e-3)
            B.append(b.ravel())
            Rjb.append(0.0)                                                     # the die is resolved: T_j = max(T_die)

        # Sinks
        HV = np.zeros((nz, nx))
        HV[0] = 2 * h / dz
        HV[-1] = 2 * hTop / dz
        y = np.arange(nz) * dz
        ny, dyv = nz, dz
        D.update({'k': Kx.ravel(), 'ky': Kz.ravel(), 'rho': RHO.ravel(), 'Cp': (RC / RHO).ravel(), 'hv': HV.ravel(),
                  'tb': nb * dz, 'nBoard': nb})

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    dy = dx if view == 'xy' else dyv
    X, Y = np.meshgrid(x, y)
    N = X.size
    D.update({'geo': np.column_stack([X.ravel(), Y.ravel()]), 'dx': dx, 'dy': dy, 'Lx': x[-1], 'Ly': y[-1],
              'x': x, 'y': y, 'wA': wA.ravel(),
              'shape': (len(y), len(x)), 'hc': np.zeros(N), 'Ta': np.full(N, Ta), 'fl': np.zeros(N),
              'B': np.column_stack(B) if B else np.zeros((N, 0)), 'Rjb': np.asarray(Rjb), 'chips': used})

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return D


#######################################################################################################################
# References
#######################################################################################################################
