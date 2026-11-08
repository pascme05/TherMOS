#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoard
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.3
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
2) Layer-resolved model ('xz'): a 3D finite-volume model of the whole board. The x-y grid of the top view is
   repeated in every row of the stack (layers below, board layers with 1-4 rows each, die attach and die over the
   chip footprints). The coolant h acts below the lowest row, natural convection h_top on the top of every column.
   All chips are included; T_j is the hottest die cell. The matrices are returned in D['mat'] (calcBoardMat), the
   display grid is the x-z slice at y = yCut (D['sel'], -1 = air), D['idx'] maps every row to its nodes.

Layers below the board (layout['below'], top → bottom, e.g. a gap filler and an aluminium heat sink) are placed
between the board and the cooling h, which then acts on the bottom of the last layer:

1) Top view: the layers below form a second plane (the heat sink) with its own temperature T_2, lateral conductance
   Σ k_xy,i·t_i and heat capacity Σ ρ_i·c_i·t_i, so the heat sink spreads the heat of the chips sideways. The planes
   are coupled through the board and the interface layers, the heat sink is cooled from its bottom:

       R_12 = Σ_board t_l/k_z,l + Σ_i<last t_i/k_z,i + t_last/(2·k_z,last),   R_2a = t_last/(2·k_z,last) + 1/h

   The board plane is then only cooled by the natural convection on top. All plane-2 quantities are stored per
   metre of board thickness (k2 = Σ k_i·t_i / t_b, rc2, h12 = 1/(R_12·t_b), h2 = 1/(R_2a·t_b)) so that calcBoardMat
   assembles both planes with the same control volumes. For the export (PO mode, one temperature per node) the
   equivalent distributed sink of the series path hvEq = h_eff/t_b with h_eff = 1/(R_12 + R_2a) + h_top is stored.
2) Layer-resolved model: the layers below are resolved like the board layers.

Thermal vias (layout['vias'], averaged, boardLib.viaK) raise the through-plane conductivity of all FR4 layers in
the via region (under every chip plus a margin, or the whole board): in the top view through a smaller local
stack-up resistance R_z(x, y) and therefore a larger local h_eff(x, y); in the cross-section through k_z of the FR4
cells in the via columns.

Inputs:     1) layout:  board layout {'view', 'W', 'L' (mm), 'board' (preset), 'layers' (stack-up, top → bottom,
                        optional, replaces the preset), 'h' (W/m²K), 'hTop' (W/m²K), 'Ta' (°C), 'dx' (mm), 'yCut'
                        (mm), 'chips': [{'name', 'tech', 'x', 'y', 'w', 'l' (mm), ...}], 'chipLib' (custom chip
                        technologies, optional), 'vias': {'mode': 'none' | 'chips' | 'all', 'd' (mm), 'tpl' (µm),
                        'pitch' (mm), 'fill', 'margin' (mm)} (optional), 'below': layers below the board (top →
                        bottom, optional), 'pcb': additional losses in the board (optional) [{'name', 'layer'
                        (index in the stack-up, 0 = top), 'x', 'y', 'w', 'l' (mm, w or l ≤ 0: whole board), 'P' (W)}]}
Outputs:    1) D:       field data (TherMOS format, SI units): geo, k, ky, rho, Cp, hv, hc, Ta, fl, dx, dy, Lx, Ly,
                        B (N x nChips, W/m³ per W of every chip), Rjb (K/W per chip), shape (ny, nx), wA (N, control
                        volume per cell volume: 1, 1/2 on edges, 1/4 in corners), chips (used), via (via mask);
                        with layers below in the top view also k2, rc2, h12, h2 (plane 2) and hvEq; q (N, W per
                        node): constant additional losses in the board (traces, shunts, passives)

V1.1: 07.11.2026, Pascal Schirmer
- Stack-up from the layout (layers with copper coverage), custom chip technologies, averaged thermal vias

V1.2: 07.11.2026, Pascal Schirmer
- Layers below the board (interface material, heat sink): second plane in the top view, resolved rows in the
  cross-section

V1.3: 07.11.2026, Pascal Schirmer
- The cross-section is a layer-resolved 3D model of the whole board (the former 2D slice without spreading
  along y over-estimated T_j several times under thick interface layers)

V1.4: 08.11.2026, Pascal Schirmer
- Additional losses in the board (rectangular regions on a selected layer or the whole board, constant): q = P of
  the region distributed over its control volumes (layer-resolved model: over the rows of the layer; top view: over
  the lumped plane)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.boardLib import BOARDS, CHIPS, H_TOP, layerProps, viaK
from src.py.tools.toolData import MATERIALS

# ==============================================================================
# External
# ==============================================================================
import numpy as np


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _chip(c, lib):
    """Chip with the defaults of its technology."""
    t = lib.get(c.get('tech', 'Si MOSFET'), CHIPS['Si MOSFET'])
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


def _viaMask(X, Y, chips, vias):
    """Nodes in the via region (under every chip plus the margin, or everywhere)."""
    mode = (vias or {}).get('mode', 'none')
    if mode == 'all':
        return np.ones(X.shape, dtype=bool)
    m = np.zeros(X.shape, dtype=bool)
    if mode == 'chips':
        g = float(vias.get('margin', 0.0)) * 1e-3
        for c in chips:
            m |= (np.abs(X - c['x'] * 1e-3) <= c['w'] * 5e-4 + g + 1e-9) & \
                 (np.abs(Y - c['y'] * 1e-3) <= c['l'] * 5e-4 + g + 1e-9)
    return m


def _pcbMask(X, Y, r, W, L):
    """Nodes of an additional loss region of the board (w or l <= 0: whole board)."""
    w, l = float(r.get('w') or 0.0), float(r.get('l') or 0.0)
    if w <= 0 or l <= 0:
        return np.ones(X.shape, dtype=bool)
    m = (np.abs(X - float(r.get('x', W * 5e2)) * 1e-3) <= w * 5e-4 + 1e-9) &         (np.abs(Y - float(r.get('y', L * 5e2)) * 1e-3) <= l * 5e-4 + 1e-9)
    if not m.any():
        m = np.zeros(X.shape, dtype=bool)
        m[int(np.argmin(np.abs(Y[:, 0] - float(r['y']) * 1e-3))), int(np.argmin(np.abs(X[0] - float(r['x']) * 1e-3)))] = True
    return m


def _nRows(t, dzMax):
    """Number of rows of a layer of thickness t (mm) in the layer-resolved model."""
    return int(min(max(np.ceil(t * 1e-3 / dzMax - 1e-9), 1), 4))


def _rows(stack, dzMax):
    """Rows of the layer-resolved model, bottom → top: (layer, material, dz, k_xy, k_z, ρc, FR4) per row."""
    out = []
    for l in stack:
        t = l[2] * 1e-3
        n = _nRows(l[2], dzMax)
        out += [(l[0], l[1], t / n, l[3], l[4], l[5] * l[6], l[1] == 'FR4')] * n
    return out


def _model3D(layout, layers, below, chips, vias, isFR4, W, L, dx, h, hTop):
    """
    Layer-resolved 3D finite-volume model of the whole board: vertex-centred in x and y (as the top view), one to four
    cell rows per layer in z, die attach and die over every chip footprint. Returns the matrices and the x-z slice at
    y = yCut for the display.
    """
    import scipy.sparse as sp

    # Grid
    nx, ny = max(int(round(W / dx)), 1) + 1, max(int(round(L / dx)), 1) + 1
    x, y = np.arange(nx) * dx, np.arange(ny) * dx
    wx, wy = _w(nx), _w(ny)
    A = np.outer(wy, wx) * dx * dx
    X, Y = np.meshgrid(x, y)
    full = np.ones((ny, nx), dtype=bool)

    # Rows: layers below, board (bottom → top)
    rws = _rows(list(reversed(below)) + list(reversed(layers)), 0.5e-3)
    via = _viaMask(X, Y, chips, vias) if isFR4.any() else np.zeros((ny, nx), dtype=bool)
    R = []
    for (nm, mat, dz, kxy, kz, rc, fr4) in rws:
        KZ = np.full((ny, nx), kz)
        if fr4 and via.any():
            KZ[via] = viaK(vias, kz)
        R.append({'m': full, 'dz': np.full((ny, nx), dz), 'kx': np.full((ny, nx), kxy), 'kz': KZ,
                  'rc': np.full((ny, nx), rc), 'name': nm or mat})
    nBoard = len(rws)
    nBelow = len(_rows(list(reversed(below)), 0.5e-3)) if below else 0

    # Rows: die attach and die over the chip footprints
    ra = {'m': np.zeros((ny, nx), dtype=bool), 'dz': np.ones((ny, nx)), 'kx': np.ones((ny, nx)),
          'kz': np.ones((ny, nx)), 'rc': np.ones((ny, nx)), 'name': 'die attach'}
    rd = {'m': np.zeros((ny, nx), dtype=bool), 'dz': np.ones((ny, nx)), 'kx': np.ones((ny, nx)),
          'kz': np.ones((ny, nx)), 'rc': np.ones((ny, nx)), 'name': 'die'}
    masks = []
    for c in chips:
        m = (np.abs(X - c['x'] * 1e-3) <= c['w'] * 5e-4 + 1e-9) & (np.abs(Y - c['y'] * 1e-3) <= c['l'] * 5e-4 + 1e-9)
        if not m.any():
            m = np.zeros_like(m)
            m[int(np.argmin(np.abs(y - c['y'] * 1e-3))), int(np.argmin(np.abs(x - c['x'] * 1e-3)))] = True
        m = m & ~rd['m']                                                       # overlapping chips: first one wins
        kA, rA, cA = _mat(c['attach'])
        kD, rD, cD = _mat(c['mat'])
        for r, k, rc, t in ((ra, kA, rA * cA, c['t_att']), (rd, kD, rD * cD, c['t'])):
            r['m'] = r['m'] | m
            r['kx'][m], r['kz'][m], r['rc'][m], r['dz'][m] = k, k, rc, t * 1e-3
        masks.append(m)
    if chips:
        R += [ra, rd]
    nr = len(R)

    # Node numbering
    idx = -np.ones((nr, ny, nx), dtype=int)
    n = 0
    for r, row in enumerate(R):
        k = int(row['m'].sum())
        idx[r][row['m']] = np.arange(n, n + k)
        n += k
    N = n
    I, J, G = [], [], []
    gOut = np.zeros(N)
    C = np.zeros(N)

    # Assembly
    for r, row in enumerate(R):
        m, dz, kx, kz = row['m'], row['dz'], row['kx'], row['kz']
        C[idx[r][m]] = (row['rc'] * A * dz)[m]
        for (sa, sb, wl) in (((slice(None), slice(0, -1)), (slice(None), slice(1, None)), (wy * dx)[:, None]),
                             ((slice(0, -1), slice(None)), (slice(1, None), slice(None)), (wx * dx)[None, :])):
            both = m[sa] & m[sb]
            if not both.any():
                continue
            wb = np.broadcast_to(wl, m.shape)[sa]
            g = wb * 0.5 * (dz[sa] + dz[sb]) / (dx / (2 * kx[sa]) + dx / (2 * kx[sb]))
            I.append(idx[r][sa][both])
            J.append(idx[r][sb][both])
            G.append(g[both])
        if r + 1 < nr:
            up = R[r + 1]
            both = m & up['m']
            g = A / (dz / (2 * kz) + up['dz'] / (2 * up['kz']))
            I.append(idx[r][both])
            J.append(idx[r + 1][both])
            G.append(g[both])
            top = m & ~up['m']
        else:
            top = m
        if hTop > 0:
            gOut[idx[r][top]] += (A / (dz / (2 * kz) + 1.0 / hTop))[top]
        if r == 0:
            gBot = np.zeros(N)
            gBot[idx[r][m]] = (A / (dz / (2 * kz) + 1.0 / max(h, 1e-9)))[m]
            gOut += gBot
    I, J, G = np.concatenate(I), np.concatenate(J), np.concatenate(G)
    Goff = sp.coo_matrix((np.concatenate([-G, -G]), (np.concatenate([I, J]), np.concatenate([J, I]))), shape=(N, N))
    Gm = (Goff + sp.diags(-np.asarray(Goff.sum(axis=1)).ravel() + gOut)).tocsc()

    # Losses (die rows) and junction nodes
    Bp = np.zeros((N, len(chips)))
    B3 = np.zeros((N, len(chips)))
    for i, m in enumerate(masks):
        ii = idx[nr - 1][m]
        V = (A * rd['dz'])[m]
        Bp[ii, i] = V / V.sum()
        B3[ii, i] = 1.0

    # Additional losses in the board (row → layer index of the stack-up, 0 = top)
    q = np.zeros(N)
    lid = [None] * nBelow + [li for li in reversed(range(len(layers))) for _ in range(_nRows(layers[li][2], 0.5e-3))]
    for p in layout.get('pcb') or []:
        if float(p.get('P') or 0.0) == 0.0:
            continue
        m = _pcbMask(X, Y, p, W, L)
        li = int(p.get('layer', 0) or 0)
        rr = [r for r in range(nBelow, nBoard) if lid[r] == li] or [nBoard - 1]
        ii = np.concatenate([idx[r][m] for r in rr])
        V = np.concatenate([(A * R[r]['dz'])[m] for r in rr])
        q[ii] += float(p['P']) * V / V.sum()

    # Display: x-z slice at y = yCut (rows bottom → top, empty cells = air)
    yCut = float(layout.get('yCut', layout['L'] / 2)) * 1e-3
    jc = int(np.argmin(np.abs(y - yCut)))
    dzr = np.array([float(np.max(row['dz'][row['m']])) if row['m'].any() else 0.0 for row in R])
    zc = np.cumsum(dzr) - dzr / 2
    Xs, Zs = np.meshgrid(x, zc)
    tb = float(np.sum([l[2] for l in layers]) * 1e-3)
    return [{'mat': [C, Gm, Bp], 'B3': B3, 'gOut': gOut, 'sel': idx[:, jc, :], 'top': idx[nBoard - 1], 'idx': idx,
             'rowName': [row['name'] for row in R], 'nBoard': nBoard, 'nBelow': nBelow, 'tb': tb, 'yCut': y[jc],
             'geo': np.column_stack([Xs.ravel(), Zs.ravel()]), 'dx': dx, 'dy': float(np.min(dzr[dzr > 0])),
             'Lx': x[-1], 'Ly': zc[-1], 'x': x, 'y': zc, 'shape': (nr, nx), 'xg': x, 'yg': y, 'N3': N,
             'via': via.ravel(), 'A': A, 'gBot': gBot, 'q': q}, x, zc, A]


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
    stack = layout.get('layers') or BOARDS[layout['board']]
    layers = [(l.get('layer', ''), l['mat'], float(l['t'])) + layerProps(l) for l in stack if float(l['t']) > 0]
    lib = {**CHIPS, **(layout.get('chipLib') or {})}
    chips = [_chip(c, lib) for c in layout.get('chips', [])]
    vias = layout.get('vias') or {'mode': 'none'}
    isFR4 = np.array([l[1] == 'FR4' for l in layers])
    below = [(l.get('layer', ''), l['mat'], float(l['t'])) + layerProps(l) for l in (layout.get('below') or [])
             if float(l['t']) > 0]

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
        via = _viaMask(X, Y, chips, vias) if isFR4.any() else np.zeros((ny, nx), dtype=bool)
        RzV = np.sum([ti / (viaK(vias, l[4]) if f else l[4]) for l, ti, f in zip(layers, t, isFR4)]) if via.any() \
            else Rz
        hEffV = 1.0 / (RzV + 1.0 / max(h, 1e-9)) + hTop
        HV = np.where(via, hEffV, hEff) / tb

        # Plane 2: layers below the board (heat sink)
        if below:
            tl = np.array([l[2] for l in below]) * 1e-3
            kzl = np.array([l[4] for l in below])
            Rup = np.sum(tl[:-1] / kzl[:-1]) + tl[-1] / (2 * kzl[-1])          # interface layers + half the sink
            R2a = tl[-1] / (2 * kzl[-1]) + 1.0 / max(h, 1e-9)
            R12 = np.where(via, RzV, Rz) + Rup
            hEff = 1.0 / (Rz + Rup + R2a) + hTop
            hEffV = 1.0 / (RzV + Rup + R2a) + hTop
            D.update({'k2': np.full(nx * ny, np.sum([l[3] * ti for l, ti in zip(below, tl)]) / tb),
                      'rc2': np.full(nx * ny, np.sum([l[5] * l[6] * ti for l, ti in zip(below, tl)]) / tb),
                      'h12': (1.0 / R12 / tb).ravel(), 'h2': np.full(nx * ny, 1.0 / R2a / tb),
                      'hvEq': (np.where(via, hEffV, hEff) / tb).ravel(), 'R12': Rz + Rup, 'R2a': R2a,
                      'tBelow': tl.sum(), 'ktBelow': np.sum([l[3] * ti for l, ti in zip(below, tl)])})
            HV = np.full((ny, nx), hTop / tb)
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
        # Additional losses in the board (lumped plane, W per node)
        q = np.zeros((ny, nx))
        for p in layout.get('pcb') or []:
            if float(p.get('P') or 0.0) != 0.0:
                m = _pcbMask(X, Y, p, W, L)
                q[m] += float(p['P']) * wA[m] / wA[m].sum()
        D['q'] = q.ravel()
        D.update({'k': K.ravel(), 'ky': K.ravel(), 'rho': np.full(nx * ny, rho), 'Cp': RC.ravel() / rho,
                  'hv': HV.ravel(), 'tb': tb, 'kEff': kxy, 'hEff': hEff, 'Rz': Rz, 'hEffVia': hEffV, 'RzVia': RzV,
                  'via': via.ravel()})

    # ==============================================================================
    # Cross-Section (layer-resolved 3D model)
    # ==============================================================================
    else:
        [D3, x, y, wA] = _model3D(layout, layers, below, chips, vias, isFR4, W, L, dx, h, hTop)
        D.update(D3)
        used = chips

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    if view == 'xy':
        X, Y = np.meshgrid(x, y)
        N = X.size
        D.update({'geo': np.column_stack([X.ravel(), Y.ravel()]), 'dx': dx, 'dy': dx, 'Lx': x[-1], 'Ly': y[-1],
                  'x': x, 'y': y, 'wA': wA.ravel(),
                  'shape': (len(y), len(x)), 'hc': np.zeros(N), 'Ta': np.full(N, Ta), 'fl': np.zeros(N),
                  'B': np.column_stack(B) if B else np.zeros((N, 0)), 'Rjb': np.asarray(Rjb), 'chips': used})
    else:
        D.update({'Rjb': np.zeros(len(used)), 'chips': used, 'B': D['B3']})

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return D


#######################################################################################################################
# References
#######################################################################################################################
