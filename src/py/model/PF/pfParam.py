#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfParam
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This module builds the board families of the parametric PF study (one network for many boards, pfParamFit). Every board
is the IMS board of the PF tutorial (30 x 20 mm, 1 mm grid, layer-resolved 3D finite-volume model of genBoard) with
three chips (Si IGBT 6 x 6 mm, Si diode 4 x 5 mm, SiC MOSFET 4 x 5 mm) and the operating condition

    μ = (h, k_diel, s_att, s_rc)

with the heat transfer coefficient h on the bottom of the Al base (W/m²K), the conductivity of the dielectric k_diel
(W/mK), the factor s_att on the conductivity of the die attach of all chips (-) and the factor s_rc on the heat capacity
ρc of the Al base (-). The chip centres (x_c, y_c) (mm) are free (integer mm, inside the board with a margin, no
overlap). The condition enters the finite-volume model

    C·dθ/dt = −G·θ + Bp·P(t),      θ = T − T_a

only through its matrices: h the diagonal of G at the bottom nodes, k_diel and s_att the conductances of the
dielectric and die-attach rows, s_rc the capacitances C of the Al base, and the chip position which nodes carry the
input Bp (and where the die rows exist, so the number of nodes N depends on the positions).
1) Sampling: sampleMu (Latin hypercube in the scaled parameters), samplePos (rejection sampling of the chip centres,
   optionally outside or inside a band of x_c), randProfile (piece-wise constant random losses).
2) Board: buildBoard returns the nodes, C, G, Bp and the die nodes of every chip; simulate integrates the model with the
   implicit Euler method (simBoard, 4 sub-steps per sample).
3) Scaling: muHat maps μ to [−1, 1] (log for h, k_diel and s_att), chipCtx gives the descriptor of a chip (centre,
   size, type).
4) Storage: saveFam / loadFam write and read the boards of a family (npz, G as sparse triplets).
Functions:  sampleMu(n, rng, lim), samplePos(rng, band, inBand), randProfile(t, rng, holds), buildBoard(mu, pos),
            simulate(B, P, Ts), muHat(mu), chipCtx(B, c), saveFam(path, boards), loadFam(path)

V1.0: 21.11.2026, Pascal Schirmer
- Initial version
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.board.genBoard import genBoard
from src.py.board.boardLib import BOARDS
from src.py.board.simBoard import simBoard
from src.py.tools.toolData import MATERIALS

# ==============================================================================
# External
# ==============================================================================
import copy
import numpy as np
import scipy.sparse as sp

#######################################################################################################################
# Settings
#######################################################################################################################
W_MM, L_MM, MARGIN, GAP = 30.0, 20.0, 1.0, 1.0                          # board, edge margin, chip spacing (mm)
BOARD = 'IMS, Al base, 100 µm dielectric'
CHIPS = [{'name': 'T1', 'tech': 'Si IGBT', 'x': 8, 'y': 10, 'w': 6, 'l': 6, 'P': 40},
         {'name': 'D1', 'tech': 'Si diode', 'x': 17, 'y': 13, 'w': 4, 'l': 5, 'P': 15},
         {'name': 'M1', 'tech': 'SiC MOSFET', 'x': 24, 'y': 7, 'w': 4, 'l': 5, 'P': 20}]
TYPES = ['Si IGBT', 'Si diode', 'SiC MOSFET']
P_REF = 40.0                                                            # loss scale of the network input (W)
MU = {'h': (500.0, 5000.0, True), 'kDiel': (1.0, 8.0, True), 'sAtt': (0.5, 2.0, True), 'sRc': (0.7, 1.3, False)}
NAMES = list(MU.keys())
NOM = {'h': 1500.0, 'kDiel': 2.0, 'sAtt': 1.0, 'sRc': 1.0}
Z_MAX = 2.0e-3                                                          # height scale of the nodes (m)


#######################################################################################################################
# Sampling
#######################################################################################################################
def sampleMu(n, rng, lim=None):
    """n conditions (list of dicts) by Latin hypercube sampling in the scaled parameters within lim (default MU)."""
    lim = lim or MU
    u = np.column_stack([(rng.permutation(n) + rng.uniform(0, 1, n)) / n for _ in NAMES])
    out = []
    for i in range(0, n):
        m = {}
        for j, k in enumerate(NAMES):
            lo, hi, lg = lim[k][0], lim[k][1], MU[k][2]
            m[k] = float(np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo))) if lg else lo + u[i, j] * (hi - lo))
        out.append(m)
    return out


def _ok(pos, chips):
    """Chips inside the board (margin) and without overlap (spacing GAP)."""
    for (x, y), c in zip(pos, chips):
        if x - c['w'] / 2 < MARGIN or x + c['w'] / 2 > W_MM - MARGIN or y - c['l'] / 2 < MARGIN or \
                y + c['l'] / 2 > L_MM - MARGIN:
            return False
    for i in range(0, len(pos)):
        for j in range(i + 1, len(pos)):
            dx = abs(pos[i][0] - pos[j][0]) - (chips[i]['w'] + chips[j]['w']) / 2
            dy = abs(pos[i][1] - pos[j][1]) - (chips[i]['l'] + chips[j]['l']) / 2
            if dx < GAP and dy < GAP:
                return False
    return True


def samplePos(rng, band=None, inBand=None, chips=None, nTry=10000):
    """
    Chip centres (list of (x, y) in mm, integer) by rejection sampling. band = (x_lo, x_hi) (mm): with inBand = None
    no chip centre lies inside [x_lo, x_hi], with inBand = i chip i lies inside and the others outside.
    """
    chips = chips or CHIPS
    for _ in range(0, nTry):
        pos = []
        for c in chips:
            pos.append((float(rng.integers(int(np.ceil(MARGIN + c['w'] / 2)), int(np.floor(W_MM - MARGIN - c['w'] / 2))
                                           + 1)),
                        float(rng.integers(int(np.ceil(MARGIN + c['l'] / 2)), int(np.floor(L_MM - MARGIN - c['l'] / 2))
                                           + 1))))
        if band is not None:
            inside = [band[0] <= x <= band[1] for (x, _) in pos]
            want = [False] * len(chips) if inBand is None else [i == inBand for i in range(0, len(chips))]
            if inside != want:
                continue
        if _ok(pos, chips):
            return pos
    raise RuntimeError("no valid chip placement found")


def randProfile(t, rng, holds=(0.5, 4.0), chips=None):
    """Losses (Nt x nChips) (W): piece-wise constant random levels in [0, P_c] with random hold times (s)."""
    chips = chips or CHIPS
    P = np.zeros((len(t), len(chips)))
    for i, c in enumerate(chips):
        k = 0
        while k < len(t):
            n = max(int(round(rng.uniform(*holds) / (t[1] - t[0]))), 1)
            P[k:k + n, i] = c['P'] * rng.uniform(0, 1)
            k += n
    return P


#######################################################################################################################
# Board
#######################################################################################################################
def layout(mu, pos, chips=None):
    """Board layout of genBoard (layer-resolved model) for the condition mu and the chip centres pos (mm)."""
    chips = chips or CHIPS
    layers = copy.deepcopy(BOARDS[BOARD])
    for l in layers:
        if l['mat'] == 'IMS dielectric':
            l['kxy'], l['kz'] = mu['kDiel'], mu['kDiel']
        if l['mat'] == 'Al alloy (6063)':
            l['rho'] = MATERIALS[l['mat']][1] * mu['sRc']
    ch = []
    for c, (x, y) in zip(chips, pos):
        att = {'Si IGBT': 'SAC305 solder', 'Si diode': 'SAC305 solder', 'SiC MOSFET': 'Sintered Ag'}.get(c['tech'],
                                                                                                    'SAC305 solder')
        ch.append(dict(c, x=x, y=y, k_att=MATERIALS[att][0] * mu['sAtt']))
    return {'view': 'xz', 'W': W_MM, 'L': L_MM, 'layers': layers, 'h': mu['h'], 'Ta': 0.0, 'dx': 1.0,
            'yCut': L_MM / 2, 'chips': ch}


def buildBoard(mu, pos, chips=None):
    """
    Board of the condition mu with the chip centres pos: {'geo' (N x 3) (m), 'C' (N) (Ws/K), 'G' (sparse N x N) (W/K),
    'Bp' (N x nChips) (W per node per W of the chip), 'die' (list of node indices of the die cells per chip), 'mu',
    'pos', 'chips'}.
    """
    chips = chips or CHIPS
    D = genBoard(layout(mu, pos, chips))
    [C, G, Bp] = D['mat']
    n = D['nod']
    geo = np.column_stack([n['x'], n['y'], n['z'] - float(np.min(n['z']))])
    die = [np.flatnonzero(D['B3'][:, i] > 0) for i in range(0, len(chips))]
    return {'geo': geo, 'C': np.asarray(C, dtype=float), 'G': sp.csr_matrix(G), 'Bp': np.asarray(Bp, dtype=float),
            'die': die, 'mu': dict(mu), 'pos': [tuple(p) for p in pos], 'chips': [dict(c) for c in chips]}


def simulate(B, P, Ts):
    """Temperature rise (Nt x N) (K) for the losses P (Nt x nChips) (W), θ(0) = 0, implicit Euler with 4 sub-steps."""
    return simBoard(B['C'], B['G'], B['Bp'], P, Ts, 0.0, 4, steady=False)[0]


#######################################################################################################################
# Scaling
#######################################################################################################################
def muHat(mu):
    """Scaled condition (4) in [−1, 1] over the training range MU (log for h, k_diel, s_att)."""
    out = []
    for k in NAMES:
        lo, hi, lg = MU[k]
        v = (np.log(mu[k]) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (mu[k] - lo) / (hi - lo)
        out.append(2 * v - 1)
    return np.asarray(out, dtype=np.float32)


def xHat(geo):
    """Scaled node positions (N x 3): x̂ = 2x/W − 1, ŷ = 2y/L − 1, ẑ = 2z/z_max − 1."""
    s = np.array([2e3 / W_MM, 2e3 / L_MM, 2.0 / Z_MAX])
    return (np.asarray(geo, dtype=float) * s - 1.0).astype(np.float32)


def chipCtx(B, c):
    """Descriptor of chip c (7): scaled centre (x̂_c, ŷ_c), size (w/10 mm, l/10 mm) and the type (one-hot, 3)."""
    ch, (x, y) = B['chips'][c], B['pos'][c]
    typ = np.zeros(len(TYPES))
    typ[TYPES.index(ch['tech'])] = 1.0
    return np.concatenate([[2 * x / W_MM - 1, 2 * y / L_MM - 1, ch['w'] / 10.0, ch['l'] / 10.0], typ]).astype(
        np.float32)


def chipRel(B, c):
    """Position of the nodes relative to the centre of chip c (N x 2), scaled as x̂: Δx̂ = 2(x − x_c)/W."""
    (x, y) = B['pos'][c]
    g = np.asarray(B['geo'], dtype=float) * 1e3
    return np.column_stack([2 * (g[:, 0] - x) / W_MM, 2 * (g[:, 1] - y) / L_MM]).astype(np.float32)


#######################################################################################################################
# Storage
#######################################################################################################################
def saveFam(path, boards):
    """Writes the boards (incl. optional profiles 'P' and rises 'th', lists) of a family to one npz file."""
    out = {'n': len(boards)}
    for i, B in enumerate(boards):
        G = B['G'].tocoo()
        out.update({'%d_geo' % i: B['geo'], '%d_C' % i: B['C'], '%d_Gi' % i: G.row, '%d_Gj' % i: G.col,
                    '%d_Gv' % i: G.data, '%d_Bp' % i: B['Bp'], '%d_mu' % i: np.array([B['mu'][k] for k in NAMES]),
                    '%d_pos' % i: np.asarray(B['pos'], dtype=float),
                    '%d_tech' % i: np.array([c['tech'] for c in B['chips']]),
                    '%d_nP' % i: len(B.get('P', []))})
        for j, d in enumerate(B['die']):
            out['%d_die%d' % (i, j)] = d
        for j, (P, th) in enumerate(zip(B.get('P', []), B.get('th', []))):
            out['%d_P%d' % (i, j)] = P
            out['%d_th%d' % (i, j)] = th.astype(np.float32)
    np.savez_compressed(path, **out)


def loadFam(path):
    """Boards of a family file (saveFam)."""
    z = np.load(path, allow_pickle=False)
    boards = []
    for i in range(0, int(z['n'])):
        N = len(z['%d_C' % i])
        tech = [str(s) for s in z['%d_tech' % i]]
        chips = [dict(next(c for c in CHIPS if c['tech'] == t)) for t in tech]
        B = {'geo': z['%d_geo' % i], 'C': z['%d_C' % i],
             'G': sp.csr_matrix((z['%d_Gv' % i], (z['%d_Gi' % i], z['%d_Gj' % i])), shape=(N, N)),
             'Bp': z['%d_Bp' % i], 'mu': dict(zip(NAMES, [float(v) for v in z['%d_mu' % i]])),
             'pos': [tuple(p) for p in z['%d_pos' % i]], 'chips': chips,
             'die': [z['%d_die%d' % (i, j)] for j in range(0, len(tech))]}
        nP = int(z['%d_nP' % i])
        B['P'] = [z['%d_P%d' % (i, j)] for j in range(0, nP)]
        B['th'] = [z['%d_th%d' % (i, j)] for j in range(0, nP)]
        boards.append(B)
    return boards


#######################################################################################################################
# References
#######################################################################################################################
