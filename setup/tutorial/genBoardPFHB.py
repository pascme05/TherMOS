#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genBoardPFHB
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
This script generates the data sets of the half-bridge study (PINN journal paper): an IMS half-bridge module (30 x 20 mm)
with the high-side IGBT T_H and diode D_H and the low-side IGBT T_L and diode D_L (pfParam.CHIPS_HB). The losses of the
four chips follow from the operating point of the leg (current amplitude Î, power factor cos φ, output frequency f_o,
modulation index M) with the device model of src/py/tools/lossCalc.py (hbLoss): for a positive phase current T_H and
D_L conduct, for a negative one T_L and D_H, so that the four losses are physically coupled and alternate between the
high and the low side at low output frequencies (start-up, stall).

Every set is generated for three meshes (pfParam.MESH: coarse 2 mm, medium 1 mm, fine 0.5 mm) with the same seeds,
i.e. the same boards and profiles on every mesh. Files: data/board/pfhb/<mesh>/<set>.npz (pfParam.saveFam) with the
full field every DEC samples ('th', POD snapshots and field errors) and the 16 sensor nodes at every sample ('ths',
training labels and hot-spot errors; the first 4 are the hot spots, sensor version S, all 16 version L).
    train   24 boards, LHS in μ, chip centres outside the band x_c ∈ [12, 16] mm, 2 random profiles of 40 s
    val     6 boards as train, 1 random profile of 40 s (early stopping, model selection)
    phys    200 boards as train, operator only (physics loss of the network)
    t1      10 boards as train, designed profile (start-up, acceleration, nominal, regeneration, 40 s)
    t2      6 boards, extrapolated h and k_diel, nominal positions, designed profile
    t3      6 boards with one chip inside the band, designed profile
    t4      1 board, nominal positions, drive-cycle mission of 30 min from cold (field every 1 s)
Nominal layout (chips fixed at the nominal positions, only the condition μ varies; baselines of the nominal section):
    trainNom 24 boards, LHS in μ, nominal positions, 2 random profiles of 40 s
    valNom   6 boards as trainNom, 1 random profile of 40 s
    n1       10 boards as trainNom, designed profile (testing inside the training range of μ)
Usage:      python setup/tutorial/genBoardPFHB.py [mesh ...]       (default: coarse medium fine)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Path
# ==============================================================================
from os.path import dirname, abspath, isdir, join as pjoin
import sys

base = dirname(abspath(__file__))
while not isdir(pjoin(base, 'src', 'py')) and dirname(base) != base:
    base = dirname(base)
sys.path.insert(0, base)

# ==============================================================================
# Internal
# ==============================================================================
from src.py.model.PF.pfParam import CHIPS_HB, NOM, MU, sampleMu, samplePos, buildBoard, simulate, saveFam
from src.py.tools.lossCalc import cond_par
from src.py.tools.toolData import DEVICES

# ==============================================================================
# External
# ==============================================================================
import os
import time
import warnings
import numpy as np

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Settings
#######################################################################################################################
TS = 0.05                                                               # sampling time (s)
T_END = 40.0                                                            # length of the profiles (s)
BAND = (12.0, 16.0)                                                     # x_c band without training chips (mm)
DEC = 10                                                                # full field every DEC samples (0.5 s)
T4_END, T4_DEC = 1800.0, 20                                             # mission length (s), field every 1 s
NSUB = 200                                                              # sub-samples per Ts of the loss average
DEV = DEVICES['Si IGBT discrete 1200 V / 75 A (TO-247)']                # IGBT and diode of every switch
OP = {'Vdc': 400.0, 'fs': 5000.0, 'Tj': 125.0}                          # DC link (V), switching (Hz), T_j (°C)
SEG = {'I': (0.0, 30.0), 'fo': (0.5, 50.0), 'hold': (0.5, 8.0), 'ramp': (0.0, 1.0)}   # random operating points
SEEDS = {'train': 1, 'val': 2, 'phys': 3, 't1': 4, 't2': 5, 't3': 6, 't4': 7, 'trainNom': 11, 'valNom': 12, 'n1': 13}
SIZES = {'train': 24, 'val': 6, 'phys': 200, 't1': 10, 'nProf': 2}      # boards per set, profiles per training board
SETS = list(SEEDS.keys())


#######################################################################################################################
# Losses of the half-bridge leg
#######################################################################################################################
def hbLoss(t, I, cosphi, fo, M=None, d=None, op=None):
    """
    Losses (Nt x 4) (W) of T_H, D_H, T_L, D_L at the samples t (s), each the mean over [t_n, t_n + Ts) (zero-order
    hold of the thermal model). I (A), cosphi (-), fo (Hz) and M (-) are functions of time (callables of the fine time
    grid) or constants; M defaults to the V/f characteristic M = 0.1 + 0.85 f_o/50 Hz. Local average over one switching
    period (duty d = (1 + M sin θ)/2 of the high side, phase current i = Î sin(θ − φ), θ = ∫ 2π f_o dt):
        i > 0:  P_TH = d (V_0 i + r i²) + f_s (E_on + E_off)(V_dc/V_ref)^K_v (i/I_ref)^K_i
                P_DL = (1 − d)(V_F0 i + r_F i²) + f_s E_rec (V_dc/V_ref)^K_v,rec (i/I_ref)^K_i,rec
        i < 0:  the same for T_L (1 − d) and D_H (d) with |i|.
    Over one fundamental period the mean of P_TH equals lossCalc.inverter (T cond + T on + T off).
    """
    d, op = d or DEV, op or OP
    t = np.asarray(t, dtype=float)
    Ts = t[1] - t[0]
    tf = (np.arange(len(t) * NSUB) + 0.5) * Ts / NSUB + t[0]
    val = lambda f: np.broadcast_to(np.asarray(f(tf) if callable(f) else f, dtype=float), tf.shape)
    Ii, cp, ff = val(I), np.clip(val(cosphi), -1, 1), val(fo)
    Mm = val(M) if M is not None else np.clip(0.1 + 0.85 * ff / 50.0, 0.1, 0.95)
    th = np.cumsum(2 * np.pi * ff) * Ts / NSUB
    i = Ii * np.sin(th - np.arccos(cp))
    duty = 0.5 * (1 + Mm * np.sin(th))
    c = cond_par(d, op['Tj'])
    tfE = max(1 + d['TcE'] * (op['Tj'] - d['T_ref']), 0.0)
    a = np.abs(i)
    sw = op['fs'] * (d['Eon'] + d['Eoff']) * (op['Vdc'] / d['V_ref']) ** d['Kv'] * (a / d['I_ref']) ** d['Ki'] * tfE
    rec = op['fs'] * d['Erec'] * (op['Vdc'] / d['V_ref']) ** d['Kv_rec'] * (a / d['I_ref']) ** d['Ki_rec'] * tfE
    pT = lambda dt: dt * (c['V0'] * a + c['r'] * a ** 2) + sw
    pD = lambda dt: dt * (c['VF0'] * a + c['rF'] * a ** 2) + rec
    pos, neg = i > 0, i < 0
    P = np.column_stack([pos * pT(duty), neg * pD(duty), neg * pT(1 - duty), pos * pD(1 - duty)])
    return P.reshape(len(t), NSUB, 4).mean(axis=1)


def _segments(seq, t):
    """Piece-wise linear functions of time (callables) of the keys of seq (list of dicts with 'dur', 'ramp' and values)."""
    tk, keys = [0.0], [k for k in seq[0] if k not in ('dur', 'ramp')]
    vk = {k: [seq[0][k]] for k in keys}
    for j, s in enumerate(seq):
        if j > 0:
            tk.append(tk[-1] + s['ramp'])
            for k in keys:
                vk[k].append(s[k])
        tk.append(tk[-1] + s['dur'])
        for k in keys:
            vk[k].append(s[k])
    tk = np.asarray(tk)
    return {k: (lambda x, v=np.asarray(vk[k]): np.interp(x, tk, v)) for k in keys}


def randOp(t, rng):
    """Random operating points: hold 0.5–8 s, ramps 0–1 s, Î 0–30 A, cos φ −1…1, f_o 0.5–50 Hz (log)."""
    seq, tt = [], 0.0
    while tt <= t[-1] + 10.0:
        s = {'dur': rng.uniform(*SEG['hold']), 'ramp': rng.uniform(*SEG['ramp']), 'I': rng.uniform(*SEG['I']),
             'cp': rng.uniform(-1.0, 1.0), 'fo': float(np.exp(rng.uniform(*np.log(SEG['fo']))))}
        seq.append(s)
        tt += s['dur'] + s['ramp']
    f = _segments(seq, t)
    return hbLoss(t, f['I'], f['cp'], f['fo'])


def designed(t):
    """
    Designed test profile (40 s): start-up at 0.5 Hz with 30 A (0–8 s), acceleration to 50 Hz (8–14 s), nominal operation
    at 50 Hz with 25 A, cos φ = 0.9 (14–26 s), regeneration with cos φ = −0.9 (26–34 s), off (34–40 s).
    """
    seq = [{'dur': 8.0, 'ramp': 0.0, 'I': 30.0, 'cp': 0.9, 'fo': 0.5},
           {'dur': 0.0, 'ramp': 6.0, 'I': 25.0, 'cp': 0.9, 'fo': 50.0},
           {'dur': 12.0, 'ramp': 0.0, 'I': 25.0, 'cp': 0.9, 'fo': 50.0},
           {'dur': 8.0, 'ramp': 0.0, 'I': 25.0, 'cp': -0.9, 'fo': 50.0},
           {'dur': 6.0, 'ramp': 0.0, 'I': 0.0, 'cp': -0.9, 'fo': 50.0}]
    f = _segments(seq, t)
    return hbLoss(t, f['I'], f['cp'], f['fo'])


def mission(t, rng):
    """
    Drive-cycle mission: segments of 5–60 s at a random speed (f_o 0.5–50 Hz) with ramps of 1–5 s; motoring with
    Î 5–30 A and cos φ = 0.85 when accelerating or cruising, regeneration with cos φ = −0.85 when decelerating.
    """
    seq, tt, fo = [], 0.0, 0.5
    while tt <= t[-1] + 60.0:
        fn = float(np.exp(rng.uniform(*np.log(SEG['fo']))))
        cp = 0.85 if fn >= fo or rng.uniform() < 0.3 else -0.85
        s = {'dur': rng.uniform(5.0, 60.0), 'ramp': rng.uniform(1.0, 5.0), 'I': rng.uniform(5.0, 30.0), 'cp': cp,
             'fo': fn}
        seq.append(s)
        tt, fo = tt + s['dur'] + s['ramp'], fn
    f = _segments(seq, t)
    return hbLoss(t, f['I'], f['cp'], f['fo'])


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def nominal():
    return [(float(c['x']), float(c['y'])) for c in CHIPS_HB]


def labelled(boards, profs, dec=DEC):
    """Adds the profiles, the field every dec samples ('th') and the sensor rises at every sample ('ths')."""
    for B, P in zip(boards, profs):
        th = [simulate(B, p, TS) for p in P]
        B['P'], B['th'], B['ths'], B['dec'] = P, [x[::dec] for x in th], [x[:, B['sens']] for x in th], dec
    return boards


def genSet(s, mesh, t):
    """Boards of the set s on the mesh (same boards and profiles on every mesh, seed SEEDS[s])."""
    rng = np.random.default_rng(SEEDS[s])
    bb = lambda m, p: buildBoard(m, p, CHIPS_HB, mesh)
    if s in ('train', 'val', 'phys', 't1'):
        n = SIZES[s]
        B = [bb(m, samplePos(rng, BAND, chips=CHIPS_HB)) for m in sampleMu(n, rng)]
        if s == 'phys':
            return B
        if s == 't1':
            return labelled(B, [[designed(t)] for _ in B])
        return labelled(B, [[randOp(t, rng) for _ in range(0, SIZES['nProf'] if s == 'train' else 1)] for _ in B])
    if s in ('trainNom', 'valNom', 'n1'):
        n = SIZES[{'trainNom': 'train', 'valNom': 'val', 'n1': 't1'}[s]]
        B = [bb(m, nominal()) for m in sampleMu(n, rng)]
        if s == 'n1':
            return labelled(B, [[designed(t)] for _ in B])
        return labelled(B, [[randOp(t, rng) for _ in range(0, SIZES['nProf'] if s == 'trainNom' else 1)] for _ in B])
    if s == 't2':
        mus = [dict(NOM, h=250.0), dict(NOM, h=8000.0), dict(NOM, kDiel=0.5), dict(NOM, kDiel=12.0),
               dict(NOM, h=250.0, kDiel=0.5), dict(NOM, h=8000.0, kDiel=12.0)]
        return labelled([bb(m, nominal()) for m in mus], [[designed(t)] for _ in mus])
    if s == 't3':
        mus = sampleMu(6, rng)
        B = [bb(mus[i], samplePos(rng, BAND, i % 4, CHIPS_HB)) for i in range(0, 6)]
        return labelled(B, [[designed(t)] for _ in B])
    if s == 't4':
        tt = np.arange(0.0, T4_END + TS * 1e-9, TS)
        return labelled([bb(sampleMu(1, rng)[0], nominal())], [[mission(tt, rng)]], T4_DEC)
    raise ValueError(s)


#######################################################################################################################
# Main Function
#######################################################################################################################
def main(meshes=None, sets=None, skip=True):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Generating the data sets of the half-bridge study")
    print("------------------------------------------")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    t = np.arange(0.0, T_END + TS * 1e-9, TS)
    for mesh in meshes or ['coarse', 'medium', 'fine']:
        folder = pjoin(base, 'data', 'board', 'pfhb', mesh)
        os.makedirs(folder, exist_ok=True)
        for s in sets or SETS:
            f = pjoin(folder, s + '.npz')
            if skip and os.path.isfile(f):
                print("INFO: %s/%s exists, skipped" % (mesh, s))
                continue
            t0 = time.time()
            B = genSet(s, mesh, t)
            saveFam(f, B)
            thMax = max([float(np.max(x)) for Bi in B for x in Bi.get('ths', [])] or [0.0])
            print("INFO: %s/%s: %d boards, N = %d nodes, hot-spot rise max %.1f K, %.1f s" %
                  (mesh, s, len(B), len(B[0]['C']), thMax, time.time() - t0))

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Data sets written to data/board/pfhb/ (training range of μ: %s)" %
          ', '.join('%s %g–%g' % (k, v[0], v[1]) for k, v in MU.items()))


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:] or None)


#######################################################################################################################
# References
#######################################################################################################################
# [1]: D. Graovac and M. Pürschel, "IGBT power losses calculation using the data-sheet parameters," Infineon AN, 2009.
