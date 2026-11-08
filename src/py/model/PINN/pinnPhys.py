#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnPhys
# Date:         08.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This module defines the physics of the PINN (pinnFit/pinnSol). Every physics model is written as a linear thermal
network in nodal form with the temperature rises x (K) of its n states over the reference temperature T_ref:

    C·dx/dt = E·P(t) - G·x,        T̂_i = T_ref,i + (O·x)_i

with the diagonal capacitance matrix C (Ws/K), the conductance matrix G (W/K), the input matrix E (states x loss
sources) and the output matrix O (outputs x states). The network of the PINN predicts the states x; the measured
outputs enter the data loss through O, the physics residual is evaluated for every state:

    r = (C·dx/dt + G·x - E·P) / (diag(G)·ΔT)          (dimensionless, ΔT: temperature scale of the training data)

Physics models (para['Mdl']['pinn']['phys']):
    0) none:    data only (every output is a free state, no residual)
    1) Foster:  chains of RC terms between a loss source j and an output i (Z_ij(t) = Σ_k R_k·(1 - exp(-t/τ_k))),
                one state per term (τ_k·dθ_k/dt = R_k·P_j - θ_k: G = I, C = τ, E = R); single output: one chain,
                several outputs: one chain per output (self-heating) and optionally chains for the coupling i ≠ j
    2) Cauer:   nodal network (G, C, E) with the output nodes, e.g. the coupled Cauer ladders of the SF mode or the
                network of the board designer; states that are not measured are hidden (no data loss)
Outputs without physics get a free state (data loss only). The loss sources P_j are equations of the input features
x1, x2, ... (raw units), the predicted outputs T1, T2, ... (T = output 'node') and constants (calcEq), e.g. 'x4' or
'3*Rs*x4**2'. The reference temperature is a constant (TrefSrc = 0), an input feature (TrefSrc = k > 0) or the
reference channel of every output (TrefSrc = -1, setup['ref']); it is assumed to vary slowly (dT_ref/dt is neglected).
The structure is given by setup['physPI'] (GUI view 'Physics'); without it a single Foster term (Rth, Cth) of output
'node' with the loss PvEq / feature PvSrc is used (as in V1.x).
setup['physPI'] = {'type': 'foster' | 'cauer', 'src': ['x4', ...],
                   'foster': [{'out': i, 'src': j, 'R': [...], 'tau': [...]}, ...],
                   'cauer': {'G': n x n, 'C': n, 'E': n x S, 'out': [node of output 1, ...] (1-based, 0 = free),
                             'labels': [...]}, 'origin': text}
Functions:  physPara(para), physDef(para, setup, M, F), sources(X, T, ph), simPhys(ph, P, Ts, x0), tauOf(ph),
            zth(ph, t), cauer2foster(ph), foster2cauer(R, tau), ladder(...), fromRC(mdl), fromSF(mdl)

V1.0: 08.11.2026, Pascal Schirmer
- Nodal physics for Foster (single and multiple outputs) and Cauer networks, dimensionless residual
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.calcEq import calcEq

# ==============================================================================
# External
# ==============================================================================
import copy
import re
import numpy as np

#######################################################################################################################
# Settings
#######################################################################################################################
PHYS = {0: 'none (data only)', 1: 'Foster', 2: 'Cauer / nodal network'}


#######################################################################################################################
# Parameters
#######################################################################################################################
_CONST = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"     # name = value
                    r"\s*(?:\[([^\]]*)\])?\s*(.*)$")                                       # [unit] meaning


def parseConstRows(txt):
    """
    Constants of the loss equations as rows {'name', 'value', 'unit', 'meaning'} from the text of Mdl › pinn › const:
    items 'name = value [unit] meaning' separated by semicolons or new lines, e.g. 'Rs = 0.0141 [Ohm] stator
    resistance; k = 0.00393 [1/K] temperature coefficient of copper'. Unit and meaning are optional ('k=0.01; p0=50');
    items separated by commas are accepted when they hold no meaning ('k=0.01, p0=50').
    """
    rows = []
    if not isinstance(txt, str) or txt.strip() in ('', '0', '-'):
        return rows
    for item in re.split(r"[;\n]", txt):
        parts = item.split(',') if item.count('=') > 1 else [item]
        for it in parts:
            if not it.strip():
                continue
            m = _CONST.match(it)
            if m is None:
                raise ValueError(f"constant '{it.strip()}': 'name = value [unit] meaning' expected")
            rows.append({'name': m.group(1), 'value': float(m.group(2)), 'unit': (m.group(3) or '').strip(),
                         'meaning': m.group(4).strip()})
    return rows


def parseConst(txt):
    """Constants 'k=0.01; p0=50' (see parseConstRows) as a dictionary {name: value}."""
    return {r['name']: r['value'] for r in parseConstRows(txt)}


def constText(rows):
    """Text of Mdl › pinn › const for the rows {'name', 'value', 'unit', 'meaning'} (inverse of parseConstRows)."""
    out = []
    for r in rows:
        txt = "%s = %.10g" % (str(r['name']).strip(), float(r['value']))
        if str(r.get('unit') or '').strip():
            txt += " [%s]" % str(r['unit']).strip().replace(']', ')').replace('[', '(')
        if str(r.get('meaning') or '').strip():
            txt += " " + str(r['meaning']).strip().replace(';', ',').replace('\n', ' ')
        out.append(txt)
    return '; '.join(out)


def physPara(para):
    """Scalar physics settings of the PINN (Mdl › pinn) with defaults (also used by the GUI)."""
    phy = para['Mdl'].get('pinn', {}) or {}

    def get(k, d, f=float):
        v = phy.get(k)
        return d if v is None or (isinstance(v, str) and not v.strip()) else f(v)
    const = {k: get(k, 0.0) for k in ('Rth', 'Cth', 'Rs') if phy.get(k) is not None}    # V1.x parameters
    const.update(parseConst(phy.get('const', 0)))                       # the list of constants wins
    Rth, Cth = const.get('Rth', get('Rth', 0.0265)), const.get('Cth', get('Cth', 2.08e4))
    return {'Rth': Rth, 'Cth': Cth, 'Rs': const.get('Rs', get('Rs', 14.1e-3)), 'alpha': get('alpha', 1.0),
            'beta': get('beta', 0.1), 'gamma': get('gamma', 0.0), 'warm': get('warm', 0, int),
            'phys': get('phys', 1, int), 'pred': get('pred', 1, int), 'x0': get('x0', 1, int),
            'col': get('col', 0, int), 'PvSrc': get('PvSrc', 0, int),
            'PvEq': str(phy.get('PvEq') or '3*Rs*x4**2'), 'TrefSrc': get('TrefSrc', 2, int),
            'Tref': get('Tref', 25.0), 'node': max(get('node', 1, int), 1), 'const': const}


def _srcLegacy(sp):
    return 'x%d' % sp['PvSrc'] if sp['PvSrc'] > 0 else sp['PvEq']


def structure(para, setup, M):
    """Physics structure (setup['physPI'] or the single Foster term of the scalar parameters)."""
    sp = physPara(para)
    st = copy.deepcopy(setup.get('physPI') or None)
    if sp['phys'] == 0:
        return {'type': 'none', 'src': [], 'origin': 'data only'}
    if st and st.get('type') in ('foster', 'cauer') and st.get('src'):
        if sp['phys'] == 1 and st['type'] == 'foster' and st.get('foster'):
            return st
        if sp['phys'] == 2 and st['type'] == 'cauer' and st.get('cauer'):
            return st
    j = min(sp['node'], M)
    if sp['phys'] == 2:
        g, c = 1 / sp['Rth'], sp['Cth']
        return {'type': 'cauer', 'src': [_srcLegacy(sp)],
                'cauer': {'G': [[g]], 'C': [c], 'E': [[1.0]], 'out': [1 if i == j - 1 else 0 for i in range(M)],
                          'labels': ['node 1']},
                'origin': 'default: one RC node'}
    return {'type': 'foster', 'src': [_srcLegacy(sp)],
            'foster': [{'out': j, 'src': 1, 'R': [sp['Rth']], 'tau': [sp['Rth'] * sp['Cth']]}],
            'origin': 'default: one Foster term'}


#######################################################################################################################
# Nodal form
#######################################################################################################################
def fosterNodal(chains, M, S):
    """
    G, C, E, O of Foster chains [{'out', 'src', 'R', 'tau'}]: one state θ_k (K) per term, τ_k·dθ_k/dt = R_k·P_j - θ_k,
    i.e. G = I, C = τ, E = R (the coupling terms of a Cauer network converted to Foster may have R < 0).
    """
    R = np.concatenate([np.asarray(c['R'], dtype=float).ravel() for c in chains]) if chains else np.zeros(0)
    tau = np.concatenate([np.asarray(c['tau'], dtype=float).ravel() for c in chains]) if chains else np.zeros(0)
    if np.any(tau <= 0) or np.any(R == 0) or not np.all(np.isfinite(R)):
        raise ValueError("Foster: τ must be positive and R finite and non-zero")
    n = len(R)
    E, O, lab, k = np.zeros((n, S)), np.zeros((M, n)), [], 0
    for c in chains:
        i, j, N = int(c['out']), int(c['src']), len(np.ravel(c['R']))
        if len(np.ravel(c['tau'])) != N:
            raise ValueError(f"Foster: chain T{i} ← P{j} has {N} resistances but {len(np.ravel(c['tau']))} τ")
        if not 1 <= i <= M:
            raise ValueError(f"Foster: output {i} does not exist ({M} outputs)")
        if not 1 <= j <= S:
            raise ValueError(f"Foster: source {j} does not exist ({S} sources)")
        E[k:k + N, j - 1] = R[k:k + N]
        O[i - 1, k:k + N] = 1.0
        lab += [f"θ{i}{j}·{m + 1}" for m in range(N)]
        k = k + N
    return np.eye(n), tau.copy(), E, O, lab


def cauerNodal(net, M, S):
    """G, C, E, O of a nodal network {'G', 'C', 'E', 'out'}."""
    G = np.atleast_2d(np.asarray(net['G'], dtype=float))
    C = np.asarray(net['C'], dtype=float).ravel()
    E = np.asarray(net['E'], dtype=float).reshape(len(C), -1)
    n = len(C)
    if G.shape != (n, n):
        raise ValueError(f"Cauer: G must be {n} x {n}")
    if E.shape[1] != S:
        raise ValueError(f"Cauer: E has {E.shape[1]} columns but there are {S} loss sources")
    if np.any(C <= 0):
        raise ValueError("Cauer: the capacitances must be positive")
    if np.min(np.linalg.eigvalsh((G + G.T) / 2)) <= 0:
        raise ValueError("Cauer: G must be symmetric positive definite (every node connected to the reference)")
    out = list(net.get('out') or [])
    O = np.zeros((M, n))
    for i in range(0, M):
        o = int(out[i]) if i < len(out) and out[i] is not None else 0
        if o > n:
            raise ValueError(f"Cauer: output {i + 1} is mapped to node {o}, the network has {n} nodes")
        if o > 0:
            O[i, o - 1] = 1.0
    lab = list(net.get('labels') or [f"node {k + 1}" for k in range(n)])
    return (G + G.T) / 2, C, E, O, lab


def physDef(para, setup, M, F=None):
    """
    Complete physics definition for M outputs (F input features, optional, for checks): nodal matrices of the
    physics states, output matrix incl. the free states, sources, reference and the loss weights.
    """
    sp = physPara(para)
    st = structure(para, setup, M)
    src = [str(s) for s in st.get('src') or []]
    S = len(src)
    if st['type'] == 'foster':
        G, C, E, O, lab = fosterNodal(st['foster'], M, S)
    elif st['type'] == 'cauer':
        G, C, E, O, lab = cauerNodal(st['cauer'], M, S)
    else:
        G, C, E, O, lab = np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0)), np.zeros((M, 0)), []
    nP = len(C)
    free = [i for i in range(0, M) if not np.any(O[i])]
    Of = np.zeros((M, len(free)))
    for k, i in enumerate(free):
        Of[i, k] = 1.0
    if F is not None and sp['TrefSrc'] > F:
        raise ValueError(f"the reference temperature is feature {sp['TrefSrc']}, there are only {F} features")
    ph = dict(sp)
    ph.update({'type': st['type'], 'src': src, 'S': S, 'G': G, 'C': C, 'E': E, 'O': np.hstack([O, Of]),
               'nP': nP, 'n': nP + len(free), 'free': free, 'labels': lab + [f"free T{i + 1}" for i in free],
               'origin': st.get('origin', ''), 'struct': st, 'M': M})
    ph['tau'] = tauOf(ph)
    return ph


def tauOf(ph):
    """Time constants (sec) of the physics network (eigenvalues of C⁻¹·G)."""
    if ph['nP'] == 0:
        return np.zeros(0)
    S = 1 / np.sqrt(ph['C'])
    lam = np.linalg.eigvalsh(S[:, None] * ph['G'] * S[None, :])
    return np.sort(1 / np.maximum(lam, 1e-300))


#######################################################################################################################
# Sources and reference
#######################################################################################################################
def sources(X, T, ph):
    """Loss sources P (... x S) for the inputs X (... x F, raw units) and the predicted outputs T (... x M)."""
    if ph['S'] == 0:
        return X[..., 0:0]
    var = {f"x{i + 1}": X[..., i] for i in range(0, X.shape[-1])}
    var.update({f"T{i + 1}": T[..., i] for i in range(0, T.shape[-1])})
    var['T'] = T[..., min(ph['node'], T.shape[-1]) - 1]
    out = [calcEq(e, var, ph['const']) + 0 * X[..., 0] for e in ph['src']]
    try:
        import torch
        if isinstance(X, torch.Tensor):
            return torch.stack(out, dim=-1)
    except ImportError:
        pass
    return np.stack(out, axis=-1)


def reference(X, R, ph, init=1):
    """Reference temperature of every output (... x M): constant, feature or reference channel (R)."""
    shp = tuple(X.shape[:-1]) + (ph['M'],)
    if ph['TrefSrc'] > 0:
        ref = X[..., ph['TrefSrc'] - 1:ph['TrefSrc']]
    elif ph['TrefSrc'] < 0 and R is not None and init == 1:
        ref = R[..., 0:ph['M']] if R.shape[-1] >= ph['M'] else R[..., 0:1]
    elif ph['TrefSrc'] < 0:
        ref = 0 * X[..., 0:1]
    else:
        ref = 0 * X[..., 0:1] + ph['Tref']
    try:
        import torch
        if isinstance(X, torch.Tensor):
            return ref.expand(*shp)
    except ImportError:
        pass
    return np.broadcast_to(ref, shp)


#######################################################################################################################
# Simulation of the physics alone
#######################################################################################################################
def simPhys(ph, P, Ts, x0=None):
    """
    Exact simulation (zero-order hold) of the physics network for the losses P (Nt x S): states x (Nt x nP) (K)
    and the output rises O·x (Nt x M) of the physics states (free outputs are zero).
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    nP = ph['nP']
    x = np.zeros((P.shape[0], nP))
    if nP == 0:
        return x, np.zeros((P.shape[0], ph['M']))
    S = np.sqrt(ph['C'])
    lam, V = np.linalg.eigh(ph['G'] / S[:, None] / S[None, :])
    lam = np.maximum(lam, 1e-300)
    a = np.exp(-lam * Ts)
    g = -np.expm1(-lam * Ts) / lam
    Bz = V.T @ (ph['E'] / S[:, None])
    z = np.zeros(nP) if x0 is None else V.T @ (S * np.asarray(x0, dtype=float))
    b = P @ Bz.T
    from scipy.signal import lfilter
    Z = np.zeros_like(b)
    for k in range(0, nP):
        Z[:, k] = lfilter([0.0, g[k]], [1.0, -a[k]], b[:, k], zi=[z[k]])[0] if len(b) else Z[:, k]
    x = (Z @ V.T) / S[None, :]
    return x, x @ ph['O'][:, 0:nP].T


def split(d):
    """
    Profiles X2, y2, r2 of a data split, cut at every change of the session (d['sid'] of loadData, e.g. the motor
    sessions of an ID-based split that the loader concatenates into one profile; else the profile number d['id']).
    Returns the lists and the index slices of every part in the concatenated data.
    """
    X2, y2 = list(d['X2']), list(d['y2'])
    r2 = list(d.get('r2') or [None] * len(X2))
    n = [len(x) for x in X2]
    ids = d.get('sid')                                                  # session of every sample (loadData)
    if ids is None or len(np.ravel(ids)) != sum(n):
        ids = d.get('id')
    if ids is None or len(np.ravel(ids)) != sum(n):
        o = np.cumsum([0] + n)
        return X2, y2, r2, [slice(o[k], o[k + 1]) for k in range(len(n))]
    ids = np.ravel(ids)
    Xs, ys, rs, sl, off = [], [], [], [], 0
    for X, y, r in zip(X2, y2, r2):
        iv = ids[off:off + len(X)]
        cut = np.concatenate([[0], np.flatnonzero(iv[1:] != iv[:-1]) + 1, [len(X)]])
        for a, b in zip(cut[:-1], cut[1:]):
            Xs.append(np.asarray(X)[a:b])
            ys.append(np.asarray(y)[a:b])
            rs.append(None if r is None else np.asarray(r)[a:b])
            sl.append(slice(off + a, off + b))
        off = off + len(X)
    return Xs, ys, rs, sl


def jsonable(ph):
    """Physics definition without arrays (for saving / display)."""
    out = {}
    for k, v in ph.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


#######################################################################################################################
# Step responses and conversions Foster <-> Cauer
#######################################################################################################################
def _modes(ph):
    S = 1 / np.sqrt(ph['C'])
    lam, V = np.linalg.eigh(S[:, None] * ph['G'] * S[None, :])
    lam = np.maximum(lam, 1e-300)
    Op = ph['O'][:, 0:ph['nP']] @ (S[:, None] * V)                       # outputs x modes
    Bm = V.T @ (S[:, None] * ph['E'])                                    # modes x sources
    return lam, Op, Bm


def zth(ph, t):
    """Thermal impedances Z_ij(t) (nt x M x S) (K/W) of the physics: response of output i to a 1 W step of source j."""
    t = np.asarray(t, dtype=float)
    if ph['nP'] == 0:
        return np.zeros((len(t), ph['M'], ph['S']))
    lam, Op, Bm = _modes(ph)
    F = -np.expm1(-np.outer(t, lam)) / lam[None, :]
    return np.einsum('im,tm,mj->tij', Op, F, Bm)


def cauer2foster(ph, coupled=True, tol=1e-6):
    """
    Exact Foster form of a network (modal decomposition): Z_ij(t) = Σ_k r_ijk·(1 - exp(-t/τ_k)), τ_k = 1/λ_k,
    r_ijk = (O·S⁻¹·v_k)_i·(v_kᵀ·S⁻¹·E)_j/λ_k. coupled = False keeps the self-heating chains i = j only. Terms below
    tol·max|r| are dropped. Returns the Foster chains.
    """
    lam, Op, Bm = _modes(ph)
    r = Op[:, :, None] * Bm[None, :, :] / lam[None, :, None]            # M x modes x S
    out = []
    for i in range(0, ph['M']):
        for j in range(0, ph['S']):
            if not coupled and i != j:
                continue
            keep = np.abs(r[i, :, j]) > tol * max(np.abs(r).max(), 1e-300)
            if keep.any():
                k = np.argsort(1 / lam[keep])
                out.append({'out': i + 1, 'src': j + 1, 'R': r[i, keep, j][k].tolist(),
                            'tau': (1 / lam[keep])[k].tolist()})
    return out


def foster2cauer(R, tau):
    """
    Cauer ladder (R_k, C_k), k = 1 … n, with the same input impedance as the Foster chain Σ R_k/(1 + s·τ_k)
    (continued-fraction expansion; node 1 = junction, R_n to the reference). Returns R, C (arrays).
    """
    R, tau = np.asarray(R, dtype=float).ravel(), np.asarray(tau, dtype=float).ravel()
    if np.any(R <= 0) or np.any(tau <= 0):
        raise ValueError("Foster → Cauer needs positive R and τ")
    t0 = float(np.exp(np.mean(np.log(tau))))                            # time scaling s' = s·t0
    ts = tau / t0
    D = np.array([1.0])
    for x in ts:
        D = np.polymul(D, [x, 1.0])
    N = np.zeros(1)
    for k in range(0, len(R)):
        Nk = np.array([R[k]])
        for m, x in enumerate(ts):
            if m != k:
                Nk = np.polymul(Nk, [x, 1.0])
        N = np.polyadd(N, Nk)
    Rc, Cc = [], []
    num, den = D, N                                                       # admittance Y = num/den
    for _ in range(0, len(R)):
        c = num[0] / den[0]                                               # Y = s·C_k + num'/den
        num = np.polysub(num, np.polymul([c, 0.0], den))[1:]
        Cc.append(c * t0)
        r = den[0] / num[0]                                               # den/num' = R_k + den'/num'
        den = np.polysub(den, r * num)[1:]
        Rc.append(r)
        if len(den) == 0 or np.max(np.abs(den)) < 1e-12 * np.max(np.abs(num)):
            break
    return np.array(Rc), np.array(Cc)


def ladder(Rl, Cl, g=None):
    """Nodal matrices G, C of M coupled Cauer ladders (Rl, Cl: M x Nr; g: Nr x M(M-1)/2 coupling conductances)."""
    Rl, Cl = np.atleast_2d(np.asarray(Rl, dtype=float)), np.atleast_2d(np.asarray(Cl, dtype=float))
    M, Nr = Rl.shape
    from src.py.model.SF.sfNet import sfNet
    nG = Nr * M * (M - 1) // 2
    gg = np.full(nG, 1e-300) if g is None else np.maximum(np.asarray(g, dtype=float).ravel(), 1e-300)
    [G, C, E] = sfNet(np.concatenate([np.log(Rl).ravel(), np.log(Cl).ravel(), np.log(gg)]), M, Nr)
    return np.asarray(G), np.asarray(C).ravel(), np.asarray(E)


#######################################################################################################################
# Conversions from other models
#######################################################################################################################
def fromRC(mdl, out=1, src=1):
    """Foster chain from a fitted RC model (rcFit: Rth, tau)."""
    return {'out': out, 'src': src, 'R': [float(v) for v in np.ravel(mdl['Rth'])],
            'tau': [float(v) for v in np.ravel(mdl['tau'])]}


def fromSF(mdl):
    """Physics structure from a fitted SF model: Foster (self-heating chains) or the coupled Cauer network."""
    if mdl.get('type') == 'foster':
        R, tau = np.asarray(mdl['Rth_rc']), np.asarray(mdl['tau_rc'])
        M = R.shape[0]
        return {'type': 'foster', 'src': [f"x{i + 1}" for i in range(M)],
                'foster': [{'out': i + 1, 'src': i + 1, 'R': R[i].tolist(), 'tau': tau[i].tolist()}
                           for i in range(M)], 'origin': 'SF mode, nodal Foster model (self-heating chains)'}
    from src.py.model.SF.sfNet import sfNet
    M, Nr = int(mdl['M']), int(mdl['Nr'])
    [G, C, E] = sfNet(mdl['p'], M, Nr)
    return {'type': 'cauer', 'src': [f"x{i + 1}" for i in range(M)],
            'cauer': {'G': np.asarray(G).tolist(), 'C': np.asarray(C).ravel().tolist(), 'E': np.asarray(E).tolist(),
                      'out': [i * Nr + 1 for i in range(M)],
                      'labels': [f"chip {i + 1}·{k + 1}" for i in range(M) for k in range(Nr)]},
            'origin': f'SF mode, coupled Cauer network ({Nr} stage(s) per source)'}


#######################################################################################################################
# References
#######################################################################################################################
