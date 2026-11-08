#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         thermStack.py
# Date:         05.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Thermal stack designer (Theory & Tools › Thermal): one-dimensional heat conduction in z through the layers of a
power semiconductor package, board and cooler, with heat spreading, two parallel heat paths (bottom and top side)
and convection to the coolant / ambient.

Model (per layer i with thickness t_i, conductivity k_i, density rho_i and specific heat cp_i):
    footprint      A(z) = w(z) l(z),  w(z) = min(w_in + 2 z tan(theta_i), w_max,i)   (spreading angle theta_i)
    resistance     R_i = int_0^t_i dz / (k_i A(z))            capacity  C_i = rho_i cp_i int_0^t_i A(z) dz
    convection     R_conv = 1 / (h A_out)                     (outer surface of the last layer of a path)
Every layer is split into n_sub cells (R/2 – C – R/2), which gives a Cauer ladder per path. The junction node joins
both ladders. The exact Foster network follows from the eigenvalues of the network:
    C dT/dt = -G T + b P,  Z_th(t) = sum_i R_i (1 - exp(-t / tau_i)),  tau_i = 1 / lambda_i
with R_i from the modal residues (sum_i R_i = R_th,ja). A reduced Foster network with N terms is fitted to Z_th(t).

Inputs:     layer tables of both paths, die size, h per side
Outputs:    R_th, C_th per layer, heat split, steady temperature profile, Cauer and Foster networks, Z_th(t), T_j(t)

Copied from OPTELIA (RAISE, src/tools) for the Theory & Tools pages of the TherMOS GUI.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
from scipy.optimize import least_squares, nnls

#######################################################################################################################
# Parameter
#######################################################################################################################
N_Z = 60                                                                  # integration points per layer


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def lee(A_s, A_p, t, k, h):
    """
    Spreading resistance of a plate (thickness t, conductivity k, area A_p) heated by a source of area A_s and cooled
    by h on the opposite face (Lee et al. 1995, equivalent circular geometry, mean source temperature):
        eps = a/b, tau = t/b, Bi = h b/k, lambda_c = pi + 1/(sqrt(pi) eps)
        Phi_c = (tanh(lambda_c tau) + lambda_c/Bi) / (1 + lambda_c/Bi tanh(lambda_c tau))
        psi = eps tau/sqrt(pi) + 1/2 (1 - eps)^(3/2) Phi_c,  R_cond = psi / (sqrt(pi) k a)
        R = R_cond + 1/(h A_p)
    The first term of psi is the 1D conduction t/(k A_p), so R_cond contains it (R_cond = t/(k A_p) for eps = 1).
    Returns (R_conduction incl. spreading, R_convection).
    """
    a, b = np.sqrt(A_s / np.pi), np.sqrt(A_p / np.pi)
    eps = min(a / b, 1.0)
    R_conv = 1 / (h * A_p)
    if eps >= 1.0:
        return t / (k * A_p), R_conv
    tau = t / b
    Bi = h * b / k
    lc = np.pi + 1 / (np.sqrt(np.pi) * eps)
    th = np.tanh(lc * tau)
    Phi = (th + lc / Bi) / (1 + lc / Bi * th)
    psi = eps * tau / np.sqrt(np.pi) + 0.5 * (1 - eps) ** 1.5 * Phi
    return float(psi / (np.sqrt(np.pi) * k * a)), float(R_conv)


def _layer(lay, w_in, l_in, rho, cp, angle=None):
    """R, C, outgoing footprint and the mean area of one layer (truncated pyramid clipped by its footprint)."""
    t = max(float(lay['t [mm]']), 1e-9) * 1e-3
    k = max(float(lay['k [W/mK]']), 1e-9)
    ang = float(lay.get('angle [°]', 45.0) or 0.0) if angle is None else angle
    tg = np.tan(np.radians(min(max(ang, 0.0), 89.0)))
    w_max = float(lay.get('w [mm]') or 0.0) * 1e-3 or np.inf
    l_max = float(lay.get('l [mm]') or 0.0) * 1e-3 or np.inf
    w0, l0 = min(w_in, w_max), min(l_in, l_max)                       # a narrower layer constricts the heat flow
    z = (np.arange(N_Z) + 0.5) / N_Z * t
    w = np.minimum(w0 + 2 * z * tg, w_max)
    ln = np.minimum(l0 + 2 * z * tg, l_max)
    A = w * ln
    dz = t / N_Z
    R = float(np.sum(dz / (k * A)))
    C = float(rho * cp * np.sum(A * dz))
    return R, C, float(min(w0 + 2 * t * tg, w_max)), float(min(l0 + 2 * t * tg, l_max)), float(np.mean(A))


def path(layers, die, h, mats, spread='angle', lee_last=True):
    """
    Layers of one heat path from the junction outwards. The first layer starts at the die footprint.
        spread 'angle': spreading angle of every layer from its column 'angle [°]' (45° rule)
        spread 'ratio': theta_i = arctan(k_i / k_i+1) (a well conducting layer on a poor one spreads strongly),
                        limited to 10–80°
        lee_last:       the last layer (plate cooled by h) with the spreading resistance of Lee et al. (1995) on
                        its full footprint instead of the angle rule
    Returns a list of layer results, the convection resistance at the end and the cooled area.
    """
    w, ln = die[0] * 1e-3, die[1] * 1e-3
    out = []
    R_conv = np.inf
    n = len(layers)
    for i, lay in enumerate(layers):
        rho, cp = mats.get(lay['material'], (1.0, 2000, 1000))[1:3]
        rho = float(lay.get('rho [kg/m³]', lay.get('rho', rho)) or rho)
        cp = float(lay.get('cp [J/kgK]', lay.get('cp', cp)) or cp)
        ang = None
        if spread == 'ratio':
            k_next = float(layers[i + 1]['k [W/mK]']) if i + 1 < n else float(lay['k [W/mK]'])
            ang = float(np.clip(np.degrees(np.arctan(float(lay['k [W/mK]']) / max(k_next, 1e-9))), 10.0, 80.0))
        R, C, w2, l2, Am = _layer(lay, w, ln, rho, cp, ang)
        last = i == n - 1
        wp, lp = float(lay.get('w [mm]') or 0.0) * 1e-3, float(lay.get('l [mm]') or 0.0) * 1e-3
        if last and lee_last and h and h > 0 and wp > 0 and lp > 0:
            t, k = float(lay['t [mm]']) * 1e-3, float(lay['k [W/mK]'])
            A_s = min(w, wp) * min(ln, lp)
            R, R_conv = lee(A_s, wp * lp, t, k, h)
            C = rho * cp * t * wp * lp
            w2, l2 = wp, lp
        out.append({'layer': lay['layer'], 'role': lay.get('role', ''), 'material': lay['material'],
                    't [mm]': float(lay['t [mm]']), 'k [W/mK]': float(lay['k [W/mK]']), 'A_in [mm²]': w * ln * 1e6,
                    'A_out [mm²]': w2 * l2 * 1e6, 'R [K/W]': R, 'C [J/K]': C, 'w_out [mm]': w2 * 1e3,
                    'l_out [mm]': l2 * 1e3, 'rho [kg/m³]': rho, 'cp [J/kgK]': cp})
        w, ln = w2, l2
    if not np.isfinite(R_conv):
        R_conv = 1 / (h * w * ln) if h and h > 0 else np.inf
    return out, R_conv, w * ln


def cauer(res, R_conv, n_sub=4):
    """Cauer ladder of one path: cells (R/2, C, R/2) of every layer, closed by R_conv to the coolant."""
    R, C = [], []
    for r in res:
        for _ in range(n_sub):
            R += [r['R [K/W]'] / n_sub / 2, r['R [K/W]'] / n_sub / 2]
            C += [r['C [J/K]'] / n_sub, 0.0]
    R.append(R_conv)
    C.append(0.0)
    return np.array(R), np.array(C)              # R[j] from node j-1 to node j, node j has capacity C[j]


def network(paths, C_j=0.0):
    """
    Conductance matrix G and capacity vector of the network: node 0 = junction, every path is a chain of nodes,
    the last resistance of a chain connects to the coolant (reference 0 K).
    """
    nodes_C = [C_j]
    edges = []                                                            # (i, j, R), j = -1 coolant
    idx = []
    for R, C in paths:
        prev = 0
        ii = []
        for j in range(len(R)):
            if j == len(R) - 1:
                edges.append((prev, -1, R[j]))
                break
            nodes_C.append(C[j])
            cur = len(nodes_C) - 1
            edges.append((prev, cur, R[j]))
            ii.append(cur)
            prev = cur
        idx.append(ii)
    n = len(nodes_C)
    G = np.zeros((n, n))
    for i, j, r in edges:
        if not np.isfinite(r) or r <= 0:
            if r == 0:
                r = 1e-12
            else:
                continue
        g = 1 / r
        G[i, i] += g
        if j >= 0:
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g
    return G, np.array(nodes_C), idx


def foster_exact(G, Cn):
    """
    Exact Foster network of the junction node: Kron reduction of the nodes without capacity, eigen decomposition of
    the symmetric C^-1/2 G C^-1/2. Returns R_i, tau_i (sorted by tau) and the static part R_0 (tau = 0).
    """
    c = Cn > 0
    s = ~c
    e = np.zeros(len(Cn))
    e[0] = 1.0                                                            # power enters the junction
    if not c.any():
        R0 = float(np.linalg.solve(G, e)[0])
        return np.array([]), np.array([]), R0
    Gcc, Gcs, Gsc, Gss = G[np.ix_(c, c)], G[np.ix_(c, s)], G[np.ix_(s, c)], G[np.ix_(s, s)]
    if s.any():
        Gss_inv = np.linalg.inv(Gss)
        Gr = Gcc - Gcs @ Gss_inv @ Gsc
        br = e[c] - Gcs @ Gss_inv @ e[s]
    else:
        Gss_inv = np.zeros((0, 0))
        Gr, br = Gcc, e[c]
    # output: T_j = c_out^T T_c + d P
    if c[0]:
        c_out = np.zeros(c.sum())
        c_out[0] = 1.0
        d = 0.0
    else:
        j = np.where(s)[0].tolist().index(0)
        c_out = -(Gss_inv @ Gsc)[j]
        d = float(Gss_inv[j] @ e[s])
    Ci = 1 / np.sqrt(Cn[c])
    M = (Ci[:, None] * Gr) * Ci[None, :]
    lam, V = np.linalg.eigh((M + M.T) / 2)
    beta = V.T @ (Ci * br)
    gam = V.T @ (Ci * c_out)
    R = gam * beta / lam
    tau = 1 / lam
    o = np.argsort(tau)
    return R[o], tau[o], d


def zth(t, R, tau, R0=0.0):
    """Thermal impedance of a Foster network Z_th(t) = R_0 + sum R_i (1 - exp(-t/tau_i))."""
    t = np.asarray(t, dtype=float)[:, None]
    return R0 + np.sum(np.asarray(R)[None, :] * (1 - np.exp(-t / np.asarray(tau)[None, :])), axis=1)


def foster_fit(R, tau, R0=0.0, N=5, t=None):
    """
    Foster network with N terms fitted to Z_th(t) of the exact network (log-spaced times, relative error), the
    static part R_0 is merged into the fastest term. Returns R_fit, tau_fit (sorted by tau).
    """
    tau = np.asarray(tau, float)
    R = np.asarray(R, float)
    if t is None:
        t = np.logspace(np.log10(max(tau.min() / 10, 1e-7)), np.log10(tau.max() * 10), 150)
    Z = zth(t, R, tau, R0)
    Rt = float(Z[-1])

    W = 1 / np.maximum(Z, 1e-2 * Rt)                                  # relative error above 1 % of R_th

    # ------------------------------------------
    # Start: N log-spaced time constants over the significant modes, R by non-negative least squares
    # ------------------------------------------
    sig = tau[R > 1e-3 * Rt] if np.any(R > 1e-3 * Rt) else tau
    t0 = np.logspace(np.log10(sig.min()), np.log10(sig.max()), N) if N > 1 else np.array([np.median(sig)])
    A = (1 - np.exp(-t[:, None] / t0[None, :])) * W[:, None]
    r0 = np.maximum(nnls(A, Z * W)[0], 1e-6 * Rt)
    n = N

    def fun(x):
        r = np.exp(x[:n])
        return np.r_[(zth(t, r, np.exp(x[n:])) - Z) * W, 30 * (r.sum() - Rt) / Rt]       # steady value held

    x0 = np.r_[np.log(r0), np.log(t0)]
    sol = least_squares(fun, x0, method='trf', max_nfev=4000)
    x = sol.x if np.sum(fun(sol.x) ** 2) <= np.sum(fun(x0) ** 2) else x0
    r, ta = np.exp(x[:n]), np.exp(x[n:])
    keep = r > 1e-6 * Rt
    r, ta = r[keep], ta[keep]
    o = np.argsort(ta)
    return r[o], ta[o]


def response(t, P, R, tau, R0=0.0):
    """Junction temperature rise for a power profile P(t) (exact update of every Foster term, zero-order hold)."""
    t = np.asarray(t, float)
    P = np.asarray(P, float) * np.ones_like(t)
    T = np.zeros((len(t), len(R)))
    x = np.zeros(len(R))
    for i in range(1, len(t)):
        a = np.exp(-(t[i] - t[i - 1]) / np.asarray(tau))
        x = x * a + np.asarray(R) * P[i - 1] * (1 - a)
        T[i] = x
    return T.sum(axis=1) + R0 * P


def foster_to_cauer(R, tau):
    """
    Cauer ladder with the same thermal impedance as a Foster network (Lanczos tridiagonalisation, exact up to
    rounding). With lambda_i = 1/tau_i and b_i = sqrt(R_i/tau_i) the Foster impedance is
        Z(s) = sum_i R_i / (1 + s tau_i) = b^T (sI + Lambda)^-1 b = |b|^2 e_1^T (sI + T)^-1 e_1
    with the tridiagonal T of the Lanczos process. The ladder (C_1 at the junction, R_k from node k to node k+1,
    R_n from the last node to the case) has Z(s) = 1/C_1 e_1^T (sI + M)^-1 e_1 with M = C^-1/2 G C^-1/2, so
        C_1 = 1/|b|^2,  g_1 = alpha_1 C_1,  C_k+1 = g_k^2 / (beta_k^2 C_k),  g_k+1 = alpha_k+1 C_k+1 - g_k
    Returns R_c (K/W, n) and C_c (J/K, n); R_c[-1] connects to the case.
    """
    R = np.asarray(R, float)
    tau = np.asarray(tau, float)
    m = (R > 0) & (tau > 0)
    R, tau = R[m], tau[m]
    n = len(R)
    if n == 0:
        return np.array([]), np.array([])
    lam = 1 / tau
    b = np.sqrt(R / tau)
    nb = float(np.linalg.norm(b))
    Q = np.zeros((n, n))
    al, be = np.zeros(n), np.zeros(max(n - 1, 0))
    q = b / nb
    q_prev = np.zeros(n)
    beta = 0.0
    for k in range(n):
        Q[:, k] = q
        w = lam * q - beta * q_prev
        al[k] = float(q @ w)
        w = w - al[k] * q
        w = w - Q[:, :k + 1] @ (Q[:, :k + 1].T @ w)                       # full re-orthogonalisation
        if k < n - 1:
            beta = float(np.linalg.norm(w))
            if beta <= 1e-14 * np.max(lam):
                al, be, n = al[:k + 1], be[:k], k + 1                     # degenerate time constants
                break
            be[k] = beta
            q_prev, q = q, w / beta
    C = np.zeros(n)
    g = np.zeros(n)
    C[0] = 1 / nb ** 2
    g[0] = al[0] * C[0]
    for k in range(1, n):
        C[k] = g[k - 1] ** 2 / (be[k - 1] ** 2 * C[k - 1])
        g[k] = al[k] * C[k] - g[k - 1]
    return 1 / np.maximum(g, 1e-300), C


def cells(res, n_sub=4):
    """Cauer cells (R/2, C, R/2) of the layer results of path(), without the closing convection."""
    R, C = [], []
    for r in res:
        for _ in range(n_sub):
            R += [r['R [K/W]'] / n_sub / 2, r['R [K/W]'] / n_sub / 2]
            C += [r['C [J/K]'] / n_sub, 0.0]
    return R, C


def _foster_of(chain, C_j=0.0):
    """Exact Foster network (R, tau, R_0) of one chain of the junction (power into node 0)."""
    G, Cn, _ = network([chain], C_j)
    R, tau, R0 = foster_exact(G, Cn)
    keep = R > 1e-12
    return R[keep], tau[keep], R0


def side(spec, mats, spread='angle', lee_last=True, n_sub=4):
    """
    One cooled side: inside the package (junction → case) and outside (case → coolant).

    Inputs:     spec: {'inner': {'mode': 'foster', 'R': [...], 'tau': [...]} or {'mode': 'layers', 'chip': {material,
                t [mm], A [mm²], k, rho, cp}, 'layers': [...]}, 'A_case' [mm²] (heated area at the case, Foster mode,
                square) or 'case_wl' (w, l) [mm],
                'outer': [...], 'h' [W/(m2 K)], 'R_sa' [K/W] (given cooler resistance instead of h, optional)}
    Outputs:    dict with the inner and outer layer tables, R_jc, R_ca, the Cauer chain to the coolant, the junction
                capacity, Z_th,jc (case ideally cooled) and Z_th,ca (heat into the case) as Foster networks
    """
    inn = spec['inner']
    if inn['mode'] == 'layers':
        ch = inn['chip']
        a = float(np.sqrt(max(ch['A'], 1e-6)))
        chip = {'layer': 'Chip', 'role': 'chip', 'material': ch['material'], 't [mm]': ch['t'], 'w [mm]': a,
                'l [mm]': a, 'angle [°]': 45.0, 'k [W/mK]': ch.get('k', mats[ch['material']][0]),
                'rho [kg/m³]': ch.get('rho', mats[ch['material']][1]),
                'cp [J/kgK]': ch.get('cp', mats[ch['material']][2])}
        lay_in, _, _ = path([chip] + list(inn['layers']), (a, a), 0.0, mats, spread, False)
        R_in, C_in = cells(lay_in, n_sub)
        C_j = 0.0
        R_jc = float(sum(r['R [K/W]'] for r in lay_in))
        w_c, l_c = lay_in[-1]['w_out [mm]'], lay_in[-1]['l_out [mm]']
        Rf_in = None
    else:
        Rf, tf = np.asarray(inn['R'], float), np.asarray(inn['tau'], float)
        Rc, Cc = foster_to_cauer(Rf, tf)
        R_in, C_in = list(Rc), list(Cc[1:]) + [0.0]
        C_j = float(Cc[0]) if len(Cc) else 0.0
        R_jc = float(np.sum(Rf))
        a = float(np.sqrt(max(spec.get('A_case', 100.0), 1e-6)))
        w_c, l_c = spec.get('case_wl') or (a, a)                       # footprint at the case [mm] (square default)
        lay_in = []
        Rf_in = (Rf, tf)
    R_sa = spec.get('R_sa')
    h = 0.0 if R_sa else float(spec.get('h', 0.0))
    lay_out, R_conv, _ = path(list(spec.get('outer', [])), (w_c, l_c), h, mats, spread, lee_last)
    if R_sa:
        R_conv = float(R_sa)
    R_out, C_out = cells(lay_out, n_sub)
    R_ca = float(sum(r['R [K/W]'] for r in lay_out) + R_conv)
    chain = (np.array(R_in + R_out + [R_conv]), np.array(C_in + C_out + [0.0]))
    zjc = (Rf_in[0], Rf_in[1], 0.0) if Rf_in is not None else _foster_of(
        (np.array(R_in + [0.0]), np.array(C_in + [0.0])), C_j)
    zca = _foster_of((np.array(R_out + [R_conv]), np.array(C_out + [0.0]))) if np.isfinite(R_conv) else \
        (np.array([]), np.array([]), np.inf)
    C_tot = float(sum(r['C [J/K]'] for r in lay_in + lay_out) + (np.sum(np.asarray(inn['tau']) /
                                                                         np.asarray(inn['R'])) if Rf_in else 0.0))
    return {'mode': inn['mode'], 'inner': lay_in, 'outer': lay_out, 'R_jc': R_jc, 'R_ca': R_ca, 'R_conv': R_conv,
            'R': R_jc + R_ca, 'chain': chain, 'C_j': C_j, 'zjc': zjc, 'zca': zca, 'A_case': w_c * l_c,
            'w_case': w_c, 'l_case': l_c, 'C': C_tot, 'foster_in': Rf_in}


def stackModel(main, mats, second=None, extra=None, P=1.0, T_cool=25.0, n_sub=4, N_fit=5, spread='angle',
               lee_last=True):
    """
    Thermal model of a power semiconductor: the cooled side 'main' (inside + outside), optionally the second side
    of a double-side cooled package and lumped extra paths (plastic housing, pins), all in parallel from the junction
    to the coolant / ambient (common reference temperature T_cool).

    Inputs:     main, second: side specifications (see side()); extra: [{'name', 'R'}] lumped paths [K/W]
    Outputs:    sides, extra paths, R_th,ja, heat split, T_j, steady profile, exact and fitted Foster network of
                Z_th,ja (junction → coolant)
    """
    sides = {'main': side(main, mats, spread, lee_last, n_sub)}
    if second:
        sides['second'] = side(second, mats, spread, lee_last, n_sub)
    chains, C_j, Rs = [], 0.0, {}
    for k, sd in sides.items():
        if np.isfinite(sd['R']):
            chains.append(sd['chain'])
        C_j += sd['C_j']
        Rs[k] = sd['R']
    ex = []
    for e in extra or []:
        if e.get('R') and np.isfinite(e['R']) and e['R'] > 0:
            chains.append((np.array([float(e['R'])]), np.array([0.0])))
            Rs[e['name']] = float(e['R'])
            ex.append(dict(e))
    g = {k: (1 / v if np.isfinite(v) and v > 0 else 0.0) for k, v in Rs.items()}
    if sum(g.values()) <= 0:
        raise ValueError("No heat path to the coolant: set h > 0 or a cooler resistance.")
    R_ja = 1 / sum(g.values())
    share = {k: v * R_ja for k, v in g.items()}
    G, Cn, _ = network(chains, C_j)
    R, tau, R0 = foster_exact(G, Cn)
    keep = R > 1e-9 * R_ja
    R, tau = R[keep], tau[keep]
    Rf, tf = foster_fit(R, tau, R0, N=N_fit) if len(R) >= 1 else (np.array([R0]), np.array([1e-6]))

    # ==============================================================================
    # Steady temperatures at the interfaces
    # ==============================================================================
    Tj = T_cool + P * R_ja
    prof = {}
    for k, sd in sides.items():
        Pp = P * share.get(k, 0.0)
        z, T, lab, part = [0.0], [Tj], ['junction'], ['inside']
        if sd['mode'] == 'layers':
            for r in sd['inner']:
                z.append(z[-1] + r['t [mm]'])
                T.append(T[-1] - Pp * r['R [K/W]'])
                lab.append(r['layer'])
                part.append('inside')
        else:
            z.append(z[-1] + 0.5)
            T.append(T[-1] - Pp * sd['R_jc'])
            lab.append('case (datasheet Z_th,jc)')
            part.append('inside')
        for r in sd['outer']:
            z.append(z[-1] + r['t [mm]'])
            T.append(T[-1] - Pp * r['R [K/W]'])
            lab.append(r['layer'])
            part.append('outside')
        prof[k] = {'z [mm]': z, 'T [°C]': T, 'interface': lab, 'part': part, 'P [W]': Pp,
                   'T_case': Tj - Pp * sd['R_jc']}
    return {'sides': sides, 'extra': ex, 'R_ja': R_ja, 'share': share, 'Tj': Tj, 'profile': prof,
            'foster': {'R': R, 'tau': tau, 'R0': R0}, 'fit': {'R': Rf, 'tau': tf, 'C': tf / Rf},
            'C_tot': float(sum(sd['C'] for sd in sides.values()))}


#######################################################################################################################
# Main Function
#######################################################################################################################
def thermStack(bottom, top, die, h_bottom, h_top, mats, n_sub=4, N_fit=5, P=1.0, T_cool=25.0, spread='angle',
               lee_last=True):
    """
    Thermal model of a stack with a bottom and a top heat path.

    Inputs:     bottom, top: lists of layer dicts (layer, role, material, t [mm], w [mm], l [mm], angle [°], k [W/mK])
                die: (w, l) [mm] heat source footprint; h_bottom, h_top: heat transfer coefficients [W/(m2 K)]
                mats: material library {name: (k, rho, cp, ...)}; P: power [W]; T_cool: coolant temperature [°C]
    Outputs:    dict with per-path layer tables, R_th per path and total, heat split, steady temperatures,
                exact and fitted Foster networks, Cauer ladders
    """
    ###################################################################################################################
    # Paths
    ###################################################################################################################
    res, sums, ladders = {}, {}, []
    for name, layers, h in (('bottom', bottom, h_bottom), ('top', top, h_top)):
        lay, Rc, A_out = path(layers, die, h, mats, spread, lee_last)
        R_cond = sum(r['R [K/W]'] for r in lay)
        res[name] = {'layers': lay, 'R_conv': Rc, 'A_out': A_out, 'R_cond': R_cond, 'R': R_cond + Rc,
                     'C': sum(r['C [J/K]'] for r in lay)}
        if np.isfinite(Rc) and lay:
            ladders.append(cauer(lay, Rc, n_sub))
            res[name]['ladder'] = ladders[-1]
        else:
            res[name]['ladder'] = None

    ###################################################################################################################
    # Network
    ###################################################################################################################
    Rb, Rt = res['bottom']['R'], res['top']['R']
    gb = 1 / Rb if np.isfinite(Rb) else 0.0
    gt = 1 / Rt if np.isfinite(Rt) else 0.0
    if gb + gt <= 0:
        raise ValueError("No heat path to the coolant: set h > 0 on at least one side.")
    R_ja = 1 / (gb + gt)
    share = {'bottom': gb / (gb + gt), 'top': gt / (gb + gt)}
    G, Cn, idx = network(ladders)
    R, tau, R0 = foster_exact(G, Cn)
    keep = R > 1e-9 * R_ja
    R, tau = R[keep], tau[keep]
    Rf, tf = foster_fit(R, tau, R0, N=N_fit) if len(R) >= 1 else (np.array([R0]), np.array([1e-6]))

    ###################################################################################################################
    # Steady temperatures at the layer interfaces
    ###################################################################################################################
    Tj = T_cool + P * R_ja
    prof = {}
    for name in ('bottom', 'top'):
        Pp = P * share[name]
        z, T, lab = [0.0], [Tj], ['junction']
        for r in res[name]['layers']:
            z.append(z[-1] + r['t [mm]'])
            T.append(T[-1] - Pp * r['R [K/W]'])
            lab.append(r['layer'])
        prof[name] = {'z [mm]': z, 'T [°C]': T, 'interface': lab, 'P [W]': Pp}

    return {'paths': res, 'R_ja': R_ja, 'share': share, 'Tj': Tj, 'profile': prof,
            'foster': {'R': R, 'tau': tau, 'R0': R0}, 'fit': {'R': Rf, 'tau': tf, 'C': tf / Rf},
            'C_tot': res['bottom']['C'] + res['top']['C']}


#######################################################################################################################
# References
#######################################################################################################################
# [1]: F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass Transfer, 2007.
# [2]: A. Wintrich et al., Application Manual Power Semiconductors, SEMIKRON, 2015 (Foster/Cauer networks).
# [3]: S. Lee et al., "Constriction/spreading resistance model for electronics packaging," ASME/JSME, 1995.
# [4]: JEDEC JESD51-14, Transient dual interface test method (thermal impedance, structure functions), 2010.
