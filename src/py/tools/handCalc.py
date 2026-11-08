#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         handCalc.py
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
Closed-form thermal hand calculations for the design of power electronics (Theory & Tools › Hand calculator). All
inputs and outputs are in SI units (m, W, K, Pa, kg, s) unless a name says otherwise.
    1) stack:       1D series stack of layers with contact resistances → R'' per element, effective h, profile
    2) contact:     contact conductance of two rough metal surfaces (Mikic/Yovanovich) incl. gap conduction
    3) spreading:   spreading resistance of a plate heated by a smaller source (Lee et al. 1995)
    4) convection:  h of a flat plate (forced, natural), of duct flow (Hausen, Gnielinski) and of radiation
    5) fins:        plate-fin heat sink (fin efficiency, overall efficiency, R_sa)
    6) vias:        thermal via array in a PCB pad
    7) coolant:     energy balance of a coolant (temperature rise ↔ flow rate)
    8) transient:   surface temperature of a semi-infinite body under a heat flux pulse, diffusion time, lumped body,
                    pulse train on a Foster network (first pulse, build-up, periodic steady state)
    9) radiation:   grey-body exchange of two surfaces expressed as an equivalent convection coefficient h_rad
   10) skin/eddy:   skin depth, AC resistance of round and rectangular conductors (Bessel, Dowell), self-heating, eddy
                    losses induced in a metal substrate / baseplate below a trace
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

import numpy as np
from scipy.special import jve

from src.py.tools.thermStack import lee
from src.py.tools.toolData import via_k

#######################################################################################################################
# Constants and libraries
#######################################################################################################################
SIGMA_SB = 5.670374419e-8                                                 # Stefan-Boltzmann constant [W/(m2 K4)]
MU0 = 4e-7 * math.pi                                                      # vacuum permeability [H/m]
G = 9.81                                                                  # gravity [m/s2]

# fluid properties over temperature [°C]: rho [kg/m3], cp [J/(kg K)], mu [Pa s], k [W/(m K)]
# air and water: Incropera et al. 2007 (Tables A.4, A.6); water-glycol 50/50: approximate values (ASHRAE)
FLUIDS = {
    'Air (1 atm)': {'T': [-23.15, 26.85, 76.85, 126.85, 176.85],
                    'rho': [1.3947, 1.1614, 0.9950, 0.8711, 0.7740], 'cp': [1006, 1007, 1009, 1014, 1021],
                    'mu': [159.6e-7, 184.6e-7, 208.2e-7, 230.1e-7, 250.7e-7],
                    'k': [22.3e-3, 26.3e-3, 30.0e-3, 33.8e-3, 37.3e-3], 'gas': True},
    'Water': {'T': [6.85, 26.85, 46.85, 66.85, 86.85],
              'rho': [1000, 997, 989, 979, 967], 'cp': [4198, 4179, 4180, 4188, 4203],
              'mu': [1422e-6, 855e-6, 577e-6, 420e-6, 324e-6], 'k': [0.582, 0.613, 0.640, 0.660, 0.674],
              'gas': False},
    'Water-glycol 50/50': {'T': [-20, 0, 20, 40, 60, 80, 100],
                           'rho': [1095, 1085, 1077, 1069, 1059, 1048, 1036],
                           'cp': [3150, 3250, 3330, 3400, 3480, 3550, 3620],
                           'mu': [18e-3, 7.0e-3, 3.8e-3, 2.3e-3, 1.5e-3, 1.1e-3, 0.8e-3],
                           'k': [0.355, 0.365, 0.375, 0.385, 0.395, 0.405, 0.415], 'gas': False},
}

# contact materials: k [W/(m K)], micro-hardness H_c [Pa] (typical values)
CONTACT_MATERIALS = {
    'Al alloy': (200.0, 1.0e9), 'Cu': (390.0, 0.9e9), 'Steel (stainless)': (16.0, 3.0e9),
    'Steel (carbon)': (50.0, 2.5e9), 'Brass': (110.0, 1.4e9), 'Ni-plated Cu': (390.0, 3.5e9),
}
# arithmetic mean roughness Ra [µm] of typical surface finishes (rms σ ≈ 1.25 Ra for a Gaussian surface)
FINISHES = {'Lapped / polished': 0.2, 'Ground': 0.8, 'Fine milled': 1.6, 'Milled / extruded': 3.2,
            'Die-cast (as cast)': 6.3}
# gap media of a joint: k [W/(m K)], gas rarefaction parameter M [m] (0 for a liquid)
GAPS = {'Vacuum': (0.0, 0.0), 'Air (1 atm)': (0.026, 0.26e-6), 'Helium (1 atm)': (0.15, 1.6e-6),
        'Oil / grease fill (k ≈ 0.2)': (0.2, 0.0)}

# measured interface resistances R'' [K mm²/W] (Incropera et al. 2007, Tables 3.1 and 3.2)
INTERFACES = [
    ('Al / Al, vacuum, 0.1 MPa', '150–500'), ('Al / Al, vacuum, 10 MPa', '20–40'),
    ('Cu / Cu, vacuum, 0.1 MPa', '10–100'), ('Cu / Cu, vacuum, 10 MPa', '1–5'),
    ('Stainless / stainless, vacuum, 0.1 MPa', '600–2500'),
    ('Al / Al (10 µm roughness), air, 0.1 MPa', '275'), ('… same with silicone oil', '52.5'),
    ('… same with glycerine', '26.5'), ('Si chip / lapped Al, air, 0.03–0.5 MPa', '30–60'),
    ('Al / Al with indium foil, 0.1 MPa', '≈ 7'), ('Al / Al with grease (DC340), 0.1 MPa', '≈ 7'),
    ('Al / Al with Pb coating', '1–10'), ('Si chip / Al with 20 µm epoxy', '20–90'),
    ('Brass / brass with 15 µm tin solder', '2.5–14'),
]


#######################################################################################################################
# Helpers
#######################################################################################################################
def fluid(name, T):
    """Properties of a library fluid at T [°C] (linear interpolation, clamped): rho, cp, mu, k, nu, alpha, Pr."""
    f = FLUIDS[name]
    p = {c: float(np.interp(T, f['T'], f[c])) for c in ('rho', 'cp', 'mu', 'k')}
    p['nu'] = p['mu'] / p['rho']
    p['alpha'] = p['k'] / (p['rho'] * p['cp'])
    p['Pr'] = p['mu'] * p['cp'] / p['k']
    p['gas'] = f['gas']
    return p


#######################################################################################################################
# 1) Layer stack
#######################################################################################################################
def stack(layers, h=None, A=None, P=None):
    """
    1D series stack (heat flows through all layers on the same area):
        R''_i = t_i/k_i (layer) and R''_c,i (contact on the hot side of layer i),  R''_conv = 1/h
        R''_tot = sum R'' ,  h_eff = 1/R''_tot ,  R_th = R''_tot/A ,  ΔT = q'' R''_tot with q'' = P/A
    layers: list of {'name', 't' [m], 'k' [W/(m K)], 'Rc' [m2 K/W]}; h = None: the far side is held at T_ref.
    Returns {'items': [(name, kind, R'', t)], 'R_tot', 'R_cond', 'h_eff', 'h_cond', 'R_th', 'dT', 'q'}.
    """
    items = []
    for x in layers:
        if x.get('Rc', 0.0) > 0:
            items.append((f"contact → {x['name']}", 'contact', float(x['Rc']), 0.0))
        if x['t'] > 0 and x['k'] > 0:
            items.append((x['name'], 'layer', x['t'] / x['k'], x['t']))
    R_cond = sum(it[2] for it in items)
    if h:
        items.append(('convection 1/h', 'convection', 1.0 / h, 0.0))
    R_tot = sum(it[2] for it in items)
    out = {'items': items, 'R_tot': R_tot, 'R_cond': R_cond, 'h_eff': 1 / R_tot if R_tot > 0 else np.inf,
           'h_cond': 1 / R_cond if R_cond > 0 else np.inf}
    if A:
        out['R_th'] = R_tot / A
        if P is not None:
            out['q'] = P / A
            out['dT'] = P / A * R_tot
    return out


def profile(layers, q, h=None):
    """
    Temperature rise above T_ref along the stack for the heat flux q [W/m2]: points (z [m], ΔT [K]) from the hot side
    to the far side (contacts are jumps at constant z, the convection film is drawn as a jump at the end).
    """
    st = stack(layers, h)
    T = q * st['R_tot']
    z, Ts = [0.0], [T]
    for _, _, R, t in st['items']:
        T -= q * R
        z.append(z[-1] + t)
        Ts.append(T)
    return np.array(z), np.array(Ts)


#######################################################################################################################
# 2) Contact resistance
#######################################################################################################################
def slope(sigma):
    """Mean absolute asperity slope m from the rms roughness σ [m] (Antonetti et al. 1991, 0.2–9.6 µm)."""
    return 0.125 * (sigma * 1e6) ** 0.402


def contact(P, k1, k2, H_c, sigma1, sigma2, k_gap=0.0, M=0.0, m=None):
    """
    Conductance of a joint of two nominally flat, rough metal surfaces (Mikic 1974, Yovanovich 1981):
        k_s = 2 k1 k2/(k1 + k2),  σ = sqrt(σ1² + σ2²),  m = 0.125 (σ/µm)^0.402
        h_c = 1.25 k_s m/σ (P/H_c)^0.95                                    (plastic contact of the asperities)
        Y = 1.53 σ (P/H_c)^-0.097,  h_g = k_gap/(Y + M)                    (conduction through the gap)
        h_j = h_c + h_g,  R''_j = 1/h_j
    P: apparent pressure [Pa], H_c: micro-hardness of the softer surface [Pa], σ: rms roughness [m], M: gas
    rarefaction parameter [m] (0 for a liquid fill). Valid for 1e-5 < P/H_c < 2e-2.
    """
    P = np.asarray(P, dtype=float)
    k_s = 2 * k1 * k2 / (k1 + k2)
    sig = math.hypot(sigma1, sigma2)
    m = slope(sig) if m is None else m
    ratio = np.clip(P / H_c, 1e-12, None)
    h_c = 1.25 * k_s * m / sig * ratio ** 0.95
    Y = 1.53 * sig * ratio ** -0.097
    h_g = k_gap / (Y + M) if k_gap > 0 else np.zeros_like(h_c)
    h_j = h_c + h_g
    return {'h_c': h_c, 'h_g': h_g, 'h_j': h_j, 'R_j': 1 / h_j, 'k_s': k_s, 'sigma': sig, 'm': m, 'Y': Y,
            'ratio': ratio}


def tim(blt, k_tim, Rc_each=0.0):
    """Interface with a TIM: R'' = BLT/k_TIM + 2 R''_c (two boundary resistances, [m2 K/W])."""
    return blt / k_tim + 2 * Rc_each


#######################################################################################################################
# 3) Spreading
#######################################################################################################################
def spreading(A_s, A_p, t, k, h):
    """
    Plate of area A_p, thickness t and conductivity k, heated on A_s and cooled by h (Lee et al. 1995):
        R_1D = t/(k A_p),  R_sp = R_cond - R_1D,  R_conv = 1/(h A_p),  R_tot = R_cond + R_conv
    Returns {'R_1D', 'R_sp', 'R_conv', 'R_tot'} [K/W].
    """
    R_cond, R_conv = lee(A_s, A_p, t, k, h)
    R1 = t / (k * A_p)
    return {'R_1D': R1, 'R_sp': max(R_cond - R1, 0.0), 'R_conv': R_conv, 'R_tot': R_cond + R_conv}


def spread_angle(a, b, t, k, angle=45.0):
    """
    Conduction resistance of a layer with the spreading-angle rule (heated area a x b grows by 2 t tan θ per side):
        R = ∫ dz/(k (a + 2cz)(b + 2cz)) = ln[b (a + 2ct)/(a (b + 2ct))]/(2 c k (b - a)),  c = tan θ
        a = b: R = t/(k a (a + 2ct))
    """
    c = math.tan(math.radians(angle))
    if abs(b - a) < 1e-12 * max(a, b):
        return t / (k * a * (a + 2 * c * t))
    return math.log(b * (a + 2 * c * t) / (a * (b + 2 * c * t))) / (2 * c * k * (b - a))


#######################################################################################################################
# 4) Convection
#######################################################################################################################
def h_forced_plate(u, L, p):
    """
    Average h of a flat plate of length L [m] in a parallel flow u [m/s] (fluid properties p):
        Re = u L/ν;  laminar (Re ≤ 5e5): Nu = 0.664 Re^½ Pr^⅓;  mixed: Nu = (0.037 Re^0.8 - 871) Pr^⅓;  h = Nu k/L
    """
    Re = u * L / p['nu']
    Nu = 0.664 * Re ** 0.5 * p['Pr'] ** (1 / 3) if Re <= 5e5 else (0.037 * Re ** 0.8 - 871) * p['Pr'] ** (1 / 3)
    return {'Re': Re, 'Nu': Nu, 'h': Nu * p['k'] / L, 'regime': 'laminar' if Re <= 5e5 else 'mixed / turbulent'}


def h_natural(dT, L, p, T_inf, orient='vertical'):
    """
    Average h of natural convection (surface ΔT = T_s - T_∞ > 0), L: height (vertical) or A/perimeter (horizontal):
        Ra = g β ΔT L³/(ν α), β = 1/T_film (gas)
        vertical (Churchill & Chu): Nu = {0.825 + 0.387 Ra^(1/6)/[1 + (0.492/Pr)^(9/16)]^(8/27)}²
        horizontal, hot side up: Nu = 0.54 Ra^¼ (Ra < 1e7), 0.15 Ra^⅓;  hot side down: Nu = 0.52 Ra^(1/5)
    """
    beta = 1 / (T_inf + dT / 2 + 273.15) if p['gas'] else 3e-4
    Ra = G * beta * max(dT, 1e-9) * L ** 3 / (p['nu'] * p['alpha'])
    if orient == 'vertical':
        Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / p['Pr']) ** (9 / 16)) ** (8 / 27)) ** 2
    elif orient == 'horizontal, hot side up':
        Nu = 0.54 * Ra ** 0.25 if Ra < 1e7 else 0.15 * Ra ** (1 / 3)
    else:
        Nu = 0.52 * Ra ** 0.2
    return {'Ra': Ra, 'Nu': Nu, 'h': Nu * p['k'] / L}


def h_radiation(eps, T_s, T_inf):
    """Linearised radiation h_rad = ε σ (T_s² + T_∞²)(T_s + T_∞) with temperatures in K (input in °C)."""
    a, b = T_s + 273.15, T_inf + 273.15
    return eps * SIGMA_SB * (a ** 2 + b ** 2) * (a + b)


def h_duct(V_dot, w, hgt, n, L, p):
    """
    Flow of V_dot [m3/s] through n parallel rectangular channels w x hgt [m] of length L [m]:
        D_h = 4 A/P_w,  u = V_dot/(n A),  Re = u D_h/ν
        laminar (Re < 2300, Hausen, developing): Nu = 3.66 + 0.0668 Gz/(1 + 0.04 Gz^⅔), Gz = Re Pr D_h/L, f = 64/Re
        turbulent (Re > 3000, Gnielinski): f = (0.790 ln Re - 1.64)^-2,
            Nu = (f/8)(Re - 1000) Pr/(1 + 12.7 (f/8)^½ (Pr^⅔ - 1));  2300…3000 linear blend
        h = Nu k/D_h,  Δp = f L/D_h ρ u²/2,  wetted area A_w = n P_w L
    """
    A, Pw = w * hgt, 2 * (w + hgt)
    Dh = 4 * A / Pw
    u = V_dot / (n * A)
    Re = u * Dh / p['nu']

    def lam(Re):
        Gz = Re * p['Pr'] * Dh / L
        return 3.66 + 0.0668 * Gz / (1 + 0.04 * Gz ** (2 / 3)), 64 / Re

    def turb(Re):
        f = (0.790 * math.log(Re) - 1.64) ** -2
        return (f / 8) * (Re - 1000) * p['Pr'] / (1 + 12.7 * math.sqrt(f / 8) * (p['Pr'] ** (2 / 3) - 1)), f

    if Re < 2300:
        Nu, f = lam(Re)
        reg = 'laminar'
    elif Re > 3000:
        Nu, f = turb(Re)
        reg = 'turbulent'
    else:
        x = (Re - 2300) / 700
        (N1, f1), (N2, f2) = lam(2300), turb(3000)
        Nu, f, reg = (1 - x) * N1 + x * N2, (1 - x) * f1 + x * f2, 'transition'
    h = Nu * p['k'] / Dh
    return {'Dh': Dh, 'u': u, 'Re': Re, 'Nu': Nu, 'f': f, 'h': h, 'dp': f * L / Dh * p['rho'] * u ** 2 / 2,
            'A_w': n * Pw * L, 'regime': reg}


#######################################################################################################################
# 5) Fins
#######################################################################################################################
def fin(h, k, t_f, H, L):
    """
    Straight rectangular fin (thickness t_f, height H, depth L), corrected tip length H_c = H + t_f/2:
        m = sqrt(2 h/(k t_f)),  η_f = tanh(m H_c)/(m H_c),  A_f = 2 L H_c
    """
    Hc = H + t_f / 2
    m = math.sqrt(2 * h / (k * t_f))
    eta = math.tanh(m * Hc) / (m * Hc) if m * Hc > 0 else 1.0
    return {'m': m, 'Hc': Hc, 'eta': eta, 'A_f': 2 * L * Hc, 'mH': m * Hc}


def heatsink(h, k, W, L, t_b, N, t_f, H, A_s=None):
    """
    Plate-fin heat sink (base W x L x t_b, N fins of thickness t_f and height H along L):
        A_b = W L - N t_f L,  A_tot = A_b + N A_f,  η_o = 1 - N A_f/A_tot (1 - η_f)
        R_fins = 1/(η_o h A_tot),  R_base = t_b/(k W L) (+ spreading from the source area A_s, Lee et al.)
        spacing s = (W - N t_f)/(N - 1)
    """
    f = fin(h, k, t_f, H, L)
    A_b = max(W * L - N * t_f * L, 0.0)
    A_tot = A_b + N * f['A_f']
    eta_o = 1 - N * f['A_f'] / A_tot * (1 - f['eta'])
    R_f = 1 / (eta_o * h * A_tot)
    h_b = 1 / (R_f * W * L)                                       # equivalent h on the base area
    R_b = (lee(A_s, W * L, t_b, k, h_b)[0] if A_s and A_s < W * L else t_b / (k * W * L))
    s = (W - N * t_f) / (N - 1) if N > 1 else W - t_f
    return {**f, 'A_b': A_b, 'A_tot': A_tot, 'eta_o': eta_o, 'R_fins': R_f, 'R_base': R_b, 'R_sa': R_f + R_b,
            's': s, 'h_b': h_b}


#######################################################################################################################
# 6) Vias
#######################################################################################################################
def vias(pitch, drill, plating, t, w, l, filled=False, k_board=0.3, k_fill=0.3, k_cu=390.0):
    """
    Pad w x l [m] on a board of thickness t [m] with a square via array (pitch, drill [m], plating [m]):
        k_eff (area-weighted parallel conduction, toolData.via_k),  R_pad = t/(k_eff w l),  R_board = t/(k_board w l)
        single via R_via = t/(k_Cu π (r_o² - r_i²))  (barrel only),  n = floor(w/p) floor(l/p)
    """
    k_eff = via_k(pitch * 1e3, drill * 1e3, plating * 1e6, filled, k_board, k_cu, k_fill)
    A = w * l
    r_o, r_i = drill / 2, max(drill / 2 - plating, 0.0)
    n = max(int(w // pitch), 1) * max(int(l // pitch), 1)
    return {'k_eff': k_eff, 'R_pad': t / (k_eff * A), 'R_board': t / (k_board * A),
            'R_via': t / (k_cu * math.pi * (r_o ** 2 - r_i ** 2)), 'n': n}


#######################################################################################################################
# 7) Coolant
#######################################################################################################################
def coolant(P, p, dT=None, V_dot=None):
    """Energy balance P = ṁ c_p ΔT (ṁ = ρ V_dot): give dT [K] → V_dot [m3/s] or V_dot → dT."""
    if V_dot is None:
        V_dot = P / (p['rho'] * p['cp'] * dT)
    else:
        dT = P / (p['rho'] * p['cp'] * V_dot)
    return {'V_dot': V_dot, 'm_dot': p['rho'] * V_dot, 'dT': dT}


#######################################################################################################################
# 8) Transient
#######################################################################################################################
def semi_inf(q, t, k, rho, cp):
    """Surface temperature rise of a semi-infinite body under a constant heat flux q: ΔT = 2 q sqrt(t/(π k ρ c_p))."""
    return 2 * q * np.sqrt(np.asarray(t, dtype=float) / (math.pi * k * rho * cp))


def diffusion_time(d, k, rho, cp):
    """Time for the heat to diffuse through the thickness d: t_d = d²/α (penetration depth δ = sqrt(α t))."""
    return d ** 2 * rho * cp / k


def lumped(m, cp, R):
    """Lumped body of mass m [kg] on a resistance R [K/W]: C = m c_p, τ = R C."""
    return {'C': m * cp, 'tau': R * m * cp}


def foster_Z(R, tau, t):
    """Step response of a Foster network: Z_th(t) = Σ R_i (1 − e^(−t/τ_i)) [K/W]."""
    t = np.asarray(t, dtype=float)[..., None]
    return np.sum(np.asarray(R, dtype=float) * (1 - np.exp(-t / np.asarray(tau, dtype=float))), axis=-1)


def pulse_train(P, t_p, T, R, tau, n_show=3, pts=60):
    """
    Rectangular pulse train (power P for t_p, then 0, period T) on a Foster network (R_i, τ_i), exact for
    piecewise-constant power. With a_i = e^(−t_p/τ_i), b_i = e^(−T/τ_i) the temperature of term i at the end of
    pulse n is
        x_i(n) = P R_i (1 − a_i)(1 − b_i^n)/(1 − b_i)      (n = 1: single pulse, n → ∞: periodic steady state)
    and it decays by c_i = e^(−(T − t_p)/τ_i) until the next pulse. Returns the peak after every pulse, the periodic
    peak and valley, the mean D P R_th, the datasheet approximation P [D R_th + (1 − D) Z(T + t_p) − Z(T) + Z(t_p)]
    and the waveform of the first and of the last n_show periods (the train is followed until 99 % of the steady
    state of the slowest term).
    """
    R, tau = np.asarray(R, dtype=float), np.asarray(tau, dtype=float)
    D = t_p / T
    a, b, c = np.exp(-t_p / tau), np.exp(-T / tau), np.exp(-(T - t_p) / tau)
    x_inf = P * R * (1 - a) / (1 - b)
    bm = float(np.max(b))
    N = int(np.clip(np.ceil(math.log(0.01) / math.log(bm)) if bm > 1e-300 else 1, n_show, 1e6))
    n = np.unique(np.geomspace(1, N, 200).astype(int))
    peaks = np.sum(x_inf * (1 - b ** n[:, None]), axis=1)

    def wave(n0, m):
        x = x_inf * (1 - b ** n0) * c if n0 > 0 else np.zeros_like(R)     # valley before pulse n0 + 1
        ts, Ts = [], []
        for k in range(m):
            for on, dur in ((True, t_p), (False, T - t_p)):
                s = np.linspace(0, dur, pts)[:, None]
                y = P * R + (x - P * R) * np.exp(-s / tau) if on else x * np.exp(-s / tau)
                ts.append((n0 + k) * T + (0 if on else t_p) + s[:, 0])
                Ts.append(y.sum(axis=1))
                x = y[-1]
        return np.concatenate(ts), np.concatenate(Ts)

    Rth = float(R.sum())

    def Z(t):
        return float(foster_Z(R, tau, t))

    return {'D': D, 'N': N, 'n': n, 'peaks': peaks, 'first': float(peaks[0]), 'peak': float(x_inf.sum()),
            'valley': float((x_inf * c).sum()), 'mean': D * P * Rth, 'Rth': Rth,
            'approx': P * (D * Rth + (1 - D) * Z(T + t_p) - Z(T) + Z(t_p)), 'single': P * Z(t_p),
            'w_first': wave(0, n_show), 'w_last': wave(max(N - n_show, 0), n_show), 'terms': x_inf}


#######################################################################################################################
# 9) Radiation as an equivalent h
#######################################################################################################################
# total hemispherical emissivity at 300–400 K (Incropera et al. 2007, Table A.11; typical values)
EMISSIVITY = {'Black anodised Al': 0.85, 'Clear anodised Al': 0.75, 'Bare Al (rolled)': 0.07,
              'Polished Al / Cu': 0.04, 'Oxidised Cu': 0.6, 'Ni / Sn plated': 0.1, 'Paint / powder coat': 0.9,
              'FR4 / solder mask': 0.9, 'Ceramic (Al2O3, AlN)': 0.8, 'Mould compound (epoxy)': 0.9,
              'Steel (oxidised)': 0.8, 'Manual': None}


def radiation(eps1, T1, T2, eps2=1.0, F=1.0, A_ratio=0.0):
    """
    Grey-body exchange of surface 1 (ε1, T1 [°C], area A1) with surface 2 (ε2, T2 [°C], area A2), view factor
    F = F_12 and A_ratio = A1/A2 (0: surface 1 is small against an enclosure):
        q'' = σ (T1⁴ − T2⁴) / [(1 − ε1)/ε1 + 1/F + (1 − ε2)/ε2 · A1/A2] = ε_eff σ (T1⁴ − T2⁴)
        h_rad = q''/(T1 − T2) = ε_eff σ (T1² + T2²)(T1 + T2)          (exact: same heat as a convection with h_rad)
        h_lin = 4 ε_eff σ T_m³,  T_m = (T1 + T2)/2                     (linearised at the mean temperature)
    """
    a, b = np.asarray(T1, dtype=float) + 273.15, np.asarray(T2, dtype=float) + 273.15
    eps_eff = 1 / ((1 - eps1) / eps1 + 1 / F + (1 - eps2) / eps2 * A_ratio)
    h = eps_eff * SIGMA_SB * (a ** 2 + b ** 2) * (a + b)
    return {'eps_eff': eps_eff, 'h_rad': h, 'q': h * (a - b), 'h_lin': 4 * eps_eff * SIGMA_SB * ((a + b) / 2) ** 3}


#######################################################################################################################
# 10) Skin and eddy currents
#######################################################################################################################
# electrical resistivity at 20 °C [Ohm m] and its temperature coefficient [1/K] (non-magnetic, μ_r = 1)
CONDUCTORS = {'Cu (annealed)': (1.72e-8, 3.93e-3), 'Al (99.5 %, EC grade)': (2.82e-8, 4.03e-3),
              'Al 6063 (busbar, heat sink)': (3.3e-8, 3.5e-3), 'Al 5052 (IMS base)': (4.9e-8, 2.1e-3),
              'Brass CuZn37': (6.6e-8, 1.5e-3), 'Stainless steel 304': (7.2e-7, 0.94e-3)}


def rho_T(rho20, alpha, T):
    """Resistivity at T [°C]: ρ(T) = ρ20 (1 + α (T − 20))."""
    return rho20 * (1 + alpha * (np.asarray(T, dtype=float) - 20))


def skin_depth(rho, f, mu_r=1.0):
    """Skin depth δ = sqrt(ρ/(π f μ0 μ_r)) [m]."""
    return np.sqrt(rho / (math.pi * np.asarray(f, dtype=float) * MU0 * mu_r))


def _F1(D):
    """Dowell skin term Δ (sinh 2Δ + sin 2Δ)/(cosh 2Δ − cos 2Δ) (→ 1 for Δ → 0, → Δ for Δ ≫ 1)."""
    D = np.maximum(np.asarray(D, dtype=float), 1e-4)
    x = np.minimum(2 * D, 700)
    return D * (np.sinh(x) + np.sin(x)) / (np.cosh(x) - np.cos(x))


def _F2(D):
    """Dowell proximity term (sinh Δ − sin Δ)/(cosh Δ + cos Δ)."""
    D = np.minimum(np.maximum(np.asarray(D, dtype=float), 1e-6), 700)
    return (np.sinh(D) - np.sin(D)) / (np.cosh(D) + np.cos(D))


def skin_round(d, delta):
    """
    R_ac/R_dc of an isolated round wire of diameter d (exact Bessel solution):
        Z' = k ρ/(2π a) · J0(k a)/J1(k a),  k = (1 − j)/δ,  R_ac/R_dc = Re[(k a/2) J0(k a)/J1(k a)]
    (scaled Bessel functions avoid an overflow; → 1 + (a/δ)⁴/48 for a ≪ δ and → a/(2δ) + 1/4 for a ≫ δ).
    """
    a = np.asarray(d, dtype=float) / 2
    ka = (1 - 1j) * np.maximum(a / np.asarray(delta, dtype=float), 1e-6)
    return np.real(ka / 2 * jve(0, ka) / jve(1, ka))


def skin(shape, w, t, delta):
    """R_ac/R_dc of a 'round' wire (d = w) or a 'rect' bar w × t: round-wire solution at the hydraulic diameter
    d_h = 4A/P (exact for round; the high-frequency limit A/(P δ) holds for any shape; ±10 % in the transition)."""
    return skin_round(w if shape == 'round' else 2 * w * t / (w + t), delta)


def proximity(shape, w, t, delta, m, pitch=None):
    """
    Additional AC factor of a winding or bundle of m layers (Dowell). A round wire d is replaced by a square of equal
    area (h = d √π/2), porosity η = conductor width/pitch (1 for foil and touching bars):
        Δ = h/δ · √η,  F_prox = Δ · 2(m² − 1)/3 · (sinh Δ − sin Δ)/(cosh Δ + cos Δ)
    Returns (F_prox, Δ).
    """
    h = w * math.sqrt(math.pi) / 2 if shape == 'round' else t
    width = h if shape == 'round' else w
    eta = 1.0 if not pitch else min(width / pitch, 1.0)
    D = float(h / delta * math.sqrt(eta))
    return float(D * 2 * (m ** 2 - 1) / 3 * _F2(D)), D


def conductor(shape, w, t, I, f, mat, T_amb, h, m=1, pitch=None, it=200):
    """
    Self-heating of a conductor carrying I_rms at f (w = d for a round wire), cooled on its perimeter with h:
        A, U = π d²/4, π d (round) or w t, 2(w + t);  R'_dc = ρ(T)/A,  F = R_ac/R_dc = F_skin + F_prox
        P' = I² F R'_dc,  T = T_amb + P'/(h U)                     (fixed point over T because ρ grows with T)
    Returns a dict at the conductor temperature (thermal runaway or > 1000 °C: T = inf).
    """
    rho20, alpha = CONDUCTORS[mat]
    A, U = (math.pi * w ** 2 / 4, math.pi * w) if shape == 'round' else (w * t, 2 * (w + t))

    def at(T):
        rho = float(rho_T(rho20, alpha, T))
        dlt = float(skin_depth(rho, max(f, 1e-9)))
        Fs = float(skin(shape, w, t, dlt))
        Fp, D = proximity(shape, w, t, dlt, m, pitch)
        return rho, dlt, Fs, Fp, D, I ** 2 * (Fs + Fp) * rho / A

    T = T_amb
    for _ in range(it):
        Tn = T_amb + at(T)[-1] / (h * U)
        if Tn > 1000:
            T = math.inf
            break
        if abs(Tn - T) < 1e-7:
            T = Tn
            break
        T = Tn
    rho, dlt, Fs, Fp, D, Pl = at(T if math.isfinite(T) else 1000)
    return {'T': T, 'dT': T - T_amb, 'delta': dlt, 'F_skin': Fs, 'F_prox': Fp, 'F': Fs + Fp, 'Delta': D,
            'R_dc': rho / A, 'R_ac': (Fs + Fp) * rho / A, 'P': Pl, 'P_dc': I ** 2 * rho / A, 'A': A, 'U': U,
            'rho': rho}


def plate_eddy(I, f, w, h, t_p, rho_p, s=None):
    """
    Eddy loss per length [W/m] induced in a conductive plate (metal substrate, baseplate, housing) of thickness t_p by
    a trace of width w carrying I_rms at the height h above it; s = centre distance to the return conductor at the
    same height (None: return far away). The plate carries the image current; its density is that of a line
    current at h_eff = h + w/(2π) (exact limits K = I/w for h → 0 and the filament for h ≫ w):
        K(x) = β I h_eff/π · [1/(h_eff² + x²) − 1/(h_eff² + (x − s)²)]
        R_s = ρ/δ · (sinh 2Δ + sin 2Δ)/(cosh 2Δ − cos 2Δ),  Δ = t_p/δ        (→ ρ/δ thick plate, → ρ/t_p thin)
        β = X/√(1 + X²),  X = 2 h_eff t_p/δ²                                  (a thin plate shields only partly)
        P' = R_s ∫ K² dx = β² R_s I²/(2π h_eff) · {1 (single trace),  2 s²/(s² + 4 h_eff²) (go and return)}
    (the cross term of the two images is ∫ dx/((h² + x²)(h² + (x − s)²)) = 2π/(h (4h² + s²))).
    """
    dlt = float(skin_depth(rho_p, f))
    he = h + w / (2 * math.pi)
    Dl = t_p / dlt
    Rs = rho_p / dlt * float(_F1(Dl)) / Dl
    X = 2 * he * t_p / dlt ** 2
    beta = X / math.sqrt(1 + X ** 2)
    pair = 1.0 if s is None else 2 * s ** 2 / (s ** 2 + 4 * he ** 2)
    P = beta ** 2 * Rs * I ** 2 / (2 * math.pi * he) * pair
    return {'P': P, 'delta': dlt, 'R_s': Rs, 'beta': beta, 'h_eff': he, 'Delta': Dl}
