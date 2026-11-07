#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         sketch.py
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
Sketches of the problems on the Theory & Tools pages (inline SVG in the style of pkg_art, strokes in currentColor so
that they work in the light and the dark theme). The sketches take the current inputs of the page:
- rod():        1D rod with heat input, convection, nodes, control volumes and the equivalent RC ladder (Numerics)
- mor():        full model → snapshots → SVD → POD modes → reduced model (MOR & POD)
- heat_path():  layers of a power module with the Foster chain and the Cauer ladder (Thermal networks)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import streamlit as st

#######################################################################################################################
# Settings
#######################################################################################################################
MAT_COL = {'Cu': '#b87333', 'Al alloy (6063)': '#a9b4c0', 'Si': '#6b7a8f', 'AlN': '#d9d2bd', 'Al2O3 (96 %)': '#e8e2d0',
           'FR4 (in-plane)': '#3f8f4f'}
HOT, COLD, BLUE = '#d03b3b', '#2a78d6', '#2a78d6'


#######################################################################################################################
# Primitives
#######################################################################################################################
def _t(x, y, s, anchor='start', size=11, col='currentColor', it=False):
    st_ = " font-style='italic'" if it else ''
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{col}' text-anchor='{anchor}'{st_}>{s}</text>"


def _l(x0, y0, x1, y1, col='currentColor', w=1.2, dash=None, op=1.0):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    return (f"<line x1='{x0:.1f}' y1='{y0:.1f}' x2='{x1:.1f}' y2='{y1:.1f}' stroke='{col}' stroke-width='{w}' "
            f"stroke-opacity='{op}'{d}/>")


def _r(x, y, w, h, fill='none', col='currentColor', sw=1.0, op=1.0, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    return (f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' fill='{fill}' fill-opacity='{op}' "
            f"stroke='{col}' stroke-width='{sw}'{d}/>")


def _arrow(x0, y0, x1, y1, col='currentColor', w=1.6):
    a = np.arctan2(y1 - y0, x1 - x0)
    p = [(x1, y1), (x1 - 8 * np.cos(a - 0.4), y1 - 8 * np.sin(a - 0.4)),
         (x1 - 8 * np.cos(a + 0.4), y1 - 8 * np.sin(a + 0.4))]
    return _l(x0, y0, x1 - 6 * np.cos(a), y1 - 6 * np.sin(a), col, w) + \
        f"<polygon points='{' '.join(f'{u:.1f},{v:.1f}' for u, v in p)}' fill='{col}'/>"


def _poly(xs, ys, col='currentColor', w=1.4, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    pts = ' '.join(f"{u:.1f},{v:.1f}" for u, v in zip(xs, ys))
    return f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='{w}'{d}/>"


def _res(x0, x1, y, lab=None, above=True):
    """Resistor (box) between x0 and x1 on a horizontal wire at y."""
    m, w = (x0 + x1) / 2, min(26, (x1 - x0) * 0.6)
    s = _l(x0, y, m - w / 2, y) + _r(m - w / 2, y - 5, w, 10, 'none', sw=1.1) + _l(m + w / 2, y, x1, y)
    if lab:
        s += _t(m, y - 9 if above else y + 19, lab, 'middle', 10)
    return s


def _cap(x, y0, y1, lab=None, gnd=True):
    """Capacitor from a node at y0 down to ground at y1."""
    m = (y0 + y1) / 2
    s = _l(x, y0, x, m - 3) + _l(x - 8, m - 3, x + 8, m - 3, w=1.8) + _l(x - 8, m + 3, x + 8, m + 3, w=1.8) + \
        _l(x, m + 3, x, y1)
    if gnd:
        s += _l(x - 7, y1, x + 7, y1, w=1.4) + _l(x - 4, y1 + 3, x + 4, y1 + 3) + _l(x - 1.5, y1 + 6, x + 1.5, y1 + 6)
    if lab:
        s += _t(x + 11, m + 4, lab, size=10)
    return s


def _dot(x, y, col='currentColor', r=3):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{col}'/>"


def _svg(body, w, h, maxw=760):
    return (f"<div style='color:inherit'><svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{maxw}px' "
            f"xmlns='http://www.w3.org/2000/svg' font-family='sans-serif'>{body}</svg></div>")


def show(svg, caption=None):
    st.markdown(svg, unsafe_allow_html=True)
    if caption:
        st.caption(caption)


#######################################################################################################################
# Rod (Numerics)
#######################################################################################################################
def rod(N=21, L=20.0, mat='Cu', h=5000.0, P=10.0):
    """1D rod with heat input P at x = 0, convection h at x = L, N nodes and the equivalent RC ladder."""
    x0, x1, y0, y1 = 90, 560, 40, 76
    col = MAT_COL.get(mat, '#b0b0b0')
    n = min(N, 9)
    xs = np.linspace(x0, x1, n)
    b = []

    # Rod, insulation, control volumes
    b.append(_r(x0, y0, x1 - x0, y1 - y0, col, sw=1.2, op=0.75))
    for y in (y0 - 4, y1 + 4):
        b.append(_l(x0, y, x1, y, w=2.5, op=0.35))
    b.append(_t((x0 + x1) / 2, y0 - 10, "insulated sides (adiabatic)", 'middle', 10))
    for i in range(n - 1):
        xm = (xs[i] + xs[i + 1]) / 2
        b.append(_l(xm, y0, xm, y1, dash='3 3', w=0.9))
    b.append(_r(x0, y0, (xs[1] - xs[0]) / 2, y1 - y0, HOT, sw=0, op=0.18))
    b.append(_r(x1 - (xs[1] - xs[0]) / 2, y0, (xs[1] - xs[0]) / 2, y1 - y0, COLD, sw=0, op=0.18))
    for x in xs:
        b.append(_dot(x, (y0 + y1) / 2))
    b.append(_t(xs[0], y1 + 18, "node 1", 'middle', 9))
    b.append(_t(xs[-1], y1 + 18, f"node {N}", 'middle', 9))
    if N > n:
        b.append(_t((x0 + x1) / 2, y1 + 18, f"{N} nodes ({n} drawn)", 'middle', 9))

    # Dimensions
    b.append(_l(xs[2], y1 + 26, xs[3], y1 + 26, w=0.8) + _l(xs[2], y1 + 22, xs[2], y1 + 30, w=0.8) +
             _l(xs[3], y1 + 22, xs[3], y1 + 30, w=0.8))
    b.append(_t((xs[2] + xs[3]) / 2, y1 + 40, "Δx = L/(N−1)", 'middle', 10))
    b.append(_t(x0, y1 + 40, "x = 0", 'middle', 10) + _t(x1, y1 + 40, f"x = L = {L:g} mm", 'middle', 10))

    # Boundary conditions
    b.append(_arrow(20, (y0 + y1) / 2, x0 - 2, (y0 + y1) / 2, HOT, 2.2))
    b.append(_t(20, (y0 + y1) / 2 - 8, f"P = {P:g} W", size=11, col=HOT))
    for dy in (-10, 0, 10):
        yy = (y0 + y1) / 2 + dy
        xx = np.linspace(x1 + 4, x1 + 44, 20)
        b.append(_poly(xx, yy + 2.5 * np.sin((xx - x1) / 4), COLD, 1.3))
    b.append(_t(x1 + 50, (y0 + y1) / 2 - 4, f"h = {h:g}", size=10, col=COLD))
    b.append(_t(x1 + 50, (y0 + y1) / 2 + 10, "W/m²K, T_a", size=10, col=COLD))

    # RC ladder
    yw, yg = 170, 215
    b.append(_t(20, yw - 26, "equivalent RC ladder (one C per node, one G between nodes)", size=10, it=True))
    b.append(_arrow(30, yw, xs[0] - 3, yw, HOT, 1.6) + _t(30, yw - 6, "P", col=HOT))
    for i in range(n - 1):
        b.append(_res(xs[i], xs[i + 1], yw, "1/G" if i == 2 else None))
    for i, x in enumerate(xs):
        b.append(_dot(x, yw, r=2.5))
        b.append(_cap(x, yw, yg, "C_i" if i == 2 else None))
    b.append(_res(xs[-1], x1 + 60, yw, "1/(hA)"))
    b.append(_l(x1 + 60, yw, x1 + 60, yg) + _l(x1 + 53, yg, x1 + 67, yg, w=1.4) + _t(x1 + 66, yw + 4, "T_a", size=10))
    b.append(_t(xs[0] - 4, yg + 22, "C_i = ρc·A·Δx  (½ at the ends),   G = k·A/Δx", size=10))
    return _svg(''.join(b), 680, 245)


#######################################################################################################################
# Model order reduction (MOR & POD)
#######################################################################################################################
def mor(N=101, K=3, train='step'):
    """Full model → snapshots → SVD → POD modes → reduced model."""
    b = []
    bw, bh, y0 = 138, 120, 30
    xb = [10, 172, 334, 496]
    titles = [f"full model: {N} states", "snapshots Θ (training)", "SVD → POD modes", f"reduced model: K = {K}"]
    for x, tt in zip(xb, titles):
        b.append(_r(x, y0, bw, bh, sw=1.0, op=0.0, dash=None))
        b.append(_t(x + bw / 2, y0 - 8, tt, 'middle', 11))
    for i in range(3):
        b.append(_arrow(xb[i] + bw + 4, y0 + bh / 2, xb[i + 1] - 4, y0 + bh / 2, w=1.5))
    u = np.linspace(0, 1, 40)

    # Full model: rod with nodes and heat flow
    b.append(_r(xb[0] + 14, y0 + 48, bw - 28, 18, MAT_COL['Cu'], sw=1.0, op=0.75))
    for x in np.linspace(xb[0] + 14, xb[0] + bw - 14, 11):
        b.append(_dot(x, y0 + 57, r=1.8))
    b.append(_arrow(xb[0] + 2, y0 + 57, xb[0] + 13, y0 + 57, HOT, 1.4))
    b.append(_t(xb[0] + bw / 2, y0 + 88, "C θ̇ = −G θ + b P", 'middle', 10))
    b.append(_t(xb[0] + bw / 2, y0 + 104, f"θ ∈ ℝ^{N}", 'middle', 10))

    # Snapshots θ(x, t_j)
    xs = xb[1] + 12 + u * (bw - 24)
    for j, lt in enumerate([0.08, 0.16, 0.3, 0.5, 0.9]):
        th = np.exp(-u / lt) * (1 - np.exp(-(j + 1) / 2.0))
        b.append(_poly(xs, y0 + bh - 14 - th * (bh - 34), HOT, 1.0 + 0.2 * j))
    b.append(_t(xb[1] + bw / 2, y0 + bh - 2, f"θ(x, t_j), {train} load", 'middle', 9))

    # Modes
    xs = xb[2] + 12 + u * (bw - 24)
    for k in range(min(K, 4)):
        ph = np.cos((k + 0.5) * np.pi * u * 0.9) * np.exp(-0.8 * u)
        b.append(_poly(xs, y0 + bh / 2 - 6 - ph * 34, ['#2a78d6', '#e07b00', '#3f8f4f', '#8e44ad'][k], 1.6))
    b.append(_t(xb[2] + bw / 2, y0 + bh - 16, "Θ = U Σ Vᵀ", 'middle', 10))
    b.append(_t(xb[2] + bw / 2, y0 + bh - 2, "Φ = U(:, 1…K)", 'middle', 10))

    # Reduced model
    b.append(_t(xb[3] + bw / 2, y0 + 36, "θ ≈ Φ a", 'middle', 13))
    b.append(_t(xb[3] + bw / 2, y0 + 62, "ΦᵀCΦ ȧ = −ΦᵀGΦ a", 'middle', 10))
    b.append(_t(xb[3] + bw / 2, y0 + 76, "+ Φᵀb P", 'middle', 10))
    b.append(_t(xb[3] + bw / 2, y0 + 100, f"{N} → {K} equations", 'middle', 10, col=BLUE))
    return _svg(''.join(b), 650, 165)


#######################################################################################################################
# Heat path with Foster and Cauer networks (Thermal networks)
#######################################################################################################################
def heat_path(n=4):
    """Layers of a power module next to the Foster chain and the Cauer ladder with n RC elements."""
    n = int(max(1, min(n, 6)))
    b = []

    # Layer stack
    lay = [('chip (junction)', '#d03b3b', 14), ('die attach', '#7d7d7d', 12), ('Cu / ceramic / Cu', '#b87333', 16),
           ('TIM', '#eda100', 12), ('heat sink', '#a9b4c0', 26)]
    y = 30
    b.append(_t(70, 18, "heat path", 'middle', 11))
    for name, c, hgt in lay:
        w = 60 if name.startswith('chip') else 110
        b.append(_r(70 - w / 2, y, w, hgt, c, sw=0.6, op=0.85))
        b.append(_t(132, y + hgt / 2 + 4, name, size=9))
        y += hgt
    b.append(_arrow(16, 34, 16, y - 4, HOT, 1.6) + _t(10, y + 14, "heat flow", size=9, col=HOT))

    # Foster chain
    x0, x1, yf = 215, 630, 52
    b.append(_t(x0, 18, "Foster: parallel RC pairs in series — inner nodes have no physical meaning", size=10,
                it=True))
    xs = np.linspace(x0 + 30, x1 - 30, n + 1)
    b.append(_t(x0, yf + 4, "T_j", size=10, col=HOT) + _l(x0 + 18, yf, xs[0], yf))
    for i in range(n):
        a, c = xs[i] + 6, xs[i + 1] - 6
        b.append(_l(xs[i], yf, a, yf) + _l(a, yf - 12, a, yf + 12) + _l(c, yf - 12, c, yf + 12) + _l(c, yf, xs[i + 1],
                                                                                                         yf))
        b.append(_res(a, c, yf - 12, f"R{i + 1}"))
        m = (a + c) / 2
        b.append(_l(a, yf + 12, m - 3, yf + 12) + _l(m - 3, yf + 5, m - 3, yf + 19, w=1.8) +
                 _l(m + 3, yf + 5, m + 3, yf + 19, w=1.8) + _l(m + 3, yf + 12, c, yf + 12))
        b.append(_t(m, yf + 32, f"C{i + 1}", 'middle', 9))
        if 0 < i:
            b.append(_dot(xs[i], yf, r=2.2))
    b.append(_l(xs[-1], yf, x1, yf) + _t(x1 + 4, yf + 4, "T_a", size=10, col=COLD))

    # Cauer ladder
    yc, yg = 150, 192
    b.append(_t(x0, 112, "Cauer: series R, grounded C — every node is a temperature inside the stack", size=10,
                it=True))
    xs = np.linspace(x0 + 30, x1 - 30, n + 1)
    b.append(_t(x0, yc + 4, "T_j", size=10, col=HOT) + _l(x0 + 18, yc, xs[0], yc))
    for i in range(n):
        b.append(_res(xs[i], xs[i + 1], yc, f"R'{i + 1}"))
    for i in range(n):
        b.append(_dot(xs[i], yc, r=2.5) + _cap(xs[i], yc, yg, f"C'{i + 1}" if i < n else None))
    b.append(_l(xs[-1], yc, x1, yc) + _t(x1 + 4, yc + 4, "T_a", size=10, col=COLD))
    b.append(_t(x0, yg + 24, "same Z_th(t); from the junction outwards the Cauer elements follow the layers on the left",
                size=9))
    return _svg(''.join(b), 660, 222)
