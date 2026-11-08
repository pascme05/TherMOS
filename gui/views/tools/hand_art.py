#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         hand_art.py
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
Small sketches of the problems of the hand calculator (inline SVG, strokes in currentColor for the light and the dark
theme, primitives of sketch.py). Each sketch takes the current inputs of its tab:
    stack()  contact()  spreading()  convection()  radiation()  fins()  vias()  coolant()  pulse()  pulse_train()
    skin()  proximity()  plate()
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np

from gui.views.tools.sketch import _t, _l, _r, _arrow, _poly, _svg, _dot, HOT, COLD
from src.py.tools import toolData as td

#######################################################################################################################
# Settings
#######################################################################################################################
GROUP_COL = {'Metal': '#a9b4c0', 'Semiconductor': '#6b7a8f', 'Joining': '#8d8d8d', 'Ceramic': '#e3dcc6',
             'Board': '#3f8f4f', 'Package': '#4a4a4a', 'TIM': '#eda100', 'Gas': '#dfe9f5', 'Fluid': '#9cc3ef'}
CU, AL, FR4, FLUID, TIMC = '#b87333', '#a9b4c0', '#3f8f4f', '#9cc3ef', '#eda100'
W_SVG, MAXW = 420, 440


def _col(mat):
    if mat == 'Cu' or (mat or '').startswith('Cu'):
        return CU
    m = td.MATERIALS.get(mat)
    return GROUP_COL.get(m[3], '#b0b0b0') if m else '#b0b0b0'


def _vres(x, y0, y1, lab=None, col='currentColor'):
    """Vertical resistor between y0 and y1 at x."""
    m, hgt = (y0 + y1) / 2, min(18, (y1 - y0) * 0.6)
    s = _l(x, y0, x, m - hgt / 2, col) + _r(x - 5, m - hgt / 2, 10, hgt, 'none', col, 1.1) + _l(x, m + hgt / 2, x, y1,
                                                                                                 col)
    return s + (_t(x + 10, m + 4, lab, size=9) if lab else '')


def _waves(x0, x1, y, n=3, col=COLD):
    """Wavy lines of a fluid."""
    s = ''
    for i in range(n):
        xs = np.linspace(x0, x1, 40)
        s += _poly(xs, y + 7 * i + 2 * np.sin((xs - x0) / 9 + i), col, 1.0)
    return s


def _down(xs, y0, y1, col=HOT):
    return ''.join(_arrow(x, y0, x, y1, col, 1.4) for x in xs)


def _grad(gid, c0, c1, vertical=False):
    xy = "x1='0' y1='0' x2='0' y2='1'" if vertical else "x1='0' y1='0' x2='1' y2='0'"
    return (f"<defs><linearGradient id='{gid}' {xy}><stop offset='0' stop-color='{c0}'/>"
            f"<stop offset='1' stop-color='{c1}'/></linearGradient></defs>")


#######################################################################################################################
# Sketches
#######################################################################################################################
def stack(layers, h):
    """Layer bands (thickness on a √ scale), contacts as dashed lines, cooling below, series chain on the right."""
    b = []
    x0, w, y = 30, 190, 34
    b.append(_t(x0 + w / 2, 13, "q'' = P/A", 'middle', 10, HOT) + _down([x0 + 40, x0 + w / 2, x0 + w - 40], 16, 32))
    tmax = max(x['t'] for x in layers)
    for x in layers:
        if x.get('Rc', 0) > 0:
            b.append(_l(x0 - 6, y, x0 + w + 6, y, TIMC, 2.2, '4 3'))
        hgt = float(np.clip(8 + 34 * np.sqrt(x['t'] / tmax), 9, 42))
        b.append(_r(x0, y, w, hgt, _col(x.get('mat')), sw=0.6, op=0.85))
        b.append(_t(x0 + w / 2, y + hgt / 2 + 4, x['name'][:24], 'middle', 9))
        y += hgt
    if h:
        b.append(_r(x0, y, w, 26, FLUID, sw=0, op=0.35) + _waves(x0 + 6, x0 + w - 6, y + 7) +
                 _t(x0 + w / 2, y + 40, "fluid: h, T_ref", 'middle', 9, COLD))
    else:
        b.append(''.join(_l(x0 + i, y + 10, x0 + i + 10, y, COLD, 1) for i in range(0, w, 8)) +
                 _t(x0 + w / 2, y + 24, "surface held at T_ref", 'middle', 9, COLD))
    ytot = y + 44
    # series chain
    items = [('R″c', TIMC) if x.get('Rc', 0) > 0 else None for x in layers]
    labs = []
    for x, c in zip(layers, items):
        if c:
            labs.append(c)
        labs.append(('t/k', 'currentColor'))
    if h:
        labs.append(('1/h', COLD))
    xr, ya, yb = 300, 32, ytot - 26
    ys = np.linspace(ya, yb, len(labs) + 1)
    b.append(_t(xr, ya - 8, "T_hot", 'middle', 10, HOT))
    for i, (lab, c) in enumerate(labs):
        b.append(_vres(xr, ys[i], ys[i + 1], lab, c))
    b.append(_t(xr, yb + 14, "T_ref", 'middle', 10, COLD))
    b.append(_l(350, ya, 356, ya) + _l(356, ya, 356, yb) + _l(350, yb, 356, yb) +
             _t(362, (ya + yb) / 2 - 2, "R″_tot", size=10) + _t(362, (ya + yb) / 2 + 12, "= 1/h_eff", size=10))
    return _svg(''.join(b), W_SVG, max(ytot, 150), MAXW)


def contact(m1, m2, gap, p):
    """Zoom into the joint: two rough surfaces touch at spots; heat through the spots (h_c) and the gap (h_g)."""
    rng = np.random.default_rng(3)
    xs = np.linspace(40, 300, 66)
    ya = 92 + 5 * np.sin(xs / 7) + rng.normal(0, 2.2, xs.size)
    yb = 100 - 5 * np.sin(xs / 5 + 1) + rng.normal(0, 2.2, xs.size)
    touch = ya >= yb
    ya = np.where(touch, (ya + yb) / 2, ya)
    yb = np.where(touch, ya, yb)
    gcol = {'Vacuum': 'none', 'Air (1 atm)': '#dfe9f5', 'Helium (1 atm)': '#e6f2e6'}.get(gap, '#f2d98a')
    pts = lambda yy, ytop: ' '.join([f"40,{ytop}"] + [f"{u:.1f},{v:.1f}" for u, v in zip(xs, yy)] + [f"300,{ytop}"])
    b = [f"<polygon points='{pts(yb, 105)} 300,105 40,105' fill='{gcol}' opacity='0.8'/>" if gcol != 'none' else '',
         f"<polygon points='{pts(ya, 30)}' fill='{CU if 'Cu' in m1 else AL}' opacity='0.85' stroke='currentColor' "
         f"stroke-width='0.6'/>",
         f"<polygon points='{pts(yb, 165)}' fill='{CU if 'Cu' in m2 else AL}' opacity='0.85' stroke='currentColor' "
         f"stroke-width='0.6'/>"]
    for u, v, tch in zip(xs, ya, touch):
        if tch:
            b.append(_dot(u, v, HOT, 2.6))
    b.append(_t(48, 48, m1, size=10) + _t(48, 156, m2, size=10))
    b.append(''.join(_arrow(x, 4, x, 26, 'currentColor', 1.3) for x in (90, 170, 250)) +
             _t(306, 18, f"p = {p:g} MPa", size=10))
    b.append(''.join(_arrow(x, 192, x, 170, 'currentColor', 1.3) for x in (90, 170, 250)))
    i = int(np.argmax(touch))
    b.append(_l(xs[i], 60, xs[i], 136, HOT, 1.8) + _t(308, 76, "● contact spots: h_c", size=9, col=HOT))
    b.append(_t(308, 92, f"gap ({gap.split(' (')[0].lower()}):", size=9) + _t(308, 104, "h_g = k_gap/(Y+M)", size=9))
    b.append(_t(308, 128, "only p/H_c of the", size=9) + _t(308, 140, "area touches", size=9))
    b.append(_t(170, 186 + 14, "zoom ≈ 50 µm wide", 'middle', 9, it=True))
    return _svg(''.join(b), W_SVG, 206, MAXW)


def spreading(a, W, t):
    """Cross-section of the plate: small source on top, heat fanning out, 45° cone dashed, h below."""
    x0, w, y0 = 30, 300, 40
    ws = float(np.clip(a / W * w, 8, w))
    ht = float(np.clip(t / W * w, 8, 70))
    xc = x0 + w / 2
    b = [_r(xc - ws / 2, y0 - 12, ws, 12, HOT, sw=0.6, op=0.85), _t(xc, y0 - 18, "source a", 'middle', 10, HOT),
         _r(x0, y0, w, ht, CU, sw=0.7, op=0.75), _t(x0 + w + 6, y0 + ht / 2 + 4, "t", size=10)]
    for f in np.linspace(-1, 1, 7):                                         # heat flow lines
        xa = xc + f * ws / 2
        xb = xc + f * w / 2 * 0.95
        b.append(f"<path d='M{xa:.1f},{y0:.1f} Q{xa:.1f},{y0 + ht * 0.7:.1f} {xb:.1f},{y0 + ht:.1f}' fill='none' "
                 f"stroke='{HOT}' stroke-width='1' stroke-opacity='0.8'/>")
    for s in (-1, 1):
        b.append(_l(xc + s * ws / 2, y0, xc + s * (ws / 2 + ht), y0 + ht, 'currentColor', 1, '3 3'))
    yb = y0 + ht
    b.append(_r(x0, yb, w, 18, FLUID, sw=0, op=0.35) + _waves(x0 + 6, x0 + w - 6, yb + 5, 2))
    b.append(''.join(_arrow(x, yb + 4, x, yb + 22, COLD, 1.2) for x in np.linspace(x0 + 20, x0 + w - 20, 6)))
    b.append(_t(xc, yb + 38, "h on the back of the plate W", 'middle', 9, COLD))
    b.append(_t(x0 + w + 6, y0 + ht + 16, "--- 45° rule", size=9))
    b.append(_t(x0, y0 + ht + 52, "the heat must first flow sideways: R_sp adds to t/(kA) + 1/(hA)", size=9, it=True))
    return _svg(''.join(b), W_SVG, y0 + ht + 60, MAXW)


def convection(case, n=10):
    b = []
    if case.startswith('Forced'):
        b.append(_r(60, 120, 260, 12, AL, sw=0.6, op=0.85) + _t(190, 148, "plate, length L", 'middle', 9))
        for y in (40, 62, 84, 104):
            b.append(_arrow(10, y, 50, y, COLD, 1.3))
        b.append(_t(12, 30, "u, T_∞", size=10, col=COLD))
        xs = np.linspace(60, 320, 40)
        b.append(_poly(xs, 120 - 3.6 * np.sqrt(xs - 60), 'currentColor', 1, '4 3') +
                 _t(326, 66, "boundary", size=9) + _t(326, 78, "layer δ ∝ √x", size=9))
        b.append(''.join(_arrow(x, 118, x, 96, HOT, 1.2) for x in (100, 170, 240, 300)))
        b.append(_t(190, 166, "h falls along the plate as the boundary layer grows", 'middle', 9, it=True))
        return _svg(''.join(b), W_SVG, 172, MAXW)
    if case.startswith('Natural'):
        b.append(_r(60, 20, 14, 140, AL, sw=0.6, op=0.85) + _t(40, 94, "T_s", 'middle', 10, HOT))
        ys = np.linspace(160, 20, 40)
        b.append(_poly(74 + 3.2 * np.sqrt(160 - ys), ys, 'currentColor', 1, '4 3'))
        for x in (84, 98):
            b.append(f"<path d='M{x},150 Q{x + 4},90 {x + 10},24' fill='none' stroke='{HOT}' stroke-width='1.3'/>"
                     + _arrow(x + 9, 34, x + 10, 22, HOT, 1.3))
        for y in (50, 95, 140):
            xs = np.linspace(150, 250, 40)
            b.append(_poly(xs, y + 4 * np.sin((xs - 150) / 5), HOT, 1.1) + _arrow(248, y, 262, y, HOT, 1.1))
        b.append(_t(270, 90, "radiation", size=10) + _t(270, 104, "ε σ (T_s⁴ − T_∞⁴)", size=10))
        b.append(_t(110, 176, "warm air rises along the plate (buoyancy)", 'start', 9, it=True))
        return _svg(''.join(b), W_SVG, 182, MAXW)
    nn = int(min(max(n, 1), 12))
    b.append(_r(140, 8, 120, 16, HOT, sw=0.6, op=0.8) + _t(200, 20, "module (losses)", 'middle', 9, '#fff'))
    b.append(_r(30, 24, 340, 74, AL, sw=0.7, op=0.8))
    wc = 300 / nn * 0.6
    for i in range(nn):
        x = 50 + i * 300 / nn + (300 / nn - wc) / 2
        b.append(_r(x, 38, wc, 46, FLUID, sw=0.6) + f"<circle cx='{x + wc / 2:.1f}' cy='61' r='3' fill='none' "
                                                    f"stroke='{COLD}' stroke-width='1'/>" + _dot(x + wc / 2, 61, COLD, 1))
    b.append(_t(200, 116, f"{n} channels w × h, flow out of the page (⊙), D_h = 4A/P_w", 'middle', 9))
    b.append(_t(200, 132, "laminar Re < 2300 · turbulent Re > 3000", 'middle', 9, it=True))
    return _svg(''.join(b), W_SVG, 140, MAXW)


def fins(N, W, H, tf, tb):
    """Front view of a plate-fin heat sink with the module below the base."""
    x0, w = 30, 300
    sc = w / W
    hb = float(np.clip(tb * sc, 6, 24))
    hf = float(np.clip(H * sc, 30, 110))
    nd = int(min(N, 30))
    wf = float(np.clip(tf * sc, 1.5, w / nd * 0.7))
    yb = 20 + hf
    b = [_r(x0, yb, w, hb, AL, sw=0.6, op=0.85)]
    xs = np.linspace(x0, x0 + w - wf, nd)
    for x in xs:
        b.append(_r(x, 20, wf, hf, AL, sw=0.5, op=0.85))
    b.append(_r(x0 + w / 2 - 50, yb + hb, 100, 12, HOT, sw=0.6, op=0.85) +
             _t(x0 + w / 2, yb + hb + 26, "module / source", 'middle', 9, HOT))
    if nd > 1:
        g0, g1 = xs[0] + wf, xs[1]
        b.append(_l(g0, 30, g1, 30, 'currentColor', 1) + _t((g0 + g1) / 2, 14, "s", 'middle', 10))
    b.append(_l(x0 + w + 8, 20, x0 + w + 8, yb) + _t(x0 + w + 14, 20 + hf / 2, "H", size=10))
    b.append(_t(x0 + w + 14, yb + hb, "t_b", size=9))
    b.append(_t(x0, yb + hb + 44, f"{N} fins: heat flows up each fin and leaves on both sides by h", size=9, it=True))
    return _svg(''.join(b), W_SVG, yb + hb + 50, MAXW)


def vias(w, pitch, filled):
    """Cross-section: device on the pad, plated vias through the FR4, bottom copper to the TIM and heat sink."""
    x0, wb = 30, 340
    n = int(min(max(w // pitch, 1), 12))
    b = [_r(130, 10, 140, 18, '#4a4a4a', sw=0.6, op=0.9) + _t(200, 23, "power device", 'middle', 9, '#fff'),
         _r(120, 28, 160, 5, CU, sw=0.4), _r(x0, 33, wb, 50, FR4, sw=0.6, op=0.65), _r(x0, 83, wb, 5, CU, sw=0.4),
         _r(x0, 88, wb, 8, TIMC, sw=0.4, op=0.8), _r(x0, 96, wb, 20, AL, sw=0.4, op=0.8)]
    xs = np.linspace(135, 265, n) if n > 1 else [200]
    for x in xs:
        b.append(_r(x - 3, 33, 6, 50, CU if filled else 'none', CU, 1.4, 0.9 if filled else 1))
    b.append(_down(xs[::max(1, n // 4)], 36, 80, HOT))
    b.append(_t(x0 + wb + 6, 60, "FR4", size=9) + _t(x0 + wb + 6, 95, "TIM", size=9) +
             _t(x0 + wb + 6, 110, "sink", size=9))
    b.append(_t(200, 132, f"{n} vias across the pad · the Cu barrels carry the heat, FR4 hardly conducts", 'middle', 9,
                it=True))
    return _svg(''.join(b), W_SVG, 138, MAXW)


def coolant(fluid):
    """Pipe with components in series: the coolant warms up along the flow."""
    b = [_grad('hc-cool', '#2a78d6', '#d03b3b'),
         "<rect x='30' y='60' width='320' height='26' fill='url(#hc-cool)' opacity='0.6' stroke='currentColor' "
         "stroke-width='0.7'/>"]
    for i, x in enumerate((70, 160, 250)):
        b.append(_r(x, 30, 60, 22, '#4a4a4a', sw=0.6, op=0.85) + _t(x + 30, 45, f"P{i + 1}", 'middle', 10, '#fff') +
                 _arrow(x + 30, 52, x + 30, 64, HOT, 1.4))
    b.append(_arrow(4, 73, 28, 73, COLD, 1.6) + _arrow(352, 73, 376, 73, HOT, 1.6))
    b.append(_t(4, 104, "T_in", size=10, col=COLD) + _t(300, 104, "T_out = T_in + P/(ṁ c_p)", size=10, col=HOT))
    b.append(_t(190, 124, f"{fluid}: the last component sees the warmest coolant", 'middle', 9, it=True))
    return _svg(''.join(b), W_SVG, 130, MAXW)


def pulse(delta, d):
    """Chip cross-section: the heat of a short pulse penetrates δ = √(α t_p) into the chip of thickness d."""
    x0, w, y0, hc = 60, 260, 34, 70
    f = float(np.clip(delta / d, 0.02, 1.0))
    b = [_grad('hc-pulse', '#d03b3b', '#6b7a8f', vertical=True),
         _r(x0, y0, w, hc, '#6b7a8f', sw=0.7, op=0.8),
         f"<rect x='{x0}' y='{y0}' width='{w}' height='{hc * f:.1f}' fill='url(#hc-pulse)' opacity='0.9'/>",
         _r(x0, y0 + hc, w, 8, '#8d8d8d', sw=0.5, op=0.9), _r(x0 - 20, y0 + hc + 8, w + 40, 16, CU, sw=0.5, op=0.8)]
    b.append(_down(np.linspace(x0 + 30, x0 + w - 30, 5), 6, y0 - 2))
    b.append(_t(x0 + w / 2, 6 + 0, "", 'middle') + _t(x0 + w + 6, 20, "pulse P, t_p", size=10, col=HOT))
    b.append(_l(x0 - 10, y0, x0 - 10, y0 + hc * f, HOT, 1.4) + _t(x0 - 14, y0 + hc * f / 2 + 4, "δ", 'end', 11, HOT))
    b.append(_l(x0 + w + 10, y0, x0 + w + 10, y0 + hc) + _t(x0 + w + 14, y0 + hc / 2 + 4, "d (chip)", size=10))
    b.append(_t(x0 + w + 14, y0 + hc + 6, "die attach", size=9) + _t(x0 + w + 24, y0 + hc + 20, "substrate", size=9))
    b.append(_t(x0 + w / 2, y0 + hc + 42, "δ < d: only the chip buffers the pulse" if f < 0.5 else
                "δ ≈ d: the heat reaches the die attach — use Z_th", 'middle', 9, it=True))
    return _svg(''.join(b), W_SVG, y0 + hc + 48, MAXW)


def radiation(case, eps1, eps2):
    """Hot surface radiating to a colder one (left) ≡ the same heat flow as a convection with h_rad (right)."""
    b = []
    hot, cold = '#d03b3b', '#6aa0d8'
    if case.startswith('Small'):
        b.append(_r(10, 10, 230, 112, 'none', COLD, 1.2, dash='5 3') + _t(16, 24, "surroundings T₂", size=9, col=COLD))
        b.append(_r(85, 50, 80, 40, hot, sw=0.7, op=0.85) + _t(125, 74, f"T₁, ε₁ = {eps1:g}", 'middle', 9, '#fff'))
        for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
            x0, y0 = 125 + 48 * np.cos(a), 70 + 28 * np.sin(a)
            b.append(_arrow(x0, y0, x0 + 24 * np.cos(a), y0 + 15 * np.sin(a), HOT, 1.1))
    else:
        b.append(_r(20, 22, 210, 16, hot, sw=0.6, op=0.85) + _t(125, 34, f"T₁, ε₁ = {eps1:g}", 'middle', 9, '#fff'))
        b.append(_r(20, 96, 210, 16, cold, sw=0.6, op=0.85) + _t(125, 108, f"T₂, ε₂ = {eps2:g}", 'middle', 9, '#fff'))
        for x in np.linspace(45, 205, 6):
            ys = np.linspace(42, 86, 30)
            b.append(_poly(x + 4 * np.sin((ys - 42) / 4), ys, HOT, 1.1) + _arrow(x, 86, x, 95, HOT, 1.1))
        if case.startswith('Custom'):
            b.append(_t(125, 128, "view factor F₁₂ < 1: part of the rays misses", 'middle', 9, it=True))
    b.append(_t(262, 74, "≡", 'middle', 22))
    b.append(_r(290, 40, 110, 14, hot, sw=0.6, op=0.85) + _t(345, 34, "T₁", 'middle', 9))
    b.append(_waves(292, 398, 74, 2, COLD) + _down([315, 345, 375], 56, 70, HOT))
    b.append(_t(345, 104, "q'' = h_rad (T₁ − T₂)", 'middle', 10, HOT) + _t(345, 118, "convection to T₂", 'middle', 9,
                                                                              it=True))
    return _svg(''.join(b), W_SVG, 136, MAXW)


def pulse_train(tp, T, first, peak, valley):
    """Rectangular power pulses (top) and the junction temperature building up to its periodic band (bottom)."""
    b, x0, x1, n = [], 30, 405, 6
    pw = (x1 - x0) / n
    D = float(np.clip(tp / T, 0.05, 0.95))
    xs, ys = [x0], [40]
    for i in range(n):
        xa = x0 + i * pw
        b.append(_r(xa, 14, pw * D, 26, HOT, sw=0, op=0.25))
        xs += [xa, xa, xa + pw * D, xa + pw * D]
        ys += [40, 14, 14, 40]
    b.append(_poly(xs + [x1], ys + [40], HOT, 1.4) + _t(8, 30, "P", size=10, col=HOT))
    b.append(_l(x0, 9, x0 + pw * D, 9, w=0.8) + _t(x0 + pw * D + 3, 11, "t_p", size=9))
    b.append(_l(x0 + pw, 47, x0 + 2 * pw, 47, w=0.8) + _t(x0 + 1.5 * pw, 57, "period T", 'middle', 9))
    yb, ya = 128, 66
    top = max(peak, 1e-12)
    r1, rv = float(np.clip(first / top, 0.05, 1)), float(np.clip(valley / top, 0, 0.95))
    xs, ys, prev = [], [], 0.0
    for i in range(n):
        g = 1 - (1 - r1) * (0.45 ** i)                                      # build-up towards the periodic peak
        pk, va = r1 + (1 - r1) * (1 - 0.45 ** i), rv * g
        xa = x0 + i * pw
        xs += [xa, xa + pw * D, xa + pw]
        ys += [yb - (yb - ya) * prev, yb - (yb - ya) * pk, yb - (yb - ya) * va]
        prev = va
    b.append(_poly(xs, ys, 'currentColor', 1.6))
    b.append(_l(x0, ya, x1, ya, HOT, 0.8, dash='4 3') + _t(x1, ya - 4, "periodic peak", 'end', 9, HOT))
    b.append(_l(x0, yb, x1, yb, w=0.8) + _t(8, yb - 28, "ΔT_j", size=10))
    b.append(_t(x0 + pw * D, yb + 12, "1st pulse", 'middle', 9, it=True) +
             _t(x1 - pw * (1 - D), yb + 12, "periodic steady state", 'middle', 9, it=True))
    return _svg(''.join(b), W_SVG, 146, MAXW)


def skin(shape, w, t, delta):
    """Cross-section of the conductor: the current flows in a shell of the skin depth δ (red), the core carries less."""
    b, cx, cy, S = [], 110, 70, 110.0
    sc = S / (w if shape == 'round' else max(w, t))
    dl = min(delta * sc, S / 2)
    if shape == 'round':
        R = S / 2
        rr = max(0.0, 1 - dl / R)
        b.append(f"<defs><radialGradient id='hc-skin'><stop offset='0' stop-color='{CU}'/><stop offset='{rr:.3f}' "
                 f"stop-color='{CU}'/><stop offset='1' stop-color='#d03b3b'/></radialGradient></defs>")
        b.append(f"<circle cx='{cx}' cy='{cy}' r='{R}' fill='url(#hc-skin)' stroke='currentColor' stroke-width='1'/>")
        if R - dl > 2:
            b.append(f"<circle cx='{cx}' cy='{cy}' r='{R - dl:.1f}' fill='none' stroke='currentColor' "
                     f"stroke-dasharray='4 3' stroke-width='0.9'/>")
        b.append(_t(cx + R * 0.72 + 6, cy - R * 0.72, "δ", size=12, col=HOT))
        b.append(_t(cx, cy + R + 14, f"d = {w * 1e3:.3g} mm, δ = {delta * 1e3:.3g} mm", 'middle', 9))
    else:
        ww, hh = w * sc, t * sc
        x, y = cx - ww / 2, cy - hh / 2
        b.append(_r(x, y, ww, hh, '#d03b3b', sw=1.0, op=0.9))
        ix, iy = ww - 2 * dl, hh - 2 * dl
        if ix > 1 and iy > 1:
            b.append(_r(x + dl, y + dl, ix, iy, CU, sw=0.9, dash='4 3'))
        b.append(_t(x + ww + 6, y + 10, "δ", size=12, col=HOT))
        b.append(_t(cx, y + hh + 14, f"{w * 1e3:.3g} × {t * 1e3:.3g} mm, δ = {delta * 1e3:.3g} mm", 'middle', 9))
    b.append(_t(240, 34, "current density J", size=10) + _r(240, 42, 12, 10, '#d03b3b', sw=0.4) +
             _t(256, 51, "high (skin)", size=9) + _r(240, 58, 12, 10, CU, sw=0.4) + _t(256, 67, "low (core)", size=9))
    b.append(_t(240, 92, "R_ac ≈ ρ/(perimeter · δ)", size=10))
    b.append(_t(240, 108, "dashed: depth δ = √(ρ/(π f μ))", size=9, it=True))
    return _svg(''.join(b), W_SVG, 146, MAXW)


def proximity(shape, w, t, m, pitch, D, Fp):
    """Winding cross-section of the Dowell model: m layers stacked, conductors side by side at the pitch p in each
    layer, the field H along the layers builds up layer by layer (MMF on the right). The red tint of a layer is its
    share of the proximity loss at large Δ (∝ k² + (k − 1)²)."""
    b, md, n = [], min(int(m), 6), 4
    wd, ht = (w, w) if shape == 'round' else (w, t)
    p = max(pitch or wd, wd)
    sc = min(190 / (n * p), 96 / (md * 1.3 * ht))
    cw, ch, cp = wd * sc, ht * sc, p * sc
    if shape != 'round':                                             # thin foils stay visible
        ch = min(max(ch, 6.0), 96 / (1.3 * md))
    lp = 1.3 * ch
    x0, y0 = 50 + (190 - n * cp) / 2 + (cp - cw) / 2, 22
    share = [(k ** 2 + (k - 1) ** 2) / (md ** 2 + (md - 1) ** 2) for k in range(1, md + 1)]
    for k in range(md):
        y = y0 + k * lp
        op = 0.1 + 0.8 * share[k]
        for j in range(n):
            x = x0 + j * cp
            if shape == 'round':
                b.append(f"<circle cx='{x + cw / 2:.1f}' cy='{y + ch / 2:.1f}' r='{cw / 2:.1f}' fill='{CU}' "
                         f"stroke='currentColor' stroke-width='0.6'/>"
                         f"<circle cx='{x + cw / 2:.1f}' cy='{y + ch / 2:.1f}' r='{cw / 2:.1f}' fill='#d03b3b' "
                         f"fill-opacity='{op:.2f}'/>")
            else:
                b.append(_r(x, y, cw, ch, CU, sw=0.6) + _r(x, y, cw, ch, '#d03b3b', 'none', 0, op))
        lab = str(m) if (k == md - 1 and m > md) else str(k + 1)
        b.append(_t(40, y + ch / 2 + min(9, lp) / 2.5, lab, 'end', round(min(9, lp), 1)))
    yb = y0 + (md - 1) * lp + ch
    b.append(_t(40, y0 - 8, "layer k", 'end', 8, it=True))
    if m > md:
        b.append(_t(x0 + n * cp + 4, y0 + (md - 1.5) * lp + ch / 2 + 4, "⋮", size=11))
    if n > 1:                                                        # pitch dimension in the first layer
        ya = y0 - 6
        b.append(_l(x0 + cw / 2, ya, x0 + cp + cw / 2, ya, w=0.8) + _l(x0 + cw / 2, ya - 3, x0 + cw / 2, ya + 3, w=0.8)
                 + _l(x0 + cp + cw / 2, ya - 3, x0 + cp + cw / 2, ya + 3, w=0.8)
                 + _t(x0 + cp, ya - 3, "p", 'middle', 9))
    xe = x0 + n * cp - (cp - cw) + 5                                 # layer height h
    b.append(_l(xe, y0, xe + 6, y0, w=0.7) + _l(xe, y0 + ch, xe + 6, y0 + ch, w=0.7)
             + _l(xe + 3, y0, xe + 3, y0 + ch, w=0.7) + _t(xe + 8, y0 + ch / 2 + 4, "h", size=9))
    xa, xw = 300, 90                                                 # MMF profile
    xs, ys = [xa], [y0]
    for k in range(md):
        y = y0 + k * lp
        xs += [xa + xw * k / md, xa + xw * (k + 1) / md]
        ys += [y, y + ch]
        if k < md - 1:
            xs.append(xa + xw * (k + 1) / md)
            ys.append(y + lp)
    b.append(_l(xa, y0 - 4, xa, yb + 4, w=0.8) + _poly(xs, ys, HOT, 1.6))
    b.append(_t(xa + 4, y0 - 6, "field H(y) along the layers", size=8) +
             _t(xa + xw, yb + 12, "H_max ∝ m I", 'middle', 8) + _t(xa - 4, y0 + 4, "0", 'end', 8))
    eta = min(wd / p, 1.0) if shape != 'round' else min(ht * np.sqrt(np.pi) / 2 / p, 1.0)
    reg = "inner layers carry most of the loss" if D > 1 else "field penetrates, proximity loss small"
    b.append(_t(210, yb + 26, f"m = {m}, η = {eta:.2f}, Δ = {D:.2f} → F_prox = {Fp:.3g}: {reg}", 'middle', 9,
                it=True))
    b.append(_t(210, yb + 39, "red tint: share of the proximity loss in layer k (∝ k² + (k − 1)² for Δ > 1)",
                'middle', 8))
    return _svg(''.join(b), W_SVG, yb + 45, MAXW)


def plate(h, tp, delta, pair):
    """Trace(s) on a dielectric above a metal substrate: the image (eddy) current flows in the plate below."""
    b, x0, W, y_d, hd, hp = [], 20, 380, 50, 10, 46
    yp = y_d + hd
    dp = hp * min(1.0, delta / max(tp, 1e-12))
    b.append(_r(x0, y_d, W, hd, '#e3dcc6', sw=0.5, op=0.9) + _r(x0, yp, W, hp, AL, sw=0.6, op=0.9))
    ww = 60
    for i, xc in enumerate([150, 270] if pair else [210]):
        b.append(_r(xc - ww / 2, y_d - 9, ww, 9, CU, sw=0.6) + _t(xc, y_d - 13, "I " + ('⊙' if i == 0 else '⊗'),
                                                                   'middle', 10))
        b.append(f"<ellipse cx='{xc}' cy='{yp + dp / 2:.1f}' rx='{ww * 0.9:.1f}' ry='{max(dp / 2, 2):.1f}' "
                 f"fill='#d03b3b' fill-opacity='0.55'/>")
        b.append(_t(xc, yp + dp / 2 + 4, '⊗' if i == 0 else '⊙', 'middle', 10, '#fff'))
    b.append(_t(x0 + 4, y_d + 8, f"dielectric h = {h * 1e3:.3g} mm", size=8))
    b.append(_t(x0 + 4, yp + hp - 6, f"metal base t = {tp * 1e3:.3g} mm, δ = {delta * 1e3:.3g} mm", size=9))
    b.append(_t(x0 + W / 2, 16, "trace and return" if pair else "single trace (return far away)", 'middle', 10))
    b.append(_t(x0 + W / 2, yp + hp + 15, "red: induced image (eddy) current — it heats the base plate", 'middle', 9,
                it=True))
    return _svg(''.join(b), W_SVG, yp + hp + 22, MAXW)
