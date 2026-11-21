#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         design_art.py
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
Sketches of the tool 'Design effects' (inline SVG, strokes in currentColor, primitives of sketch.py):
    top(pos, chips, ok, sel, h)     top view of the board with the chips at their current positions
    stack(rows, key)                layer stack (node rows) with the rows touched by a design change highlighted
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.views.tools.sketch import _t, _l, _r, _arrow, _svg, HOT, COLD

#######################################################################################################################
# Settings
#######################################################################################################################
CU, AL, DIEL, ATT, DIE = '#b87333', '#a9b4c0', '#eda100', '#8d8d8d', '#6b7a8f'
ROWS = [('Al base (bottom row)', AL), ('Al base', AL), ('Al base (top row)', AL), ('dielectric', DIEL),
        ('copper', CU), ('die attach', ATT), ('die', DIE)]                 # node rows of the FV model, bottom → top
WHAT = {'h': "G diagonal (h·A)", 'kDiel': "G series conductances", 'sAtt': "G series conductances",
        'sRc': "C (ρ·c·V)", 'pos': "G links die ↔ copper"}
MAXW = 440


#######################################################################################################################
# Sketches
#######################################################################################################################
def top(pos, chips, ok=True, sel=None, h=1500.0):
    """Top view 30 x 20 mm, chips as rectangles at their centres, the selected chip outlined, cooling below."""
    s, x0, y0 = 11.0, 40, 16                                          # px per mm, origin of the board
    b = [_r(x0, y0, 30 * s, 20 * s, CU, sw=1.2, op=0.25)]
    for i, ((x, y), c) in enumerate(zip(pos, chips)):
        X, Y = x0 + (x - c['w'] / 2) * s, y0 + (20 - y - c['l'] / 2) * s
        b.append(_r(X, Y, c['w'] * s, c['l'] * s, DIE if ok else HOT, col=HOT if i == sel else 'currentColor',
                    sw=2.4 if i == sel else 1.0, op=0.85))
        b.append(_t(X + c['w'] * s / 2, Y + c['l'] * s / 2 + 4, c['name'], 'middle', 11, '#ffffff'))
    b.append(_t(x0, y0 + 20 * s + 16, "0", 'middle', 9) + _t(x0 + 30 * s, y0 + 20 * s + 16, "30 mm", 'middle', 9))
    b.append(_t(x0 - 6, y0 + 4, "20", 'end', 9))
    b.append(_t(x0 + 15 * s, y0 + 20 * s + 30, f"cold plate below: h = {h:g} W/(m²K)", 'middle', 10, COLD, it=True))
    if not ok:
        b.append(_t(x0 + 15 * s, y0 - 4, "invalid layout: chips overlap or leave the 1 mm margin", 'middle', 10, HOT,
                    it=True))
    return _svg(''.join(b), 420, 270, MAXW)


def stack(rows, key):
    """Seven node rows of the FV model (bottom → top); rows with a changed entry are outlined in red and labelled."""
    b, x0, w, hgt, y0 = [], 20, 200, 18, 14
    n = len(ROWS)
    for i, (name, col) in enumerate(ROWS):
        y = y0 + (n - 1 - i) * (hgt + 3)
        hit = i < len(rows) and rows[i] > 0
        wid = w if name != 'die' and name != 'die attach' else w * 0.35
        xs = x0 if wid == w else x0 + (w - wid) / 2
        b.append(_r(xs, y, wid, hgt, col, HOT if hit else 'currentColor', 2.2 if hit else 0.6, 0.8 if hit else 0.35))
        b.append(_t(x0 + w + 10, y + 13, name, size=10))
        if hit:
            b.append(_t(x0 + w + 115, y + 13, f"{int(rows[i])} nodes", size=10, col=HOT))
    yb = y0 + n * (hgt + 3)
    for x in range(x0 + 20, x0 + w, 40):
        b.append(_arrow(x, yb, x, yb + 14, COLD, 1.3))
    b.append(_t(x0 + w / 2, yb + 28, "cooling h", 'middle', 10, COLD))
    b.append(_t(x0 + w + 10, yb + 28, f"changes: {WHAT.get(key, '')}", size=10, col=HOT, it=True))
    b.append(_l(x0, y0 - 4, x0 + w, y0 - 4, dash='3,3', op=0.4))
    return _svg(''.join(b), 420, yb + 40, MAXW)
