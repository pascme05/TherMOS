#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         pkg_art.py
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
Small schematic cross-sections of the package presets of the thermal stack designer (inline SVG, not to scale):
chip (red), copper (brown), ceramic (light), mold / gel (grey), board (green), interface material (yellow), cooler
(blue with fins or coolant). The dashed line marks the case (end of the datasheet Z_th,jc) and the arrow the main heat
flow. Text uses the theme colour (currentColor), so the pictures work in light and dark mode.
"""

#######################################################################################################################
# Parameter
#######################################################################################################################
COL = {'chip': '#d03b3b', 'cu': '#b87333', 'cer': '#e8e2d0', 'mold': '#7d7d7d', 'gel': '#c9d6e3', 'board': '#3f8f4f',
       'tim': '#eda100', 'cool': '#2a78d6', 'al': '#a9b4c0', 'lead': '#b87333'}
W, H = 300, 170


#######################################################################################################################
# Helpers
#######################################################################################################################
def _r(x, y, w, h, c, op=1.0, rx=0):
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{COL.get(c, c)}' fill-opacity='{op}' "
            f"stroke='#444' stroke-width='0.5'/>")


def _t(x, y, s, anchor='start', size=10):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='currentColor' text-anchor='{anchor}'>{s}</text>"


def _fins(x, y, w, n=9, h=22, up=False):
    out = [_r(x, y, w, 6, 'al')]
    dx = w / n
    for i in range(n):
        out.append(_r(x + i * dx + dx * 0.3, y - h if up else y + 6, dx * 0.4, h, 'al'))
    return ''.join(out)


def _coolant(x, y, w, h=16):
    return _r(x, y, w, h, 'cool', 0.35) + _t(x + w / 2, y + h / 2 + 4, 'coolant', 'middle', 9)


def _case(x0, x1, y):
    return (f"<line x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' stroke='{COL['chip']}' stroke-dasharray='4 3' "
            f"stroke-width='1.2'/>" + _t(x1 + 4, y + 3, 'case', size=9))


def _arrow(x, y0, y1):
    d = 1 if y1 > y0 else -1
    return (f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y1}' stroke='{COL['chip']}' stroke-width='2'/>"
            f"<polygon points='{x - 5},{y1 - 7 * d} {x + 5},{y1 - 7 * d} {x},{y1}' fill='{COL['chip']}'/>")


#######################################################################################################################
# Pictures
#######################################################################################################################
def _tht():
    s = _r(90, 30, 120, 50, 'mold', 0.9, 3) + _r(85, 80, 130, 14, 'cu') + _r(135, 66, 30, 8, 'chip')
    s += ''.join(_r(30, 40 + 12 * i, 60, 5, 'lead') for i in range(3))
    s += _r(70, 94, 160, 4, 'tim') + _fins(50, 98, 200)
    return s + _case(85, 215, 94) + _arrow(20, 80, 125) + _t(8, 20, 'TO-247 on heat sink')


def _smd():
    s = _r(100, 40, 100, 34, 'mold', 0.9, 3) + _r(105, 74, 90, 8, 'cu') + _r(135, 62, 30, 8, 'chip')
    s += _r(60, 84, 180, 16, 'board') + ''.join(_r(120 + 14 * i, 84, 4, 16, 'cu') for i in range(5))
    s += _r(80, 100, 140, 6, 'tim') + _r(40, 106, 220, 12, 'al') + _coolant(40, 118, 220, 14)
    return s + _case(105, 195, 82) + _arrow(30, 60, 120) + _t(8, 20, 'SMD on board (vias, pad, housing)')


def _topcool():
    s = _fins(70, 34, 160, up=True) + _r(80, 40, 140, 5, 'tim') + _r(100, 45, 100, 8, 'cu') + _r(135, 53, 30, 8,
                                                                                                   'chip')
    s += _r(100, 61, 100, 26, 'mold', 0.9, 3) + _r(60, 92, 180, 16, 'board')
    s += ''.join(_r(102 + 88 * i, 82, 8, 10, 'lead') for i in range(2))
    return s + _case(100, 200, 45) + _arrow(30, 80, 25) + _t(8, 160, 'Top-side cooled SMD (heat sink above)')


def _module():
    s = _r(70, 30, 160, 40, 'gel', 0.8) + _r(120, 58, 30, 6, 'chip') + _r(170, 58, 30, 6, 'chip')
    s += _r(100, 64, 120, 5, 'cu') + _r(95, 69, 130, 6, 'cer') + _r(100, 75, 120, 5, 'cu')
    s += _r(60, 82, 180, 14, 'cu') + _r(60, 96, 180, 4, 'tim') + _r(40, 100, 220, 14, 'al') + _coolant(40, 114, 220)
    return s + _case(60, 240, 96) + _arrow(30, 60, 125) + _t(8, 20, 'Module: DBC on baseplate, cold plate')


def _pinfin():
    s = _r(70, 30, 160, 40, 'gel', 0.8) + _r(120, 58, 30, 6, 'chip') + _r(170, 58, 30, 6, 'chip')
    s += _r(100, 64, 120, 5, 'cu') + _r(95, 69, 130, 6, 'cer') + _r(100, 75, 120, 5, 'cu')
    s += _r(60, 82, 180, 10, 'cu') + ''.join(_r(66 + 12 * i, 92, 5, 22, 'cu') for i in range(15))
    s += _r(55, 92, 190, 30, 'cool', 0.25)
    return s + _case(60, 240, 82) + _arrow(30, 60, 125) + _t(8, 20, 'Module: pin-fin baseplate, direct cooling')


def _dsc():
    s = _coolant(40, 20, 220) + _r(40, 36, 220, 10, 'al') + _r(70, 46, 160, 3, 'tim')
    s += _r(90, 49, 120, 4, 'cu') + _r(85, 53, 130, 5, 'cer') + _r(90, 58, 120, 4, 'cu') + _r(135, 62, 30, 10, 'cu')
    s += _r(135, 72, 30, 7, 'chip')
    s += _r(90, 79, 120, 4, 'cu') + _r(85, 83, 130, 5, 'cer') + _r(90, 88, 120, 4, 'cu') + _r(70, 92, 160, 3, 'tim')
    s += _r(40, 95, 220, 10, 'al') + _coolant(40, 105, 220)
    return (s + _case(90, 210, 92) + _case(90, 210, 49) + _arrow(25, 75, 120) + _arrow(25, 75, 25) +
            _t(8, 160, 'Double-side cooled module'))


def _qfn():
    s = _r(115, 46, 70, 22, 'mold', 0.9, 3) + _r(125, 68, 50, 6, 'cu') + _r(137, 60, 26, 8, 'chip')
    s += _r(60, 76, 180, 16, 'board') + ''.join(_r(126 + 12 * i, 76, 4, 16, 'cu') for i in range(5))
    s += _r(80, 92, 140, 6, 'tim') + _r(40, 98, 220, 12, 'al') + _coolant(40, 110, 220, 14)
    return s + _case(125, 175, 74) + _arrow(30, 60, 118) + _t(8, 20, 'GaN QFN on board with vias')


def _embedded():
    s = _r(50, 50, 200, 40, 'board', 0.9) + _r(130, 60, 40, 8, 'chip')
    s += ''.join(_r(134 + 8 * i, 68, 3, 6, 'cu') for i in range(5)) + _r(100, 74, 100, 6, 'cu')
    s += _r(50, 90, 200, 4, 'cu') + _r(70, 94, 160, 4, 'tim') + _r(40, 98, 220, 12, 'al') + _coolant(40, 110, 220, 14)
    return s + _case(100, 200, 80) + _arrow(30, 60, 118) + _t(8, 20, 'Die embedded in the PCB')


DRAW = {'tht': _tht, 'smd': _smd, 'topcool': _topcool, 'module': _module, 'pinfin': _pinfin, 'dsc': _dsc,
        'qfn': _qfn, 'embedded': _embedded}


def svg(kind):
    """Inline SVG of a package kind (empty string for an unknown kind)."""
    f = DRAW.get(kind)
    if f is None:
        return ''
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:340px' xmlns='http://www.w3.org/2000/svg' "
            f"font-family='sans-serif'>{f()}</svg>")
