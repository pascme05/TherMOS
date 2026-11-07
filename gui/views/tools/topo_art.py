#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         topo_art.py
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
Schematics of the topologies of the power loss calculator with the notation of the variables (inline SVG): B6 and B4
inverter, buck and boost converter and the general single device. Switches are drawn as a transistor box with its
antiparallel / body diode; the highlighted switch (red) is the one whose losses are shown. Text uses the theme colour
(currentColor), so the schematics work in light and dark mode.
"""

#######################################################################################################################
# Parameter
#######################################################################################################################
RED, BLUE, GREEN, INK = '#d03b3b', '#2a78d6', '#1baf7a', 'currentColor'
W, H = 420, 200


#######################################################################################################################
# Primitives
#######################################################################################################################
def _l(x1, y1, x2, y2, c=INK, w=1.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{c}' stroke-width='{w}'{d}/>"


def _t(x, y, s, c=INK, size=11, anchor='middle', it=False):
    st = " font-style='italic'" if it else ''
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{c}' text-anchor='{anchor}'{st}>{s}</text>"


def _sw(x, y, lab, hot=False, diode=True):
    """Switch centred at (x, y): transistor box (24 x 30) with the antiparallel diode on its right."""
    c = RED if hot else INK
    s = (f"<rect x='{x - 12}' y='{y - 15}' width='24' height='30' rx='3' fill='none' stroke='{c}' "
         f"stroke-width='{2 if hot else 1.5}'/>" + _t(x, y + 4, lab, c, 10))
    if diode:
        s += (_l(x + 12, y - 12, x + 24, y - 12, c, 1) + _l(x + 12, y + 12, x + 24, y + 12, c, 1) +
              _l(x + 24, y - 12, x + 24, y + 12, c, 1) +
              f"<polygon points='{x + 19},{y + 5} {x + 29},{y + 5} {x + 24},{y - 4}' fill='{c}'/>" +
              _l(x + 19, y - 5, x + 29, y - 5, c, 1.2))
    return s


def _arrow(x1, y1, x2, y2, c=BLUE):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - 7 * math.cos(a - 0.4), y2 - 7 * math.sin(a - 0.4))
    p2 = (x2 - 7 * math.cos(a + 0.4), y2 - 7 * math.sin(a + 0.4))
    return (_l(x1, y1, x2, y2, c, 1.5) +
            f"<polygon points='{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}' fill='{c}'/>")


def _cap(x, y_top, y_bot, lab):
    ym = (y_top + y_bot) / 2
    return (_l(x, y_top, x, ym - 4) + _l(x - 10, ym - 4, x + 10, ym - 4, w=2) + _l(x - 10, ym + 4, x + 10, ym + 4, w=2)
            + _l(x, ym + 4, x, y_bot) + _t(x - 14, ym + 4, lab, anchor='end', it=True))


def _ind(x1, x2, y, lab):
    n = 4
    dx = (x2 - x1) / n
    arcs = ''.join(f"<path d='M {x1 + i * dx} {y} a {dx / 2} {dx / 2} 0 0 1 {dx} 0' fill='none' stroke='{INK}' "
                   f"stroke-width='1.5'/>" for i in range(n))
    return arcs + _t((x1 + x2) / 2, y - dx / 2 - 6, lab, it=True)


def _src(x, y_top, y_bot, lab):
    ym = (y_top + y_bot) / 2
    return (_l(x, y_top, x, ym - 14) + f"<circle cx='{x}' cy='{ym}' r='14' fill='none' stroke='{INK}' "
            f"stroke-width='1.5'/>" + _t(x, ym - 3, '+', size=10) + _t(x, ym + 10, '−', size=10) +
            _l(x, ym + 14, x, y_bot) + _t(x - 20, ym + 4, lab, anchor='end', it=True))


def _leg(x, top, bot, labs, hot_top=False):
    """Half bridge between the rails: upper and lower switch, output node at the middle."""
    mid = (top + bot) / 2
    return (_l(x, top, x, mid - 45) + _sw(x, mid - 30, labs[0], hot_top) + _l(x, mid - 15, x, mid + 15) +
            _sw(x, mid + 30, labs[1]) + _l(x, mid + 45, x, bot) + f"<circle cx='{x}' cy='{mid}' r='2.5' "
            f"fill='{INK}'/>"), mid


#######################################################################################################################
# Topologies
#######################################################################################################################
def _b6():
    top, bot = 20, 180
    s = _l(30, top, 330, top) + _l(30, bot, 330, bot) + _cap(30, top, bot, 'V<tspan font-size="8">dc</tspan>')
    for i, (x, ph) in enumerate(zip((110, 190, 270), 'abc')):
        g, mid = _leg(x, top, bot, (f"T{2 * i + 1}", f"T{2 * i + 2}"), hot_top=i == 0)
        s += g
        s += _l(x, mid, x + 40, mid)
        s += _t(x + 46, mid + 4, ph, anchor='start')
    s += _arrow(124, 108, 150, 108) + _t(142, 124, 'i<tspan font-size="8">a</tspan>(t)', BLUE, 10, 'start')
    s += _t(345, 60, 'i<tspan font-size="8">a</tspan> = Î sin(ωt − φ)', BLUE, 10, 'start')
    s += _t(345, 78, 'duty of T1:', INK, 10, 'start') + _t(345, 92, '½ (1 + M sin ωt)', INK, 10, 'start')
    s += _t(345, 120, 'D1 = diode of T1', RED, 10, 'start') + _t(345, 136, 'losses shown per', INK, 10, 'start')
    s += _t(345, 150, 'switch position', INK, 10, 'start')
    return s


def _b4():
    top, bot = 20, 180
    s = _l(40, top, 260, top) + _l(40, bot, 260, bot) + _cap(40, top, bot, 'V<tspan font-size="8">dc</tspan>')
    g1, m = _leg(120, top, bot, ('T1', 'T2'), hot_top=True)
    g2, _ = _leg(220, top, bot, ('T3', 'T4'))
    s += g1 + g2 + _l(120, m, 145, m) + _l(195, m, 220, m)
    s += f"<rect x='145' y='{m - 8}' width='50' height='16' fill='none' stroke='{INK}' stroke-width='1.5'/>"
    s += _t(170, m + 4, 'load', size=9) + _arrow(150, m - 16, 175, m - 16) + _t(162, m - 22, 'i(t)', BLUE, 10)
    s += _t(290, 60, 'i = Î sin(ωt − φ)', BLUE, 10, 'start') + _t(290, 80, 'bipolar PWM,', INK, 10, 'start')
    s += _t(290, 94, 'duty of T1: ½ (1 + M sin ωt)', INK, 10, 'start')
    return s


def _buck():
    top, bot = 30, 170
    s = _src(50, top, bot, 'V<tspan font-size="8">in</tspan>') + _l(50, top, 120, top) + _l(50, bot, 330, bot)
    s += _l(120, top, 120, 55) + _sw(120, 70, 'S', True) + _l(120, 85, 120, 115) + _sw(120, 130, 'R') + \
        _l(120, 145, 120, bot)
    s += f"<circle cx='120' cy='100' r='2.5' fill='{INK}'/>" + _l(120, 100, 170, 100) + _ind(170, 230, 100, 'L')
    s += _l(230, 100, 290, 100) + _cap(260, 100, bot, 'C') + _l(290, 100, 330, 100)
    s += _t(338, 140, 'V<tspan font-size="8">out</tspan>', anchor='start', it=True)
    s += _arrow(175, 115, 225, 115) + _t(200, 130, 'i<tspan font-size="8">L</tspan> = I ± ΔI/2', BLUE, 10)
    s += _t(225, 30, 'D = V<tspan font-size="8">out</tspan> / V<tspan font-size="8">in</tspan>', INK, 10, 'start')
    s += _t(225, 46, 'S: on for D·T, R for (1 − D)·T', INK, 10, 'start')
    s += _t(225, 62, 'R = diode or sync. MOSFET', INK, 10, 'start')
    return s


def _boost():
    top, bot = 50, 170
    s = _src(50, top, bot, 'V<tspan font-size="8">in</tspan>') + _l(50, top, 60, top) + _ind(60, 120, top, 'L')
    s += _l(120, top, 160, top) + _l(50, bot, 330, bot)
    s += f"<circle cx='160' cy='{top}' r='2.5' fill='{INK}'/>" + _l(160, top, 160, 95) + _sw(160, 110, 'S', True) + \
        _l(160, 125, 160, bot)
    s += _l(160, top, 190, top) + f"<rect x='190' y='{top - 15}' width='40' height='30' rx='3' fill='none' " \
                                  f"stroke='{INK}' stroke-width='1.5'/>" + _t(210, top + 4, 'R', size=10)
    s += _l(230, top, 300, top) + _cap(270, top, bot, 'C') + _l(300, top, 330, top)
    s += _t(338, 120, 'V<tspan font-size="8">out</tspan>', anchor='start', it=True)
    s += _arrow(65, top + 15, 115, top + 15) + _t(90, top + 30, 'i<tspan font-size="8">L</tspan> = I ± ΔI/2', BLUE, 10)
    s += _t(372, 80, 'D = 1 − V<tspan font-size="8">in</tspan> / V<tspan font-size="8">out</tspan>', INK, 10,
            'start')
    s += _t(372, 96, 'S: on for D·T', INK, 10, 'start') + _t(372, 112, 'R: conducts (1 − D)·T', INK, 10, 'start')
    s += _t(372, 128, 'R = diode or sync. MOSFET', INK, 10, 'start')
    return s


def _device():
    s = _l(80, 20, 80, 75) + _sw(80, 100, 'T', True) + _l(80, 115, 80, 180)
    s += _l(40, 100, 68, 100) + _t(36, 104, 'gate', anchor='end', size=10)
    s += _arrow(60, 30, 60, 70, BLUE) + _t(54, 52, 'i<tspan font-size="8">F</tspan>', BLUE, 11, 'end')
    s += _arrow(120, 70, 120, 30, GREEN) + _t(126, 52, 'i<tspan font-size="8">R</tspan>', GREEN, 11, 'start')
    s += _t(140, 104, 'v (blocking V<tspan font-size="8">sw</tspan>)', INK, 10, 'start')
    x0, y0, w = 220, 150, 180
    s += _l(x0, y0, x0 + w, y0, w=1) + _l(x0, y0, x0, 30, w=1) + _t(x0 + w, y0 + 14, 't', it=True)
    s += (f"<polyline points='{x0},{y0} {x0},{70} {x0 + 70},{70} {x0 + 70},{y0} {x0 + 110},{y0} {x0 + 110},{y0 + 25} "
          f"{x0 + 140},{y0 + 25} {x0 + 140},{y0} {x0 + w},{y0}' fill='none' stroke='{BLUE}' stroke-width='2'/>")
    s += _t(x0 + 35, 62, 'I<tspan font-size="8">F</tspan> for D<tspan font-size="8">F</tspan>·T', BLUE, 10)
    s += _t(x0 + 125, y0 + 38, '−I<tspan font-size="8">R</tspan> for D<tspan font-size="8">R</tspan>·T', GREEN, 10)
    s += _t(x0 + 2, 82, 'I<tspan font-size="8">on</tspan>', RED, 9, 'start') + \
        _t(x0 + 72, 82, 'I<tspan font-size="8">off</tspan>', RED, 9, 'start')
    s += _t(x0 + 90, 24, 'one switching period T = 1/f<tspan font-size="8">s</tspan>', INK, 10)
    return s


DRAW = {'B6': _b6, 'B4': _b4, 'Buck': _buck, 'Boost': _boost, 'Device': _device}


def svg(topo):
    """Inline SVG schematic of a topology ('' for an unknown one)."""
    f = DRAW.get(topo)
    if f is None:
        return ''
    return (f"<svg viewBox='0 0 {W + 120} {H}' width='100%' style='max-width:560px' "
            f"xmlns='http://www.w3.org/2000/svg' font-family='sans-serif'>{f()}</svg>")
