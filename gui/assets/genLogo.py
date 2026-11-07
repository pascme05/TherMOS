#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         genLogo
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
This function draws the TherMOS logo: a chip on a board whose heat spreads in isotherms from red (hot) to blue
(cold), plus the word mark. It writes gui/assets/logo.png (sidebar, wide) and gui/assets/logo_icon.png (browser tab,
collapsed sidebar).
Usage:      python gui/assets/genLogo.py
Inputs:     None
Outputs:    None
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge

#######################################################################################################################
# Parameters
#######################################################################################################################
OUT = os.path.dirname(os.path.abspath(__file__))
INK, RED, NAVY = '#0b0b0b', '#b32d2d', '#0d366b'


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _icon(ax, x0, y0, s):
    """Icon in the square (x0, y0, s): rounded tile, isotherms below a chip, board line."""
    ax.add_patch(FancyBboxPatch((x0 + 0.04 * s, y0 + 0.04 * s), 0.92 * s, 0.92 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.2 * s}", fc='white', ec=NAVY,
                                lw=0.035 * s * 72 / 25))
    cmap = plt.get_cmap('RdBu_r')
    cx, cy = x0 + 0.5 * s, y0 + 0.66 * s
    n = 6
    for i in range(n, 0, -1):
        r = 0.08 * s + 0.055 * s * i
        ax.add_patch(Wedge((cx, cy), r, 180, 360, fc=cmap(1 - (i - 0.5) / n * 0.95), ec='white', lw=0.6))
    ax.add_patch(Rectangle((x0 + 0.14 * s, cy - 0.006 * s), 0.72 * s, 0.03 * s, fc=NAVY, ec='none'))
    ax.add_patch(Rectangle((cx - 0.12 * s, cy + 0.024 * s), 0.24 * s, 0.1 * s, fc=INK, ec='none'))
    for k in range(-2, 3):
        ax.add_patch(Rectangle((cx + k * 0.045 * s - 0.008 * s, cy + 0.124 * s), 0.016 * s, 0.04 * s, fc=INK,
                               ec='none'))


#######################################################################################################################
# Main Function
#######################################################################################################################
def genLogo():
    ###################################################################################################################
    # Icon
    ###################################################################################################################
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    _icon(ax, 0, 0, 1)
    fig.savefig(os.path.join(OUT, 'logo_icon.png'), transparent=True)
    plt.close(fig)

    ###################################################################################################################
    # Word Mark
    ###################################################################################################################
    W, H = 6.0, 1.5
    fig = plt.figure(figsize=(W, H), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    _icon(ax, 0.05, 0.05, 1.4)
    t1 = ax.text(1.62, 0.62, 'Ther', fontsize=44, fontweight='bold', color=INK, va='baseline', family='DejaVu Sans')
    fig.canvas.draw()
    bb = t1.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.02, 0.62, 'MOS', fontsize=44, fontweight='bold', color=RED, va='baseline',
            family='DejaVu Sans')
    ax.text(1.66, 0.28, 'Thermal Model Order Reduction & Simulation', fontsize=10.5, color='#52514e',
            va='baseline', family='DejaVu Sans')
    fig.savefig(os.path.join(OUT, 'logo.png'), transparent=True)
    plt.close(fig)


#######################################################################################################################
# Calculations
#######################################################################################################################
if __name__ == '__main__':
    genLogo()
