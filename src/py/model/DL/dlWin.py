#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         dlWin
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function cuts the input (and output) sequence of one profile into windows of W samples. For seq2point (seq > 0)
window i contains the samples i-(seq-1) ... i+(W-seq) and the target is y(i), i.e. the target is the seq-th sample of
the window; the profile is padded with its first/last value so that there is one window per sample. For seq2seq
(seq = 0) the windows are consecutive blocks with the given stride, the last block is padded with the last value.
Inputs:     1) X:       input features (Nt x F)
            2) y:       output (Nt x M) or None
            3) W:       window length (samples)
            4) seq:     (0) seq2seq, (>0) position of the target in the window (1..W)
            5) stride:  step between windows (seq2seq only)
            6) pad:     (1) pad (inference), (0) only complete windows (training)
Outputs:    1) Xw:      windows (Nw x W x F)
            2) yw:      targets (Nw x M) or (Nw x W x M), None if y is None

V1.1: 07.11.2026, Pascal Schirmer
- Seq2seq windows taken from a sliding window view; before, y was converted once per window and every window kept
  its own full copy of y alive (out of memory for long profiles, e.g. tutorial_PI with 630k samples)
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
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


#######################################################################################################################
# Main Function
#######################################################################################################################
def dlWin(X, y, W, seq, stride=1, pad=1):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    X = np.asarray(X, dtype=np.float32)
    Nt = X.shape[0]

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Seq2Point
    # ==============================================================================
    if seq > 0:
        pre, post = seq - 1, W - seq
        if pad:
            Xp = np.concatenate([np.repeat(X[0:1], pre, 0), X, np.repeat(X[-1:], post, 0)])
            idx = np.arange(Nt)
        else:
            Xp = X
            idx = np.arange(0, Nt - W + 1, stride)
        Xw = sliding_window_view(Xp, W, axis=0).transpose(0, 2, 1)[idx]
        yw = None if y is None else np.asarray(y, dtype=np.float32)[idx + (0 if pad else pre)]

    # ==============================================================================
    # Seq2Seq
    # ==============================================================================
    else:
        if pad:
            n = int(np.ceil(Nt / W))
            Xp = np.concatenate([X, np.repeat(X[-1:], n * W - Nt, 0)])
            Xw = Xp.reshape(n, W, -1)
            yw = None
        else:
            starts = np.arange(0, Nt - W + 1, stride)
            Xw = sliding_window_view(X, W, axis=0).transpose(0, 2, 1)[starts]
            yw = None if y is None else                 sliding_window_view(np.asarray(y, dtype=np.float32), W, axis=0).transpose(0, 2, 1)[starts]

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [np.ascontiguousarray(Xw), yw]


#######################################################################################################################
# References
#######################################################################################################################
