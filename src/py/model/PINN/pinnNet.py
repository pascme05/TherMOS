#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pinnNet
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
This class wraps the sequence network of the PINN (dlNet: built-in CNN / LSTM / CNN-LSTM or a user-defined
architecture, seq2seq). The network predicts the n states x of the physics network (pinnPhys) as temperature rises
scaled with ΔT (x = ΔT·f(X)); the outputs are T̂ = T_ref + O·x. The inputs and outputs of the module are in the units
of the data (normalised if normX/normY are active); the normalisation of TherMOS is undone inside, so that the
physics works on raw units:

    X_raw = a_X·X + b_X,      T_data = (T̂ - b_y) / a_y

With x0 = 1 the measured temperature rise at the start of the window, (T(0) - T_ref)/ΔT, is appended to the inputs
as M extra channels (repeated over the window): the initial thermal state of the window.

Inputs:     1) body:    dlNet (F + M·x0 inputs, n outputs, seq2seq)
            2) O:       output matrix (M x n)
            3) dT:      temperature scale ΔT (K)
            4) aff:     affine maps of the normalisation {'aX', 'bX', 'ay', 'by'}
            5) ref:     reference settings (TrefSrc, Tref, M, init)
            6) x0:      initial state as input (0/1)
Outputs:    1) net:     torch module with forward(X, R, y0) -> T (data units), outputs(X, R, y0) -> states x (K),
                        T̂ (raw) and the raw inputs
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
import torch
import torch.nn as nn


#######################################################################################################################
# Class
#######################################################################################################################
class pinnNet(nn.Module):
    def __init__(self, body, O, dT, aff, ref, x0=0):
        super().__init__()
        self.body = body
        self.x0 = int(x0)
        self.register_buffer('O', torch.as_tensor(O, dtype=torch.float32))
        self.register_buffer('aX', torch.as_tensor(aff['aX'], dtype=torch.float32))
        self.register_buffer('bX', torch.as_tensor(aff['bX'], dtype=torch.float32))
        self.register_buffer('ay', torch.as_tensor(aff['ay'], dtype=torch.float32))
        self.register_buffer('by', torch.as_tensor(aff['by'], dtype=torch.float32))
        self.dT = float(dT)
        self.TrefSrc, self.Tref, self.M, self.init = int(ref['TrefSrc']), float(ref['Tref']), int(ref['M']), \
            int(ref.get('init', 1))

    # ==============================================================================
    # Units
    # ==============================================================================
    def raw(self, X):
        """Inputs in raw units."""
        return X * self.aX + self.bX

    def toData(self, T):
        return (T - self.by) / self.ay

    def fromData(self, y):
        return y * self.ay + self.by

    def reference(self, Xr, R=None):
        """Reference temperature of every output (B x W x M) from the raw inputs Xr and the reference channels R."""
        if self.init != 1:                                              # outputs are rises (offset removed)
            ref = torch.zeros_like(Xr[..., 0:1])
        elif self.TrefSrc > 0:
            ref = Xr[..., self.TrefSrc - 1:self.TrefSrc]
        elif self.TrefSrc < 0 and R is not None:
            ref = R[..., 0:self.M] if R.shape[-1] >= self.M else R[..., 0:1]
        elif self.TrefSrc < 0:
            ref = torch.zeros_like(Xr[..., 0:1])
        else:
            ref = torch.zeros_like(Xr[..., 0:1]) + self.Tref
        return ref.expand(*Xr.shape[:-1], self.M)

    # ==============================================================================
    # Forward
    # ==============================================================================
    def augment(self, X, Xr, R, y0):
        """
        Inputs of the body: the features and (x0 = 1) the measured temperature rise at the start of the window
        (y0: B x M, data units) over the reference, scaled with ΔT and repeated over the window.
        """
        if not getattr(self, 'x0', 0):
            return X
        if y0 is None:
            raise ValueError("the PINN needs the initial temperatures y0 (x0 = 1)")
        d = (self.fromData(y0) - self.reference(Xr[:, 0:1], None if R is None else R[:, 0:1])[:, 0]) / self.dT
        return torch.cat([X, d[:, None, :].expand(X.shape[0], X.shape[1], self.M)], dim=-1)

    def outputs(self, X, R=None, y0=None):
        """States x (K), outputs T̂ (°C, raw) and raw inputs."""
        Xr = self.raw(X)
        x = self.body(self.augment(X, Xr, R, y0)) * self.dT
        T = self.reference(Xr, R) + x @ self.O.T
        return x, T, Xr

    def forward(self, X, R=None, y0=None):
        return self.toData(self.outputs(X, R, y0)[1])


#######################################################################################################################
# References
#######################################################################################################################
