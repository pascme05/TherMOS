#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfNet
# Date:         16.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This module defines the networks of the PF mode (physics-informed field network). All networks return the temperature
rise scaled with the temperature scale ΔT of the training data, θ̂/ΔT (-).
1) pfMlp: fully connected network with tanh activations (smooth, so that the second derivatives of the heat equation
   exist). Optional Fourier features: every input x̂ ∈ [−1, 1] is extended by sin(2^i·π·x̂) and cos(2^i·π·x̂),
   i = 0 … nF − 1, which lets the network represent sharp changes (thin layers, chip edges).
2) pfOperator (variant 1): the field is a sum of K products of a time function and a space function

       θ̂(x, t) = ΔT·Σ_k b_k(t)·φ_k(x),      k = 1 … K

   Branch: sequence network (dlNet, seq2seq) from the source amplitudes p(t) (batch x W x S) to the coefficients b(t)
   (batch x W x K); it carries the thermal memory. Trunk: pfMlp from the scaled position x̂ (n x d) to the basis
   functions φ(x) (n x K). The trunk plays the role of the POD modes and the branch the role of the state-space model
   of the PS mode, both are learned together.
3) pfCoord (variant 2): pfMlp from (x̂, t̂) (n x (d + 1)) to θ̂/ΔT (n), t̂ = 2·t/t_end − 1.
Inputs:     1) branch:  dlNet (S inputs, K outputs, seq2seq)
            2) d:       number of space dimensions (2 or 3)
            3) K:       number of basis functions
            4) width:   neurons per hidden layer
            5) depth:   number of hidden layers
            6) nF:      number of Fourier frequencies per input (0: none)
Outputs:    1) net:     torch module

V1.0: 16.11.2026, Pascal Schirmer
- Initial version
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
import math
import torch
import torch.nn as nn


#######################################################################################################################
# Fully Connected Network
#######################################################################################################################
class pfMlp(nn.Module):
    def __init__(self, nIn, nOut, width=64, depth=4, nF=0):
        super().__init__()
        self.nF = int(nF)
        if self.nF > 0:
            self.register_buffer('freq', math.pi * 2.0 ** torch.arange(0, self.nF, dtype=torch.float32))
        c = nIn * (1 + 2 * self.nF)
        layers = []
        for _ in range(0, max(int(depth), 1)):
            layers += [nn.Linear(c, int(width)), nn.Tanh()]
            c = int(width)
        layers.append(nn.Linear(c, nOut))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.nF > 0:
            a = x[..., None] * self.freq
            x = torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)
        return self.net(x)


#######################################################################################################################
# Operator Network
#######################################################################################################################
class pfOperator(nn.Module):
    def __init__(self, branch, d, K, width=64, depth=4, nF=0):
        super().__init__()
        self.branch = branch
        self.trunk = pfMlp(d, K, width, depth, nF)
        self.K = int(K)

    def coeff(self, p):
        """Coefficients b (batch x W x K) for the source amplitudes p (batch x W x S)."""
        return self.branch(p)

    def basis(self, x):
        """Basis functions φ (n x K) at the scaled positions x̂ (n x d)."""
        return self.trunk(x)

    def forward(self, p, x):
        """θ̂/ΔT (batch x W x n)."""
        return self.coeff(p) @ self.basis(x).T


#######################################################################################################################
# Coordinate Network
#######################################################################################################################
class pfCoord(nn.Module):
    def __init__(self, d, width=64, depth=4, nF=0):
        super().__init__()
        self.mlp = pfMlp(d + 1, 1, width, depth, nF)

    def forward(self, z):
        """θ̂/ΔT (n) for z = (x̂, t̂) (n x (d + 1))."""
        return self.mlp(z)[..., 0]


#######################################################################################################################
# References
#######################################################################################################################
# [1] L. Lu et al., Learning nonlinear operators via DeepONet, Nature Machine Intelligence 3, 2021.
# [2] M. Tancik et al., Fourier features let networks learn high frequency functions, NeurIPS 2020.
