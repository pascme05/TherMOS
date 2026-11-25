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
4) pfModal (parametric study, pfParamFit): one network for many boards; space functions per chip that depend on the
   chip descriptor and the operating condition, and first-order lags with learned rates (modal structure).
Inputs:     1) branch:  dlNet (S inputs, K outputs, seq2seq)
            2) d:       number of space dimensions (2 or 3)
            3) K:       number of basis functions
            4) width:   neurons per hidden layer
            5) depth:   number of hidden layers
            6) nF:      number of Fourier frequencies per input (0: none)
Outputs:    1) net:     torch module

V1.0: 16.11.2026, Pascal Schirmer
- Initial version

V1.1: 21.11.2026, Pascal Schirmer
- Parametric modal network pfModal (with an optional bound τ_cap of the time constants)

V1.2: 22.11.2026, Pascal Schirmer
- Physics-informed DeepONet baseline pfDeepONet (trunk of pfModal, recurrent branch instead of the exact lags)

V1.3: 24.11.2026, Pascal Schirmer
- Log-linear condition heads of pfModal (lin = True): log λ_k += ω_kᵀ·μ̂ and φ_k ·= exp(ν_kᵀ·μ̂) with the scaled
  condition μ̂ (last nMu context inputs), outside the tanh layers and zero at the start, i.e., the rates and amplitudes
  follow a power law of h beyond the training range instead of saturating
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
# Parametric Modal Network
#######################################################################################################################
class pfModal(nn.Module):
    """
    Parametric operator with the structure of the modal solution of a linear board (pfParamFit): every chip c adds K
    space functions weighted with first-order lags of its loss,

        θ̂(x, t)/ΔT = Σ_c Σ_k φ_k(x, z_c, μ)·b_{c,k}(t),      db_{c,k}/dt = λ_k(z_c, μ)·(p̂_c(t) − b_{c,k})

    with the scaled loss p̂_c = P_c/P_ref, the chip descriptor z_c and the scaled condition μ̂ (context, nCtx). Trunk:
    Fourier features of the nSp space inputs (absolute and relative to the chip) plus the context → φ (K); rates:
    small network of the context → λ_k = 1/τ_cap + exp(a_k + Δa_k) with a_k log-spaced over 1/τ_max … 1/τ_min
    (Δa = 0 at the start). The floor 1/τ_cap bounds the slowest mode (no time constant above τ_cap): a mode slower
    than the training profiles is not seen in them and acts like an integrator over long profiles (drift). b is the
    exact zero-order-hold solution (coeff), so it is stable for any profile length.
    """
    def __init__(self, nSp=5, nCtx=11, K=16, width=128, depth=4, nF=4, tau=(0.02, 40.0), rel=True, tauCap=None,
                 lin=False, nMu=4):
        super().__init__()
        # lin: log-linear heads of the scaled condition μ̂ (last nMu context inputs, log h, log k_diel, ...) outside the
        # tanh layers: log λ_k and log of a gain g_k of φ_k get a linear term in μ̂ (zero at the start), i.e., the
        # rates and amplitudes follow a power law of h beyond the training range instead of saturating.
        self.lin, self.nMu = bool(lin), int(nMu)
        if self.lin:
            self.linR = nn.Linear(self.nMu, int(K))
            self.linG = nn.Linear(self.nMu, int(K))
            for m in (self.linR, self.linG):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.nF, self.K, self.rel = int(nF), int(K), bool(rel)
        self.lamMin = 0.0 if not tauCap else 1.0 / float(tauCap)
        if self.nF > 0:
            self.register_buffer('freq', math.pi * 2.0 ** torch.arange(0, self.nF, dtype=torch.float32))
        c = nSp * (1 + 2 * self.nF) + nCtx
        layers = []
        for _ in range(0, max(int(depth), 1)):
            layers += [nn.Linear(c, int(width)), nn.Tanh()]
            c = int(width)
        layers.append(nn.Linear(c, self.K))
        self.trunk = nn.Sequential(*layers)
        self.rate = nn.Sequential(nn.Linear(nCtx, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.K))
        nn.init.zeros_(self.rate[-1].weight)
        nn.init.zeros_(self.rate[-1].bias)
        self.register_buffer('a0', -torch.linspace(math.log(tau[1]), math.log(tau[0]), self.K))

    def basis(self, xs, ctx):
        """Space functions φ (... x n x K) for the space inputs xs (... x n x nSp) and the context ctx (... x nCtx)."""
        if not self.rel:
            xs = torch.cat([xs[..., 0:3], torch.zeros_like(xs[..., 3:])], dim=-1)
        if self.nF > 0:
            a = xs[..., None] * self.freq
            xs = torch.cat([xs, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)
        phi = self.trunk(torch.cat([xs, ctx[..., None, :].expand(*xs.shape[:-1], ctx.shape[-1])], dim=-1))
        if self.lin:
            phi = phi * torch.exp(self.linG(ctx[..., -self.nMu:]))[..., None, :]
        return phi

    def rates(self, ctx):
        """Decay rates λ (... x K) (1/s)."""
        a = self.a0 + self.rate(ctx)
        if self.lin:
            a = a + self.linR(ctx[..., -self.nMu:])
        return self.lamMin + torch.exp(a)

    @staticmethod
    def coeff(p, lam, Ts):
        """
        Coefficients b (... x L x K) for the scaled losses p (... x L) and rates λ (... x K): zero-order hold with the
        loss p_k on the interval (t_{k−1}, t_k] and b_0 = 0,

            b_k = a·b_{k−1} + (1 − a)·p_k,      a = e^{−λ·Ts}

        evaluated as a convolution with the kernel (1 − a)·a^n (FFT, exact).
        """
        L = p.shape[-1]
        n = 1 << int(math.ceil(math.log2(2 * L)))
        a = torch.exp(-lam * Ts)                                                    # ... x K
        k = torch.arange(L, device=p.device, dtype=p.dtype)
        h = (1 - a)[..., None, :] * torch.exp(k[:, None] * torch.log(a.clamp_min(1e-30))[..., None, :])
        p0 = torch.cat([torch.zeros_like(p[..., 0:1]), p[..., 1:]], dim=-1)          # p_0 does not act (θ(0) = 0)
        b = torch.fft.irfft(torch.fft.rfft(p0, n=n)[..., None] * torch.fft.rfft(h, n=n, dim=-2), n=n, dim=-2)
        return b[..., 0:L, :]

    def seq(self, p, ctx, Ts):
        """Coefficients b (... x L x K) and their rates db/dt (... x L x K) for the scaled losses p (... x L)."""
        lam = self.rates(ctx)
        b = self.coeff(p, lam, Ts)
        p0 = torch.cat([torch.zeros_like(p[..., 0:1]), p[..., 1:]], dim=-1)
        return b, lam[..., None, :] * (p0[..., None] - b)


#######################################################################################################################
# Physics-Informed DeepONet (baseline)
#######################################################################################################################
class pfDeepONet(pfModal):
    """
    Physics-informed DeepONet baseline [1, 3] with the same trunk as pfModal (space functions per chip φ_k(x, z_c, μ))
    and a learned branch instead of the exact lags: a recurrent network (GRU, causal, any profile length) of the scaled
    loss p̂_c(t) and the context gives the coefficients

        θ̂(x, t)/ΔT = Σ_c Σ_k φ_k(x, z_c, μ)·b_{c,k}(t),      b_c(t) = GRU(p̂_c(0 … t), z_c, μ)

    with b_0 = 0 (θ(0) = 0) and db/dt from the backward difference (b_n − b_{n−1})/T_s (implicit Euler).
    """
    def __init__(self, nSp=5, nCtx=11, K=16, width=128, depth=4, nF=4, rel=True, hid=64, layers=2, **kw):
        super().__init__(nSp, nCtx, K, width, depth, nF, rel=rel)
        del self.rate
        self.gru = nn.GRU(1 + nCtx, int(hid), int(layers), batch_first=True)
        self.out = nn.Linear(int(hid), self.K)

    def seq(self, p, ctx, Ts):
        sh = p.shape                                                                # ... x L
        x = torch.cat([p[..., None], ctx[..., None, :].expand(*sh, ctx.shape[-1])], dim=-1)
        x = torch.cat([torch.zeros_like(x[..., 0:1, :]), x[..., 1:, :]], dim=-2)    # p_0 does not act
        h, _ = self.gru(x.reshape(-1, sh[-1], x.shape[-1]))
        y = self.out(h).reshape(*sh, self.K)
        b = torch.cat([torch.zeros_like(y[..., 0:1, :]), y[..., 1:, :] - y[..., 0:1, :]], dim=-2)   # b_0 = 0
        db = torch.cat([torch.zeros_like(b[..., 0:1, :]), b[..., 1:, :] - b[..., :-1, :]], dim=-2) / Ts
        return b, db


#######################################################################################################################
# References
#######################################################################################################################
# [1] L. Lu et al., Learning nonlinear operators via DeepONet, Nature Machine Intelligence 3, 2021.
# [2] M. Tancik et al., Fourier features let networks learn high frequency functions, NeurIPS 2020.
# [3] S. Wang, H. Wang, P. Perdikaris, Learning the solution operator of parametric partial differential equations
#     with physics-informed DeepONets, Science Advances 7, 2021.
