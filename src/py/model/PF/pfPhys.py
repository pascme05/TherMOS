#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         pfPhys
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
This module evaluates the physics residual of the PF mode (physics-informed field network). The physics is the heat
equation for the temperature rise θ = T − T_a (K) over the ambient

    ρc·∂θ/∂t = Σ_i ∂/∂x_i (k_i·∂θ/∂x_i) − h_v·θ + Q

with the heat capacity ρc (J/m³K), the conductivities k_i (W/mK) in the directions i = x, y(, z), the distributed heat
sink h_v (W/m³K) and the heat generation Q (W/m³). Both residuals are dimensionless and read as a temperature error
relative to the temperature scale ΔT of the training data:

1) Strong form with automatic differentiation (phys = 1), at collocation points x_c inside the domain

       r = (ρc·∂θ̂/∂t − Σ_i k_i·∂²θ̂/∂x_i² + h_v·θ̂ − Q) / (g·ΔT),      g = Σ_i 2·k_i/ℓ_i² + h_v   (W/m³K)

   with ℓ_i (m) half the extent of the domain in direction i: g is the conductance per volume of a temperature
   error that is spread over the whole domain, so r = 0.01 means that the heat balance of the point is violated as
   much as a smooth temperature error of 1 % of ΔT would violate it. Errors on the scale of the mesh weigh more (by
   (ℓ/h)², h: node spacing): a mesh-free residual cannot be scaled correctly for all error shapes at once, which is
   why this form converges slower than 2).
   The material of a collocation point is the material of its nearest node (piece-wise constant), i.e. the term
   ∂k/∂x·∂θ/∂x at material interfaces is neglected, and boundary conditions are not part of the residual (the FEM
   data do not store them): the behaviour at interfaces and boundaries is learned from the temperature data.
   The space derivatives are taken in scaled coordinates, ∂²/∂x_i² = s_i²·∂²/∂x̂_i², κ_i = k_i·s_i².
2) Finite-volume operator of the data (phys = 2), at the nodes n of the data

       r = G⁻¹·(C·dθ̂/dt + G·θ̂ − V·Q) / ΔT = (G⁻¹·C·dθ̂/dt + θ̂ − θ_ss) / ΔT,      θ_ss = G⁻¹·V·Q

   with the capacitances C (Ws/K), the conductance matrix G (W/K, incl. the cooling at the boundary) and the control
   volumes V (m³) stored by the board designer. Exact at material interfaces and boundaries. The heat balance
   (W per node) is multiplied by G⁻¹ (one sparse factorisation, solveG): the residual is then the temperature error
   that this imbalance causes in the whole field, r = 0.01 is an error of 1 % of ΔT, also for smooth errors.
   θ_ss is the steady-state rise for the present sources; for the encoded sources θ_ss = Σ_j Θ_j·p_j with the
   steady-state fields Θ = G⁻¹·(V·U) (N x S) of the source footprints.
The functions take the coefficients / time derivatives they are given; the time discretisation (residual in the
middle of every sampling step, sources of the step) is chosen in pfFit.
Functions:  basisLap(trunk, x, kap):    Σ_i κ_i·∂²φ_k/∂x̂_i² (n x K) of the trunk of the operator network
            resOperator(...):           residual 1) of the operator network (batch x W x n)
            resOperatorFV(...):         residual 2) of the operator network (batch x W x n)
            solveG(lu):                 differentiable function x → G⁻¹·x for the factorisation lu of G
            coordDeriv(net, z):         value (n), gradient (n x (d + 1)) and second derivatives (n x (d + 1)) of the
                                        coordinate network with respect to its inputs z = (x̂, t̂)
            coordRate(net, x, t):       value and derivative with respect to t̂ (nT x N) of the coordinate network at
                                        all nodes x for the times t

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
import torch
from torch.func import vmap, hessian, grad, jvp


#######################################################################################################################
# Operator Network
#######################################################################################################################
def basisLap(trunk, x, kap):
    """Conduction operator of the basis functions Σ_i κ_i·∂²φ_k/∂x̂_i² (n x K); x (n x d), κ (n x d)."""
    H = vmap(hessian(lambda v: trunk(v)))(x)                            # n x K x d x d
    return torch.einsum('nkii,ni->nk', H, kap)


def resOperator(b, db, phi, lap, p, rc, hv, Uc, g, dT):
    """
    Residual of the heat equation (batch x W x n, dimensionless) for the coefficients b and their time derivative db
    (batch x W x K), the basis φ and its conduction operator (n x K), the source amplitudes p (batch x W x S) and
    the material ρc, h_v, local conductance g (n) and source footprints U (n x S) at the collocation points.
    """
    w = (1.0 / g)[:, None]
    return db @ (phi * rc[:, None] * w).T + b @ ((phi * hv[:, None] - lap) * w).T - p @ (Uc * w / dT).T


def resOperatorFV(b, db, phi, Mphi, p, Tss, dT):
    """
    Residual of the finite-volume operator (batch x W x n, dimensionless): φ and G⁻¹·C·φ (n x K) at the selected
    nodes, steady-state fields Θ = G⁻¹·V·U (n x S) of the source footprints.
    """
    return db @ Mphi.T + b @ phi.T - p @ (Tss / dT).T


def solveG(lu):
    """
    Function x → G⁻¹·x (x: N x m tensor on any device) with the sparse LU factorisation lu of the symmetric
    conductance matrix G (scipy splu); the gradient is G⁻ᵀ·grad, so the solve can be part of the loss.
    """
    class Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return torch.as_tensor(lu.solve(x.detach().cpu().double().numpy()), dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, g):
            return torch.as_tensor(lu.solve(g.detach().cpu().double().numpy(), trans='T'), dtype=g.dtype,
                                   device=g.device)
    return Solve.apply


#######################################################################################################################
# Coordinate Network
#######################################################################################################################
def coordDeriv(net, z):
    """Value f (n), gradient ∂f/∂z (n x (d + 1)) and second derivatives ∂²f/∂z_i² (n x (d + 1))."""
    fnc = lambda v: net(v[None])[0]
    f = net(z)
    g = vmap(grad(fnc))(z)
    H = vmap(hessian(fnc))(z)
    return f, g, torch.diagonal(H, dim1=-2, dim2=-1)


def coordRate(net, x, t):
    """Value f and ∂f/∂t̂ (nT x N) at the nodes x (N x d) for the scaled times t (nT)."""
    def fnc(tt):
        z = torch.cat([x[None].expand(len(tt), -1, -1), tt[:, None, None].expand(-1, x.shape[0], 1)], dim=-1)
        return net(z)
    return jvp(fnc, (t,), (torch.ones_like(t),))


#######################################################################################################################
# References
#######################################################################################################################
# [1] S. Cai et al., Physics-informed neural networks for heat transfer problems, J. Heat Transfer 143 (6), 2021.
# [2] M. Raissi et al., Physics-informed neural networks, J. Comput. Phys. 378, 2019.
