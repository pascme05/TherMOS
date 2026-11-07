#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         plotMdl
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
This function plots the structure of the trained models:
RC:     transient thermal impedance Zth(t) = Σ Rth_k·(1 - exp(-t/τ_k)) (K/W) and the RC pairs (Rth, τ, Cth).
SS/PS:  poles of the discrete system in the complex plane (unit circle = stability limit).
SF:     resistance matrix Rth (K/W).
PO/PS:  singular values σ_i of the POD (log scale), captured energy and the first spatial modes φ_i (-).
DL/PI:  training and validation loss over the iterations/epochs.
Inputs:     1) data:    training, validation and testing data
            2) mdls:    dictionary of the models {tag: mdl}
            3) setup:   includes all simulation variables
Outputs:    None

V1.0: 07.11.2026, Pascal Schirmer
- Python port of plotMdl.m; one figure per model, also for multiple selected models
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
import matplotlib.pyplot as plt


#######################################################################################################################
# Main Function
#######################################################################################################################
def plotMdl(data, mdls, setup):
    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for tag, m in mdls.items():
        # ==============================================================================
        # Foster
        # ==============================================================================
        if tag == 'rc':
            t = np.logspace(np.log10(m['Ts']), np.log10(max(m['tau']) * 10), 400)
            Zth = np.sum(m['Rth'][:, None] * (1 - np.exp(-t[None, :] / m['tau'][:, None])), axis=0)
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1]})
            fig.suptitle('Foster Network (K = %d)' % m['K'])
            ax[0].semilogx(t, Zth)
            ax[0].set_xlabel('Time (sec)')
            ax[0].set_ylabel('Zth (K/W)')
            ax[0].set_title('Transient Thermal Impedance')
            ax[0].grid(True, which='both')
            ax[1].axis('off')
            rows = [['%d' % (k + 1), '%.4g' % m['Rth'][k], '%.4g' % m['tau'][k], '%.4g' % m['Cth'][k]]
                    for k in range(0, m['K'])]
            ax[1].table(rows, colLabels=['k', 'Rth (K/W)', 'τ (sec)', 'Cth (Ws/K)'], loc='center')

        # ==============================================================================
        # State Space
        # ==============================================================================
        if 'sys' in m and isinstance(m['sys'], dict) and 'A' in m['sys']:
            lam = np.linalg.eigvals(m['sys']['A'])
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), 'k--', lw=0.8)
            ax.plot(np.real(lam), np.imag(lam), 'rx', ms=8)
            ax.set_xlabel('Real Part (-)')
            ax.set_ylabel('Imaginary Part (-)')
            ax.set_title('Poles of the discrete %s model (n = %d)' % (tag.upper(), len(lam)))
            ax.set_aspect('equal')
            ax.grid(True)

        # ==============================================================================
        # Nodal Foster
        # ==============================================================================
        if tag == 'sf':
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(m['Rth'], cmap='viridis')
            fig.colorbar(im, ax=ax, label='Rth (K/W)')
            ax.set_title('Thermal Resistance Matrix')
            ax.set_xlabel('Heated Node (-)')
            ax.set_ylabel('Node (-)')

        # ==============================================================================
        # Proper Orthogonal Decomposition
        # ==============================================================================
        if 'rPhi' in m:
            K = m['K']
            n = min(K, 4)
            fig, ax = plt.subplots(2, max(n, 2), figsize=(4 * max(n, 2), 7))
            fig.suptitle('POD of the %s model (K = %d, E = %.4f %%)' % (tag.upper(), K, m['E'] * 100))
            sv = np.asarray(m['lam'])
            ax[0, 0].semilogy(np.arange(1, len(sv) + 1), sv, 'o-')
            ax[0, 0].axvline(K, color='r', ls='--')
            ax[0, 0].set_xlabel('Mode i (-)')
            ax[0, 0].set_ylabel('Singular Value σ_i (K)')
            ax[0, 0].grid(True)
            ax[0, 1].plot(np.arange(1, len(sv) + 1), np.cumsum(sv ** 2) / np.sum(sv ** 2) * 100, 'o-')
            ax[0, 1].set_xlabel('Number of Modes K (-)')
            ax[0, 1].set_ylabel('Captured Energy Σσ² (%)')
            ax[0, 1].grid(True)
            for a in ax[0, 2:]:
                a.axis('off')
            S = m['sPhi']
            for i in range(0, n):
                F = S[..., S.shape[2] // 2, i] if S.ndim == 4 else S[..., i]
                im = ax[1, i].imshow(F, origin='lower', cmap='RdBu_r')
                ax[1, i].set_title('Spatial Mode φ_%d (-)' % (i + 1))
                fig.colorbar(im, ax=ax[1, i])
            for a in ax[1, n:]:
                a.axis('off')
            fig.tight_layout()

        # ==============================================================================
        # Deep Learning
        # ==============================================================================
        if tag in ('dl', 'pi'):
            info = m['info']
            fig, ax = plt.subplots(figsize=(9, 4.5))
            if 'iter' in info:
                ax.semilogy(info['iter'], info['loss'], lw=0.6, label='Training')
                ax.semilogy(info['valIter'], info['valLoss'], 'o-', label='Validation')
                ax.set_xlabel('Iteration (-)')
            else:
                ax.semilogy(np.arange(1, len(info['loss']) + 1), info['loss'], label='Training')
                ax.semilogy(np.arange(1, len(info['valLoss']) + 1), info['valLoss'], label='Validation')
                ax.set_xlabel('Epoch (-)')
            ax.set_ylabel('Loss (normalised units)')
            ax.set_title('Convergence of the %s model' % tag.upper())
            ax.legend()
            ax.grid(True, which='both')

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    plt.show()


#######################################################################################################################
# References
#######################################################################################################################
