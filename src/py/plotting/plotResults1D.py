#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         plotResults1D
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
This function plots the results of the 1D models:
1) Input data of the first training/validation/testing profile (losses and temperatures).
2) Optional (setup['featRank'] = 1): box plot of the training inputs, Pearson correlation of the inputs and outputs
   and feature ranking with the mutual information (MATLAB: ReliefF).
3) Per model: measured and predicted temperature, error e = T - T̂ (K) over time for every output channel, scatter of
   prediction vs. measurement, residual distribution and the error metrics.
Inputs:     1) data:    training, validation and testing data
            2) pred:    dictionary of the predictions {tag: data}
            3) grt:     dictionary of the ground truth {tag: data}
            4) results: dictionary of the performance results {tag: results}
            5) setup:   includes all simulation variables
Outputs:    None

V1.0: 07.11.2026, Pascal Schirmer
- Python port of plotResults1D.m; time axis in seconds, units on all axes, one figure per model, common scale in the
  scatter plot and legends (F-PL3, F-PL4)
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
# Additional Functions
#######################################################################################################################
def _names(setup, key, n, pre):
    lst = list(setup.get(key, []))
    return [lst[i] if i < len(lst) else pre + str(i + 1) for i in range(0, n)]


#######################################################################################################################
# Main Function
#######################################################################################################################
def plotResults1D(data, pred, grt, results, setup):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    nIn = np.shape(data['tr']['X'])[1]
    nOut = np.shape(data['tr']['y'])[1]
    namesIn = _names(setup, 'inp', nIn, 'P')
    namesOut = _names(setup, 'out', nOut, 'T')

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Input Data
    # ==============================================================================
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex='col')
    fig.suptitle('Input Data (first profile of every split)')
    for j, (s, t) in enumerate((('tr', 'Training'), ('vl', 'Validation'), ('te', 'Testing'))):
        d = data[s]
        X, y = np.asarray(d['X2'][0]), np.asarray(d['y2'][0])
        tt = np.arange(len(X)) * d['Ts']
        ax[0, j].plot(tt, X)
        ax[0, j].set_title(t + ' Data-1')
        ax[0, j].set_ylabel('Input (W)')
        ax[0, j].legend(namesIn, fontsize=8)
        ax[1, j].plot(tt, y)
        ax[1, j].set_ylabel('Temperature (°C)')
        ax[1, j].set_xlabel('Time (sec)')
        ax[1, j].legend(namesOut, fontsize=8)
        for a in ax[:, j]:
            a.grid(True)
    fig.tight_layout()

    # ==============================================================================
    # Feature Analysis
    # ==============================================================================
    if setup.get('featRank', 0) == 1:
        from sklearn.feature_selection import mutual_info_regression
        X, y = np.asarray(data['tr']['X']), np.asarray(data['tr']['y'])
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle('Input Feature Analysis')
        ax[0].boxplot(X, tick_labels=namesIn)
        ax[0].set_title('Training Input Features')
        R = np.corrcoef(np.hstack([X, y]).T)[0:nIn, nIn:]
        im = ax[1].imshow(R, vmin=-1, vmax=1, cmap='coolwarm')
        ax[1].set_xticks(range(nOut), namesOut)
        ax[1].set_yticks(range(nIn), namesIn)
        ax[1].set_title('Pearson Correlation Input/Output (-)')
        fig.colorbar(im, ax=ax[1])
        ax[2].bar(namesIn, mutual_info_regression(X, y[:, 0], random_state=42))
        ax[2].set_title('Feature Ranking (Mutual Information with ' + namesOut[0] + ')')
        fig.tight_layout()

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    for tag in pred:
        yT, yP = np.asarray(grt[tag]['y']), np.asarray(pred[tag]['y'])
        n = yT.shape[1]
        tt = np.arange(len(yT)) * grt[tag]['Ts']
        e = yT - yP

        # ==============================================================================
        # Time Domain
        # ==============================================================================
        fig, ax = plt.subplots(2, n, figsize=(5 * n, 6), sharex=True, squeeze=False)
        fig.suptitle('Prediction of the ' + tag.upper() + ' model (testing data)')
        for i in range(0, n):
            ax[0, i].plot(tt, yT[:, i], 'k', label='Measured')
            ax[0, i].plot(tt, yP[:, i], 'r--', label='Predicted')
            ax[0, i].set_title(namesOut[i] if n > 1 else namesOut[0])
            ax[0, i].set_ylabel('Temperature (°C)')
            ax[0, i].legend()
            ax[1, i].plot(tt, e[:, i], 'b')
            ax[1, i].set_ylabel('Error T - T̂ (K)')
            ax[1, i].set_xlabel('Time (sec)')
            for a in ax[:, i]:
                a.grid(True)
        fig.tight_layout()

        # ==============================================================================
        # Error Analysis
        # ==============================================================================
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle('Error Analysis of the ' + tag.upper() + ' model (testing data)')
        ax[0].scatter(yT.ravel(), yP.ravel(), s=4)
        lim = [np.nanmin(yT), np.nanmax(yT)]
        ax[0].plot(lim, lim, 'k--')
        ax[0].set_xlabel('Measured Temperature (°C)')
        ax[0].set_ylabel('Predicted Temperature (°C)')
        ax[0].set_title('Prediction vs. Measurement')
        for i in range(0, n):
            ax[1].hist(e[:, i], bins=50, density=True, alpha=0.5, label=namesOut[i])
        ax[1].set_xlabel('Error T - T̂ (K)')
        ax[1].set_ylabel('Probability Density (1/K)')
        ax[1].set_title('Residual Distribution')
        ax[1].legend()
        r = results[tag]['err']['tot']
        mets = ['MAE', 'RMSE', 'MAX']
        w = 0.8 / n
        for i in range(0, n):
            ax[2].bar(np.arange(3) + i * w, [r[m][i] for m in mets], w, label=namesOut[i])
        ax[2].set_xticks(np.arange(3) + 0.4 - w / 2, ['MAE (K)', 'RMSE (K)', 'MAX (K)'])
        ax[2].set_title('Error Metrics (all samples)')
        ax[2].legend()
        for a in ax:
            a.grid(True)
        fig.tight_layout()

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    plt.show()


#######################################################################################################################
# References
#######################################################################################################################
