#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         dlNet
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
This class defines the deep learning networks of the DL model (PyTorch). The input is a window of W samples with F
features (batch x W x F); it is normalised with the z-score of the training features (as sequenceInputLayer with
Normalization 'zscore'). The output is one value per temperature node M (seq2point, batch x M) or per sample of the
window (seq2seq, batch x W x M).
Inputs:     1) sel:     (1) CNN, (2) LSTM, (3) CNN-LSTM
            2) F:       number of input features
            3) M:       number of outputs (temperature nodes)
            4) seq:     (0) seq2seq, (>0) seq2point
            5) mu:      mean of the input features (F)
            6) sd:      standard deviation of the input features (F)
Outputs:    1) net:     torch module
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
class dlNet(nn.Module):
    ###################################################################################################################
    # Constructor
    ###################################################################################################################
    def __init__(self, sel, F, M, seq, mu, sd):
        super().__init__()
        self.sel = sel
        self.seq = seq
        self.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))

        # ==============================================================================
        # CNN
        # ==============================================================================
        if sel == 1:
            self.conv = nn.Sequential(nn.Conv1d(F, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                                      nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, M))

        # ==============================================================================
        # LSTM
        # ==============================================================================
        elif sel == 2:
            self.lstm = nn.LSTM(F, 64, batch_first=True)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M))

        # ==============================================================================
        # CNN-LSTM
        # ==============================================================================
        else:
            self.conv = nn.Sequential(nn.Conv1d(F, 32, 3, padding=1), nn.ReLU())
            self.lstm = nn.LSTM(32, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, M))

    ###################################################################################################################
    # Forward
    ###################################################################################################################
    def forward(self, x):
        # ==============================================================================
        # Normalisation
        # ==============================================================================
        x = (x - self.mu) / self.sd

        # ==============================================================================
        # Features
        # ==============================================================================
        if self.sel == 1:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h = h.mean(dim=1) if self.seq > 0 else h
        elif self.sel == 2:
            h = self.lstm(x)[0]
            h = h[:, -1] if self.seq > 0 else h
        else:
            h = self.lstm(self.conv(x.transpose(1, 2)).transpose(1, 2))[0]
            h = h[:, -1] if self.seq > 0 else h

        # ==============================================================================
        # Return
        # ==============================================================================
        return self.head(h)
