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
Inputs:     1) sel:     (1) CNN, (2) LSTM, (3) CNN-LSTM (built-in networks, used if arch is None)
            2) F:       number of input features
            3) M:       number of outputs (temperature nodes)
            4) seq:     (0) seq2seq, (>0) seq2point
            5) mu:      mean of the input features (F)
            6) sd:      standard deviation of the input features (F)
            7) arch:    user-defined architecture (optional, replaces sel) {'body': [layer, ...], 'head': [units of
                        the hidden dense layers], 'pool': 'auto' | 'last' | 'mean' | 'max'}; a layer is a dict with
                        'type' ('conv', 'lstm', 'gru', 'transformer', 'dense') and 'units', 'kernel', 'dilation',
                        'heads', 'bidir', 'norm', 'act' ('relu', 'gelu', 'tanh', 'none'), 'drop'. The body works on
                        the window (batch x W x channels, 'same' padding of the convolutions); seq2point reduces the
                        window with the pooling (auto: last step after a recurrent layer, else the mean); the head is
                        applied per time step (seq2seq) or once (seq2point) and ends with a linear layer to M.
Outputs:    1) net:     torch module

V1.1: 08.11.2026, Pascal Schirmer
- User-defined architectures (layer list) and presets (ARCHS): CNN, LSTM, CNN-LSTM, GRU, stacked LSTM,
  bidirectional LSTM, temporal convolutional network (dilated), transformer encoder
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
# Presets of the user-defined architectures
#######################################################################################################################
LAYER_TYPES = ['conv', 'lstm', 'gru', 'transformer', 'dense']
ACTS = ['relu', 'gelu', 'tanh', 'none']
POOLS = ['auto', 'last', 'mean', 'max']


def _L(type, units, kernel=3, dilation=1, heads=4, bidir=False, norm=False, act='relu', drop=0.0):
    return {'type': type, 'units': units, 'kernel': kernel, 'dilation': dilation, 'heads': heads, 'bidir': bidir,
            'norm': norm, 'act': act, 'drop': drop}


ARCHS = {
    'CNN': {'body': [_L('conv', 64, 5, norm=True), _L('conv', 32, 3)], 'head': [32], 'pool': 'mean',
            'note': "Two 1D convolutions (local filters over time), mean over the window. Same as the built-in CNN."},
    'LSTM': {'body': [_L('lstm', 64)], 'head': [64], 'pool': 'last',
             'note': "One LSTM layer carries a state through the window. Same as the built-in LSTM."},
    'CNN-LSTM': {'body': [_L('conv', 32, 3), _L('lstm', 64, act='none')], 'head': [32], 'pool': 'last',
                 'note': "Convolution for fast transients, LSTM for the slow heating. Same as the built-in CNN + "
                         "LSTM."},
    'GRU': {'body': [_L('gru', 64)], 'head': [64], 'pool': 'last',
            'note': "Gated recurrent unit: like the LSTM with fewer parameters (no separate cell state)."},
    'Stacked LSTM': {'body': [_L('lstm', 64, drop=0.1), _L('lstm', 32, act='none')], 'head': [32], 'pool': 'last',
                     'note': "Two LSTM layers: the second one works on the features of the first (deeper memory)."},
    'Bidirectional LSTM': {'body': [_L('lstm', 48, bidir=True)], 'head': [48], 'pool': 'mean',
                           'note': "Reads the window forward and backward (seq2seq: uses later inputs of the "
                                   "window, not causal)."},
    'TCN (dilated CNN)': {'body': [_L('conv', 32, 3, d) for d in (1, 2, 4, 8)], 'head': [32], 'pool': 'last',
                          'note': "Temporal convolutional network: the dilations 1, 2, 4, 8 grow the receptive field "
                                  "exponentially (31 samples with k = 3) at few parameters."},
    'Transformer': {'body': [_L('dense', 32, act='none'), _L('transformer', 32, heads=4, drop=0.1),
                             _L('transformer', 32, heads=4, drop=0.1)], 'head': [32], 'pool': 'mean',
                    'note': "Linear embedding + sinusoidal position + two self-attention encoder layers: every "
                            "sample of the window can attend to every other one."},
}
BUILTIN = {1: 'CNN', 2: 'LSTM', 3: 'CNN-LSTM'}


def _act(a):
    return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh()}.get(str(a).lower(), nn.Identity())


class Block(nn.Module):
    """One layer of a user-defined architecture on batch x W x C."""
    def __init__(self, l, cin):
        super().__init__()
        t, u = str(l['type']).lower(), int(l['units'])
        self.type = t
        self.out = u
        if u < 1:
            raise ValueError(f"{t}: units must be >= 1")
        if t == 'conv':
            k, d = int(l.get('kernel') or 3), int(l.get('dilation') or 1)
            self.core = nn.Conv1d(cin, u, k, dilation=d, padding='same')
            self.norm = nn.BatchNorm1d(u) if l.get('norm') else nn.Identity()
        elif t in ('lstm', 'gru'):
            bi = bool(l.get('bidir'))
            self.core = (nn.LSTM if t == 'lstm' else nn.GRU)(cin, u, batch_first=True, bidirectional=bi)
            self.out = u * (2 if bi else 1)
            self.norm = nn.LayerNorm(self.out) if l.get('norm') else nn.Identity()
        elif t == 'transformer':
            h = int(l.get('heads') or 4)
            if u % h:
                raise ValueError(f"transformer: units ({u}) must be a multiple of the heads ({h})")
            self.proj = nn.Linear(cin, u) if cin != u else nn.Identity()
            self.core = nn.TransformerEncoderLayer(u, h, dim_feedforward=2 * u, dropout=float(l.get('drop') or 0.0),
                                                   batch_first=True, norm_first=True)
            self.norm = nn.LayerNorm(u) if l.get('norm') else nn.Identity()
        elif t == 'dense':
            self.core = nn.Linear(cin, u)
            self.norm = nn.LayerNorm(u) if l.get('norm') else nn.Identity()
        else:
            raise ValueError(f"unknown layer type '{t}'")
        self.act = _act(l.get('act', 'relu')) if t != 'transformer' else nn.Identity()
        self.drop = nn.Dropout(float(l['drop'])) if t != 'transformer' and float(l.get('drop') or 0) > 0 else \
            nn.Identity()

    def forward(self, x):
        if self.type == 'conv':
            x = self.norm(self.core(x.transpose(1, 2))).transpose(1, 2)
        elif self.type in ('lstm', 'gru'):
            x = self.norm(self.core(x)[0])
        elif self.type == 'transformer':
            x = self.proj(x)
            n, d = x.shape[1], x.shape[2]
            pos = torch.arange(n, device=x.device, dtype=x.dtype)[:, None]
            div = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype) * (-math.log(1e4) / d))
            pe = torch.zeros(n, d, device=x.device, dtype=x.dtype)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)[:, :d // 2]
            x = self.norm(self.core(x + pe))
        else:
            x = self.norm(self.core(x))
        return self.drop(self.act(x))


class Pool(nn.Module):
    """Reduction of the window to one vector (seq2point)."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'last':
            return x[:, -1]
        if self.mode == 'max':
            return x.max(dim=1)[0]
        return x.mean(dim=1)


def buildArch(arch, F, M, seq):
    """Body (ModuleList), pooling (None for seq2seq) and head of a user-defined architecture."""
    body, c = nn.ModuleList(), F
    for l in arch.get('body') or []:
        b = Block(l, c)
        body.append(b)
        c = b.out
    pool = None
    if seq > 0:
        mode = str(arch.get('pool') or 'auto')
        if mode == 'auto':
            mode = 'last' if len(body) and body[-1].type in ('lstm', 'gru') else 'mean'
        pool = Pool(mode)
    head = []
    for u in arch.get('head') or []:
        head += [nn.Linear(c, int(u)), nn.ReLU()]
        c = int(u)
    head.append(nn.Linear(c, M))
    return body, pool, nn.Sequential(*head)


#######################################################################################################################
# Class
#######################################################################################################################
class dlNet(nn.Module):
    ###################################################################################################################
    # Constructor
    ###################################################################################################################
    def __init__(self, sel, F, M, seq, mu, sd, arch=None):
        super().__init__()
        self.sel = sel
        self.seq = seq
        self.arch = arch or None
        self.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))

        # ==============================================================================
        # User-defined architecture
        # ==============================================================================
        if self.arch:
            self.body, self.pool, self.head = buildArch(self.arch, F, M, seq)
            return

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
        if getattr(self, 'arch', None):
            h = x
            for b in self.body:
                h = b(h)
            h = self.pool(h) if self.pool is not None else h
        elif self.sel == 1:
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
