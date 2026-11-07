#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         dataset.py
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
Information about the TherMOS data sets under data/ for the Setup page and the input viewers: file list, kind (1D
table / 2D / 3D field), channels (columns), sheets, profile IDs and the raw content. The files are read with the same
conventions as src/py/data/DataLoader.py (1D .mat: struct 'data' of columns; 2D/3D .mat: geo, k, rho, Cp, X, y, ...;
.xlsx: one table per sheet, unit rows below the header).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A

import os

import numpy as np
import pandas as pd
import streamlit as st

FIELD_KEYS = ['X', 'y', 'r', 't', 'Ts', 'geo', 'dx', 'dy', 'dz', 'Lx', 'Ly', 'Lz', 'k', 'rho', 'Cp', 'hc', 'fl', 'Ta']


#######################################################################################################################
# Files
#######################################################################################################################
def files(fmt=None):
    """Data set names (relative to data/, no extension), optionally only of one format ('mat' or 'xlsx')."""
    return [n for n, e in A.data_files() if fmt is None or e == fmt]


def path(name, fmt):
    """File of a data set; like the DataLoader, a bare name is searched recursively under data/."""
    base = os.path.join(A.ROOT, 'data')
    p = os.path.join(base, *str(name).replace(os.sep, '/').strip('/').split('/')) + '.' + fmt
    if os.path.isfile(p):
        return p
    f = os.path.basename(str(name)) + '.' + fmt
    for r, _, fs in os.walk(base):
        if f in fs:
            return os.path.join(r, f)
    return p


def _mtime(name, fmt):
    p = path(name, fmt)
    return os.path.getmtime(p) if os.path.isfile(p) else 0.0


#######################################################################################################################
# Content
#######################################################################################################################
@st.cache_data(show_spinner="Reading data set …", max_entries=8)
def _info(name, fmt, _m):
    import scipy.io as sio
    p = path(name, fmt)
    if not os.path.isfile(p):
        return {'kind': 'missing'}
    if fmt == 'xlsx':
        xl = pd.ExcelFile(p)
        sheets = xl.sheet_names
        cols = [str(c) for c in pd.read_excel(p, sheet_name=sheets[0], nrows=0).columns]
        return {'kind': '1D', 'sheets': sheets, 'cols': cols, 'ids': []}
    keys = sio.whosmat(p)
    names = [k[0] for k in keys]
    if 'data' in names:
        raw = sio.loadmat(p, squeeze_me=True, struct_as_record=False)['data']
        cols = list(raw._fieldnames)
        ids = sorted(set(np.atleast_1d(raw.id).astype(int).tolist())) if 'id' in cols else []
        return {'kind': '1D', 'sheets': [], 'cols': cols, 'ids': ids, 'n': len(np.atleast_1d(getattr(raw, cols[0])))}
    shape = dict((k[0], k[1]) for k in keys)
    geo = shape.get('geo', (0, 2))
    X = shape.get('X', (0, 0))
    return {'kind': '3D' if len(geo) > 1 and geo[1] == 3 else '2D', 'sheets': [], 'cols': [], 'ids': [],
            'nodes': X[1] if len(X) > 1 else 0, 'steps': X[0]}


def info(name, fmt):
    """{'kind': '1D'|'2D'|'3D'|'missing', 'sheets', 'cols', 'ids', ...} of a data set."""
    try:
        return _info(name, fmt, _mtime(name, fmt))
    except Exception as e:
        return {'kind': 'error', 'error': str(e), 'sheets': [], 'cols': [], 'ids': []}


@st.cache_data(show_spinner="Loading data …", max_entries=4)
def _table(name, fmt, sheet, _m):
    import scipy.io as sio
    p = path(name, fmt)
    if fmt == 'xlsx':
        df = pd.read_excel(p, sheet_name=str(sheet))
        num = df.apply(pd.to_numeric, errors='coerce')
        n = 0
        while n < len(df) and num.iloc[n].isna().all() and df.iloc[n].notna().any():
            n += 1
        return num.iloc[n:].reset_index(drop=True)
    raw = sio.loadmat(p, squeeze_me=True, struct_as_record=False)['data']
    return pd.DataFrame({f: np.atleast_1d(getattr(raw, f)) for f in raw._fieldnames})


def table(name, fmt, sheet=None):
    """1D data set as DataFrame (xlsx: one sheet)."""
    return _table(name, fmt, sheet, _mtime(name, fmt))


@st.cache_data(show_spinner="Loading field data …", max_entries=3)
def _field(name, _m):
    import scipy.io as sio
    raw = sio.loadmat(path(name, 'mat'), variable_names=FIELD_KEYS, squeeze_me=True)
    return {k: (raw[k].item() if np.ndim(raw[k]) == 0 else np.asarray(raw[k])) for k in FIELD_KEYS if k in raw}


def field(name):
    """2D/3D data set {geo, k, rho, Cp, hc, X, y, t, Ts, Lx, Ly, (Lz), dx, dy, (dz), ...}."""
    return _field(name, _mtime(name, 'mat'))
