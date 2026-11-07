#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         viz_inputs.py
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
This function shows the input workbooks. workbook() is the generic viewer (one tab per sheet: table, and a chart of
the numeric columns over the first column). Add dedicated charts per operating mode to INPUTS
(mode -> function(cfg)); they are shown in the Inputs view of the mode page.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import files
from gui.core.style import show, line, downsample, table

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


#######################################################################################################################
# Generic viewer
#######################################################################################################################
def arrow_safe(df):
    """Mixed text/number columns (typical for parameter sheets) as text, so the table renders without warnings."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and not all(isinstance(v, str) for v in df[c].dropna()):
            df[c] = df[c].map(lambda v: '' if v is None or (isinstance(v, float) and v != v) else str(v))
    return df


def sheet(df, key):
    """Table of a sheet and a line chart of its numeric columns over the first column."""
    df = arrow_safe(df.dropna(how='all').dropna(axis=1, how='all'))
    if df.empty:
        st.info("Empty sheet.")
        return
    num = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    x = df.columns[0]
    if num and len(df) > 5 and pd.api.types.is_numeric_dtype(df[x]):
        pick = st.multiselect("Columns", num, default=num[:min(4, len(num))], key=f"{key}|cols")
        if pick:
            s = downsample(len(df))
            fig = go.Figure([line(df[x].iloc[::s], df[c].iloc[::s], str(c), i) for i, c in enumerate(pick)])
            fig.update_layout(xaxis_title=str(x))
            show(fig, key=f"{key}|ts")
    with st.expander(f"Table ({len(df)} rows × {len(df.columns)} columns)", expanded=not num):
        table(df, hide_index=True, height=320)


def workbook(kind, name, key, field=None):
    """
    All sheets of an input workbook, one tab per sheet. field: run setting the workbook is selected by (None in the
    browser) — dedicated viewers/editors per kind can use it to offer 'use in the configuration'.
    """
    if not name or name not in files.list_files(kind):
        st.info(f"Workbook '{name}' not found in {kind}/.")
        return
    try:
        sheets = files.load_workbook(kind, name, files.mtime(kind, name))
    except Exception as e:
        st.error(f"The workbook could not be read: {e}")
        return
    names = [s for s in sheets if s.lower() != 'readme'] or list(sheets)
    for tab, s in zip(st.tabs(names), names):
        with tab:
            sheet(sheets[s], key=f"{key}|{s}")


#######################################################################################################################
# Inputs per mode: mode -> function(cfg)
#######################################################################################################################
INPUTS = {}
