#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         inputs.py
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
Input data page: the input files of the configuration (grouped as in adapter.FILE_GROUPS, the group used by the
current settings first) and a browser of any workbook of the input folders (adapter.FOLDERS).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import files, forms
from gui.core import viz_inputs as vi
from gui.views.project import file_groups

import streamlit as st

ALL = 'All workbooks'


#######################################################################################################################
# Views
#######################################################################################################################
def _browse(cfg):
    kinds = [k for k in A.FOLDERS if files.list_files(k)]
    if not kinds:
        st.info("No input workbooks found in " + ", ".join(f"{v}/" for v in A.FOLDERS.values()) + ".")
        return
    c1, c2 = st.columns(2)
    with c1:
        kind = st.selectbox("Folder", kinds, format_func=lambda k: f"{k} ({A.FOLDERS[k]}/)", key='in|kind')
    names = files.list_files(kind)
    cur = next((cfg[A.RUN_SECTION].get(v) for v, k, _, _ in A.RUN_FIELDS if k == f"file:{kind}"), None)
    with c2:
        name = st.selectbox("Workbook", names, index=names.index(cur) if cur in names else 0, key=f"in|{kind}")
    vi.workbook(kind, name, key=f"in|{kind}|{name}")


def _group(cfg, g, fields, active):
    st.caption((g.get('text', '') + ' — ' if g.get('text') else '') +
               ("used by the current settings." if active else "**not used** by the current settings (topology)."))
    if not fields:
        st.info("No input files in this group.")
        return
    labels = [f"{label} · {cfg[A.RUN_SECTION].get(var)}" for var, _, label, _ in fields]
    pick = st.segmented_control("File", list(range(len(fields))), default=0, key=f"in|grp|{g['title']}",
                                format_func=lambda i: labels[i], label_visibility='collapsed')
    var, kind, label, _ = fields[pick or 0]
    name = cfg[A.RUN_SECTION].get(var)
    vi.workbook(kind[5:], name, key=f"in|{kind[5:]}|{name}", field=var)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Input data")
    st.caption("The data sets selected on the Setup page, as stored in data/ (before resampling and normalisation).")
    cfg = forms.cfg()
    with st.expander("Data selection", icon=':material/database:'):
        from gui.views.project import data_form
        data_form(cfg)
    from gui.core.viz_data import view
    view(cfg, key='in')
