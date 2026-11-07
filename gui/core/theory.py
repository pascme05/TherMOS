#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         theory.py
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
Theory of the operating modes with interactive examples (view 'Theory' of a mode page, shown when the mode has an
entry in THEORY). Each function gets the current config and renders equations (st.latex), a short explanation and
explorers: widgets with keys 'th<mode>|…' (not bound to the config), key-figure tiles and charts via style.show.
Use the functions of the toolkit (pre-processing, models, loaders) instead of re-implementing them, cache expensive
calls (st.cache_resource / st.cache_data on a JSON dump of the config) and state units in every label and table.
Reference: PyPowerSim gui/core/theory.py (+ gui/core/toolkit.py), OPTELIA gui/core/theory.py.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import streamlit as st


def _intro(text):
    st.caption(text + " Values start from the current configuration; changes here do not modify it.")


#######################################################################################################################
# Theory per mode: mode -> function(cfg)
#######################################################################################################################
THEORY = {}
