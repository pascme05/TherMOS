#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         needs.py
# Date:         21.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
What every mode needs: the data to build (train) the model, the data to predict a new case, and what it cannot do
without new data. Shown in the Inputs view of every mode (render) and as one table on the MOR page (table), so that
comparisons between the modes say which information each of them was given.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import pandas as pd
import streamlit as st

#######################################################################################################################
# Needs per mode: (build / train, predict a new case, new design needs, information level)
#######################################################################################################################
NEEDS = {
    'RC': ("Measured losses P(t) and temperatures T(t) at the outputs (one board, one cooling).",
           "Losses P(t) only.",
           "New measurement and fit for every board or cooling.", 'data, 1D'),
    'SS': ("Measured losses P(t) and temperatures T(t) of all outputs (one board, one cooling).",
           "Losses P(t) only.",
           "New measurement and identification for every board or cooling.", 'data, 1D'),
    'SF': ("Losses and temperatures; nodal Foster: one single-source experiment per source.",
           "Losses P(t) only.",
           "New experiments for every board or cooling.", 'data, 1D'),
    'PO': ("Temperature fields T(x, t) (snapshots), heat generation Q(x, t) and the geometry and materials of one "
           "board (Galerkin projection).",
           "Heat generation Q(x, t) of the same board.",
           "New snapshots (simulation) for every layout; the POD basis does not follow a moving source.",
           'snapshots + geometry'),
    'PS': ("Temperature fields T(x, t) (snapshots) and heat generation Q(x, t) of one board.",
           "Heat generation Q(x, t) of the same board.",
           "New snapshots for every layout.", 'snapshots'),
    'ML': ("Losses P(t) and temperatures T(t) (one board).", "Losses P(t) only.",
           "New training data for every board.", 'data, 1D'),
    'DL': ("Losses P(t) and temperatures T(t) (one board).", "Losses P(t) only.",
           "New training data for every board.", 'data, 1D'),
    'PI': ("Losses P(t), temperatures T(t) and the parameters of a lumped RC model (physics residual).",
           "Losses P(t) only.", "New training data for every board.", 'data + lumped physics'),
    'PF': ("Temperature fields of the training profiles (all nodes or a few sensors) and, for physics 2/3, the "
           "finite-volume operator C, G, V of the training board.",
           "Heat generation Q(x, t) only: no operator, no matrix solve.",
           "Single-board network: retraining. Parametric network (paper study): new conditions and positions inside "
           "the trained range without retraining.", 'data + operator (training only)'),
    'KR': ("No temperatures. The finite-volume operator C, G, V and the heat source footprints of the board "
           "(board designer data).",
           "The operator C, G, V of the board to be predicted, its heat generation Q(x, t) and one sparse LU "
           "factorisation of G.",
           "Nothing beyond the operator: every new layout, material or cooling is reduced again (exact, but only "
           "because its operator is known).", 'operator of every board'),
}
HEAD = ("To build the model", "To predict a new case", "For a new design")


#######################################################################################################################
# Views
#######################################################################################################################
def render(mode):
    """Box 'What this mode needs' (three columns) above the inputs of the mode."""
    n = NEEDS.get(mode)
    if not n:
        return
    with st.container(border=True):
        st.markdown(f"**What this mode needs** · information level: *{n[3]}*")
        c = st.columns(3)
        for col, h, t in zip(c, HEAD, n[0:3]):
            col.caption(h)
            col.markdown(t)
    if mode in ('KR', 'PO', 'PS', 'PF'):
        st.caption("Accuracy comparisons between the modes are only fair at the same information level: a mode that "
                   "receives the operator of the testing board (KR) solves a different, easier task than one that "
                   "only receives the heat sources (PF) — see Theory & Tools › MOR & POD.")


def table(modes=None):
    """All modes as one table (mode, build, predict, new design, information)."""
    ms = modes or list(NEEDS)
    return pd.DataFrame([(m, *NEEDS[m][0:3], NEEDS[m][3]) for m in ms],
                        columns=['mode', *HEAD, 'information level'])
