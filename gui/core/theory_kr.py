#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         theory_kr.py
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
Theory of the KR mode (block Krylov reduction of the finite-volume operator): what the mode needs, the idea of
moment matching, a small tool on the IMS board (error over the number of moment blocks q, for a moved chip and another
cooling), the equations with all symbols and the interpretation.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gui.core import needs
from gui.core.style import show, fmt, line
from gui.views.tools import _common as cm
from src.py.tools import theoryPF as tp


#######################################################################################################################
# Cached calculations
#######################################################################################################################
@st.cache_data(show_spinner="Krylov reductions of the board …")
def _order(dx, h):
    return tp.krylovOrder(dx, h)


#######################################################################################################################
# Page
#######################################################################################################################
def theory_kr(cfg):
    q0 = int(((cfg.get('Mdl') or {}).get('kr') or {}).get('q', 4) or 4)
    st.caption("KR mode: model order reduction of the finite-volume (FV) model of a board by moment matching. The "
               "reduced model is computed from the matrices of the board, not from temperature data.")
    cm.question("How few states reproduce the field of a board when its matrices C and G are known — and what does "
                "this cost for every new design?")
    needs.render('KR')

    cm.section(1, "The idea in three steps", divider=False)
    st.markdown(
        "1. **Steady state first.** For a constant loss the field is Θ·P with Θ = G⁻¹·B: one sparse solve per heat "
        "source. These fields are the first basis vectors.\n"
        "2. **Then the delay.** Multiplying with M = G⁻¹·C (time constants, s) gives how the field lags behind the "
        "loss; the vectors M·Θ, M²·Θ, … are the next moments of the transfer function H(s) = (G + s·C)⁻¹·B around "
        "s = 0. q blocks span r = q·S states (S: sources).\n"
        "3. **Project and solve.** The FV model is projected onto these vectors (Galerkin) and solved exactly. The "
        "reduced model matches the first q moments, i.e. the steady state and the slow response exactly; fast "
        "details converge with growing q.")

    cm.section(2, "Tool: error over the number of moment blocks", "IMS board (30 × 20 mm, 3 chips), pulse losses "
               "(period 4 s) for 20 s. Move T1 or change the cooling: the reduction is simply repeated for the new "
               "operator — this is why Krylov MOR is exact for every design, and why it needs that design's matrices.")
    c = st.columns(3)
    dx = c[0].slider("Displacement of T1 Δx (mm)", -4, 12, 0, 2, key='thKR|dx',
                     help="T1 starts at x = 8 mm; D1 and M1 sit at the right edge. Every position is a new operator.")
    h = c[1].select_slider("Cooling h (W/(m²K))", [250, 500, 1000, 1500, 3000, 5000, 8000], 1500, key='thKR|h',
                           help="Heat transfer coefficient of the cold plate; enters the diagonal of G.")
    qs = c[2].select_slider("Moment blocks q (chart marker)", [1, 2, 3, 4, 6, 8], min(q0, 8) if q0 in (1, 2, 3, 4, 6, 8)
                            else 4, key='thKR|q', help="Number of moment blocks of the basis (Parameters › kr › q).")
    res = _order(float(dx), float(h))
    rows = {r['q']: r for r in res['rows']}
    R = rows[qs]
    cm.tiles([("States r", f"{R['r']}", f"r = q·S with S = 3 sources; full FV model: {res['N']} nodes."),
              ("Field RMSE", f"{fmt(R['rmse'])} K", f"{100 * R['rmse'] / res['peak']:.3f} % of the peak rise "
                                                    f"{fmt(res['peak'])} K, all nodes and samples."),
              ("Junction RMSE", f"{fmt(R['rmseTj'])} K", "Hottest die node of each chip."),
              ("Reduction", f"{fmt(1e3 * R['tRed'])} ms", "Sparse LU of G, q block solves, orthonormalisation and "
                                                          "projection — needed again for every new design.")])
    q = [r['q'] for r in res['rows']]
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(line(q, [r['rmse'] for r in res['rows']], 'field RMSE', 0, mode='lines+markers'), secondary_y=False)
    fig.add_trace(line(q, [r['rmseTj'] for r in res['rows']], 'junction RMSE', 1, mode='lines+markers'),
                  secondary_y=False)
    fig.add_trace(line(q, [r['r'] for r in res['rows']], 'states r', 2, dash='dot', mode='lines+markers'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=[qs], y=[R['rmse']], mode='markers', marker=dict(color='red', size=11),
                             name='selected q'), secondary_y=False)
    fig.update_yaxes(type='log', title_text='RMSE vs FV (K)', secondary_y=False)
    fig.update_yaxes(title_text='states r', secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text='moment blocks q')
    show(fig, height=340, key='thKR|fig', info="**Error of the Krylov model against the full FV model** (log scale) "
         "over the number of moment blocks q, for the current chip position and cooling. Dotted, right axis: number "
         "of states r. The error drops by about one order of magnitude per block until it reaches the accuracy of "
         "the time integration; for large q dependent moments are dropped, so r stops growing.")
    st.caption(f"Δx = {dx} mm, h = {h} W/(m²K): q = 1 (only the steady-state fields) gives "
               f"{fmt(rows[1]['rmse'])} K, q = 2 {fmt(rows[2]['rmse'])} K and q = 4 {fmt(rows[4]['rmse'])} K "
               f"with {rows[4]['r']} states; the reduction takes {fmt(1e3 * rows[4]['tRed'])} ms.")

    cm.equations([
        ("FV model of the board", r"\mathbf C\,\dot{\boldsymbol\theta} = -\mathbf G\,\boldsymbol\theta + "
                                  r"\mathbf B\,\boldsymbol p(t),\qquad \mathbf B = \mathrm{diag}(\mathbf V)\,\mathbf U",
         "From the board designer: one equation per node."),
        ("Transfer function and moments", r"\mathbf H(s) = (\mathbf G + s\mathbf C)^{-1}\mathbf B = \sum_{j\ge 0} "
                                          r"(-s)^j\,\mathbf M^j\,\boldsymbol\Theta,\quad \boldsymbol\Theta = "
                                          r"\mathbf G^{-1}\mathbf B,\quad \mathbf M = \mathbf G^{-1}\mathbf C",
         "Expansion around s = 0 (slow dynamics first)."),
        ("Krylov basis", r"\mathbf V_r = \mathrm{orth}\,[\boldsymbol\Theta,\ \mathbf M\boldsymbol\Theta,\ \dots,\ "
                         r"\mathbf M^{q-1}\boldsymbol\Theta],\quad r = q\,S", "One sparse LU of G, q block solves."),
        ("Reduced model", r"\mathbf C_r\,\dot{\boldsymbol x} = -\mathbf G_r\,\boldsymbol x + \mathbf B_r\,"
                          r"\boldsymbol p,\quad \mathbf C_r = \mathbf V_r^T\mathbf C\mathbf V_r,\ \mathbf G_r = "
                          r"\mathbf V_r^T\mathbf G\mathbf V_r,\ \boldsymbol\theta \approx \mathbf V_r\boldsymbol x",
         "Symmetric positive definite: stable and passive for every q (PRIMA)."),
    ], sym=[('θ', 'temperature rise over the reference at every node', 'K'),
            ('C', 'heat capacities of the nodes (diagonal)', 'Ws/K'),
            ('G', 'conductance matrix incl. the cooling to the ambient', 'W/K'),
            ('V', 'control volume of every node', 'm³'),
            ('U, p', 'source footprints (N × S) and their amplitudes over time', '–, W/m³'),
            ('S', 'number of independent heat sources (chips)', '–'),
            ('Θ', 'steady-state fields of the sources per unit amplitude', 'K/(W/m³)'),
            ('M', 'time-constant operator G⁻¹·C', 's'),
            ('q', 'number of moment blocks (Parameters › kr › q)', '–'),
            ('r', 'states of the reduced model, r = q·S', '–'),
            ('s', 'Laplace variable', '1/s')], expanded=True)
    cm.refs(['Odabasioglu1998', 'Antoulas2005', 'Benner2015'])
    ok = rows[4]['rmse'] < 0.01 * res['peak']
    cm.interpret([
        f"**With q = 4 blocks ({rows[4]['r']} states) the error is {fmt(rows[4]['rmse'])} K** "
        f"({100 * rows[4]['rmse'] / res['peak']:.3f} % of the rise) for this design — "
        + ("far below any measurement uncertainty." if ok else "larger than usual; increase q."),
        "Why: the basis contains the exact steady-state fields of every source and the first moments of the delay, "
        "so slow loads, which dominate thermal mission profiles, are reproduced exactly. Fast load changes need "
        "more blocks.",
        "What it needs: the matrices C, G and the source footprints of exactly the board to be predicted, and one "
        "sparse factorisation per design. Moving a chip or changing the cooling means a new reduction (here "
        f"{fmt(1e3 * rows[4]['tRed'])} ms; for large 3D models seconds to minutes).",
        "Fair comparison: KR receives the full physics of the new design. The PF network receives only the chip "
        "positions and the condition; its larger error (about 1 K) is the price for not needing the operator. "
        "Theory & Tools › MOR & POD compares both at equal settings."], title="How to read this")
