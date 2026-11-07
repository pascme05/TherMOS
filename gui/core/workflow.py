#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         workflow.py
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
This function draws the workflow of every operating mode (and sub-mode) as a flow chart (Graphviz, rendered in the
browser) and explains the steps. The nodes show the settings of the current configuration that control the step;
steps that are skipped with the current settings are drawn dashed. Add one function per mode to WORKFLOWS.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms

import streamlit as st

#######################################################################################################################
# Style
#######################################################################################################################
COL = {'inp': ('#e8f1fc', '#2a78d6'), 'step': ('#ffffff', '#52514e'), 'out': ('#e7f6ef', '#1baf7a'),
       'opt': ('#fdf1e8', '#eb6834'), 'off': ('#f6f6f3', '#c3c2b7')}


def opt(sec, cat, var):
    """'value) label' of a parameter with options."""
    return f"{forms.value(sec, cat, var)}) {forms.option_label(sec, cat, var).split(' (')[0]}"


def val(sec, cat, var, unit=''):
    v = forms.value(sec, cat, var)
    if isinstance(v, float) and forms.find(sec, cat, var)['percent']:
        return f"{100 * v:g} %"
    return f"{v:g}{(' ' + unit) if unit else ''}" if isinstance(v, (int, float)) else str(v)


class Flow:
    """Small builder of a Graphviz flow chart."""

    def __init__(self, rankdir='TB'):
        self.nodes, self.edges, self.ranks = [], [], []
        self.rankdir = rankdir

    def node(self, nid, title, lines=(), kind='step', active=True):
        fill, line = COL[kind if active else 'off']
        txt = f"<<B>{_esc(title)}</B>" + ''.join(f"<BR/><FONT POINT-SIZE='10' COLOR='#52514e'>{_esc(l)}</FONT>"
                                                  for l in lines if l) + ">"
        shape = {'inp': 'note', 'out': 'box', 'opt': 'box', 'step': 'box'}[kind]
        style = 'filled,rounded' + (',dashed' if not active else '') + (',bold' if kind == 'out' else '')
        self.nodes.append(f'"{nid}" [label={txt}, shape={shape}, style="{style}", fillcolor="{fill}", '
                          f'color="{line}", fontcolor="{"#0b0b0b" if active else "#898781"}"];')
        return self

    def edge(self, a, b, label='', active=True):
        lab = f', label="{label}", fontsize=9, fontcolor="#52514e"' if label else ''
        self.edges.append(f'"{a}" -> "{b}" [color="{"#52514e" if active else "#c3c2b7"}"'
                          f'{", style=dashed" if not active else ""}{lab}];')
        return self

    def same(self, *ids):
        self.ranks.append('{rank=same; ' + ' '.join(f'"{i}";' for i in ids) + '}')
        return self

    def dot(self):
        return ('digraph G {\n'
                f'  rankdir={self.rankdir}; bgcolor="transparent"; nodesep=0.35; ranksep=0.4;\n'
                '  node [fontname="Helvetica", fontsize=12, margin="0.18,0.08"];\n'
                '  edge [arrowsize=0.7, penwidth=1.2];\n  ' + '\n  '.join(self.nodes + self.edges + self.ranks) +
                '\n}')


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _legend():
    st.caption(":blue[■] inputs (workbooks, config) · ■ processing step · :orange[■] option / loop · "
               ":green[■] results · dashed: skipped with the current settings")


#######################################################################################################################
# Workflows: mode -> function(cfg, sub) returning (Flow, [(step title, explanation), ...])
#######################################################################################################################
def _g(cfg):
    return cfg[A.RUN_SECTION]


def _pre(f, cfg):
    """Common pre-processing chain of main(): data → load/resample → offset → loss correction → normalise → dim."""
    G = _g(cfg)
    dim = int(G.get('datDim', 1))
    files = f"train {len(G.get('trFile') or [])} · test {G.get('teFile')}"
    f.node('dat', 'Data', [f"{dim}D · .{G.get('format')}", files], kind='inp')
    f.node('cfg', 'Config', ["Exp · Dat · Mdl · Par"], kind='inp')
    f.same('dat', 'cfg')
    f.node('load', 'Load & resample', [f"Ts = {val('Exp', 'gen', 'Ts', 's')}", opt('Exp', 'gen', 'samp')])
    f.node('off', 'Offset', [opt('Exp', 'gen', 'init')], active=forms.value('Exp', 'gen', 'init') != 1)
    fw = forms.value('Par', 'loss', 'fw') == 1
    f.node('loss', 'Loss correction θ(T)', [f"Rslope = {val('Par', 'loss', 'Rslope')}"], kind='opt', active=fw)
    nx, ny = forms.value('Par', 'gen', 'normX'), forms.value('Par', 'gen', 'normY')
    f.node('norm', 'Normalise', [f"X: {opt('Par', 'gen', 'normX')}", f"y: {opt('Par', 'gen', 'normY')}"],
           active=nx != 1 or ny != 1)
    dOut = int(forms.value('Dat', 'gen', 'dOut'))
    probe = [f"probe ({val('Dat', 'gen', 'outX', 'm')}, {val('Dat', 'gen', 'outY', 'm')})"] if dOut == 1 < dim else []
    f.node('dim', 'Dimension', [f"{dim}D → {dOut}D"] + probe, active=dim != dOut)
    f.edge('dat', 'load').edge('cfg', 'load').edge('load', 'off').edge('off', 'loss', active=fw)
    f.edge('loss' if fw else 'off', 'norm').edge('norm', 'dim')
    return 'dim'


def _post(f, cfg, last, tag):
    G = _g(cfg)
    bw = forms.value('Par', 'loss', 'bw') == 1
    f.node('pred', 'Predict test data', ["loss iteration θ(T)" if bw else "",
                                         "trained model" if G.get('train') else f"mdl_{tag}_{G.get('name')}"],
           kind='opt' if bw else 'step')
    f.node('met', 'Metrics', ["MAE · RMSE · MAX · FIT", f"transient: |dT| > {val('Par', 'gen', 'th')}·max"])
    f.node('res', 'Results', [f"results/{G.get('name')}/", "figures · gui run"], kind='out')
    f.edge(last, 'pred').edge('pred', 'met').edge('met', 'res')


PRE_STEPS = [('Data / Config', "the data sets of the Setup page and the parameters (sheets Exp, Dat, Mdl, Par)."),
             ('Load & resample', "every profile is interpolated to Ts (loadData, resmpData)."),
             ('Offset', "start values or reference temperatures are removed (offRemove) and added back at the end."),
             ('Loss correction', "optional: the losses are scaled with θ(T) of the measured temperature (updLoss)."),
             ('Normalise', "optional scaling of inputs and outputs (normData)."),
             ('Dimension', "a 2D/3D field is kept (PO/PS) or reduced to one probe node (1D models, dimData).")]
POST_STEPS = [('Predict', "the model simulates the testing data from its initial state; with Par › loss bw = 1 the "
                          "losses are iterated with the predicted temperature."),
              ('Metrics', "errors per output on all / transient / steady-state samples (performance)."),
              ('Results', "figures (results/<name>/fig_*.png), model file mdl/mdl_<tag>_<name> and this GUI run.")]


def _rc(cfg, sub):
    f = Flow()
    last = _pre(f, cfg)
    leg = int(_g(cfg).get('legacy', 0)) == 1
    o = forms.value('Mdl', 'gen', 'opt') == 1
    k = f"K = {val('Mdl', 'rc', 'K')}" + (f" (search 1…{val('Mdl', 'gen', 'Kmax')})" if o else '')
    f.node('fit', 'Fit Foster network', [k, "Zth curve fit (legacy)" if leg else "output-error fit of ΔT"])
    f.edge(last, 'fit')
    _post(f, cfg, 'fit', 'rc')
    return f, PRE_STEPS + [('Fit Foster network', "Rᵢ and τᵢ of K RC pairs, so that the network driven by the "
                                                  "measured losses reproduces the temperature rise.")] + POST_STEPS


def _ss(cfg, sub):
    f = Flow()
    last = _pre(f, cfg)
    pem = forms.value('Mdl', 'ss', 'dis') == 2
    f.node('n4', 'Subspace identification', [f"n = {val('Mdl', 'ss', 'K')}", "N4SID / MOESP"])
    f.node('pem', 'Output-error fit', [opt('Mdl', 'ss', 'sta'), f"≤ {val('Mdl', 'gen', 'iterMax')} iterations"],
           kind='opt', active=pem)
    f.edge(last, 'n4').edge('n4', 'pem', active=pem)
    _post(f, cfg, 'pem' if pem else 'n4', 'ss')
    return f, PRE_STEPS + [('Subspace identification', "A, B, C from the Hankel matrices of inputs and outputs "
                                                       "(fitN4sid)."),
                           ('Output-error fit', "the simulated output is fitted to the data (fitPem), optionally "
                                                "with stable poles.")] + POST_STEPS


def _sf(cfg, sub):
    f = Flow()
    last = _pre(f, cfg)
    f.node('zth', 'Self-heating Zthₖₖ', [f"{val('Mdl', 'rc', 'K')} RC pairs per node"])
    f.node('rth', 'Rth matrix', ["ΔTᵢ(end)/Pₖ(end)", "G = Rth⁻¹"])
    f.edge(last, 'zth').edge('zth', 'rth')
    _post(f, cfg, 'rth', 'sf')
    return f, PRE_STEPS + [('Self-heating', "one Foster network per node from the experiment that heats only this "
                                            "node."),
                           ('Rth matrix', "coupling from the steady-state end values of all experiments.")] + POST_STEPS


def _po(cfg, sub, ps=False):
    f = Flow()
    last = _pre(f, cfg)
    sel = forms.value('Mdl', 'pod', 'sel')
    kk = f"K = {val('Mdl', 'pod', 'K')}" if sel != 2 else \
        f"energy ≥ {val('Mdl', 'gen', 'Emax')}, K ≤ {val('Mdl', 'gen', 'Kmax')}"
    f.node('svd', 'POD (SVD of snapshots)', [kk])
    f.edge(last, 'svd')
    if ps:
        f.node('ss', 'SS identification of a(t)', [f"n = {val('Mdl', 'ss', 'K')}", opt('Mdl', 'ss', 'dis')])
        f.edge('svd', 'ss')
        _post(f, cfg, 'ss', 'ps')
        return f, PRE_STEPS + [('POD', "spatial modes φₖ from the SVD of the training snapshots."),
                               ('SS identification', "a state-space model from the losses to the modal coordinates "
                                                     "aₖ(t) = φₖᵀ(T − T̄).")] + POST_STEPS
    pr = int(forms.value('Mdl', 'pod', 'protocol'))
    f.node('gal', 'Galerkin projection', ["C = ∫ρc φφ, G = ∫k ∇φ∇φ", "q = ∫Q φ"])
    f.node('cal', 'Calibration', [opt('Mdl', 'pod', 'protocol')], kind='opt', active=pr >= 4)
    f.edge('svd', 'gal').edge('gal', 'cal', active=pr >= 4)
    _post(f, cfg, 'cal' if pr >= 4 else 'gal', 'po')
    return f, PRE_STEPS + [('POD', "spatial modes φₖ from the SVD of the training snapshots (energy criterion or "
                                   "fixed K)."),
                           ('Galerkin projection', "the heat equation is projected onto the modes: K ODEs instead of "
                                                   "one per node."),
                           ('Calibration', "protocols 4–6 correct C, G and α with the training trajectories.")] + \
        POST_STEPS


def _ml(cfg, sub):
    f = Flow()
    last = _pre(f, cfg)
    G = _g(cfg)
    var = A.MODES['ML']['pick'][1].get(int(G.get('selML') or 1), '')
    f.node('feat', 'Features', ["EWMA/EWMS" if G.get('featRoll') else "inputs at t"], kind='opt',
           active=bool(G.get('featRoll')))
    f.node('fit', 'Regressor', [var, "random search, 3-fold CV" if forms.value('Mdl', 'gen', 'opt') == 1 else ""])
    f.edge(last, 'feat').edge('feat', 'fit')
    _post(f, cfg, 'fit', 'ml')
    return f, PRE_STEPS + [('Features', "optional rolling means/variances give the static regressor a memory."),
                           ('Regressor', "maps the features at time t to the temperatures at time t.")] + POST_STEPS


def _dl(cfg, sub, pinn=False):
    f = Flow()
    last = _pre(f, cfg)
    G = _g(cfg)
    m = 'PI' if pinn else 'DL'
    var = A.MODES[m]['pick'][1].get(int(G.get(A.SEL[m]) or 1), '')
    seq = 0 if pinn else int(forms.value('Mdl', 'dl', 'seq'))
    f.node('win', 'Windows', [f"W = {val('Mdl', 'dl', 'W')}", "seq2seq" if seq == 0 else f"seq2point @ {seq}"])
    f.node('net', 'Train network', [var, f"{val('Mdl', 'dl', 'epoch')} epochs, lr {val('Mdl', 'dl', 'initLr')}"])
    f.edge(last, 'win').edge('win', 'net')
    if pinn:
        b = forms.value('Mdl', 'pinn', 'beta')
        f.node('phy', 'Physics loss', [f"α = {val('Mdl', 'pinn', 'alpha')}, β = {val('Mdl', 'pinn', 'beta')}"],
               kind='opt', active=b > 0)
        f.edge('phy', 'net', active=b > 0)
    _post(f, cfg, 'net', m.lower())
    steps = [('Windows', "the profiles are cut into windows of W samples (no window across two profiles)."),
             ('Train network', "Adam, learning-rate drops, early stopping on the validation data.")]
    if pinn:
        steps.append(('Physics loss', "residual of the lumped model Cth·T' = (T_amb − T)/Rth + 3·Rs·Is² in the loss."))
    return f, PRE_STEPS + steps + POST_STEPS


WORKFLOWS = {'RC': _rc, 'SS': _ss, 'SF': _sf, 'PO': _po, 'PS': lambda c, s: _po(c, s, True), 'ML': _ml,
             'DL': _dl, 'PI': lambda c, s: _dl(c, s, True)}


#######################################################################################################################
# View
#######################################################################################################################
def view(m):
    cfg = forms.cfg()
    sub = None
    spec = A.MODES[m].get('sub')
    if spec:
        p = forms.find(*spec)
        cur = forms.value(*spec)
        if p['options']:
            names = p['options']
            sub = st.segmented_control("Sub-mode", list(names), default=cur, key=f"wf|{m}",
                                       format_func=lambda k: f"{names[k]}" + (' (selected)' if k == cur else ''))
            sub = cur if sub is None else sub
            if sub != cur:
                st.caption(f"Showing sub-mode {sub}; the configuration runs sub-mode {cur} — change it in "
                           f"*Parameters*.")
    f, steps = WORKFLOWS[m](cfg, sub)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(f.dot(), width='content')
        _legend()
    with c2:
        st.markdown("##### What happens")
        for i, (t, txt) in enumerate(steps, start=1):
            st.markdown(f"**{i}. {t}** — {txt}")
        st.caption("The boxes show the settings of the current configuration; change them in *Parameters*, the "
                   "diagram follows.")
