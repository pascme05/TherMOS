#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        GUI
# File:         theory_dl.py
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
Theory views of the data-driven models: ML (static regression with rolled EWMA/EWMS features, calcFeat), DL
(windowing and the network architecture, drawn from the live dlNet with forward hooks: layers, tensor shapes,
parameter counts) and PI (the physics-informed loss of pinnFit as a computation graph and an explorer of the lumped
stator model that defines the residual). arch_dot() is also used by the results view of DL/PI.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, show, line, fmt
from gui.core.theory_lumped import intro, num, _get
from gui.views.tools import _common as cm
from src.py.general.calcFeat import calcFeat
from src.py.model.RC.rcSim import rcSim

#######################################################################################################################
# Settings
#######################################################################################################################
NETS = {1: 'CNN', 2: 'LSTM', 3: 'CNN + LSTM'}


#######################################################################################################################
# Architecture diagram
#######################################################################################################################
def _shape(o):
    if isinstance(o, tuple):
        o = o[0]
    return '×'.join(['B'] + [str(d) for d in tuple(o.shape)[1:]])


def _label(m):
    n = type(m).__name__
    if n == 'Conv1d':
        return f"Conv1d {m.in_channels}→{m.out_channels}, k = {m.kernel_size[0]}"
    if n in ('LSTM', 'GRU'):
        return f"{'Bidirectional ' if m.bidirectional else ''}{n} {m.input_size}→{m.hidden_size}"
    if n == 'TransformerEncoderLayer':
        a = m.self_attn
        return f"Transformer encoder d = {a.embed_dim}, {a.num_heads} heads, FF {m.linear1.out_features}"
    if n == 'Pool':
        return {'last': 'last time step', 'mean': 'mean over time', 'max': 'max over time'}[m.mode] + " (seq2point)"
    if n == 'Conv1d' and m.dilation[0] > 1:
        return f"Conv1d {m.in_channels}→{m.out_channels}, k = {m.kernel_size[0]}, d = {m.dilation[0]}"
    if n == 'Dropout':
        return f"Dropout {m.p:g}"
    if n == 'LayerNorm':
        return "LayerNorm"
    if n == 'Linear':
        return f"Linear {m.in_features}→{m.out_features}"
    if n == 'BatchNorm1d':
        return f"BatchNorm {m.num_features}"
    return n


def arch_dot(net, W, F, names=None, outs=None, physics=None):
    """
    Graphviz DOT of a dlNet: input window → normalisation → layers (shape after each layer, parameters) → output.
    physics: None or dict(Rth, Cth, alpha, beta, PvTxt, TrefTxt, node) to append the PINN loss graph.
    """
    import torch
    rec = []
    hooks = [m.register_forward_hook(lambda mod, i, o, mod_=m: rec.append((mod_, _shape(o))))
             for m in _atoms(net)]
    net.eval()
    with torch.no_grad():
        y = net(torch.zeros(2, W, F))
    for h in hooks:
        h.remove()
    seq = getattr(net, 'seq', 0)
    nP = sum(p.numel() for p in net.parameters())
    inp = ', '.join(map(str, names)) if names else f"{F} features"
    L = ['digraph G {', 'rankdir=TB; nodesep=0.25; ranksep=0.22;',
         'node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6", fontname="Helvetica", '
         'fontsize=11];', 'edge [color="#666666", fontname="Helvetica", fontsize=9];',
         f'x [label="input window\\n{inp}\\nB×{W}×{F}", fillcolor="#fff4e5", color="#e07b00"];',
         'nrm [label="normalisation (x − μ)/σ"];', 'x -> nrm;']
    fill = {'Conv1d': '#e8f4ea', 'LSTM': '#f3e8f7', 'GRU': '#f3e8f7', 'Linear': '#eef3fb',
            'TransformerEncoderLayer': '#fff4e5', 'Pool': '#fffbe6'}
    blocks = []                                                       # [label lines, out shape, kind, parameters]
    for m, sh in rec:
        n = type(m).__name__
        p = sum(q.numel() for q in m.parameters())
        if n in ('Identity',):
            continue
        if n in ('ReLU', 'GELU', 'Tanh', 'BatchNorm1d', 'LayerNorm', 'Dropout') and blocks:
            blocks[-1][0].append('+ ' + _label(m))
            blocks[-1][1], blocks[-1][3] = sh, blocks[-1][3] + p
        else:
            blocks.append([[_label(m)], sh, n, p])
    prev = 'nrm'
    for i, (lab, sh, n, p) in enumerate(blocks):
        txt = '\\n'.join(lab) + f"\\nout: {sh}" + ('  (channels first)' if n == 'Conv1d' else '') + \
            (f"\\n{p:,} parameters" if p else '')
        L.append(f'n{i} [label="{txt}", fillcolor="{fill.get(n, "#eef3fb")}"];')
        L.append(f'{prev} -> n{i};')
        prev = f'n{i}'
        nxt = blocks[i + 1][2] if i + 1 < len(blocks) else None
        if seq > 0 and not getattr(net, 'arch', None) and (n == 'LSTM' or (n == 'Conv1d' and nxt == 'Linear')):
            L.append(f'sel{i} [label="{"last time step" if n == "LSTM" else "mean over time"} (seq2point)", '
                     f'shape=note, fillcolor="#fffbe6"];')
            L.append(f'{prev} -> sel{i};')
            prev = f'sel{i}'
    o = ', '.join(map(str, outs)) if outs else 'outputs'
    if physics is None:
        L.append(f'y [label="T̂ ({o})\\n{_shape(y)}  ·  {nP:,} parameters in total", fillcolor="#fde8e8", '
                 f'color="#d03b3b"];')
        L.append(f'{prev} -> y;')
        L.append('t [label="target T", shape=ellipse, fillcolor="#ffffff"]; ld [label="loss  mean((T̂ − T)²)", '
                 'shape=hexagon, fillcolor="#fde8e8"]; y -> ld; t -> ld;')
    else:
        ph = physics
        src = ', '.join(f"P{j + 1} = {e}" for j, e in enumerate(ph.get('src') or [])) or 'no loss source'
        L.append(f'xs [label="states x of the physics network (K)\\nx = ΔT·f(X), {_shape(y)}  ·  {nP:,} '
                 f'parameters", fillcolor="#f3e8f7", color="#8a4fb3"];')
        L.append(f'{prev} -> xs;')
        L += [f'y [label="T̂ = T_ref + O·x  ({o})", fillcolor="#fde8e8", color="#d03b3b"];', 'xs -> y;',
              't [label="measured T", shape=ellipse, fillcolor="#ffffff"];',
              'ld [label="data loss\\nL_data = mean(((T̂ − T)/ΔT)²)", shape=hexagon, fillcolor="#fde8e8"];',
              'y -> ld; t -> ld;',
              'dt [label="dx/dt\\n(central differences)", fillcolor="#f3e8f7"];',
              f'phx [label="{src}\\nfrom the input window (and T̂)", shape=ellipse, fillcolor="#fff4e5", '
              f'color="#e07b00"];',
              f'res [label="residual of the {ph.get("type", "")} network ({ph.get("nP", 0)} states)\\n'
              f'r = (C·dx/dt + G·x − E·P)/(diag(G)·ΔT)", fillcolor="#e8f4ea", color="#3f8f4f"];',
              'lp [label="physics loss\\nL_phys = mean(r²)", shape=hexagon, fillcolor="#e8f4ea", color="#3f8f4f"];',
              'li [label="initial condition\\nL_init = Σw·((T̂ − T(0))/ΔT)²/Σw", shape=hexagon, fillcolor="#eef3fb"];',
              f'tot [label="L = α·L_data + β·L_phys + γ·L_init\\nα = {fmt(ph["alpha"])}, β = {fmt(ph["beta"])}, '
              f'γ = {fmt(ph.get("gamma", 0))}  (first {int(ph.get("warm", 0))} samples skipped)", '
              f'shape=doubleoctagon, fillcolor="#fffbe6", color="#b38600"];',
              'xs -> dt; dt -> res; xs -> res; phx -> res; y -> phx [style=dotted]; res -> lp; ld -> tot; '
              'lp -> tot; y -> li; li -> tot;', 'x -> phx [style=dashed];']
        if ph['beta'] == 0 or not ph.get('nP'):
            L.append('note [label="β = 0 or no physics: the physics loss has no influence", shape=note, '
                     'fillcolor="#ffffff", color="#b38600"]; tot -> note [style=dotted];')
    L.append('}')
    return '\\n'.join(L)


ATOMS = ('Conv1d', 'LSTM', 'GRU', 'Linear', 'BatchNorm1d', 'LayerNorm', 'ReLU', 'GELU', 'Tanh', 'Dropout',
         'TransformerEncoderLayer', 'Pool', 'Identity')


def _atoms(net):
    """Modules drawn as one block, in definition order (a transformer layer is not split into its parts)."""
    out = []

    def walk(m):
        for c in m.children():
            if type(c).__name__ in ATOMS or not list(c.children()):
                out.append(c)
            else:
                walk(c)
    walk(net)
    return out


def summary(net, W, F):
    """Rows (layer, output shape, parameters) of a dlNet from a forward pass with a window of W samples."""
    import torch
    rec = []
    hooks = [m.register_forward_hook(lambda mod, i, o, mod_=m: rec.append((mod_, _shape(o)))) for m in _atoms(net)]
    net.eval()
    with torch.no_grad():
        y = net(torch.zeros(2, W, F))
    for h in hooks:
        h.remove()
    rows = [{'layer': _label(m), 'output shape': sh, 'parameters': sum(q.numel() for q in m.parameters())}
            for m, sh in rec if type(m).__name__ != 'Identity']
    return rows, _shape(y)


def _net(sel, F, M, seq, arch=None):
    from src.py.model.DL.dlNet import dlNet
    return dlNet(int(sel), int(F), int(M), int(seq), np.zeros(F), np.ones(F), arch=arch)


def _feat_names(cfg):
    inp = list(_get(cfg, 'Run', 'inp', default=[]) or [])
    if int(_get(cfg, 'Run', 'featRoll', default=0) or 0):
        a = list(_get(cfg, 'Run', 'EWMA', default=[]) or [])
        s = list(_get(cfg, 'Run', 'EWMS', default=[]) or [])
        inp = inp + [f"EWMA{v}({x})" for v in a for x in inp] + [f"EWMS{v}({x})" for v in s for x in inp]
    return inp or ['P']


#######################################################################################################################
# Synthetic profile (ML, DL window sketch)
#######################################################################################################################
@st.cache_data(show_spinner=False)
def _profile():
    Ts = 1.0
    rng = np.random.default_rng(42)
    P = np.repeat(rng.uniform(0, 100, 60), 40)
    T, _ = rcSim(np.array([0.1, 0.3]), np.array([20.0, 300.0]), P, Ts)
    return np.arange(len(P)) * Ts, P, T + 25


#######################################################################################################################
# ML
#######################################################################################################################
def theory_ml(cfg):
    from sklearn.linear_model import LinearRegression
    intro("ML mode: a static regression T(k) = f(x(k)) from the features of one sample; the dynamics enter only "
          "through rolled features (EWMA/EWMS, Run › featRoll).")
    cm.question("How can a static regression predict a temperature that depends on the history of the losses?")
    cm.equations([
        ("Static regression", r"\hat T(k) = f\left(x(k)\right),\quad x(k) = [P(k),\; \mu_{s_1}(k), \dots,\; "
                              r"\sigma_{s_1}(k), \dots]",
         "f: linear regression, tree, SVR, GPR, ensemble or MLP (setup selML)."),
        ("Rolled features (calcFeat)", [r"\mu_s(k) = \alpha P(k) + (1-\alpha)\,\mu_s(k-1),\quad \alpha = "
                                        r"\frac{2}{s+1}",
                                        r"\sigma_s^2(k) = (1-\alpha)\left(\sigma_s^2(k-1) + \alpha\,(P(k) - "
                                        r"\mu_s(k-1))^2\right)"],
         "An EWMA with span s is a first-order low pass with time constant ≈ s/2 samples — like one RC pair."),
    ], sym=[('x(k)', 'feature vector of sample k', 'W, …'), ('s', 'span of the moving average (Run › EWMA/EWMS)',
                                                             'samples'),
            ('μ_s, σ_s', 'exponentially weighted mean and standard deviation', 'W')], expanded=True)
    cm.section(None, "Explorer: linear regression with and without rolled features")
    spans = list(_get(cfg, 'Run', 'EWMA', default=[]) or []) or [10, 100, 500]
    txt = st.text_input("EWMA spans (samples)", ', '.join(str(int(v)) for v in spans), key='thML|spans')
    try:
        spans = [int(float(v)) for v in txt.replace(';', ',').split(',') if v.strip()]
    except ValueError:
        st.warning("Comma separated integers expected.")
    t, P, T = _profile()
    n = len(t) // 2
    X0 = P[:, None]
    X1 = calcFeat(P[:, None], spans, [], None, 0)
    res = {}
    for lab, X in (('P only', X0), (f'P + EWMA {spans}', X1)):
        m = LinearRegression().fit(X[0:n], T[0:n])
        res[lab] = m.predict(X)
    e = {k: np.abs(v[n:] - T[n:]).mean() for k, v in res.items()}
    cm.tiles([(f"MAE test, {k}", f"{fmt(v)} K", None) for k, v in e.items()])
    fig = go.Figure([line(t, T, 'temperature (2 RC pairs, τ = 20 s, 300 s)', 0, width=3)] +
                    [line(t, v, k, i + 1, dash='dot') for i, (k, v) in enumerate(res.items())])
    fig.add_vline(x=t[n], line_dash='dash')
    fig.update_layout(xaxis_title='t (s)', yaxis_title='T (°C)')
    show(fig, key='thML|fit')
    st.caption("Left of the dashed line: training, right: test.")
    e0, e1 = list(e.values())
    cm.interpret([
        f"With the loss alone the regression can only learn a static gain and misses every transient (MAE {fmt(e0)} K). "
        f"The EWMA features act as low-pass filters of the loss with time constants of about s/2 samples; a linear "
        f"combination of them is close to a Foster network. With the spans {spans} the MAE is {fmt(e1)} K"
        + (" — the spans cover the time constants of the profile (20 s and 300 s)." if e1 < 0.5 * e0 else
           " — the improvement is small because the spans do not reach the slow time constant of 300 s; try "
           "spans like 40, 200, 600."),
        "Choose spans that cover the thermal time constants (from τ_min to τ_max in samples). Non-linear models "
        "(tree, SVR, MLP) add flexibility for temperature-dependent losses, but need the same rolled features to see "
        "the history."])


#######################################################################################################################
# DL
#######################################################################################################################
def theory_dl(cfg, sel=None, physics=None):
    sel = int(sel or _get(cfg, 'Run', 'selDL', default=3) or 3)
    dl = _get(cfg, 'Mdl', 'dl', default={}) or {}
    W, seq, stride = int(dl.get('W', 600) or 600), int(dl.get('seq', 0) or 0), int(dl.get('stride', 1) or 1)
    names = _feat_names(cfg)
    outs = list(_get(cfg, 'Run', 'out', default=[]) or []) or ['T']
    if physics is None:
        intro("DL mode: a neural network maps a window of W samples of the input features to the temperatures.")
        cm.question("What does the network see, how is it built and how many parameters does it learn?")
    cm.equations([
        ("Windowing (dlWin)", [r"X_i = [x(i - s + 1), \dots, x(i + W - s)] \in \mathbb{R}^{W\times F}",
                               r"\text{seq2point } (s > 0):\; \hat y_i = T(i),\qquad \text{seq2seq } (s = 0):\; "
                               r"\hat y_i = [T(i), \dots, T(i + W - 1)]"],
         "W: window length (Mdl › dl › W), s: position of the target (Mdl › dl › seq), stride between windows."),
        ("Training", r"\min_\theta \frac{1}{N}\sum_i \lVert f_\theta(X_i) - y_i\rVert^2,\quad \text{Adam, lr} \cdot "
                     r"\text{lrDropFa}^{\lfloor ep/\text{lrDropPr}\rfloor},\;\text{early stopping after valPat}",
         "The network with the lowest validation loss is kept."),
    ], sym=[('x(k)', 'input features of sample k (F)', '–'), ('W', 'window length', 'samples'),
            ('θ', 'weights of the network', '–'), ('B', 'batch size (Mdl › dl › batch)', '–')],
        expanded=physics is None)
    cm.section(None, "Architecture" + (" of the PINN (network of the view *Network*, physics of the view "
                                       "*Physics*)" if physics else ""))
    c = st.columns([1, 1, 1, 1])
    if physics is None:
        arch = _get(cfg, 'Run', 'archDL', default=None)
        if arch:
            c[0].markdown("**Network:** user-defined (view *Network*)")
        else:
            sel = c[0].selectbox("Network", list(NETS), index=list(NETS).index(sel if sel in NETS else 3),
                                 format_func=NETS.get, key='thDL|sel')
    Wd = int(c[1].number_input("Window W (shown)", 2, 100000, min(W, 100000), key=f'th{"PI" if physics else "DL"}|W',
                               help="Window length of the diagram; the config value is the default."))
    st.caption(f"{len(names)} input features: {', '.join(names[0:12])}{' …' if len(names) > 12 else ''} → "
               f"{len(outs)} outputs: {', '.join(map(str, outs))}. Shapes: B = batch, then time × channels.")
    if physics is not None:
        names = names + ([f"ΔT0({o})" for o in outs] if physics.get('x0') else [])
    try:
        net = _net(sel, len(names), physics['n'] if physics else len(outs), 0 if physics else seq,
                   _get(cfg, 'Run', 'archPI' if physics else 'archDL', default=None))
        st.graphviz_chart(arch_dot(net, Wd, len(names), names if len(names) <= 6 else None, outs, physics),
                          width='content')
    except Exception as e:                                                         # never let a diagram kill the page
        st.warning(f"Architecture could not be drawn: {e}")
    if physics is not None:
        return
    cm.section(None, "Windows over a profile")
    t, P, T = _profile()
    Wp, sp = min(max(W, 20), 400), min(max(stride, 1), 400)
    fig = go.Figure([line(t, P, 'loss P (input)', 0), line(t, T, 'temperature (target)', 1, yaxis='y2')])
    for k in range(3):
        x0 = 600 + k * max(sp, Wp // 3 if seq else Wp)
        fig.add_vrect(x0=x0, x1=x0 + Wp, fillcolor=['#2a78d6', '#e07b00', '#3f8f4f'][k], opacity=0.12, line_width=0)
        if seq:
            fig.add_vline(x=x0 + min(seq, Wp) - 1, line_dash='dot', line_color=['#2a78d6', '#e07b00', '#3f8f4f'][k])
    fig.update_layout(xaxis_title='sample', yaxis_title='P (W)', yaxis2=dict(title='T (°C)', overlaying='y',
                                                                             side='right'))
    show(fig, key='thDL|win')
    st.caption(f"Shaded: three windows of W = {Wp} samples{' (shortened for the plot)' if Wp != W else ''}; "
               + (f"dotted: target sample s = {seq} (seq2point, one window per sample)." if seq else
                  f"seq2seq: consecutive blocks with stride {stride}, the network predicts the whole block."))
    table(pd.DataFrame([(k, v) for k, v in dl.items()], columns=['parameter (Mdl › dl)', 'value']), hide_index=True)
    cm.interpret([
        f"The {NETS.get(sel, 'network')} sees {W} samples of {len(names)} features at once; the window must be longer "
        f"than the slowest thermal time constant (in samples) that matters, otherwise the network cannot know the "
        f"thermal history. Rolled features (featRoll) carry the long history at a small W.",
        "CNN layers learn local filters (fast transients), the LSTM carries a state through the window (slow "
        "heating), the CNN + LSTM combines both. More parameters need more training data — compare the parameter "
        "count with the number of training windows.",
        "Training on CPU can take long: reduce epoch or W for first tests (tutorial dl_window)."])


#######################################################################################################################
# PI
#######################################################################################################################
def theory_pi(cfg):
    from gui.core import pinn_phys
    from src.py.model.PINN.pinnPhys import physDef
    sel = int((cfg.get('Run') or {}).get('selPI', 3) or 3)
    intro("PI mode: a sequence network predicts the states of a thermal network (Foster terms or Cauer nodes, one or "
          "several outputs); the loss adds the residual of that network to the data error. The physics regularises "
          "the network where data are scarce or noisy, the data correct the simplified physics.")
    cm.question("How do the data and the physics enter the training, and how should α, β and γ be chosen?")
    cm.equations([
        ("Physics network (view Physics)", r"C\,\dot x = E\,P - G\,x,\qquad \hat T = T_{ref} + O\,x",
         "Foster: one state θ_k per term, τ_k·dθ_k/dt = R_k·P_j − θ_k (G = I, C = τ, E = R); Cauer: node "
         "temperatures with capacitances C and conductances G, hidden nodes allowed."),
        ("Network", r"x = \Delta T\cdot f_\theta(X, \Delta T_0)",
         "f_θ: CNN/LSTM/own architecture on a window of W samples; ΔT₀: measured rise at the window start (x0)."),
        ("Residual", r"r = \frac{C\,\dot x + G\,x - E\,P}{\mathrm{diag}(G)\,\Delta T}",
         "dx/dt by central differences; dimensionless, so β does not depend on the units."),
        ("Loss", r"L = \alpha L_{data} + \beta L_{phys} + \gamma L_{init}",
         "Each part is a mean square relative to ΔT (see Physics › How the parameters enter the PINN)."),
    ], sym=[('x', 'states of the physics network (rises over T_ref)', 'K'),
            ('C, G, E, O', 'capacitance, conductance, input and output matrix', 'Ws/K, W/K, –, –'),
            ('P_j', 'loss sources (equations of the inputs, view Physics)', 'W'),
            ('T_ref', 'reference temperature (constant, feature or reference channel)', '°C'),
            ('ΔT', '99th percentile of |T − T_ref| of the training data', 'K'),
            ('α, β, γ', 'weights of the data, physics and initial-condition loss', '–'),
            ('W', 'window length (Mdl › dl › W)', 'samples')], expanded=True)
    pinn_phys.equation_panel(cfg, expanded=False)
    try:
        ph = physDef(cfg, cfg.get('Run') or {}, len(list((cfg.get('Run') or {}).get('out') or ['T'])))
        ph['beta'] = float(ph['beta'])
        theory_dl(cfg, sel=sel, physics=ph)
    except Exception as e:
        st.warning(f"The physics of the configuration is not valid ({e}) — fix it in the view *Physics*.",
                   icon=':material/warning:')
    pinn_phys.lab(cfg)


