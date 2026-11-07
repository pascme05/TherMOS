#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Theory & Tools
# File:         _common.py
# Date:         05.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Shared building blocks of the Theory & Tools pages: the question a tool answers, symbol tables, the box 'What the
result means', literature lines, key figures and export buttons. Every tool follows the same order: question →
inputs → equation with symbols → key figures → charts with 'How to read' → what the result means → export.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from gui.core.style import table, fmt, SERIES
from gui.core import literature as lit

KEY = 'tl'                                                                     # prefix of all widget keys


#######################################################################################################################
# Functions
#######################################################################################################################
def page_head(title, text):
    st.title(title)
    st.caption(text + " The tools are self-contained calculators: they do not change the configuration or the "
                      "workbooks. Results can be exported and copied into the input workbooks of the modes. Typical "
                      "values are taken from the literature and must be checked for the specific part.")


def question(q, text=None, art=None):
    """
    The engineering question the tool answers (first line of every tool); art: name of a task picture
    (gui/views/tools/task_art.py) shown next to it.
    """
    pic = ''
    if art:
        from gui.views.tools import task_art
        pic = task_art.svg(art)
    cols = st.columns([3, 2], vertical_alignment='center') if pic else [st.container()]
    with cols[0].container(border=True):
        st.markdown(f":material/help: **Question.** {q}")
        if text:
            st.caption(text)
    if pic:
        cols[1].markdown(pic, unsafe_allow_html=True)


def section(n, title, text=None, divider=True):
    """
    Numbered step heading with a coloured bar (n = None: unnumbered, e.g. 'Result'). Separates the inputs, the result
    and the interpretation of a tool so that a page reads as a sequence of steps.
    """
    if divider:
        st.divider()
    lab = f"{n} · {title}" if n is not None else title
    col = SERIES[0] if n is not None else SERIES[2]
    st.markdown(f"<div style='border-left:4px solid {col};padding:2px 0 2px 10px;margin:0 0 6px 0;"
                f"font-weight:600;font-size:1.08rem'>{lab}</div>", unsafe_allow_html=True)
    if text:
        st.caption(text)


def expert():
    """True when the global sidebar toggle 'Expert mode' is on (advanced inputs are shown)."""
    return bool(st.session_state.get('expert', False))


def hidden(n, what="advanced inputs"):
    """Caption for inputs that are hidden outside the expert mode (they keep their default values)."""
    if n and not expert():
        st.caption(f":material/visibility_off: {n} {what} use their default values — switch on *Expert mode* in "
                   f"the sidebar to change them.")


def equations(steps, sym=None, title="Equations", expanded=False):
    """
    Numbered equation steps [(title, latex, meaning)] in the order input → intermediate → result, followed by the
    symbol table (rows symbol, meaning, unit).
    """
    with st.expander(title, icon=':material/function:', expanded=expanded):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i} · {s[0]}**")
            for tex in ([s[1]] if isinstance(s[1], str) else s[1]):
                st.latex(tex)
            if len(s) > 2 and s[2]:
                st.caption(s[2])
        if sym:
            st.markdown("**Symbols**")
            table(pd.DataFrame(sym, columns=['symbol', 'meaning', 'unit']), hide_index=True,
                  column_config={'meaning': st.column_config.TextColumn(width='large')})


def interpret(paragraphs, title="Interpretation"):
    """
    Plain-language interpretation at the bottom of a tool: a few paragraphs of prose built from the current results
    (what was computed, the key number and its meaning, which input drives it, what to do).
    """
    st.divider()
    with st.container(border=True):
        st.markdown(f"**:material/lightbulb: {title}**")
        for p in paragraphs:
            if p:
                st.markdown(p)


def symbols(rows, title="Symbols of the equation", expanded=False):
    """Symbol table (symbol, meaning, unit) below an equation."""
    with st.expander(title, icon=':material/function:', expanded=expanded):
        table(pd.DataFrame(rows, columns=['symbol', 'meaning', 'unit']), hide_index=True,
              column_config={'meaning': st.column_config.TextColumn(width='large')})


def findings(lines, title="What the result means"):
    """Plain-language interpretation of the result (list of markdown sentences)."""
    with st.container(border=True):
        st.markdown(f"**:material/lightbulb: {title}**")
        st.markdown("\n".join(f"- {x}" for x in lines if x))


def refs(keys, prefix="Literature: "):
    """Linked literature line."""
    keys = [k for k in dict.fromkeys(keys) if k in lit.REFS]
    if keys:
        st.caption(prefix + " · ".join(f"[{lit.label(k)}]({lit.url(k)})" for k in keys))


def tiles(items):
    """Row of key figures [(label, value, help)]."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        lab, val = it[0], it[1]
        c.metric(lab, val, help=it[2] if len(it) > 2 else None)


def num(c, label, value, key, help=None, **kw):
    """Number input with the generic %g format (integer input for format '%d')."""
    f = kw.pop('format', '%g')
    cast = int if f == '%d' else float
    kw = {k: cast(v) if k in ('min_value', 'max_value', 'step') else v for k, v in kw.items()}
    if f == '%d':
        return c.number_input(label, value=int(value), key=f"{KEY}|{key}", help=help, format=f, **kw)
    return c.number_input(label, value=float(value), key=f"{KEY}|{key}", help=help, format=f, **kw)


def export(df, name, label="Download CSV", key=None, note=None):
    """CSV download of a table (with an optional note on where to paste it)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=f"{name}.csv", mime='text/csv',
                       icon=':material/download:', key=key or f"{KEY}|dl|{name}")
    if note:
        st.caption(note)


def export_json(obj, name, label="Download JSON", key=None):
    st.download_button(label, json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)), file_name=f"{name}.json",
                       mime='application/json', icon=':material/download:', key=key or f"{KEY}|dj|{name}")


def chain():
    """Hand-over between the tools (losses → thermal → lifetime → test) in the session."""
    return st.session_state.setdefault(f"{KEY}|chain", {})


def f_life(x, unit):
    """Life with its unit and a conversion to years for hours."""
    if unit == 'h' and np.isfinite(x):
        return f"{fmt(x)} h ({fmt(x / 8760)} a)"
    return f"{fmt(x)} {unit}"
