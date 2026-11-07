#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         tutorials.py
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
Tutorials of the GUI (page Help › Tutorials, shown when TUTORIALS is not empty): guided scenarios per operating mode.
Every tutorial is a config (config/tutorials/<id>.json) made of a base config and 'changes' {'Sec.cat.var': value}
(run settings: '<RUN_SECTION>.var'). Write the configs with 'python -m gui.core.tutorials'. The 'look' texts must
come from runs of the tutorial configs — never invent the numbers.

Entry: {'id', 'mode', 'level' (Beginner | Intermediate | Advanced), 'minutes', 'base' (config name, default
'default'), 'title', 'story', 'goals' [..], 'changes' {..}, 'steps' [..], 'look' (markdown), 'exercise' [..]}
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core.config import CONFIG_DIR, from_json, to_json, default_config, read_config

import copy
import os

TUT_DIR = os.path.join(CONFIG_DIR, 'tutorials')
RUN = {}                                                              # run settings of all tutorials (e.g. save off)

#######################################################################################################################
# Tutorials
#######################################################################################################################
TUTORIALS = []


#######################################################################################################################
# Functions
#######################################################################################################################
def merge_changes(cfg, t):
    """Config of a tutorial: base config + run settings + changes 'Section.cat.var' (or '<RUN_SECTION>.var')."""
    cfg = copy.deepcopy(cfg)
    cfg[A.RUN_SECTION].update(RUN)
    for k, v in t['changes'].items():
        *path, last = k.split('.')
        d = cfg
        for p in path:
            d = d.setdefault(p, {})
        d[last] = copy.deepcopy(v)
    if A.MODE_VAR:
        cfg[A.RUN_SECTION][A.MODE_VAR] = t['mode']
    return cfg


def base(t):
    return default_config() if t.get('base', 'default') == 'default' else read_config(t['base'])


def config_file(tid):
    return os.path.join(TUT_DIR, f"{tid}.json")


def get(tid):
    return next(t for t in TUTORIALS if t['id'] == tid)


def load(tid):
    """Config of a tutorial (config/tutorials/<id>.json, generated from base and changes if the file is missing)."""
    f = config_file(tid)
    if os.path.isfile(f):
        with open(f, 'r', encoding='utf-8') as fh:
            return from_json(fh.read())
    return merge_changes(base(get(tid)), get(tid))


def write_all():
    """Writes config/tutorials/<id>.json for all tutorials."""
    os.makedirs(TUT_DIR, exist_ok=True)
    for t in TUTORIALS:
        with open(config_file(t['id']), 'w', encoding='utf-8') as fh:
            fh.write(to_json(merge_changes(base(t), t)))


if __name__ == '__main__':
    write_all()
