#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         help.py
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
This function holds the extended help of the parameters (shown in the question marks of the forms), the parameters of
the basic (non-expert) mode and the conditions under which a parameter or category is used at all.

HELP:            'Section.Category.Variable' -> markdown text (what it does, effect of the value, typical values)
BASIC:           parameters shown when the expert mode is off (empty: all parameters are basic)
ACTIVE:          'Section.Category.Variable' -> function(cfg) that is False if the parameter has no effect with the
                 current settings (hidden in the basic mode, marked as inactive in the expert mode)
RELEVANT:        section -> {category: function(cfg)} False if the category is not used by the selected sub-mode
CATEGORY_TITLES: category -> title of its expander
"""

#######################################################################################################################
# Help texts
#######################################################################################################################
HELP = {
    # ==================================================================================================================
    # Section
    # ==================================================================================================================
    # 'General.Numeric.numMC':
    #     "Number of Monte Carlo repetitions. More runs give smoother distributions, the run time grows linearly. "
    #     "Start with 20-100 for exploration and use >= 1000 for final results.",
}

#######################################################################################################################
# Basic parameters (expert mode off)
#######################################################################################################################
BASIC = set()

#######################################################################################################################
# Activity rules
#######################################################################################################################
ACTIVE = {
    # 'Section.Statistic.parUnc': lambda c: c['Section']['Statistic']['unc'] == 1,
}

#######################################################################################################################
# Categories
#######################################################################################################################
RELEVANT = {
    # 'Section': {'Opti': lambda c: c['Section']['Basic']['mode'] == 2},
}
CATEGORY_TITLES = {}


#######################################################################################################################
# Helper
#######################################################################################################################
def key(section, p):
    return f"{section}.{p['cat']}.{p['var']}"


def is_basic(section, p):
    return not BASIC or key(section, p) in BASIC


def is_active(section, p, cfg):
    rule = ACTIVE.get(key(section, p))
    if rule is None:
        return True
    try:
        return bool(rule(cfg))
    except (KeyError, TypeError, ValueError):
        return True
