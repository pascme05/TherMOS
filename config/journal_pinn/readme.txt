%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: config/journal_pinn/readme                                        %
% Date: 21.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.0                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Configurations of the PINN journal paper, executed by setup/journal_pinn/journal_pinn_*.py:
  journal_pinn_net.json       hyperparameters of the network (all runs; a run overrides single values with 'cfg')
  journal_pinn_data.json      board families (Section III-A)
  journal_pinn_Sec_IV_A.json  ... journal_pinn_Sec_VI.json, journal_pinn_figures.json: steps of every section

Step types: data, run, baseline, layout, single, script (see setup/journal_pinn/runPaper.py).
A run is defined by
  name    results/pfparam/<name>.json and .pt
  train   number of labelled boards (the first n boards of the family)
  phys    family of physics-only boards (phys, physAll, physExt) or null (no physics loss)
  data    [labelled family, profiles per board, profile length in s]; omitted: family 'train', all profiles (2 x 40 s)
  cfg     overrides of journal_pinn_net.json (e.g. {"beta": 0.0, "gamma": 0.0} = data only, {"K": 8})
Keys starting with '_' are comments.
