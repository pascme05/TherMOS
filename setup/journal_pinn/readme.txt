%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: setup/journal_pinn/readme                                         %
% Date: 24.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.2                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: setups named after the figures and tables of the paper        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Setups of the PINN journal paper (half-bridge module), one per group of figures and tables (numbers of the draft of
22.11.2026). Every setup has the same parameters (like start.py); the lines to adapt are marked CHANGE. The parameters
of the methods and data sets are in config/journal_pinn/<name>.json (see config/journal_pinn/readme.txt).
Start with: python setup/journal_pinn/<setup>.py, in this order:

  1 journal_pinn_Tab_IV_V_data.py                  data sets on 3 meshes (Tables IV, V; Figs. 4, 5)           ~2 h
  2 journal_pinn_Tab_VII_nominal.py                nominal layout N1, T2, T4, T5: Foster, POD, networks       ~30 h
  3 journal_pinn_Tab_X_Fig_10_variable.py          variable layouts T1, T3: Foster, POD (shared network runs) ~1 h
  4 journal_pinn_ablation.py                       ablation of the proposed network (coarse, medium)          ~3 h
  5 journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium.py medium mesh details, field map (runs of 2 and 3)           ~10 min
  6 journal_pinn_Fig_11_samples.py                 number of training boards (medium, sensors L)              ~5 h
  7 journal_pinn_Tab_XI_Fig_12_load.py             memory and runtime per mesh (idle computer)                ~30 min
  8 journal_pinn_Fig_13_layout.py                  layout optimisation (network pinn_medium_L)                ~10 min
  9 journal_pinn_figures.py                        data of all figures and tables, figure PDFs (docu/paper)   ~5 min
  journal_pinn_all.py                              all setups 1-9 in one go (list setup['setups'])            ~42 h
  mainPINN.py                                      main function of the setups (no need to edit)
  journal_pinn_operator.py, journal_pinn_coord.py  single-board PF networks (main.py, config pinn_*.xlsx; earlier draft,
                                                   not in journal_pinn_all.py; the V1.2 extensions do not apply)

Times: RTX 3070 (8 GB), estimates; most of the time is the training of the networks on the fine mesh (~115 s per
epoch, ~3-5 h per run). Data: data/board/pfhb/<mesh>/ (incl. the extended physics-only boards physExt and the testing
set T5; existing data sets are kept). Results: setup['out'] (default results/pfhb_v2; the first version is in
results/pfhb). Existing results are skipped (setup['skip'] = 1), so the setups share the network runs and an
interrupted setup restarts where it stopped. Rerun of all results: python setup/journal_pinn/journal_pinn_all.py
(an empty results/pfhb_v2 is created; do not reuse results/pfhb, since pinn is a new network there).
Results per run: <method>_<mesh>_<sensors>[_n<boards>].json (+ .pt for the networks), methods foster, fosterNom,
pod, podNom (sensors F: full field), pinn, nophys, deeponet, deeponetExt, pinnBase, pinnLin, pinnLinBoth, pinnExt.
Proposed network (pinn, V1.2): base network plus (i) log-linear condition heads (decay rates and amplitudes follow a
power law of h beyond the training range), (ii) the standard (h = 500-5000 W/(m2K)) plus the extended physics-only
boards (h = 250-5000, no simulation needed), (iii) the moment loss (moments 1, 2 of the transfer function of every
physics-only board, i.e., the slow time constants). The labelled boards cover h = 500-5000: T2 (h = 250) lies outside
the labels but inside the physics, T5 (h = 150) outside both. Ablation (setup 4): pinnBase (first version) ->
pinnLin (i) -> pinnLinBoth (i, ii) -> pinn (i, ii, iii); pinnExt: base network with physExt only. deeponetExt: DeepONet
with the same physics-only boards as pinn (no moment loss, the recurrent branch has no decay rates).
A test on another computer: set setup['out'] (e.g. 'results/pfhb_test') and setup['name'], for a quick check of the
whole chain setup['epoch'] = 2; adapt the GPU memory in config/journal_pinn/pinn*.json, nophys.json, deeponet*.json
(entries with CHANGE: nCache, nbP, nR, Mesh/fine).
GUI: python start_gui.py, page Journals > PINN journal (all setups, their settings, runs, logs and results).
