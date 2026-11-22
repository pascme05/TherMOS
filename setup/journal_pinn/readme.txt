%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: setup/journal_pinn/readme                                         %
% Date: 22.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: setups named after the figures and tables of the paper        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Setups of the PINN journal paper (half-bridge module), one per group of figures and tables (numbers of the draft of
22.11.2026). Every setup has the same parameters (like start.py); the lines to adapt are marked CHANGE. The parameters
of the methods and data sets are in config/journal_pinn/<name>.json (see config/journal_pinn/readme.txt).
Start with: python setup/journal_pinn/<setup>.py, in this order:

  1 journal_pinn_Tab_IV_V_data.py                  data sets on 3 meshes (Tables IV, V; Figs. 4, 5)          ~2 h
  2 journal_pinn_Tab_VII_nominal.py                nominal layout N1, T2, T4: Foster, POD, networks           ~20 h
  3 journal_pinn_Tab_X_Fig_10_variable.py          variable layouts T1, T3: Foster, networks (shared runs)    ~1 h
  4 journal_pinn_Tab_VIII_IX_Fig_6_7_8_9_medium.py medium mesh details, field map (runs of 2 and 3)           ~10 min
  5 journal_pinn_Fig_11_samples.py                 number of training boards (medium, sensors L)              ~5 h
  6 journal_pinn_Tab_XI_Fig_12_load.py             memory and runtime per mesh (idle computer)                ~30 min
  7 journal_pinn_Fig_13_layout.py                  layout optimisation (network pinn_medium_L)                ~10 min
  8 journal_pinn_figures.py                        data of all figures and tables, figure PDFs (docu/paper)   ~5 min
  mainPINN.py                                      main function of the setups (no need to edit)

Times: RTX 3070 (8 GB), estimates; most of the time is the training of the networks on the fine mesh (~115 s per
epoch, ~3-5 h per run). Data: data/board/pfhb/<mesh>/. Results: setup['out'] (default results/pfhb). Existing
results are skipped (setup['skip'] = 1), so the setups share the network runs and an interrupted setup restarts where
it stopped.
Results per run: <method>_<mesh>_<sensors>[_n<boards>].json (+ .pt for the networks), methods foster, fosterNom,
podNom (sensors F: full field), pinn, nophys, deeponet.
A test on another computer: set setup['out'] (e.g. 'results/pfhb_test') and setup['name'], for a quick check of the
whole chain setup['epoch'] = 2; adapt the GPU memory in config/journal_pinn/pinn.json, nophys.json, deeponet.json
(entries with CHANGE: nCache, nbP, nR, Mesh/fine).
