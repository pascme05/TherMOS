%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: setup/journal_pinn/readme                                         %
% Date: 21.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.0                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Setups of the PINN journal paper (half-bridge module). Every setup has the same parameters (like start.py) and
reproduces one section; the parameters of the methods and data sets are in config/journal_pinn/<name>.json.
Start with: python setup/journal_pinn/<setup>.py

  journal_pinn_data.py        data sets on 3 meshes (III-B), run first                    ~1 h
  journal_pinn_Sec_IV_A.py    Foster, POD, network, network w/o physics on 6 data sets     ~6 h
  journal_pinn_Sec_IV_B.py    extrapolation T2-T4 (same runs as IV-A)                      -
  journal_pinn_Sec_IV_C.py    number of training boards (medium mesh, sensors L)          ~1.5 h
  journal_pinn_Sec_IV_D.py    memory and computational load per mesh                      ~30 min
  journal_pinn_Sec_VI.py      layout optimisation                                          ~10 min
  journal_pinn_figures.py     figure data and figure PDFs (needs docu/paper, pdflatex)     ~5 min
  mainPINN.py                 main function of the setups (no need to edit)
Times: RTX 3070 (8 GB). Data: data/board/pfhb/<mesh>/. Results: results/pfhb/. Existing results are skipped
(setup['skip'] = 1).
