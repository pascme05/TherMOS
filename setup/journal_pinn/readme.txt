%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: setup/journal_pinn/readme                                         %
% Date: 21.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.0                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Setups of the PINN journal paper (parametric modal operator network for the transient temperature field of power
electronic boards). Every setup runs one section; its steps are defined in config/journal_pinn/<setup>.json and the
hyperparameters of the network in config/journal_pinn/journal_pinn_net.json.

Order (each setup needs the ones above it):
  journal_pinn_00_data.py     III-A  board families data/board/pfparam/*.npz, single-board data   ~30 min
  journal_pinn_Sec_IV_A.py    IV-A   base, base2, nophys, nophys2, K8, K16, norel, noSS + baselines ~2.5 h
  journal_pinn_Sec_IV_B.py    IV-B   physAll, physExt, sensitivity (figsens)                       ~1 h
  journal_pinn_Sec_IV_C.py    IV-C   n3 ... n12np, tL10/tL20/tP1, pos1 ... pos24, mu1 ... mu24     ~4.5 h
  journal_pinn_Sec_IV_D.py    IV-D   runtime and memory over grid, length, sampling time (figscale) ~30 min
  journal_pinn_Sec_VI.py      VI     layout optimisation (VI-C), single-board extensions (VI-E)    ~2 h
  journal_pinn_figures.py     -      figure data and standalone figure PDFs (docu/paper, pdflatex)  ~5 min
  journal_pinn_all.py         all of the above in this order                                       ~11 h
Times: RTX 3070 (8 GB), one network training ~17 min.

Usage:  python setup/journal_pinn/journal_pinn_Sec_IV_A.py [options]
  --dry          list the steps and whether their results exist
  --force        recompute existing results (otherwise they are skipped, so a setup can be restarted)
  --only a,b     only these steps (run/baseline/single names or step types, e.g. --only base,krylov)
  --epoch N      maximum number of epochs (quick check of the chain, e.g. --epoch 2 --out <scratch folder>)
  --out DIR      results folder (default results/pfparam)

Results: results/pfparam/<run>.json (metrics per test set and board, training history), <run>.pt (network),
fv/pmor/pmorPos/krylov.json (baselines), layout_physAll.json, single.json (single-board runs).
The figure scripts (sensitivity, scaling, figure data) are part of the paper folder docu/paper; without it these steps
are skipped with a warning.
