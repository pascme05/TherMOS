%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: config/journal_pinn/readme                                        %
% Date: 24.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.2                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: sections as in config/gui/start.json                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Configs of the PINN journal paper (half-bridge module), in the style of config/gui/start.json: sections (Info, Net,
Loss, Train, Samp, Mesh, ...) with the parameters as "Variable": Value. Every parameter has a comment "_Variable"
(description and unit); keys starting with '_' are ignored. Edit the values only.
CHANGE in a comment marks the parameters that depend on the computer (GPU memory, runtime); all other values are the
ones of the paper and should not be changed to reproduce it.
  data.json      Mesh (coarse, medium, fine), Sets (sizes of the data sets), Prof (profiles), Op (operating points)
  foster.json    coupled Foster model (number of RC pairs)
  pod.json       POD model (energy criterion of the rank)
  pinn.json      proposed physics-informed network: log-linear condition heads (Net/lin), standard plus extended
                 physics-only boards (Loss/physSet = phys+physExt, h = 250-5000 W/(m2K)), moment loss (Loss/delta,
                 nMom: moments 1, 2 of the transfer function of every physics-only board)
  nophys.json    the same network without physics (ablation)
  pinnBase.json  base network of the first version (no heads, standard physics-only boards, no moment loss; ablation)
  pinnLin.json, pinnLinBoth.json, pinnExt.json  ablation steps: base + heads; base + heads + both physics sets;
                 base with the extended physics-only boards only (setup journal_pinn_ablation.py)
  deeponet.json  physics-informed DeepONet (baseline: recurrent branch instead of the exact lags), standard
                 physics-only boards
  deeponetExt.json  DeepONet with the same physics-only boards as pinn (phys+physExt)
fosterNom and podNom (setup['methods']) use foster.json and pod.json, but are fitted on the nominal layout (trainNom).
The section Mesh of a method config replaces values on one mesh, e.g. "Mesh": {"fine": {"nR": 2048, "nCache": 8}}.
A new variant: copy a method config (e.g. pinn.json to pinn2.json), change the values, and add its name to
setup['methods'] of a setup; the results are then pinn2_<mesh>_<sensors>.json.
