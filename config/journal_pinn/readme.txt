%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: config/journal_pinn/readme                                        %
% Date: 21.11.2026                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.1.0                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Configs of the PINN journal paper (half-bridge module): every parameter is one entry with
Name | Category | Description | Variable | Value | Unit (like the config workbooks). Edit 'Value' only.
  data.json     meshes (coarse, medium, fine), data sets, operating points of the half-bridge
  foster.json   coupled Foster model (number of RC pairs)
  pod.json      POD model (energy criterion of the rank)
  pinn.json     proposed physics-informed network
  nophys.json   the same network without physics (ablation)
A new variant: copy a method config, change the values, add its name to setup['methods'] of a setup.
