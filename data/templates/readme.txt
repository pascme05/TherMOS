%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: data/readme                                                       %
% Date: 18.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TherMOS can take 1D or 2D/3D input data. For 1D either .xlsx or .mat files
are available for 2D/3D only .mat files can be used. The following data
structure must be followed:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description 1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The size of the data is Tx(M + 2N + 3) where T is the number of samples M 
is the number of input signals, N the number of output/reference signals.
Three addtional signals are needed for time, time per id, and the id 
label. The structure is as follows:

1) data.time:       Continuous time starting from 0 to (Ts*T-Ts)
2) data.time_id:    ID based time starting from 0 for each new ID
3) data.id:         ID label to seperate several profiles within one file
4) data.Fx:         Input feature x
5) data.Ox:         Output temperature x
6) data.Rx:         Reference temperature x


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The 2D/3D data file contains several variables, the variable name, meaning,
size, and unit are listed below assuming a 2D problem with T temporal
samples and M/N spatial samples:

1) Cp:      Spezific spatial heat capacitance of size MN (J/KgK)
2) dx:      Spatial resolution x direction (m)
3) dy:      Spatial resolution y direction (m)
4) geo:     Geometrical position in x,y,z of size MNx3 (m)
5) k:       Spatial thermal conductivity of size MN (W/mK)
6) Lx:      Length x direction (m)
7) Ly:      Length y direction (m)
8) r:       Spatial temperature reference of size TxMN (degC)
9) rho:     Spatial material density of size MN (kg/m3)
10) t:      Time vector of size Tx1 (sec)
11) Ts:     Sampling time (sec)
12) X:      Spatial input features of size TxMN (W)
13) y:      Spatial output features of size TxMN (degC)

The following three parameters are optional. If provided they will be used 
to calculate the results, otherwise the boundary conditions are calculated
based on the derivate of the temperatures at the boundary.

14) Ta:     Ambient temperature for convection at the boundary (degC)
15) h:      Heat transfer coefficient at the boundary (W/m2K)
16) f:      Heat flux boundary conditions (W/m2)