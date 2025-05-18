%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: tutorial_DL                                                       %
% Date: 18.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This tutorial exemplarly shows the usage of DL models based on
% temperature measurement of a induction motor with different input
% features.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Toolkit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% #########################################################################
% #              Thermal Model Order Reduction (TherMOS)                  #
% #                                                                       #
% # This MATLAB script/toolkit focuses on the reduction of high-order     #
% # thermal models to more manageable lower-order models while preserving #
% # essential thermal dynamics. Thermal model order reduction is critical #
% # in simulations and real-time control applications where computational #
% # efficiency and accuracy are paramount. This process is typically      #
% # necessary for systems with a large number of thermal nodes, where     #
% # full-scale modeling would be computationally prohibitive.             #
% #                                                                       #
% # Thermal systems can be described by high-dimensional differential     #
% # equations of the form:                                                #
% #                                                                       #
% #        C*dT/dt + G*T = P(t),                                          #
% #                                                                       #
% # where C is the capacitance matrix, G is the conductance matrix,       #
% # T is the vector of node temperatures, and P(t) is the vector of       #
% # external heat inputs.                                                 #
% #                                                                       #
% # In many practical applications, this high-order system must be        #
% # reduced while maintaining the key thermal characteristics. Some       #
% # common methods for model order reduction in thermal systems include:  #
% #                                                                       #
% # 1. **Foster Models:** These are RC network models that approximate    #
% #    the thermal impedance of the system. By fitting an equivalent RC   #
% #    network to the thermal impedance, the order of the model can be    #
% #    significantly reduced. Foster models are particularly effective    #
% #    for lumped-parameter thermal systems.                              #
% #                                                                       #
% # 2. **State-Space Models:** State-space representation allows the      #
% #    system to be expressed in terms of a set of first-order            #
% #    differential equations:                                            #
% #                                                                       #
% #        dx/dt = A*x + B*u,                                             #
% #        y = C*x + D*u,                                                 #
% #                                                                       #
% #    where x is the state vector, u is the input vector, y is the       #
% #    output vector, and A, B, C, D are matrices defining the system.    #
% #    Techniques such as Balanced Truncation, Moment Matching, and       #
% #    Krylov Subspace methods are used to reduce the order of state-     #
% #    space models while preserving essential dynamics.                  #
% #                                                                       #
% # 3. **Proper Orthogonal Decomposition (POD):** POD is a statistical    #
% #    method that reduces the model order by decomposing the system into #
% #    a set of orthogonal modes, capturing the most energetic modes.     #
% #    This method is particularly useful in spatially distributed        #
% #    systems, where the temperature field can be approximated by a few  #
% #    dominant modes. The system is represented as:                      #
% #                                                                       #
% #        T(x,t) ≈ Σ φ_i(x) * a_i(t),                                    #
% #                                                                       #
% #    where φ_i(x) are the spatial modes and a_i(t) are the time-        #
% #    dependent coefficients.                                            #
% #                                                                       #
% # 4. **Other Techniques:** Methods such as Arnoldi-based approaches,    #
% #    Reduced Basis Methods, and Component Mode Synthesis are also       #
% #    employed in thermal model reduction, depending on the specific     #
% #    characteristics and requirements of the thermal system.            #
% #                                                                       #
% # This script/toolkit facilitates the application of these methods to   #
% # achieve efficient and accurate reduced-order models suitable for      #
% # simulation and control of complex thermal systems.                    #
% #########################################################################

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Format (do not change)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear variables
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init (do not change)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Path
%===================================================
path = initPath();

%===================================================
% Others
%===================================================
setup = struct();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Simulation Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% General
%===================================================
setup.user = "Pascal Schirmer";                                             % name of the user (string)
setup.name = "tutorial_DL";                                                 % name of the simulation (string)

%===================================================
% Experiment
%===================================================
%----------------------------------------
% Configuration
%----------------------------------------
setup.config = "tutorial_DL";                                               % name of the config file (string)

%----------------------------------------
% Plotting and Saving
%----------------------------------------
setup.plotOut = 2;                                                          % 0) model output is not plotted, 1) model output is plotted, 2) plotting one figure per output
setup.plotMdl = 0;                                                          % 0) model structure is not plotted, 1) model structure is plotted
setup.featRank = 0;                                                         % 0) No feature ranking, 1) feature ranking
setup.save = 1;                                                             % 0) output data not saved, 1) output data saved

%----------------------------------------
% Training and Testing
%----------------------------------------
setup.train = 1;                                                            % 0) no model training, 1) model training
setup.test = 1;                                                             % 0) no model testing, 1) model testing


%===================================================
% Data
%===================================================
%----------------------------------------
% Format and Dimension
%----------------------------------------
setup.format = 'mat';                                                       % input data format: "xlsx" or "mat"
setup.datDim = 1;                                                           % input data dimension: 1) 1D data, 2) 2D data, 3) 3D data
setup.datSep = 3;                                                           % input data separation: 1) File based, 2) Sheet based, 3) ID based, 4) Ratio based

%----------------------------------------
% Training and Testing Selection
%----------------------------------------
% File based
setup.trFile = ["motorTemp"];                                               % list of training files (strings) at least one
setup.teFile = "motorTemp";                                                 % testing file (string) exactly one
setup.vlFile = ["motorTemp"];                                               % list of validation files (strings) at least one

% Sheet based
setup.trSheet = ["OP1", "OP2"];                                             % list of training sheets (strings) at least one
setup.teSheet = "OP3";                                                      % testing sheet (string) exactly one
setup.vlSheet = ["OP3"];                                                    % list of validation sheets (strings) at least one

% ID based
setup.trID = [];                                                            % list of training IDs (if empty all remaining ones are used)
setup.teID = [60];                                                  % list of testing IDs 60, 62, 74
setup.vlID = [10, 48, 63];                                                  % list of validation IDS

% Ratio based
setup.rTr = 0.7;                                                            % percentag of training data (testing data is 1 - rTr)
setup.rVl = 0.2;                                                            % percentag of validation data (this is taken from the training data)

%----------------------------------------
% Inputs and Output Mapping
%----------------------------------------
setup.inp = ["Ta", "Tc", "Us", "Is"];                                       % list of input feature (strings) "Ta", "Tc", "Us", "Is", "Ss", "Tm", "Wm", "Iw"
setup.out = ["T_sw", "T_st", "T_so", "T_rm"];                               % list of temperature outputs (strings) 
setup.ref = ["Tc", "Tc", "Tc", "Tc"];                                       % list of reference temperatures (strings) 

%===================================================
% Model
%===================================================
%----------------------------------------
% Analytical 1D Models
%----------------------------------------
% Single-Input Single-Output (SISO)
setup.selRC = 0;                                                            % RC-Model (Foster) 0) de-activated, 1) activated

% Multiple-Input Multiple-Output (MIMO)
setup.selSS = 0;                                                            % SS-Model (State-Space) 0) de-activated, 1) activated

%----------------------------------------
% Analytical 2D Models
%----------------------------------------
setup.selPO = 0;                                                            % PO-Model (Proper-Orthogonal) 0) de-activated, 1) activated
setup.selPS = 0;                                                            % PO-SS-Model (Proper-Orthogonal State Space) 0) de-activated, 1) activated

%----------------------------------------
% Machine/Deep Learning (tbi)
%----------------------------------------
setup.selML = 0;                                                            % ML-Model (Machine Learning) 0) de-activated, 1) LR, 2) RF, 3) SVR, 4) GRP, 5) EN, 6) NN
setup.selDL = 3;                                                            % DL-Model (Machine Learning) 0) de-activated, 1) CNN, 2) LSTM, 3) CNN+LSTM


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculation (do not change)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
main(setup, path);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1