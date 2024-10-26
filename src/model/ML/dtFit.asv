%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: dtFit                                                             %
% Date: 13.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function fits a decision tree model.                
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = dtFit(data, val, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting DT ML model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % General Parameter
    %===================================================
    Nt = min(data.Nt);                                                      % minimum number of training time steps
    [~, N] = size(data.X2);                                                 % number of training datasets used for fitting
    Nt_vl = min(val.Nt);                                                    % minimum number of validation time steps
    [~, N_vl] = size(val.X2);                                               % number of validation datasets used for fitting
    
    %===================================================
    % Model Parameter
    %===================================================
    leaf = para.Mdl.dt.leaf;
    split = para.Mdl.dt.split;
    parent = para.Mdl.dt.parent;

    %===================================================
    % Variables
    %===================================================
    Pv = data.X;
    Pv_vl = val.X;
    T = data.y;
    T_vl = val.y;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Averaging Losses
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    for i = 1:N
        if i == 1
            T = data.y2{1,i}(1:Nt,1);
        else
            T = T + data.y2{1,i}(1:Nt,1);
        end
    end
    T = T / N;
    
    %----------------------------------------
    % Validation
    %----------------------------------------
    for i = 1:N_vl
        if i == 1
            T_vl = val.y2{1,i}(1:Nt_vl,1);
        else
            T_vl = T_vl + val.y2{1,i}(1:Nt_vl,1);
        end
    end
    T_vl = T_vl / N_vl;

    %===================================================
    % Init Value
    %===================================================
    T_vl = T_vl - T_vl(1);
    T = T - T(1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Optimal Model
    %===================================================
    if para.Mdl.gen.opt == 1
        %----------------------------------------
        % Optimise
        %----------------------------------------
        mdl = fitrtree(Pv_vl,T_vl,'OptimizeHyperparameters','auto',...
                                  'HyperparameterOptimizationOptions',...
                                   struct('AcquisitionFunctionName', ...
                                          'expected-improvement-plus'));
        
        %----------------------------------------
        % Extract Parameters
        %----------------------------------------
        leaf = mdl.ModelParameters.MinLeaf;
        split = mdl.ModelParameters.MaxSplits;
        parent = mdl.ModelParameters.MinParent;
    end

    %===================================================
    % Fixed Order Model
    %===================================================
    mdl = fitrtree(Pv, T, ...
                   'MinLeafSize', leaf, ...
                   'MaxNumSplits', split, ...
                   'MinParentSize', parent, ...
                   'SplitCriterion', 'mse', ...
                   'Prune', 'on');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting DT ML model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1