%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: knnFit                                                            %
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
% This function fits a k-nearest-neighbor model.                
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = knnFit(data, val, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting KNN ML model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % General Parameter
    %===================================================
    K = para.Mdl.knn.K;                                                     % number of neighbors
    Nt = min(data.Nt);                                                      % minimum number of training time steps
    [~, N] = size(data.X2);                                                 % number of training datasets used for fitting
    Nt_vl = min(val.Nt);                                                    % minimum number of validation time steps
    [~, N_vl] = size(val.X2);                                               % number of validation datasets used for fitting
    
    %===================================================
    % General Parameter
    %===================================================
    Pv = data.X;
    Pv_vl = val.X;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Optimiser
    %===================================================
    if para.Mdl.knn.ns == 1
        NSMethod = 'exhaustive';
    else
        NSMethod = 'kdtree';
    end

    %===================================================
    % Distance Metric
    %===================================================
    if para.Mdl.knn.dist == 1
        Distance = 'cityblock';
    elseif para.Mdl.knn.dist == 2
        Distance = 'chebychev';
    elseif para.Mdl.knn.dist == 3
        Distance = 'euclidean';
    elseif para.Mdl.knn.dist == 4
        Distance = 'minkowski';
    else
        Distance = 'euclidean';
    end

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
        mdl = fitcknn(Pv_vl,T_vl,'OptimizeHyperparameters','auto',...
                                 'HyperparameterOptimizationOptions',...
                                  struct('AcquisitionFunctionName', ...
                                         'expected-improvement-plus'));
        %----------------------------------------
        % Extract Parameters
        %----------------------------------------
        K = mdl.NumNeighbors;
    end

    %===================================================
    % Fixed Order Model
    %===================================================
    mdl = fitcknn(Pv,T,'NumNeighbors',K,...
                       'NSMethod',NSMethod, ...
                       'Distance',Distance,...
                       'Standardize',1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting KNN ML model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1