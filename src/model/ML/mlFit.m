%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: mlFit                                                             %
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
% This function fits a machine learning model.                
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
%       4) setup:   All setup files for the simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = mlFit(data, val, para, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting Machine Learning model")

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
    %----------------------------------------
    % K-Nearest Neigbours
    %----------------------------------------
    K = para.Mdl.knn.K;                                                     % number of neighbors

    %----------------------------------------
    % Random Forest
    %----------------------------------------
    leaf = para.Mdl.dt.leaf;                                                % number of leaves in the tree
    split = para.Mdl.dt.split;                                              % maximum number of splits
    parent = para.Mdl.dt.parent;                                            % maximum number of parents

    %----------------------------------------
    % Support Vector Machines
    %----------------------------------------
    C = para.Mdl.svm.C;                                                     % Box constraint SVM
    eps = para.Mdl.svm.eps;                                                 % Epsilon SVM

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
    % Kernel Method
    %===================================================
    if para.Mdl.svm.kernel == 1
        kernel = 'linear';
    elseif para.Mdl.knn.dist == 2
        kernel = 'polynomial';
    elseif para.Mdl.knn.dist == 3
        kernel = 'rbf';
    else
        kernel = 'linear';
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
        % K-Nearest Neigbours
        %----------------------------------------
        if setup.selML == 1
            % Optimise
            mdl = fitcknn(Pv_vl,T_vl,'OptimizeHyperparameters','auto',...
                                 'HyperparameterOptimizationOptions',...
                                  struct('AcquisitionFunctionName', ...
                                         'expected-improvement-plus'));

            % Parameter
            K = mdl.NumNeighbors;

        %----------------------------------------
        % Random Forest
        %----------------------------------------
        elseif setup.selML == 2
            % Optimise
            mdl = fitrtree(Pv_vl,T_vl,'OptimizeHyperparameters','auto',...
                                  'HyperparameterOptimizationOptions',...
                                   struct('AcquisitionFunctionName', ...
                                          'expected-improvement-plus'));

            % Parameter
            leaf = mdl.ModelParameters.MinLeaf;
            split = mdl.ModelParameters.MaxSplits;
            parent = mdl.ModelParameters.MinParent;
    
        %----------------------------------------
        % Support Vector Machines
        %----------------------------------------
        elseif setup.selML == 3
            % Optimise
            mdl = fitrsvm(Pv_vl,T_vl,'OptimizeHyperparameters','auto',...
                                 'HyperparameterOptimizationOptions',...
                                  struct('AcquisitionFunctionName', ...
                                         'expected-improvement-plus'));

            % Parameter
            C = max(mdl.BoxConstraints);
            eps = mdl.Epsilon;
            kernel = mdl.KernelParameters.Function;
        
        %----------------------------------------
        % Invalid
        %----------------------------------------
        else
            disp("ERROR: Invalid model for optimisation")
        end
    end

    %===================================================
    % Fixed Order Model
    %===================================================
    %----------------------------------------
    % K-Nearest Neigbours
    %----------------------------------------
    if setup.selML == 1
        mdl = fitcknn(Pv,T,'NumNeighbors',K,...
                       'NSMethod',NSMethod, ...
                       'Distance',Distance,...
                       'Standardize',1);
    
    %----------------------------------------
    % Random Forest
    %----------------------------------------
    elseif setup.selML == 2
        mdl = fitrtree(Pv, T, ...
                       'MinLeafSize', leaf, ...
                       'MaxNumSplits', split, ...
                       'MinParentSize', parent, ...
                       'SplitCriterion', 'mse', ...
                       'Prune', 'on');

    %----------------------------------------
    % Support Vector Machines
    %----------------------------------------
    elseif setup.selML == 3
        mdl = fitrsvm(Pv, T, ...
                  'KernelFunction', kernel, ...
                  'KernelScale', 'auto', ...
                  'BoxConstraint', C, ...
                  'Epsilon', eps, ...
                  'Standardize', true, ...
                  'Solver', 'SMO', ...
                  'Verbose', 1);
    
    %----------------------------------------
    % Invalid
    %----------------------------------------
    else
        disp("ERROR: Invalid model for machine learning")
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting Machine Learning model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1