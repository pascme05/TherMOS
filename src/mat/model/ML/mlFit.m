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
    
    %----------------------------------------
    % Gaussian Process Regression
    %----------------------------------------
    iterGPR = para.Mdl.gpr.iterMax;                                         % Maximum iterations of the GPR model

    %----------------------------------------
    % Ensemble of Boosted Trees
    %----------------------------------------
    cycle = para.Mdl.en.cycle;                                              % Maximum number of learning cycles EN
    splitEN = para.Mdl.en.split;                                            % Maximum number of splits EN

    %----------------------------------------
    % Shallow NN
    %----------------------------------------
    iterNN = para.Mdl.nn.iterMax;                                           % Maximum number of iterations NN
    nNodes = para.Mdl.nn.nodes;                                             % Maximum number of nodes NN

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
    % Kernel Methods
    %===================================================
    %----------------------------------------
    % SVM
    %----------------------------------------
    if para.Mdl.svm.kernel == 1
        kernelSVM = 'linear';
    elseif para.Mdl.svm.kernel == 2
        kernelSVM = 'polynomial';
    elseif para.Mdl.svm.kernel == 3
        kernelSVM = 'rbf';
    else
        kernelSVM = 'linear';
    end

    %----------------------------------------
    % GPR
    %----------------------------------------
    if para.Mdl.gpr.kernel == 1
        kernelGPR = 'exponential';
    elseif para.Mdl.gpr.kernel == 2
        kernelGPR = 'squaredexponential';
    else
        kernelGPR = 'squaredexponential';
    end
    
    %===================================================
    % Basis Functions and Aggregation
    %===================================================
    %----------------------------------------
    % GPR
    %----------------------------------------
    if para.Mdl.gpr.basis == 1
        basis = 'constant';
    elseif para.Mdl.gpr.basis == 2
        basis = 'linear';
    elseif para.Mdl.gpr.basis == 3
        basis = 'pureQuadratic';
    else
        basis = 'linear';
    end

    %----------------------------------------
    % EN
    %----------------------------------------
    if para.Mdl.en.method == 1
        method = 'LSBoost';
    elseif para.Mdl.en.method == 2
        method = 'Bag';
    else
        method = 'LSBoost';
    end
    
    %===================================================
    % Layer Structure
    %===================================================
    %----------------------------------------
    % Layers
    %----------------------------------------
    if para.Mdl.nn.layer == 1
        layerNN = [nNodes];
    elseif para.Mdl.nn.layer == 2
        layerNN = [nNodes, nNodes];
    elseif para.Mdl.nn.layer == 3
        layerNN = [nNodes, nNodes, nNodes];
    else
        layerNN = [nNodes, nNodes];
    end

    %----------------------------------------
    % Activation
    %----------------------------------------
    if para.Mdl.nn.act == 1
        act = 'relu';
    elseif para.Mdl.nn.act == 2
        act = 'tanh';
    elseif para.Mdl.nn.act == 3
        act = 'sigmoid';
    else
        act = 'relu';
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

    % %===================================================
    % % Init Value
    % %===================================================
    % T_vl = T_vl - T_vl(1);
    % T = T - T(1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Optimal Model
    %===================================================
    if para.Mdl.gen.opt == 1
        %----------------------------------------
        % Random Forest
        %----------------------------------------
        if setup.selML == 2
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
            kernelSVM = mdl.KernelParameters.Function;
        
        %----------------------------------------
        % Gaussian Process Regression
        %----------------------------------------
        elseif setup.selML == 4
        
        %----------------------------------------
        % Ensemble of Boosted Trees
        %----------------------------------------
        elseif setup.selML == 5
            % Optimise
            t = templateTree('Surrogate','on');
            mdl = fitrensemble(Pv_vl,T_vl,'MPG','Learners',t, ...
                                          'OptimizeHyperparameters',...
                                          {'NumLearningCycles',...
                                           'MaxNumSplits'});

            % Parameter
            cycle = mdl.NumLearningCycles;
            splitEN = mdl.MaxNumSplits;


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
    % Linear Regression
    %----------------------------------------
    if setup.selML == 1
        mdl = fitlm(Pv, T);
    
    %----------------------------------------
    % Decision Tree
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
                  'KernelFunction', kernelSVM, ...
                  'KernelScale', 'auto', ...
                  'BoxConstraint', C, ...
                  'Epsilon', eps, ...
                  'Standardize', true, ...
                  'Solver', 'SMO', ...
                  'Verbose', 1);
    
    %----------------------------------------
    % Gaussian Process Regression
    %----------------------------------------
    elseif setup.selML == 4
        mdl = fitrgp(Pv, T,'KernelFunction',kernelGPR,...
                           'BasisFunction', basis,...
                           'IterationLimitBCD', iterGPR, ...
                           'Verbose', 1);

    %----------------------------------------
    % Ensemble of Boosted Trees
    %----------------------------------------
    elseif setup.selML == 5
        mdl = fitrensemble(Pv, T, 'Method', method, ...
                                  'NumLearningCycles', cycle, ...
                                  'Learners', templateTree('MaxNumSplits', splitEN));

    %----------------------------------------
    % Shallow NN
    %----------------------------------------
    elseif setup.selML == 6
        mdl = fitrnet(Pv, T, 'Standardize', true, ...
                             'LayerSizes', layerNN, ...
                             'Activations', act, ...
                             'Verbose', 1,...
                             'IterationLimit', iterNN);
    
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