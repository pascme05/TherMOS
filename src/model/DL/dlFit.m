%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: dlFit                                                             %
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
% This function fits a deep learning model based on inputs (power losses),
% X (NtxF) with Nt samples and F features, and outputs (temperatures) y
% (NtxN) with Nt samples and N nodes using a regression function r()
% parameterized by a set of free parameters.
%
%                              T = r(Pv)
%
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
%       4) setup:   All setup files for the simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = dlFit(data, val, para, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Training a Deep Learning Model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % General Parameter
    %===================================================
    [T, M] = size(data.y);                                                  % number of training samples Nt and temperature nodes N
    [T_val, ~] = size(val.y);                                               % number of validation samples Nt and temperature nodes N
    [~, F] = size(data.X);                                                  % number of features F
    W = para.Mdl.dl.W;                                                      % Length of each sequence (window size)
    stride = para.Mdl.dl.stride;                                            % Step size to slide the window

    %===================================================
    % Solver Parameter
    %===================================================
    epoch = para.Mdl.dl.epoch;
    gradTh = para.Mdl.dl.gradTh;
    initLr = para.Mdl.dl.initLr;
    lrDropPr = para.Mdl.dl.lrDropPr;
    lrDropFa = para.Mdl.dl.lrDropFa;
    valFreq = para.Mdl.dl.valFreq;
    batch = para.Mdl.dl.batch;
    shuOpt  = para.Mdl.dl.shu;

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    X = data.X;
    Y = data.y;

    %----------------------------------------
    % Validation
    %----------------------------------------
    X_val = val.X;
    Y_val = val.y;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Window Data
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    % Calculate number of windows
    numWin = floor((T - W) / stride) + 1;
    
    % Preallocate arrays
    Xtrain = zeros(W, F, numWin);   % [time, features, batch]
    Ytrain = zeros(M, numWin);      % [response, batch]
    
    % Efficient windowing loop
    for idx = 1:numWin
        i = (idx - 1) * stride + 1;
        Xtrain(:, :, idx) = X(i:i+W-1, :);        % W x F
        Ytrain(:, idx) = Y(i+W-1, :)';            % M x 1
    end

    %----------------------------------------
    % Validation
    %----------------------------------------
    % Calculate number of windows
    numWin_val = floor((T_val - W) / stride) + 1;
    
    % Preallocate arrays
    Xval = zeros(W, F, numWin_val);     % [time, features, batch]
    Yval = zeros(M, numWin_val);        % [response, batch]
    
    % Efficient windowing loop
    for idx = 1:numWin_val
        i = (idx - 1) * stride + 1;
        Xval(:, :, idx) = X_val(i:i+W-1, :);     % W x F
        Yval(:, idx) = Y_val(i+W-1, :)';         % M x 1
    end

    %===================================================
    % Shuffel
    %===================================================
    switch shuOpt
        case 1, shu = 'never';
        case 2, shu = 'once';
        case 3, shu = 'every-epoch';
        otherwise, shu = 'every-epoch';
    end

    %===================================================
    % Training Options
    %===================================================
    options = trainingOptions('adam', ...
                              'MaxEpochs', epoch, ...
                              'MiniBatchSize', batch, ...
                              'Shuffle', shu, ...
                              'GradientThreshold', gradTh, ...
                              'InitialLearnRate', initLr, ...
                              'LearnRateSchedule', 'piecewise', ...
                              'LearnRateDropPeriod', lrDropPr, ...
                              'LearnRateDropFactor', lrDropFa, ...            
                              'Verbose', 1, ...
                              'ValidationData', {Xval, Yval'}, ...
                              'ValidationFrequency', valFreq, ...
                              'ValidationPatience', 10, ...   
                              'OutputNetwork', 'best-validation-loss',...
                              'VerboseFrequency', 20, ...                 
                              'ExecutionEnvironment', 'auto', ...
                              'Plots', 'none');   


                              

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Model
    %===================================================
    if setup.selDL == 1
        layers = [
            sequenceInputLayer(F, 'Normalization', 'zscore')
            convolution1dLayer(5, 64, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            convolution1dLayer(3, 32, 'Padding', 'same')
            reluLayer
            globalAveragePooling1dLayer
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(M)
        ];
    
    elseif setup.selDL == 2
        layers = [ ...
            sequenceInputLayer(F, Normalization="zscore")
            lstmLayer(64, OutputMode="last")
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(M)
        ];

    elseif setup.selDL == 3
        layers = [
            sequenceInputLayer(F, 'Normalization', 'zscore')
            convolution1dLayer(3, 32, 'Padding', 'same') 
            reluLayer
            lstmLayer(64, 'OutputMode', 'last')
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(M)
        ];

    else
        disp('Error: No valid choice of DL model.')
    end

    %===================================================
    % Fitting
    %===================================================
    mdl = trainnet(Xtrain, Ytrain',layers,"mse",options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Training a Deep Learning Model")
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1