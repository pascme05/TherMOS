%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: cnnFit                                                            %
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
% This function solves a deep learning model using CNNs.   
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = cnnFit(data, val, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Training a CNN")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % General Parameter
    %===================================================
    [~, N] = size(data.y);                                                  % number of samples Nt and temperature nodes N
    
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
    valPat = para.Mdl.dl.valPat;

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    Pv = data.X;
    T = data.y;

    %----------------------------------------
    % Validation
    %----------------------------------------
    Pv_vl = val.X;
    T_vl = val.y;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Reshape Data
    %===================================================
    trainX = reshape(Pv', [size(Pv, 2), size(Pv, 1), 1]);
    valX = reshape(Pv_vl', [size(Pv_vl, 2), size(Pv_vl, 1), 1]);
    trainY = reshape(T', [size(T, 2), size(T, 1), 1]);
    valY = reshape(T_vl', [size(T_vl, 2), size(T_vl, 1), 1]);

    %===================================================
    % Shuffel
    %===================================================
    if para.Mdl.dl.shu == 1
        shu = 'never';
    elseif para.Mdl.dl.shu == 2
        shu = 'once';
    elseif para.Mdl.dl.shu == 3
        shu = 'every-epoch';
    else
        shu = 'every-epoch';
    end

    %===================================================
    % Training Options
    %===================================================
    options = trainingOptions('adam', ...
                              'MaxEpochs', epoch, ...
                              'MiniBatchSize', batch, ...
                              'Shuffle', shu, ...
                              'GradientThreshold', gradTh, ...
                              'ValidationPatience', valPat, ...
                              'InitialLearnRate', initLr, ...
                              'LearnRateSchedule', 'piecewise', ...
                              'LearnRateDropPeriod', lrDropPr, ...
                              'LearnRateDropFactor', lrDropFa, ...
                              'ValidationData', {valX, valY}, ...
                              'ValidationFrequency', valFreq, ...
                              'Verbose', 0, ...
                              'Plots', 'training-progress');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Model
    %===================================================
    layers = [ ...
            sequenceInputLayer(size(trainX, 1))
            convolution1dLayer(3, 64, 'Padding', 'same')
            reluLayer
            convolution1dLayer(3, 64, 'Padding', 'same')
            reluLayer
            flattenLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(N)
            regressionLayer];

    %===================================================
    % Fitting
    %===================================================
    mdl = trainNetwork(trainX, trainY, layers, options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Training a CNN")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1