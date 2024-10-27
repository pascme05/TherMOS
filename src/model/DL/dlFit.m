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
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = dlFit(data, val, para)
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
    [Nt, N] = size(data.y);                                                 % number of training samples Nt and temperature nodes N
    [Nt_vl, ~] = size(val.y);                                               % number of training samples Nt and temperature nodes N
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
    % Window Data
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    % Init
    numWindows = floor((Nt - W) / stride) + 1;
    trainX = zeros(F, W, numWindows);
    trainY = zeros(N, numWindows);

    % Calc
    for i = 1:numWindows
        startIdx = (i - 1) * stride + 1;
        endIdx = startIdx + W - 1;
        trainX(:, :, i) = Pv(startIdx:endIdx, :)';
        trainY(:, i) = T(endIdx, :)';
    end

    %----------------------------------------
    % Validation
    %----------------------------------------
    % Init
    numWindows = floor((Nt_vl - W) / stride) + 1;
    valX = zeros(F, W, numWindows);
    valY = zeros(N, numWindows);

    % Calc
    for i = 1:numWindows
        startIdx = (i - 1) * stride + 1;
        endIdx = startIdx + W - 1;
        valX(:, :, i) = Pv_vl(startIdx:endIdx, :)';
        valY(:, i) = T_vl(endIdx, :)';
    end

    %===================================================
    % Reshape Data
    %===================================================
    trainX = squeeze(mat2cell(trainX, F, W, ones(1, length(trainY))));
    valX = squeeze(mat2cell(valX, F, W, ones(1, length(valY))));

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
    % options = trainingOptions("adam", ...
    % MaxEpochs=epoch, ...
    % GradientThreshold = gradTh, ...
    % MiniBatchSize = batch, ...
    % Shuffle = shu, ...
    % LearnRateSchedule="piecewise", ...
    % InitialLearnRate=initLr, ...
    % SequenceLength="shortest", ...
    % Plots="none", ...
    % ValidationData = {valX, valY'}, ...
    % LearnRateDropPeriod = lrDropPr, ...
    % LearnRateDropFactor = lrDropFa, ...
    % ValidationFrequency = valFreq, ...
    % Verbose= 1);

    options = trainingOptions('adam', ...
                              'MaxEpochs', epoch, ...
                              'MiniBatchSize', batch, ...
                              'Shuffle', shu, ...
                              'GradientThreshold', gradTh, ...
                              'InitialLearnRate', initLr, ...
                              'LearnRateSchedule', 'piecewise', ...
                              'LearnRateDropPeriod', lrDropPr, ...
                              'LearnRateDropFactor', lrDropFa, ...
                              'ValidationData', {valX, valY'}, ...
                              'ValidationFrequency', valFreq, ...
                              'Verbose', 1, ...
                              'Plots', 'none');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Model
    %===================================================
    layers = [ ...
              sequenceInputLayer(F, Normalization="zscore")
              lstmLayer(32, OutputMode="last")
              reluLayer
              fullyConnectedLayer(32)
              reluLayer
              fullyConnectedLayer(N)
              regressionLayer];

    %===================================================
    % Fitting
    %===================================================
    mdl = trainNetwork(trainX, trainY', layers, options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Training a Deep Learning Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1