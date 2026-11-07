%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: pinnFit                                                           %
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
% This function fits a physics informed deep learning model based on inputs 
% (power losses), X (NtxF) with Nt samples and F features, and outputs 
% (temperatures) y (NtxN) with Nt samples and N nodes using a regression 
% function r() parameterized by a set of free parameters. The fitting is
% constrained by a pair of thermal resistances and capitances.
%
%                              T = r(Pv)
%
%                 s.t. Cth*dT/dt = (Ta - T)/Rth + Pv
%
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
%       4) setup:   All setup files for the simulation
% Out:  1) mdl:     Trained model
%       2) info:    Training info

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mdl, info] = pinnFit(data, val, para, ~)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Training a physics informed Deep Learning Model")

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
    % gradTh = para.Mdl.dl.gradTh;
    initLr = para.Mdl.dl.initLr;
    lrDropPr = para.Mdl.dl.lrDropPr;
    lrDropFa = para.Mdl.dl.lrDropFa;
    % valFreq = para.Mdl.dl.valFreq;
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
    Ytrain = zeros(W, M, numWin);      % [time, response, batch]
    
    % Efficient windowing loop
    for idx = 1:numWin
        i = (idx - 1) * stride + 1;
        Xtrain(:, :, idx) = X(i:i+W-1, :);        % W x F
        Ytrain(:, :, idx) = Y(i:i+W-1, :)';       % W x M
    end

    %----------------------------------------
    % Validation
    %----------------------------------------
    % Calculate number of windows
    numWin_val = floor((T_val - W) / stride) + 1;
    
    % Preallocate arrays
    Xval = zeros(W, F, numWin_val);     % [time, features, batch]
    Yval = zeros(W, M, numWin_val);     % [time, response, batch]
    
    % Efficient windowing loop
    for idx = 1:numWin_val
        i = (idx - 1) * stride + 1;
        Xval(:, :, idx) = X_val(i:i+W-1, :);     % W x F
        Yval(:, :, idx) = Y_val(i:i+W-1, :)';    % M x 1
    end
    
    %===================================================
    % Convert Data
    %===================================================
    Xtrain = dlarray(single(Xtrain), 'TCB');                                % T: time, C: channels (features), B: batch
    Ytrain = dlarray(single(Ytrain), 'TCB');                                % T: time, C: output dimension, B: batch
    Xval = dlarray(single(Xval), 'TCB');                                    % T: time, C: channels (features), B: batch
    Yval = dlarray(single(Yval), 'TCB');                                    % T: time, C: output dimension, B: batch

    %===================================================
    % Shuffel
    %===================================================
    switch shuOpt
        case 1, shu = 'never';
        case 2, shu = 'once';
        case 3, shu = 'every-epoch';
        otherwise, shu = 'every-epoch';
    end

                        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Model
    %===================================================
    %----------------------------------------
    % Layer Structure
    %----------------------------------------
    layers = [
            sequenceInputLayer(F, 'Normalization', 'zscore')
            convolution1dLayer(3, 32, 'Padding', 'same') 
            reluLayer
            lstmLayer(64, 'OutputMode', 'sequence')
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(M)
        ];
    
    %----------------------------------------
    % Convert to Model
    %----------------------------------------
    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);

    %===================================================
    % Fitting
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    trailingAvg = [];
    trailingAvgSq = [];
    info.Loss = zeros(epoch,1);
    info.ValLoss = zeros(epoch,1);
    info.learnRate = zeros(epoch,1);
    bestValLoss = inf;
    patience = 0;
    maxPatience = 10;
    learnRate = initLr;
    bestNet = dlnet;
    
    %----------------------------------------
    % Shuffle
    %----------------------------------------
    if shu == "once"
        idx = randperm(size(Xtrain, 2));
        Xtrain = Xtrain(:,idx,:);
        Ytrain = Ytrain(:,idx,:);
    end
    
    %----------------------------------------
    % Enable GPU if available
    %----------------------------------------
    executionEnvironment = "auto";  
    useGPU = canUseGPU && executionEnvironment == "auto";

    %----------------------------------------
    % Iterating
    %----------------------------------------
    for e = 1:epoch
        lossEpoch = 0;

        if shu == "every-epoch"
            idx = randperm(size(Xtrain, 2));
            Xtrain = Xtrain(:,idx,:);
            Ytrain = Ytrain(:,idx,:);
        end

        for i = 1:batch:size(Xtrain, 2)
            bIdx = i:min(i+batch-1, size(Xtrain, 2));
            XBatch = Xtrain(:,bIdx,:);
            YBatch = Ytrain(:,bIdx,:);

            % Move to GPU if required
            if useGPU
                XBatch = gpuArray(XBatch);
                YBatch = gpuArray(YBatch);
            end

            % Evaluate model and gradients
            [loss, gradients] = dlfeval(@modelLoss, dlnet, XBatch, YBatch);

            % Adam update
            [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
                trailingAvg, trailingAvgSq, e, learnRate);

            lossEpoch = lossEpoch + double(gather(extractdata(loss)));
        end

        % Learning rate scheduling
        if mod(e, lrDropPr) == 0
            learnRate = learnRate * lrDropFa;
        end

        % Validation
        valLoss = 0;
        for i = 1:batch:size(Xval, 2)
            bIdx = i:min(i+batch-1, size(Xval, 2));
            XBatchVal = Xval(:,bIdx,:);
            YBatchVal = Yval(:,bIdx,:);
            if useGPU
                XBatchVal = gpuArray(XBatchVal);
                YBatchVal = gpuArray(YBatchVal);
            end
            YpredVal = forward(dlnet, XBatchVal);
            YpredVal = stripdims(YpredVal);
            YBatchVal = stripdims(YBatchVal);
            valLoss = valLoss + mse(dlarray(YpredVal, 'TCB'), dlarray(YBatchVal, 'TCB'));
        end
        valLoss = double(gather(extractdata(valLoss)));

        % Save info
        info.Loss(e) = lossEpoch;
        info.ValLoss(e) = valLoss;
        info.learnRate(e) = learnRate;

        % Print
        disp("Epoch " + e + ": TrainLoss = " + lossEpoch + " | ValLoss = " + valLoss + " | LearnRate = " + learnRate);

        % Early stopping and model saving
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestNet = dlnet;
            patience = 0;
        else
            patience = patience + 1;
        end

        if patience >= maxPatience
            disp("Early stopping at epoch " + e);
            break;
        end
    end
    
    %===================================================
    % Output
    %===================================================
    info.TrainingHistory.Iteration = (1:epoch)';
    info.TrainingHistory.Loss = info.Loss(1:epoch);
    info.TrainingHistory.LearnRate = info.learnRate(1:epoch);
    info.ValidationHistory.Iteration = (1:epoch)';
    info.ValidationHistory.Loss = info.ValLoss(1:epoch);
    mdl = bestNet;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Training a physics informed Deep Learning Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss, gradients] = modelLoss(dlnet, X, Ytrue)
    % Constants
    Rth = 0.0265;                                                           % Thermal resistance (K/W)
    Cth = 2.08e4;                                                           % Thermal capacitance (Ws/K)
    Rs = 14.1e-3;                                                           % Stator resistance 20 degC
    Ts = 1;                                                                 % Sampling time (sec)
    
    % Forward pass
    Ypred = forward(dlnet, X);
    
    % Adapt dimension
    Ypred = stripdims(Ypred); 
    Ytrue = stripdims(Ytrue);
    X = stripdims(X);
    Ypred = squeeze(permute(Ypred, [3, 2, 1]));                             % → [T × B]
    Ytrue = squeeze(permute(Ytrue, [3, 2, 1]));                             % → [T × B]

    % Extract inputs
    Is    = squeeze(permute(X(4,:,:), [3, 2, 1]));                          % Stator current [T × B]
    T_amb = squeeze(permute(X(2,:,:), [3, 2, 1]));                          % Ambient temp   [T × B]
    Q_in = 3 * Rs * Is.^2;                                                  % Stator losses (W)

    % Compute dT/dt (gradient w.r.t. time)
    dTdt = zeros(size(Ypred), 'like', Ypred);
    dTdt(1:end-1, :) = (Ypred(2:end, :) - Ypred(1:end-1, :)) / Ts;
    dTdt(end, :) = dTdt(end-1, :);

    % dTdt2 = zeros(size(Ytrue), 'like', Ytrue);
    % dTdt2(1:end-1, :) = (Ytrue(2:end, :) - Ytrue(1:end-1, :)) / Ts;
    % dTdt2(end, :) = dTdt2(end-1, :);


    % Physics residual from thermal RC equation
    physResidual = Cth .* dTdt - ((T_amb - Ypred) ./ Rth + Q_in);
    % physResidual2 = Cth .* dTdt2 - ((T_amb - Ytrue) ./ Rth + Q_in);

    % Assign Labels
    Ypred = dlarray(Ypred, 'TB');
    Ytrue = dlarray(Ytrue, 'TB');
    physResidual = dlarray(physResidual, 'TB');
    physTrue = dlarray(zeros(size(physResidual), 'like', physResidual), 'BT');

    % Physics-informed loss
    dataLoss = mse(Ypred, Ytrue);                                           % Supervised loss
    physLoss = mse(physResidual, physTrue);

    % Total loss
    alpha = 1.0;                                                            % Weight on data loss
    beta = 0.0;                                                             % Weight on physics loss
    loss = alpha * dataLoss + beta * physLoss;

    % Compute gradients
    gradients = dlgradient(loss, dlnet.Learnables);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1