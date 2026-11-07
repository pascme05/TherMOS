%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: dnnSol                                                            %
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
% This function solves a deep learning model based on inputs (power losses),
% X (NtxF) with Nt samples and F features, and outputs (temperatures) y
% (NtxN) with Nt samples and N nodes using a regression function r()
% parameterized by a set of free parameters.
%
%                              T = r(Pv)
%
% -------------------------------------------------------------------------
% Inp:  1) mdl:     Fitted model parameters
%       2) data:    Testing input data struct
%       3) para:    All simulation parameters of the current simulation
%       4) setup:   All setup files for the simulation
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = dlSol(mdl, data, para, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Solving a Deep Learning Model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameter
    %===================================================
    Rcon = para.Par.loss.Rcon;                                              % contact resistance (Ohm)
    Rohm = para.Par.loss.Rohm;                                              % conduction resistance (Ohm)
    Rslope = para.Par.loss.Rslope;                                          % slope specific resistance change (1/K)
    Tref = para.Par.loss.Tref;                                              % reference temperature losses (°C)
    eps = para.Par.gen.eps;                                                 % lower numerical bound
    [Nt, N] = size(data.y);                                                 % number of samples Nt and temperature nodes N
    [~, F] = size(data.X);                                                  % number of features F
    errMax = para.Mdl.gen.err;                                              % error bound
    err = Inf;                                                              % initial error
    maxSteps = para.Mdl.gen.nSub;                                           % maximum amount of optimisation steps
    W = para.Mdl.dl.W;                                                      % Length of each sequence (window size)
    seq = para.Mdl.dl.seq;                                                  % Selector for seq2seq and seq2point        

    %===================================================
    % Variables
    %===================================================
    Pv = data.X;                                                            % power losses input (W)
    Toff = data.r;                                                          % temperature offset/reference (°C)
    T_est = zeros(size(Toff));                                              % init nodes temperatures
    out = data;     

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Rolling Features
    %===================================================
    if setup.feature_roll ~= 0
        Pv = calcFeat(Pv, setup.EWMA, setup.EWMS);
        [~, F] = size(Pv);                                                  % number of features F
    else
        disp("INFO: Features deactivated")
    end

    %===================================================
    % Calc Stride
    %===================================================
    if seq == 0
        stride = W;                                                         % Step size to slide the window (Seq2Seq)
    else
        stride = 1;                                                         % Step size to slide the window (Seq2Point)
    end

    %===================================================
    % Padding Data
    %===================================================
    if seq == 0
        remainder = mod(Nt, W);                                             % Leftover if not divisible
        if remainder ~= 0
            padLen = W - remainder;
            padPv = [Pv; repmat(Pv(end, :), padLen, 1)];  
            Toff = [Toff; repmat(Toff(end, :), padLen, 1)];
        end
        Nt_padded = size(padPv, 1);
    else
        padLen = W - 1;
        padPv = padarray(Pv', [0 padLen], 'replicate', 'pre')';             % T+W-1 × F
    end

    %===================================================
    % Window Data
    %===================================================
    % Init
    if seq == 0
        numWin = Nt_padded / W; 
    else
        numWin = Nt;   
    end

    % Preallocate 3D array
    testX_mat = zeros(F, W, numWin);

    % Windowing
    for idx = 1:numWin
        startIdx = (idx - 1) * stride + 1;
        stopIdx = startIdx + W - 1;
        testX_mat(:, :, idx) = padPv(startIdx:stopIdx, :)';  % F × W
    end

    %===================================================
    % Reshape Input
    %===================================================
    testX = permute(testX_mat, [2, 1, 3]);

    %===================================================
    % Scaling Function
    %===================================================
    if Rohm == 0
        theta = @(T) 1;
    else
        theta = @(T) (Rcon + Rohm*Rslope*(T-Tref)) / (Rcon + Rohm + eps);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Scaling Function
    %===================================================
    if Rohm == 0
        T_est = predict(mdl.sys, testX);
        if seq == 0
            T_est = reshape(permute(T_est, [1 3 2]), [], N);
            T_est = T_est(1:Nt, :);
        end

    %===================================================
    % Scaling Function
    %===================================================
    else
        %----------------------------------------
        % Init
        %----------------------------------------
        iter = 1;

        %----------------------------------------
        % Define initial guesses for C and G
        %----------------------------------------
        while err > errMax
            % Predict
            temp = predict(mdl.sys, testX);
            
            % Losses
            Pv = Pv .* theta(Tj_est + Toff);
            testX = num2cell(Pv', 1);

            % Error 
            err = mean(mean(temp - T_est));
            T_est = temp;

            % End While Loop
            if iter > maxSteps
                continue;
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_est;
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving a Deep Learning Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1