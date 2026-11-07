%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: pinnSol                                                           %
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
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = pinnSol(mdl, data, para)
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
    [Nt, ~] = size(data.y);                                                 % number of samples Nt and temperature nodes N
    [~, F] = size(data.X);                                                  % number of features F
    errMax = para.Mdl.gen.err;                                              % error bound
    err = Inf;                                                              % initial error
    maxSteps = para.Mdl.gen.nSub;                                           % maximum amount of optimisation steps
    W = para.Mdl.dl.W;                                                      % Length of each sequence (window size)
    stride = W;                                                             % Step size to slide the window

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
    % Padding Data
    %===================================================
    remainder = mod(Nt, W);                                                 % Leftover if not divisible
    if remainder ~= 0
        padLen = W - remainder;
        Pv = [Pv; repmat(Pv(end, :), padLen, 1)];  
        Toff = [Toff; repmat(Toff(end, :), padLen, 1)];
    end
    Nt_padded = size(Pv, 1);             

    %===================================================
    % Window Data
    %===================================================
    % Init
    numWin = Nt_padded / W;   

    % Preallocate 3D array
    testX_mat = zeros(F, W, numWin);

    % Windowing
    for idx = 1:numWin
        startIdx = (idx - 1) * stride + 1;
        stopIdx = startIdx + W - 1;
        testX_mat(:, :, idx) = Pv(startIdx:stopIdx, :)';  % F × W
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
        T_est = reshape(permute(T_est, [1 3 2]), [], 1);
        T_est = T_est(1:Nt);
        T_est = gather(T_est);

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
            T_est_win = predict(mdl.sys, testX);  
            T_temp = reshape(permute(T_est_win, [1 3 2]), [], 1); 

            % Recompute losses using updated temperature
            Pv_scaled = Pv .* theta(T_temp + Toff);
            testX_mat_scaled = zeros(F, W, numWin);
            for idx = 1:numWin
                startIdx = (idx - 1) * W + 1;
                endIdx = startIdx + W - 1;
                testX_mat_scaled(:, :, idx) = Pv_scaled(startIdx:endIdx, :)'; 
            end
            testX = permute(testX_mat_scaled, [2 1 3]);  

            % Check convergence
            err = mean(abs(T_temp - T_est));
            T_est = T_temp;

            iter = iter + 1;
            if iter > maxSteps
                warning("Max correction steps reached in PINN solver.");
                break;
            end
        end
        
        %----------------------------------------
        % Trim padding
        %----------------------------------------
        T_est = T_est(1:Nt);
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