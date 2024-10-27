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
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = dlSol(mdl, data, para)
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
    stride = para.Mdl.dl.stride;                                            % Step size to slide the windoww

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
    padding = W - 1;
    padPv = padarray(Pv', [0 padding 0], 'replicate', 'pre')';

    %===================================================
    % Window Data
    %===================================================
    % Init
    testX = zeros(F, W, Nt);

    % Calc
    for i = 1:Nt
        startIdx = (i - 1) * stride + 1;
        endIdx = startIdx + W - 1;
        testX(:, :, i) = padPv(startIdx:endIdx, :)';
    end

    %===================================================
    % Reshape Input
    %===================================================
    testX = squeeze(mat2cell(testX, F, W, ones(1, Nt)));

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