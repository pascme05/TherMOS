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
% This function solves a deep learning model.
% -------------------------------------------------------------------------
% Inp:  1) mdl:     Fitted model parameters
%       2) data:    Testing input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = dnnSol(mdl, data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Solving a DNN")

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
    errMax = para.Mdl.gen.err;
    err = Inf;
    maxSteps = para.Mdl.gen.nSub;
    W = 96; % Length of each sequence (window size)
    stride = 1; % Step size to slide the window

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
    % Calculate the amount of padding needed
    padding = W - 1;

    % Pad the data along the time dimension
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
    % testX = num2cell(Pv', 1); 
    % testX = dlarray(Pv, 'TCB');
    % testX = dlarray(testX, 'CBT');
    testX = squeeze(mat2cell(testX, F, W, ones(1, Nt)));
    % testX = dlarray(testX, 'CTB'); 

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
    if N == 1
        % out.y = extractdata(squeeze(T_est))';
        out.y = T_est;
    else
        % out.y = extractdata(squeeze(T_est))';
        out.y = T_est;
    end
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving a DNN")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1