%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: lstmSol                                                           %
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
function out = lstmSol(mdl, data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Solving a LSTM")

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
    [~, N] = size(data.y);                                                  % number of samples Nt and temperature nodes N
    errMax = para.Mdl.gen.err;
    err = Inf;
    maxSteps = para.Mdl.gen.nSub;

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
    % Reshape Input
    %===================================================
    % testX = num2cell(Pv', 1);
    testX = dlarray(Pv, 'TCB');

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
        out.y = extractdata(squeeze(T_est));
    else
        out.y = extractdata(squeeze(T_est))';
    end
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving a LSTM")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1