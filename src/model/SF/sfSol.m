%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: sfSol                                                             %
% Date: 08.05.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function solves a structure function capturing the system dynamics 
% of the temperature response:
%
%                       C*dT/dt = P(t) - G*T
%
% where T is the temperature vector, P is the loss vector, C is the
% thermal capacitance matrix, and G the thermal conductance matrix. 
% -------------------------------------------------------------------------
% Inp:  1) mdl:     Fitted model parameters
%       2) data:    Testing input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = sfSol(mdl, data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Solving thermal structure function")

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

    %===================================================
    % Variables
    %===================================================
    t = data.t;
    Pv = data.X;                                                            % power losses input (W)
    Toff = data.r;                                                          % temperature offset/reference (°C)
    T_est = zeros(size(data.y));                                            % init nodes temperatures
    out = data;     

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Model Parameters
    %===================================================
    Cth = mdl.Cth;
    Gth = mdl.Gth;
    Cth_inv = inv(Cth);

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
    % Feedthrough
    %===================================================
    for i = 2:length(t)
        %----------------------------------------
        % Update power
        %----------------------------------------
        Pv(i,:) = Pv(i,:) .* theta(T_est(i-1,:) + Toff(i-1,:));

        
        %----------------------------------------
        % Solve Nodes
        %----------------------------------------
        dt = t(i) - t(i-1);
        T_est(i, :) = calcT(dt, Pv(i,:), Cth_inv, Gth, T_est(i-1, :));
    end

    %===================================================
    % Correcting Offset
    %===================================================
    % T_est = T_est + Toff;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_est;
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving thermal structure function")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Calc Temperature
%===================================================
function Tnew = calcT(dt, P, Cth, Gth, Told)
    %----------------------------------------
    % Init
    %----------------------------------------
    Told = Told(:);
    P = P(:);

    %----------------------------------------
    % Calculation
    %----------------------------------------
    dTdt = Cth * (P - Gth*Told);
    Tnew2 = Told + dt*dTdt;
    Tnew = (Cth/dt +Gth) / (P + Cth/dt*Told)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1