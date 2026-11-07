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
    disp("START: Solving thermal structure function (ODE)")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    Rcon   = para.Par.loss.Rcon;
    Rohm   = para.Par.loss.Rohm;
    Rslope = para.Par.loss.Rslope;
    Tref   = para.Par.loss.Tref;
    eps_   = para.Par.gen.eps;

    %===================================================
    % Variables
    %===================================================
    t     = data.t;
    Pv    = data.X;       % power losses (W)
    Toff  = data.r;       % temperature offset
    out   = data;
    [~, M] = size(data.y); 
    
    %----------------------------------------
    % Build weights from Foster
    %----------------------------------------
    weights = mdl.weights;
    Nr = mdl.Nr;

    %----------------------------------------
    % Init T0
    %----------------------------------------
    T0_node = data.y(1,:) - Toff(1,:);
    T0 = zeros(M*Nr,1);
    for i = 1:M
        idx = (i-1)*Nr + (1:Nr);
        T0(idx) = weights(i,:) * T0_node(i);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Cth = mdl.Cth;
    Gth = mdl.Gth;
    Cvec = diag(Cth);

    %===================================================
    % Scaling Function (vector-safe)
    %===================================================
    if Rohm == 0
        theta = @(T) ones(size(T));
    else
        theta = @(T) (Rcon + Rohm*Rslope.*(T - Tref)) ./ (Rcon + Rohm + eps_);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ODE Definition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
    odefun = @(tt, T) thermal_ode(tt, T, t, Pv, Toff, Cth, Gth, theta, M, Nr, weights);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Solve ODE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, T_sim] = ode15s(odefun, t, T0, opts);
    T_nodes = zeros(length(t), M);
    for i = 1:M
        idx = (i-1)*Nr + (1:Nr);
        T_nodes(:,i) = sum(T_sim(:,idx),2);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_nodes + Toff;
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving thermal structure function (ODE)")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Calc Temperature
%===================================================
function dTdt = thermal_ode(tt, T, t, Pv, Toff, Cth, Gth_nodes, theta, M, Nr, weights)

    %----------------------------------------
    % Interpolate inputs
    %----------------------------------------
    P = interp1(t, Pv, tt, 'linear', 'extrap');

    %----------------------------------------
    % Reconstruct node temperatures
    %----------------------------------------
    T_node = zeros(M,1);

    for i = 1:M
        idx = (i-1)*Nr + (1:Nr);
        T_node(i) = sum(T(idx));
    end

    %----------------------------------------
    % Node-level coupling
    %----------------------------------------
    coupling_node = Gth_nodes * T_node;   % [M x 1]

    %----------------------------------------
    % Allocate derivative
    %----------------------------------------
    dTdt = zeros(M*Nr,1);

    %----------------------------------------
    % Loop over nodes and RC branches
    %----------------------------------------
    for i = 1:M
        for k = 1:Nr

            idx = (i-1)*Nr + k;

            % Weight for this branch
            w = weights(i,k);

            % Apply BOTH power and coupling consistently
            Pin  = w * P(i);
            Pout = w * coupling_node(i);

            % Dynamics
            dTdt(idx) = (Pin - Pout) / Cth(idx,idx);
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1