%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: sfFit                                                             %
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
% This function fits a structure function capturing the system dynamics of
% the temperature response:
%
%                       C*dT/dt = P(t) - G*T
%
% where T is the temperature vector, u is the loss vector, and C is the
% thermal capacitance matrix 
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = sfFit(data, setup, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting thermal structure function")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % General Parameter
    %===================================================
    Ts = data.Ts;                                                           % sampling time (sec)
    
    %===================================================
    % Solver Parameter
    %===================================================

    %===================================================
    % Variables
    %===================================================
    Pv = data.X;
    T = data.y;
    t = data.t;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    opt = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init Parameter
    %===================================================
    %----------------------------------------
    % Get the number of nodes
    %----------------------------------------
    [Nt, M] = size(Pv);  
    [~, N] = size(T); 

    %----------------------------------------
    % Define initial guesses for C and G
    %----------------------------------------
    C0 = ones(M, N); 
    G0 = ones(M, N);
    init = [C0(:); G0(:)];

    %===================================================
    % Solve
    %===================================================
    [params_opt, fval] = fminunc(@(params) costFnc(params, Pv, T, t, M, N), init, opt);

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Values
    %===================================================

    %===================================================
    % Solve
    %===================================================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting thermal structure function")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Cost Function II
%===================================================
function error = costFnc2(params, P, T_true, time, M, L)
    %----------------------------------------
    % Extract Cth and Gth
    %----------------------------------------
    C = reshape(params(1:M*L), M, L); 
    G = reshape(params(M*L+1:end), M, L);
    
    %----------------------------------------
    % Init Variables
    %----------------------------------------
    T0 = T_true(1, :);
    
    %----------------------------------------
    % Solver
    %----------------------------------------
    options = odeset('RelTol',1e-3,'AbsTol',1e-5);
    

    %----------------------------------------
    % Solver
    %----------------------------------------
    % Define the differential equation
    ode_fun = @(t, T) (P_interp(t, P, time) - T * G') ./ C'; 

    % Solve the ODE system using ode45
    [~, T_model] = ode45(ode_fun, time, T0);

    % Interpolate model output to match measured data size
    T_mdl_int = interp1(time, T_model, time);

    % Calculate the mean squared error between model and measured data
    error = sum((T_true - T_mdl_int).^2, 'all');
end

%===================================================
% Cost Function
%===================================================
function error = costFnc(params, P, T_true, time, M, L)
    % Extract C and G from the parameter vector and reshape them to MxL matrices
    C = reshape(params(1:M*L), M, L); 
    G = reshape(params(M*L+1:end), M, L);

    % Solve the differential equation using ode45 for each node
    T0 = T_true(1, :);
    
    % Define the differential equation
    ode_fun = @(t, T) (P_interp(t, P, time) - T * G') ./ C'; 

    % Solve the ODE system using ode45
    [~, T_model] = ode45(ode_fun, time, T0);

    % Interpolate model output to match measured data size
    T_mdl_int = interp1(time, T_model, time);

    % Calculate the mean squared error between model and measured data
    error = sum((T_true - T_mdl_int).^2, 'all');
end

%===================================================
% Interpolation
%===================================================
function P_t = P_interp(t, P, time)
    P_t = interp1(time, P, t);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1