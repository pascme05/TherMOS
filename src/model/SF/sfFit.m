%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: sfFit                                                             %
% Date: 08.05.2025                                                        %
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
% where T is the temperature vector, P is the loss vector, C is the
% thermal capacitance matrix, and G the thermal conductance matrix. 
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = sfFit(data, ~, para)
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
    [~, M] = size(data.y);                                                  % number of time samples Nt and temperature nodes M
    [~, N] = size(data.X);                                                  % number of features N
    K = size(data.t2,1);                                                    % number of experiments

    %===================================================
    % Variables
    %===================================================
    maxIter = para.Mdl.gen.iterMax;                                         % maximum number of iterations
    tol = para.Mdl.gen.eps;                                                 % tolerance for optimisation

    %===================================================
    % Variables
    %===================================================
    Pv = data.X2;                                                           % power losses (W)
    Tc = data.r2;                                                           % Reference Temperature (°C)
    Tm = data.y2;                                                           % training temperatures (°C)
    t = data.t2;                                                            % time vector (sec)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Define Function
    %===================================================
    fun = @(x) objective_dgl(x, t, Pv, T, M, N);

    %===================================================
    % Optimisation options
    %===================================================
    opt = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', tol, ...
                       'MaxIter', maxIter);
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init Parameter
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    Rth_init = eye(M);      
    Cth_init = ones(M,1);  

    %----------------------------------------
    % Opti Rth
    %----------------------------------------
    for i=1:K
        Rth_init(i,i) = (Tm{i}(end,i) - Tc{i}(end,i)) / Pv{i}(end,i);
    end

    %----------------------------------------
    % Opti Cth
    %----------------------------------------
    for i=1:K
        [~,idx] = min(abs(Tm{i}(:,i)-((Tm{i}(end,i) - Tc{i}(end,i))*0.63+Tc{i}(end,i))));
        tau = t{i}(idx);
        Cth_init(i) = tau / Rth_init(i,i);
    end
    
    %----------------------------------------
    % Init Condition
    %----------------------------------------
    x0 = [reshape(Rth_init,[],1); Cth_init];


    %===================================================
    % Solve
    %===================================================
    x_opt = lsqnonlin(@(x) thermal_err_multi(x, t, Pv, Tm, Tc, M, K), x0, [], [], opt);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Values
    %===================================================
    Cth = x_opt(N^2+1:end);
    Cth = diag(Cth);
    Rth = reshape(x_opt(1:N^2), N, N);
    Gth = 1./Rth;

    %===================================================
    % Fitting
    %===================================================
    disp("INFO: Thermal resistance Rth and capacitance Cth");
    disp(Rth);
    disp(Cth);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mdl.Rth = Rth;
    mdl.Gth = Gth;
    mdl.Cth = Cth;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting thermal structure function")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Cost Function
%===================================================
function error = objective_dgl(x, t, P, T_true, M, N)
    %----------------------------------------
    % Extract parameters from x
    %----------------------------------------
    Cth = reshape(x(1:M*N), [M, N]);
    Gth = reshape(x(M*N+1:end), [M, N]);
    
    %----------------------------------------
    % Predict temperatures
    %----------------------------------------
    T_pred = calcT(t, P, Cth, Gth);
    
    %----------------------------------------
    % Compute the error
    %----------------------------------------
    error = (T_true - T_pred).^2;
end

%===================================================
% Calc Temperature
%===================================================
function T_pred = calcT(t, P, Cth, Gth)
    %----------------------------------------
    % Init
    %----------------------------------------
    Nt = length(t);
    dt = t(2) - t(1);
    M = size(Cth, 1);
    N = size(Cth, 2);
    T_pred = zeros(Nt, M);

    %----------------------------------------
    % Calc Prediction
    %----------------------------------------
    for i = 1:M
        for k = 2:Nt
            for j = 1:N
                T_pred(k, i) = T_pred(k-1, i) + (dt / Cth(i, j)) * (P(k-1, j) - Gth(i, j) * T_pred(k-1, i));
            end
        end
    end
end

%===================================================
% Cost Function
%===================================================
function err = thermal_err_multi(x, t2, P2, T2, Tc2, M, K)
    Rth = reshape(x(1:M^2),M,M);
    Cth = x(M^2+1:end);
    
    % Construct conductance matrix G = inv(Rth)
    G = Rth\eye(M);  
    C = diag(Cth);

    % ----- Simulate all experiments and accumulate error ---------------
    all_err = [];

    for k = 1:K
        % Extract Var
        t = t2{k}(:);
        P = P2{k};
        Tm = T2{k};
        Tc = Tc2{k};

        if any(diff(t) <= 0)
            error('Time vector t must be strictly increasing.');
        end

        % Make sure Tc has size [Nk x M]
        if size(Tc,2) == 1 && M > 1
            Tc = repmat(Tc, 1, M);
        end

        % ODE: C * dT/dt = P(t) - G*(T - Tc(t))
        odefun = @(tt,T) C \ ( ...
                         interp1(t, P, tt, 'linear', 'extrap')' ...
                       - G*(T - interp1(t, Tc, tt, 'linear', 'extrap')') );

        % Initial condition: measured or ambient at first time
        T0 = Tm(1,:).';     % or Tc(1,:).'

        [~, T_sim] = ode45(odefun, t, T0);

        % Compute residual
        res_k = T_sim - Tm;     % [Nk x M]
        all_err = [all_err; res_k(:)];
    end

    err = all_err;   % for lsqnonlin
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1