%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: rcFit                                                             %
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
% This function fits the thermal resistances (Rth) and the thermal 
% capcitances (Cth) for a one dimensional Foster network. The function
% that is fitted can be written as:
%
%                     Zth = Σ Rth * (1 - exp(-t/tau))
%
% and describes the transient thermal impedance where tau=Rth*Cth.
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = rcFit(data, val, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting Foster Network")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameter
    %===================================================
    %----------------------------------------
    % General
    %----------------------------------------
    K = para.Mdl.rc.K;                                                      % number of RC pairs
    bc = para.Mdl.rc.bc;                                                    % boundary conditions for Rth and Cth
    iterMax = para.Mdl.gen.iterMax;                                         % maximum number of model iterations
    tol = para.Mdl.gen.eps;                                                 % numerical toleranz
    initRes = Inf;                                                          % initial residual value
    Ts = data.Ts;                                                           % sampling time (sec)
    Kmax = para.Mdl.gen.Kmax;                                               % maximum number of RC pairs

    %----------------------------------------
    % Training
    %----------------------------------------
    Nt = min(data.Nt);                                                      % minimum number of training time steps
    [~, N] = size(data.X2);                                                 % number of training datasets used for fitting

    %----------------------------------------
    % Validation
    %----------------------------------------
    Nt_vl = min(val.Nt);                                                    % minimum number of validation time steps
    [~, N_vl] = size(val.X2);                                               % number of validation datasets used for fitting

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    Pv = [];                                                                % power losses (W)
    T = [];                                                                 % temperature response (°C)
    t = 0:Ts:Nt-1;                                                          % input time vector (sec)
    
    %----------------------------------------
    % Validation
    %----------------------------------------
    Pv_vl = [];                                                             % power losses (W)
    T_vl = [];                                                              % temperature response (°C)
    t_vl = 0:Ts:Nt_vl-1;                                                    % input time vector (sec)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Reformating
    %===================================================
    t = t';
    t_vl = t_vl';

    %===================================================
    % Averaging Losses
    %===================================================
    %----------------------------------------
    % Calc
    %----------------------------------------
    % Training
    for i = 1:N
        if i == 1
            Pv = data.X2{1,i}(1:Nt,1);
            T = data.y2{1,i}(1:Nt,1);
        else
            Pv = Pv + data.X2{1,i}(1:Nt,1);
            T = T + data.y2{1,i}(1:Nt,1);
        end
    end
    Pv = Pv / N;
    T = T / N;

    % Validation
    for i = 1:N_vl
        if i == 1
            Pv_vl = val.X2{1,i}(1:Nt_vl,1);
            T_vl = val.y2{1,i}(1:Nt_vl,1);
        else
            Pv_vl = Pv_vl + val.X2{1,i}(1:Nt_vl,1);
            T_vl = T_vl + val.y2{1,i}(1:Nt_vl,1);
        end
    end
    Pv_vl = Pv_vl / N_vl;
    T_vl = T_vl / N_vl;
    
    %----------------------------------------
    % Msg
    %----------------------------------------
    disp("INFO: Averaged Zth over training instances");

    %===================================================
    % Generating Zth
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    dT = (T - T(1));
    Zth = dT ./ Pv;

    %----------------------------------------
    % Validation
    %----------------------------------------
    dT_vl = (T_vl - T_vl(1));
    Zth_vl = dT_vl ./ Pv_vl;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Fitting Parameters
    %===================================================
    %----------------------------------------
    % Boundary Conditions
    %----------------------------------------
    if bc > 0
        % Lower Bound
        [Rth_lb, ~, tau_lb] = deal(bc, bc, bc);

        % Upper Bound
        [Rth_ub, ~, tau_ub] = deal(max(Zth), bc, max(t));
    else
        % Lower Bound
        [Rth_lb, ~, tau_lb] = deal(0, 0, 0);

        % Upper Bound
        [Rth_ub, ~, tau_ub] = deal(Inf, Inf, Inf);
    end

    %----------------------------------------
    % Solver Options
    %----------------------------------------
    options = optimoptions('lsqcurvefit', ...
                           'Display', 'off', ...          
                           'Algorithm', 'levenberg-marquardt', ...
                           'MaxFunctionEvaluations', 1e6, ...
                           'MaxIterations', iterMax, ...
                           'TolFun', tol, ...
                           'TolX', tol);

    %===================================================
    % Fitting Optimal Order
    %===================================================
    if K == -1
        for i = 1:Kmax
            %----------------------------------------
            % Init Fnc
            %----------------------------------------
            rcFnc = @(params, t) fncZth(params, t, i);
            
            %----------------------------------------
            % Init Para
            %----------------------------------------
            Rth_init = ones(1, i);                                                 
            tau_init = ones(1, i);                                                 
            init = [Rth_init, tau_init];
            
            %----------------------------------------
            % Boundary
            %----------------------------------------
            lb = [Rth_lb*ones(1,i); tau_lb*ones(1,i)];
            ub = [Rth_ub*ones(1,i); tau_ub*ones(1,i)];
            
            %----------------------------------------
            % Solve
            %----------------------------------------
            [params_temp, ~, ~, ~, ~] = lsqcurvefit(rcFnc, init, t, ...
                                                    Zth, lb, ub, options);
            
            %----------------------------------------
            % Validation
            %----------------------------------------
            Zth_fit_vl = fncZth(params_temp, t_vl, i);
            resnorm = sum((Zth_fit_vl-Zth_vl).^2);
    
            %----------------------------------------
            % Check residual
            %----------------------------------------
            if resnorm < initRes
                initRes = resnorm;
                K = i;
            end
        end
    end
    
    %===================================================
    % Fitting Fixed Order
    %===================================================
    %----------------------------------------
    % Foster Fnc
    %----------------------------------------
    rcFnc = @(params, t) fncZth(params, t, K);
    
    %----------------------------------------
    % Initial Conditions
    %----------------------------------------
    Rth_init = ones(1, K);                                                  % Initial guess for Rth
    tau_init = ones(1, K);                                                  % Initial guess for tau
    init = [Rth_init, tau_init];                                            % global init 
    
    %----------------------------------------
    % Boundary Conditions
    %----------------------------------------
    lb = [Rth_lb*ones(1,K); tau_lb*ones(1,K)];
    ub = [Rth_ub*ones(1,K); tau_ub*ones(1,K)];

    %----------------------------------------
    % Fixed Model Order
    %----------------------------------------
    [params_opt, resnorm, res, flag, out] = lsqcurvefit(rcFnc, init, t, ...
                                                  Zth, lb, ub, options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if flag > 0
        Rth_opti = params_opt(1:K);
        tau_opti = params_opt(K+1:2*K);
        fprintf('Optimized Rth (K/W) values:\n');
        disp(Rth_opti);
        fprintf('Optimized tau (sec) values:\n');
        disp(tau_opti);
    else
        disp('INFO: Optimization did not converge.');
        disp(out.message);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mdl.Rth = params_opt(1:K);
    mdl.tau = params_opt(K+1:2*K);
    mdl.Cth = mdl.tau ./ mdl.Rth;
    mdl.lb = lb;
    mdl.ub = ub;
    mdl.Ts = Ts;
    mdl.res = res;
    mdl.resnorm = resnorm;
    mdl.K = K;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting Foster Network")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Zth_fit = fncZth(params, t, K)
    % Initialize Rth and Cth   
    Rth = params(1:K);
    tau = params(K+1:end);
    
    % Initialize the thermal impedance response
    Zth_fit = zeros(size(t));
    
    % Calculate the contribution of each RC pair
    for i = 1:K
        Zth_fit = Zth_fit + Rth(i) * (1 - exp(-t / tau(i)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] Touzelbaev, Maxat N., et al. "High-efficiency transient temperature 
% calculations for applications in dynamic thermal management of electronic 
% devices." Journal of Electronic Packaging 135.3 (2013): 031001.