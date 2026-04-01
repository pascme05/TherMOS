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
    % Nodal Foster Networks
    %===================================================

    %===================================================
    % Init Parameter
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    Rth_init = eye(M);      
    Cth_init = ones(M,1);  

    %----------------------------------------
    % Opti Cth
    %----------------------------------------
    for i=1:min(M,K)
        [~,idx] = min(abs(Tm{i}(:,i)-((Tm{i}(end,i) - Tc{i}(end,i))*0.63+Tc{i}(end,i))));
        tau = t{i}(idx);
        Cth_init(i) = tau / Rth_init(i,i);
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Analytic Rth
    %===================================================
    Rth = zeros(M);
    for k = 1:min(M,K)   
        Pk = Pv{k}(end,k);
        dT = Tm{k}(end,:) - Tc{k}(end,:);
        Rth(:,k) = dT(:) / Pk;   
    end
    Rth = (Rth + Rth.') / 2;
    Gth = inv(Rth);
    Gth = (Gth + Gth.') / 2;

    %===================================================
    % Optimisation options
    %===================================================
    opt = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', tol, ...
                       'MaxIter', maxIter);
    x0 = Cth_init(:);
    lb = 1e-12 * ones(numel(x0),1);
    ub = [];

    %===================================================
    % Solve
    %===================================================
    C_opt = lsqnonlin(@(c) thermal_err_C(c, t, Pv, Tm, Tc, Gth, K), ...
                  Cth_init, lb, ub, opt);
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Values
    %===================================================
    Cth = diag(C_opt);

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
function err = thermal_err_C(Cvec, t2, P2, T2, Tc2, Gth, K)

    C = diag(Cvec);
    all_err = [];

    for k = 1:K
        t = t2{k}(:);
        P = P2{k};
        Tm = T2{k} - Tc2{k}(1,:);
        Tc = Tc2{k} - Tc2{k}(1,:);

        odefun = @(tt,T) C \ ( ...
            interp1(t, P, tt, 'linear', 'extrap')' ...
          - Gth*(T - interp1(t, Tc, tt, 'linear', 'extrap')') );

        T0 = Tm(1,:)';

        [~, T_sim] = ode15s(odefun, t, T0);

        res = T_sim - Tm;
        all_err = [all_err; res(:)];
    end

    err = all_err;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1