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
    Nr = 5;

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
    %----------------------------------------
    % Options
    %----------------------------------------
    options = optimoptions('lsqcurvefit', ...
                           'Display','off', ...
                           'Algorithm','levenberg-marquardt', ...
                           'MaxIterations', maxIter, ...
                           'MaxFunctionEvaluations', 1e5, ...
                           'TolFun', tol, ...
                           'TolX', tol);

    %----------------------------------------
    % Initial Conditions
    %----------------------------------------
    Rfoster = zeros(M, Nr);
    taufoster = zeros(M, Nr);
    Cfoster = zeros(M, Nr);
    
    %----------------------------------------
    % Fitting Model 
    %----------------------------------------
    for k = 1:M
        % Extract data (single-source experiment assumed)
        Pk = Pv{k}(:,k);
        dT = Tm{k}(:,k) - Tc{k}(:,k);
        tk = t{k}(:);

        % Safe Zth computation
        Pk_safe = max(Pk, 1e-9);
        Zth = dT ./ Pk_safe;
        Zth = Zth - Zth(1);

        % Weighting
        w = 1 ./ (tk + 1e-6);
        Zth_w = Zth .* w;

        % Initial guess
        tmin = tk(2);
        tmax = tk(end);
        tau_init = logspace(log10(tmin), log10(tmax), Nr);
        R_total = Zth(end);
        R_init = (R_total / Nr) * ones(1, Nr);
        p0 = log([R_init, tau_init]);

        % Weighted fitting
        modelFun = @(p, t) fncZth_exp(p, t, Nr) .* w;
        p_opt = lsqcurvefit(modelFun, p0, tk, Zth_w, [], [], options);

        % Extract + sort
        R = exp(p_opt(1:Nr));
        tau = exp(p_opt(Nr+1:end));
        [tau, idx] = sort(tau);
        R = R(idx);
        C = tau ./ R;

        % Store
        Rfoster(k,:) = R;
        taufoster(k,:) = tau;
        Cfoster(k,:) = C;
    end
    mdl.weights = Rfoster ./ sum(Rfoster,2); 
    mdl.Rth_rc = Rfoster;
    mdl.tau_rc = taufoster;
    mdl.Cth_rc = Cfoster;
    mdl.Nr = Nr;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Analytic Cth
    %===================================================
    Cth = diag(reshape(mdl.Cth_rc', [], 1));

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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("INFO: Thermal conductance Rth and capacitance Cth");
    disp(Rth);
    disp(Cth);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mdl.Rth = Rth;
    mdl.Gth = Gth;
    mdl.Cth = Cth;
    mdl.Nr = Nr;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting thermal structure function")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Zth Function
%===================================================
function Zth_fit = fncZth_exp(p, t, K)

    R = exp(p(1:K));
    tau = exp(p(K+1:end));

    Zth_fit = zeros(size(t));

    for i = 1:K
        Zth_fit = Zth_fit + R(i) * (1 - exp(-t / tau(i)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1