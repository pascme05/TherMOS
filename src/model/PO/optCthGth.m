%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: optCthGth                                                         %
% Date: 07.05.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here goes the description of the function.
% -------------------------------------------------------------------------
% Inp:  1) Input-1
%       2) Input-2
% Out:  1) Output-1
%       2) Output-2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C_opt, G_opt, q_est] = optCthGth(u, q, dt, C_init, G_init, lam)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("INFO: Optimizing Cth and Gth")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [K, ~] = size(u);                                                       % number of POD modes
    
    %===================================================
    % Optimizer
    %===================================================
    options = optimoptions('fminunc', ...
        'Display', 'iter', ...
        'MaxIterations', 1000, ...
        'StepTolerance', 1e-8, ...
        'OptimalityTolerance', 1e-8, ...
        'FunctionTolerance', 1e-8, ...
        'Algorithm', 'quasi-newton');

    options = optimoptions('lsqnonlin', ...
        'Display', 'iter', ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', 1e-15, ...
        'StepTolerance', 1e-15, ...
        'OptimalityTolerance', 1e-15, ...
        'Algorithm', 'trust-region-reflective');



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Normalize inputs
    %===================================================
    % u_norm = u / max(abs(u(:)));
    % q_norm = q / max(abs(q(:)));
    % lam_norm = lam;
    if ~isscalar(lam)
        lam_norm = lam / max(abs(lam(:)));
    end

    %===================================================
    % Compute du/dt using gradient
    %===================================================
    du_dt = gradient(u, dt, 2);
    u_mid = u;
    q_mid = q;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Initial guess using provided estimates
    %===================================================
    x0 = [C_init(:); G_init(:)];
    
    %===================================================
    % Loss function
    %===================================================
    % loss_fn = @(x) residual_fn(x, du_dt, u_norm, q_norm, K, lam_norm);
    loss_fn = @(x) residual_vec(x, du_dt, u, q, K, lam);

    

    
    %===================================================
    % Run optimization
    %===================================================
    % [x_opt, ~] = fminunc(loss_fn, x0, options);
    [x_opt, resnorm, residual, exitflag, output] = lsqnonlin(loss_fn, x0, [], [], options);
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Reshape results
    %===================================================
    C_opt = reshape(x_opt(1:K*K), K, K);
    G_opt = reshape(x_opt(K*K+1:end), K, K);
    
    %===================================================
    % Compute estimated q using optimized parameters
    %===================================================
    q_est = C_opt * du_dt + G_opt * u;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fprintf('Init Cth:\n'); disp(C_init);
    % fprintf('Optimized Cth:\n'); disp(C_opt);
    % fprintf('Init Gth:\n'); disp(G_init);
    % fprintf('Optimized Gth:\n'); disp(G_opt);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function err = residual_fn(x, du_dt, u, q, K, lam)
    % Unpack C and G
    C = reshape(x(1:K*K), K, K);
    G = reshape(x(K*K+1:end), K, K);

    % Predicted q
    q_pred = C * du_dt + G * u;

    % Residual error
    residual = abs(q - q_pred).*lam;
    err = sum((residual(:)).^2);  % sum of squared error
end

function r = residual_vec(x, du_dt, u, q, K, lam)
    C = reshape(x(1:K*K), K, K);
    G = reshape(x(K*K+1:end), K, K);

    q_pred = C * du_dt + G * u;
    r = (q - q_pred) .* lam;  % vector of residuals (no squaring!)
    r = r(:);  % return as vector for lsqnonlin
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1