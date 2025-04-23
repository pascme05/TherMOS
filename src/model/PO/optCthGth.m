function [C_opt, G_opt, q_est] = optCthGth(u, q, dt, C_init, G_init, lam)

    [K, ~] = size(u);

    % Compute du/dt using finite difference
    du_dt = diff(u, 1, 2) / dt;  % size: K x (T-1)
    du_dt = gradient(u, dt, 2);
    u_mid = u(:, 1:end-1);       % size: K x (T-1)
    u_mid = u;
    q_mid = q(:, 1:end-1);       % size: K x (T-1)
    q_mid = q;

    % Initial guess using provided estimates
    x0 = [C_init(:); G_init(:)];

    % Loss function
    loss_fn = @(x) residual_fn(x, du_dt, u_mid, q_mid, K, lam);
    % loss_fn = @(x) trapezoidal_residual_fn(x, u, q, dt, K);

    % Optimization options
    options = optimoptions('fminunc', ...
        'Display', 'iter', ...
        'MaxIterations', 1000, ...
        'StepTolerance', 1e-12, ...
        'OptimalityTolerance', 1e-12, ...
        'FunctionTolerance', 1e-12, ...
        'Algorithm', 'quasi-newton');

    % Run optimization
    [x_opt, ~] = fminunc(loss_fn, x0, options);

    % Reshape results
    C_opt = reshape(x_opt(1:K*K), K, K);
    G_opt = reshape(x_opt(K*K+1:end), K, K);
    
    % Compute estimated q using optimized parameters
    du_dt = gradient(u, dt, 2);  % K x T
    q_est = C_opt * du_dt + G_opt * u;  % K x T

    % Optional: show results
    fprintf('Init Cth:\n'); disp(C_init);
    fprintf('Optimized Cth:\n'); disp(C_opt);
    fprintf('Init Gth:\n'); disp(G_init);
    fprintf('Optimized Gth:\n'); disp(G_opt);
end

function err = residual_fn(x, du_dt, u, q, K, lam)
    % Unpack C and G
    C = reshape(x(1:K*K), K, K);
    G = reshape(x(K*K+1:end), K, K);

    % Predicted q
    q_pred = C * du_dt + G * u;

    % Residual error
    residual = (q - q_pred).*lam;
    err = sum((residual(:)).^2);  % sum of squared error
end

% function err = trapezoidal_residual_fn(x, u, q, dt, K)
%     C = reshape(x(1:K*K), K, K);
%     G = reshape(x(K*K+1:end), K, K);
% 
%     [~, T] = size(u);
%     num_steps = T - 1;
%     err = 0;
% 
%     for t = 1:num_steps
%         du = (u(:, t+1) - u(:, t)) / dt;
%         u_avg = 0.5 * (u(:, t+1) + u(:, t));
%         q_avg = 0.5 * (q(:, t+1) + q(:, t));
% 
%         residual = q_avg - (C * du + G * u_avg);
%         err = err + sum(residual.^2);  % squared norm
%     end
% end
