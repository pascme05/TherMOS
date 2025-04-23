function [alpha_opt, u_sim] = optCthGth3(C, G, rPhi, T, u, q, tlist)
    K = size(u, 1);

    % Initial guess
    alpha0 = ones(K, 1);

    % Loss function: simulate u given alpha .* q, compare to measured u
    loss_fn = @(alpha) u_residual_from_alpha(alpha, C, G, rPhi, T, u, q, tlist);

    % Optimizer settings
    options = optimoptions('fminunc', ...
        'Display', 'iter', ...
        'MaxIterations', 1000, ...
        'StepTolerance', 1e-12, ...
        'OptimalityTolerance', 1e-12, ...
        'FunctionTolerance', 1e-12, ...
        'Algorithm', 'quasi-newton');

    % Optimize
    [alpha_opt, ~] = fminunc(loss_fn, alpha0, options);

    % Return final simulated u
    F_adj = alpha_opt .* q;
    odeoptions = odeset('Mass', C, 'Jacobian', -G, 'JConstant', 'on', ...
                        'RelTol', 1e-6, 'AbsTol', 1e-8);
    g0 = u(:, 1);
    sol = ode23s(@(t,y) odefnc2(t,y,G,F_adj',tlist),tlist,g0,odeoptions);
    u_sim = deval(sol, tlist);
end

function loss = u_residual_from_alpha(alpha, C, G, rPhi, T, u, q, tlist)
    F_adj = alpha .* q;

    g0 = u(:, 1);
    odeoptions = odeset('Mass', C, 'JConstant', 'on', ...
                            'RelTol', 1e-5, 'AbsTol', 1e-7, ...
                            'Jacobian', -G);

    try
        sol = ode23s(@(t,y) odefnc2(t,y,G,F_adj',tlist),tlist,g0,odeoptions);
        u_sim = deval(sol, tlist);
        T_sim = u_sim' * rPhi';
        loss = sum((T - T_sim).^2, 'all');
    catch
        loss = 1e12;  % Large penalty on failure
    end
end