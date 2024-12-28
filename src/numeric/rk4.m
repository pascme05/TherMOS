function u = rk4(C, G, P, a0, time_steps, num_modes, dt)
    % MATLAB implementation of the solve function using RK4
    % Inputs:
    % C: Capacitance matrix
    % G: Conductance matrix
    % P: Forcing term matrix
    % a0: Function to initialize conditions
    % time_steps: Number of time steps
    % num_modes: Number of modes
    % dt: Time step size

    % init
    C_inverse = inv(C);
    CG = C_inverse * G;
    CP = (C_inverse * P)';

    % Solution vectors
    u = zeros(time_steps + 1, num_modes);

    % Current time solution
    a_n = a0;
    u(1, :) = a_n';
    
    % Initialize argument for interpolation
    initial = zeros(size(CP(1, :)));

    for i = 1:time_steps
        % Implement RK4
        x1 = initial;
        if i > 1
            x1 = CP(i - 1, :);
        end
        x2 = CP(i, :);
        
        curr_arg = x1;
        arg_step = (x2 - x1) / 20;
        
        for j = 1:20
            % Interpolation for RK4 arguments
            interp_arg = curr_arg + arg_step;
            next_arg = interp_arg + arg_step;
            
            % RK4 steps
            k1 = dt * f(curr_arg, CG, a_n);
            k2 = dt * f(interp_arg, CG, a_n + 0.5 * k1');
            k3 = dt * f(interp_arg, CG, a_n + 0.5 * k2');
            k4 = dt * f(next_arg, CG, a_n + k3');
            
            % Update solution
            a_vec = a_n' + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            a_n = a_vec';
            curr_arg = next_arg;
        end
        
        u(i + 1, :) = a_n';
    end
    
    % Return all solutions except the first
    u = u(2:end-1, :);
end

function val = f(arg, CG, a)
    % ODE function
    val = arg' - CG * a';
end

