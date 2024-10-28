% Sample data generation (replace this with your actual data)
Nt = 100; % Number of samples
M = 4; % Number of temperature sensors
N = 4; % Number of power sources

% Simulated true Cth, Gth, and F matrices with meaningful values
Cth_true = 0.5 + 1.5 * rand(M, N); % Thermal capacitance between 0.5 and 2 J/K
Gth_true = 0.01 + 0.04 * rand(M, N); % Thermal conductance between 0.01 and 0.05 W/K
F_true = 0.5 + 1.5 * rand(M, N); % Power mapping matrix

% Time vector
t = linspace(0, 1000, Nt); % Extending the time range for a smooth curve
dt = t(2) - t(1); % Time step

% Generate simulated temperature data based on true Cth, Gth, and F
P = 5 * ones(Nt, N); % Constant power loss of 5 W (replace with actual data if needed)
T = zeros(Nt, M);
for i = 1:M
    for k = 2:Nt
        FP = sum(F_true(i, :) .* P(k-1, :)); % Power mapping
        T(k, i) = T(k-1, i) + (dt / Cth_true(i, i)) * (FP - Gth_true(i, i) * T(k-1, i));
    end
end

% Add some noise to the temperature data
T = T + 0.05 * randn(size(T));

% Objective function to minimize (least squares)
objectiveFunction = @(x) objective_dgl(x, t, P, T, M, N);

% Initial guess for Cth, Gth, and F
x0 = [ones(M*N, 1); 0.01 * ones(M*N, 1); 0.5 * ones(M*N, 1)];

% Optimization options
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 3000);

% Perform the optimization
[x_opt, fval, exitflag, output] = fminunc(objectiveFunction, x0, options);

% Extract the optimized Cth, Gth, and F
Cth_opt = reshape(x_opt(1:M*N), [M, N]);
Gth_opt = reshape(x_opt(M*N+1:2*M*N), [M, N]);
F_opt = reshape(x_opt(2*M*N+1:end), [M, N]);

% Predict temperatures using optimized Cth, Gth, and F
T_pred = predict_temperature(t, P, Cth_opt, Gth_opt, F_opt);

% Plot true and predicted temperatures
figure;
for i = 1:M
    subplot(M, 1, i);
    plot(t, T(:, i), 'b', 'DisplayName', 'True Temperature');
    hold on;
    plot(t, T_pred(:, i), 'r--', 'DisplayName', 'Predicted Temperature');
    xlabel('Time');
    ylabel(['Temperature Sensor ' num2str(i)]);
    legend;
    title(['Temperature Sensor ' num2str(i)]);
end

% Objective function definition
function error = objective_dgl(x, t, P, T_true, M, N)
    % Extract parameters from x
    Cth = reshape(x(1:M*N), [M, N]);
    Gth = reshape(x(M*N+1:2*M*N), [M, N]);
    F = reshape(x(2*M*N+1:end), [M, N]);

    % Predict temperatures using the current parameters
    T_pred = predict_temperature(t, P, Cth, Gth, F);

    % Compute the error (sum of squared differences)
    error = sum((T_true - T_pred).^2, 'all');
end

% Function to predict temperature based on current Cth, Gth, and F estimates
function T_pred = predict_temperature(t, P, Cth, Gth, F)
    Nt = length(t);
    dt = t(2) - t(1);
    M = size(Cth, 1);
    N = size(Cth, 2);
    T_pred = zeros(Nt, M);
    for i = 1:M
        for k = 2:Nt
            FP = sum(F(i, :) .* P(k-1, :)); % Power mapping
            T_pred(k, i) = T_pred(k-1, i) + (dt / Cth(i, i)) * (FP - Gth(i, i) * T_pred(k-1, i));
        end
    end
end
