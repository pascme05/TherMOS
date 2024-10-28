% Sample data generation (replace this with your actual data)
Nt = 100; % Number of samples
M = 4; % Number of temperature sensors
N = 4; % Number of power sources

% Simulated true symmetric Cth and Gth matrices with meaningful values
Cth_true = 0.5 + 1.5 * rand(M); % Thermal capacitance between 0.5 and 2 J/K
Cth_true = (Cth_true + Cth_true') / 2; % Making Cth_true symmetric
Gth_true = 0.01 + 0.04 * rand(M); % Thermal conductance between 0.01 and 0.05 W/K
Gth_true = (Gth_true + Gth_true') / 2; % Making Gth_true symmetric

% Time vector
t = linspace(0, 1000, Nt); % Extending the time range for a smooth curve
dt = t(2) - t(1); % Time step

% Generate simulated temperature data based on true Cth and Gth
P = 5 * ones(Nt, N); % Constant power loss of 5 W (replace with actual data if needed)
T = zeros(Nt, M);
for i = 1:M
    for k = 2:Nt
        for j = 1:N
            T(k, i) = T(k-1, i) + (dt / Cth_true(i, j)) * (P(k-1, j) - Gth_true(i, j) * T(k-1, i));
        end
    end
end

% Add some noise to the temperature data
T = T + 0.05 * randn(size(T));

% Objective function to minimize (least squares)
objectiveFunction = @(x) objective_dgl(x, t, P, T, M, N);

% Initial guess for Cth and Gth
x0 = [ones(M*(M+1)/2, 1); 0.01 * ones(M*(M+1)/2, 1)];

% Optimization options
options = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 3000);

% Perform the optimization
[x_opt, resnorm, residual, exitflag, output] = lsqnonlin(objectiveFunction, x0, [], [], options);

% Extract the optimized symmetric Cth and Gth
Cth_opt = reshape_sym_matrix(x_opt(1:M*(M+1)/2), M);
Gth_opt = reshape_sym_matrix(x_opt(M*(M+1)/2+1:end), M);

% Predict temperatures using optimized Cth and Gth
T_pred = predict_temperature(t, P, Cth_opt, Gth_opt);

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
    Cth = reshape_sym_matrix(x(1:M*(M+1)/2), M);
    Gth = reshape_sym_matrix(x(M*(M+1)/2+1:end), M);

    % Predict temperatures using the current parameters
    T_pred = predict_temperature(t, P, Cth, Gth);

    % Compute the error (sum of squared differences)
    error = (T_true - T_pred).^2;
    error = error(:); % Convert to a column vector for lsqnonlin
end

% Function to reshape the upper triangular part of a vector into a symmetric matrix
function A = reshape_sym_matrix(v, M)
    A = zeros(M, M);
    ind = 1;
    for i = 1:M
        for j = i:M
            A(i, j) = v(ind);
            A(j, i) = v(ind);
            ind = ind + 1;
        end
    end
end

% Function to predict temperature based on current Cth and Gth estimates
function T_pred = predict_temperature(t, P, Cth, Gth)
    Nt = length(t);
    dt = t(2) - t(1);
    M = size(Cth, 1);
    N = size(Cth, 2);
    T_pred = zeros(Nt, M);
    for i = 1:M
        for k = 2:Nt
            for j = 1:N
                T_pred(k, i) = T_pred(k-1, i) + (dt / Cth(i, j)) * (P(k-1, j) - Gth(i, j) * T_pred(k-1, i));
            end
        end
    end
end
