% Sample data generation (replace this with your actual data)
Nt = 100; % Number of samples
M = 2; % Number of temperature sensors
N = 3; % Number of power sources

% Simulated true Cth and Gth matrices with meaningful values
Cth_true = 0.5 + 1.5 * rand(M, N); % Thermal capacitance between 0.5 and 2 J/K
Gth_true = 0.01 + 0.04 * rand(M, N); % Thermal conductance between 0.01 and 0.05 W/K

% Time vector
t = linspace(0, 1000, Nt); % Extending the time range for a smooth curve
dt = t(2) - t(1); % Time step

% Generate simulated temperature data based on true Cth and Gth
P = 5 * ones(Nt, N); % Constant power loss of 5 W (replace with actual data if needed)
T = zeros(Nt, M);
for i = 1:M
    for j = 1:N
        for k = 2:Nt
            T(k, i) = T(k-1, i) + (dt / Cth_true(i, j)) * (P(k-1, j) - Gth_true(i, j) * T(k-1, i));
        end
    end
end

% Add some noise to the temperature data
T = T + 0.05 * randn(size(T));

% Objective function to minimize (least squares)
fun = @(x) reshape(T - predict_temperature(t, P, reshape(x(1:M*N), [M, N]), reshape(x(M*N+1:end), [M, N])), [], 1);

% Initial guess for Cth and Gth
x0 = [ones(M*N, 1); 0.01 * ones(M*N, 1)];

% Optimization options
options = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', 1e-8, 'MaxIter', 1000);

% Perform the optimization
x = lsqnonlin(fun, x0, [], [], options);

% Extract the optimized Cth and Gth
Cth_opt = reshape(x(1:M*N), [M, N]);
Gth_opt = reshape(x(M*N+1:end), [M, N]);

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

% Function to predict temperature based on current Cth and Gth estimates
function T_pred = predict_temperature(t, P, Cth, Gth)
    Nt = length(t);
    dt = t(2) - t(1);
    M = size(Cth, 1);
    N = size(Cth, 2);
    T_pred = zeros(Nt, M);
    for i = 1:M
        for j = 1:N
            for k = 2:Nt
                T_pred(k, i) = T_pred(k-1, i) + (dt / Cth(i, j)) * (P(k-1, j) - Gth(i, j) * T_pred(k-1, i));
            end
        end
    end
end
