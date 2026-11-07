%% ----------------- Thermal RC Model with 2 Time-Varying Heat Sinks -----------------
clc; clear;

%% ------------------ USER SETTINGS ------------------
N = 3;                          % number of nodes
t = (0:0.1:50)';               % time vector
M = length(t);

%% ------------------ INPUT SIGNALS ------------------

% Power input (MxN)
P_all = zeros(M,N);
P_all(:,1) = 200*(t>5);          % node 1 turns on at t=5
P_all(:,2) = 100*(t>10);         % node 2 at t=10
P_all(:,3) = 150*(t>20);       % node 3 at t=20

% Time-varying ambient temperature
T_amb_vec = 45 + 0*sin(0.1*t);

% Time-varying coolant temperature
T_cool_vec = 35 + 0*cos(0.05*t);

%% ------------------ TRUE SYSTEM (for testing) ------------------

% True parameters (unknown in real case)
Rth_int_true  = [0.5 0.1 0.05;
                 0.1 0.6 0.08;
                 0.05 0.08 0.4];

Rth_amb_true  = [2; 3; 4];
Rth_cool_true = [1; 1.5; 2];
Cth_true      = [10; 8; 6];

% Convert to conductances
G_int_true  = Rth_int_true \ eye(N);
G_amb_true  = diag(1 ./ Rth_amb_true);
G_cool_true = diag(1 ./ Rth_cool_true);
C_true      = diag(Cth_true);

%% ------------------ SIMULATE TRUE SYSTEM ------------------

odefun_true = @(tt,T) thermal_ode_2sink(tt, T, t, P_all, ...
                                        C_true, G_int_true, G_amb_true, G_cool_true, ...
                                        T_amb_vec, T_cool_vec);

T0 = 35 * ones(N,1);

[~, T_all] = ode45(odefun_true, t, T0);

%% ------------------ INITIAL GUESS ------------------

Rth_int_init  = eye(N);
Rth_amb_init  = ones(N,1)*2;
Rth_cool_init = ones(N,1)*2;
Cth_init      = ones(N,1)*5;

x0 = [reshape(Rth_int_init,[],1); Rth_amb_init; Rth_cool_init; Cth_init];

%% ------------------ OPTIMIZATION ------------------

options = optimoptions('lsqnonlin','Display','iter','MaxIterations',500);

x_opt = lsqnonlin(@(x) thermal_err_2sink(x, P_all, T_all, t, ...
                                         T_amb_vec, T_cool_vec, N), ...
                  x0, [], [], options);

%% ------------------ EXTRACT PARAMETERS ------------------

Rth_int  = reshape(x_opt(1:N^2), N, N);
Rth_amb  = x_opt(N^2+1 : N^2+N);
Rth_cool = x_opt(N^2+N+1 : N^2+2*N);
Cth      = x_opt(N^2+2*N+1 : end);

disp('Estimated Rth_int:'); disp(Rth_int);
disp('Estimated Rth_amb:'); disp(Rth_amb);
disp('Estimated Rth_cool:'); disp(Rth_cool);
disp('Estimated Cth:'); disp(Cth);

%% ------------------ SIMULATE FITTED MODEL ------------------

G_int  = Rth_int \ eye(N);
G_amb  = diag(1 ./ Rth_amb);
G_cool = diag(1 ./ Rth_cool);
C      = diag(Cth);

odefun_fit = @(tt,T) thermal_ode_2sink(tt, T, t, P_all, ...
                                       C, G_int, G_amb, G_cool, ...
                                       T_amb_vec, T_cool_vec);

[~, T_sim] = ode45(odefun_fit, t, T0);

%% ------------------ PLOTS ------------------

figure;
for i = 1:N
    subplot(N,1,i);
    plot(t, T_all(:,i), 'b', 'LineWidth', 2); hold on;
    plot(t, T_sim(:,i), '--r', 'LineWidth', 2);
    grid on;
    xlabel('Time (s)');
    ylabel(['T_', num2str(i)]);
    title(['Node ', num2str(i)]);
    legend('True','Model');
end

%% ------------------ ERROR PLOT ------------------

figure;
for i = 1:N
    subplot(N,1,i);
    plot(t, T_sim(:,i) - T_all(:,i), 'k', 'LineWidth', 1.5);
    grid on;
    title(['Error Node ', num2str(i)]);
end

%% ------------------ RMSE ------------------

rmse = sqrt(mean((T_sim - T_all).^2));
disp('RMSE per node:'); disp(rmse);

%% ------------------ FUNCTIONS ------------------

function dTdt = thermal_ode_2sink(tt, T, t, P_all, ...
                                 C, G_int, G_amb, G_cool, ...
                                 T_amb_vec, T_cool_vec)

    % Interpolate inputs
    P_vec     = interp1(t, P_all, tt, 'linear', 'extrap')';
    T_amb     = interp1(t, T_amb_vec, tt, 'linear', 'extrap');
    T_cool    = interp1(t, T_cool_vec, tt, 'linear', 'extrap');

    T = T(:);

    % Heat flows
    q_int  = -G_int * T;
    q_amb  = -G_amb * (T - T_amb);
    q_cool = -G_cool * (T - T_cool);

    % ODE
    dTdt = C \ (q_int + q_amb + q_cool + P_vec);
end

function err = thermal_err_2sink(x, P_all, T_all, t, T_amb_vec, T_cool_vec, N)

    % Extract parameters
    Rth_int  = reshape(x(1:N^2), N, N);
    Rth_amb  = x(N^2+1 : N^2+N);
    Rth_cool = x(N^2+N+1 : N^2+2*N);
    Cth      = x(N^2+2*N+1 : end);

    % Build matrices
    G_int  = Rth_int \ eye(N);
    G_amb  = diag(1 ./ Rth_amb);
    G_cool = diag(1 ./ Rth_cool);
    C      = diag(Cth);

    % Simulate
    T0 = T_all(1,:)';
    odefun = @(tt,T) thermal_ode_2sink(tt, T, t, P_all, ...
                                       C, G_int, G_amb, G_cool, ...
                                       T_amb_vec, T_cool_vec);

    [~, T_sim] = ode45(odefun, t, T0);

    % Error
    err = (T_sim - T_all);
    err = err(:);
end