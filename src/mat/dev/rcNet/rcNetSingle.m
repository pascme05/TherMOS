%% ----------------- Reduced-Order Thermal Model Extraction -----------------
clc; clear;

% ------------------ USER INPUT ------------------
% N: number of nodes/components
% t: time vector (M x 1)
% T_step: cell array of transient temps for each single power step (M x N)
% P: power vector (N x 1)
% T_c: heat sink temperature

% Example placeholders (replace with your data)
N = 3;                  
t = (0:0.1:100)';        
P = [20; 30; 10];          
T_c = 25;               

% T_step{i} should be MxN for single power step i
% For testing: generate fake data (you would load real data)
T_step = cell(N,1);
for i = 1:N
    T_step{i} = zeros(length(t),N);
    for j = 1:N
        R_fake = 2.5 + 2*rand;
        tau_fake = 5 + 5*rand;
        T_step{i}(:,j) = T_c + R_fake*P(i)*(1 - exp(-t/tau_fake));
    end
end

%% ------------------ Step 1: Extract Steady-State Rth ------------------
Rth_init = zeros(N,N);
for i = 1:N
    dT = T_step{i}(end,:)' - T_c;   % steady-state temperature rise (Nx1)
    Rth_init(:,i) = dT / P(i);      % each column = effect of Pi on all nodes
end

%% ------------------ Step 2: Initial Guess for Cth ------------------
Cth_init = zeros(N,1);
for i = 1:N
    Ti = T_step{i}(:,i) - T_c;     % self-heating response
    Pi_i = P(i);
    
    % Fit 1st-order exponential: y = a*(1-exp(-t/b))
    ft = fittype('a*(1-exp(-x/b))','independent','x','coefficients',{'a','b'});
    f = fit(t, Ti, ft, 'StartPoint',[Rth_init(i,i)*Pi_i, 1]);
    tau = f.b;
    Cth_init(i) = tau / Rth_init(i,i);
end

%% ------------------ Step 3: Full Network Optimization ------------------
x0 = [reshape(Rth_init,[],1); Cth_init];  % initial guess
options = optimoptions('lsqnonlin','Display','iter','MaxIterations',1000);

x_opt = lsqnonlin(@(x) thermal_err(x, P, T_step, t, N, T_c), x0, [], [], options);

% Extract optimized Rth and Cth
Rth_opt = reshape(x_opt(1:N^2), N, N);
Cth_opt = x_opt(N^2+1:end);

disp('Optimized Rth (K/W):'); disp(Rth_opt);
disp('Optimized Cth (J/K):'); disp(Cth_opt);

%% ------------------ Step 4: Reconstruct temperatures (non-ODE model) ------------------
T_sim_all = cell(N,1);

for k = 1:N   % for each excitation case (only Pk active)
    
    T_sim = zeros(length(t), N);
    
    for j = 1:N
        tau = Cth_opt(j) * Rth_opt(j,j);
        
        % Time response (Mx1)
        phi = (1 - exp(-t/tau));
        
        % Outer product → MxN
        T_sim = T_sim + phi * (Rth_opt(:,j)' * P(j));
    end
    
    % Add ambient
    T_sim = T_sim + T_c;
    
    T_sim_all{k} = T_sim;
end

%% ------------------ Step 5: Plot comparison ------------------
for k = 1:N
    figure('Name',['Excitation P', num2str(k)]);
    
    for i = 1:N
        subplot(N,1,i);
        
        plot(t, T_step{k}(:,i), 'b', 'LineWidth', 2); hold on;
        plot(t, T_sim_all{k}(:,i), '--r', 'LineWidth', 2);
        
        grid on;
        xlabel('Time (s)');
        ylabel(['T_', num2str(i)]);
        title(['Node ', num2str(i), ' (Power at node ', num2str(k), ')']);
        
        legend('True','Model');
    end
end

%% ------------------ Step 6: Plot error ------------------
for k = 1:N
    figure('Name',['Error P', num2str(k)]);
    
    for i = 1:N
        subplot(N,1,i);
        
        err = T_sim_all{k}(:,i) - T_step{k}(:,i);
        
        plot(t, err, 'k', 'LineWidth', 1.5);
        
        grid on;
        xlabel('Time (s)');
        ylabel(['Error T_', num2str(i)]);
        title(['Error Node ', num2str(i), ' (Power at node ', num2str(k), ')']);
    end
end

%% ------------------ Step 7: RMSE ------------------
rmse = zeros(N,N);

for k = 1:N
    err = T_sim_all{k} - T_step{k};
    rmse(:,k) = sqrt(mean(err.^2))';
end

disp('RMSE (rows = node, cols = excitation):');
disp(rmse);

%% ------------------ Supporting Function ------------------
function err = thermal_err(x, P, T_step, t, N, T_c)
    Rth = reshape(x(1:N^2),N,N);
    Cth = x(N^2+1:end);
    err = [];
    M = length(t);
    
    for i = 1:N
        % Simulate network response for single power step Pi
        T_sim = zeros(M,N);
        for j = 1:N
            tau = Cth(j) * Rth(j,j);
            phi = (1 - exp(-t/tau)); % Mx1
            T_sim = T_sim + phi * (Rth(:,j)' * P(j));
        end
        err = [err; T_sim(:) - T_step{i}(:)];
    end
end