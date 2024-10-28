% Define parameters
Nx = 50; % Spatial resolution in x-direction
Ny = 50; % Spatial resolution in y-direction
Nt = 100; % Number of time steps

dx = 1.0 / (Nx - 1);
dy = 1.0 / (Ny - 1);
dt = 0.01;

% Material parameters
rho = 1.0;  % Density
Cp = 1.0;   % Specific heat capacity
k = 1.0;    % Thermal conductivity
alpha = k / (rho * Cp); % Thermal diffusivity

% Initial condition
T = zeros(Nx, Ny);

% Volumetric heat generation
Pv1 = zeros(Nx, Ny);
Pv2 = zeros(Nx, Ny);

% Define some heat sources for meaningful data
Pv1(round(Nx/4), round(Ny/4)) = 100;
Pv1(round(3*Nx/4), round(3*Ny/4)) = 100;

Pv2(round(Nx/2), round(Ny/2)) = 100;
Pv2(round(Nx/3), round(Ny/3)) = 50;

% Time integration loop
T_snapshots1 = zeros(Nx, Ny, Nt);
T_snapshots2 = zeros(Nx, Ny, Nt);

for t = 1:Nt
    % Update temperature for test case 1
    T_new1 = T;
    for i = 2:Nx-1
        for j = 2:Ny-1
            T_new1(i, j) = T(i, j) + alpha * dt * ( ...
                (T(i+1, j) - 2*T(i, j) + T(i-1, j)) / dx^2 + ...
                (T(i, j+1) - 2*T(i, j) + T(i, j-1)) / dy^2 ) + ...
                dt * Pv1(i, j) / (rho * Cp);
        end
    end
    T = T_new1;
    T_snapshots1(:, :, t) = T;

    % Update temperature for test case 2
    T_new2 = T;
    for i = 2:Nx-1
        for j = 2:Ny-1
            T_new2(i, j) = T(i, j) + alpha * dt * ( ...
                (T(i+1, j) - 2*T(i, j) + T(i-1, j)) / dx^2 + ...
                (T(i, j+1) - 2*T(i, j) + T(i, j-1)) / dy^2 ) + ...
                dt * Pv2(i, j) / (rho * Cp);
        end
    end
    T = T_new2;
    T_snapshots2(:, :, t) = T;
end

% Flatten the temperature data for POD
T1 = reshape(T_snapshots1, Nx*Ny, Nt)';
T2 = reshape(T_snapshots2, Nx*Ny, Nt)';

% Step 1: Construct the snapshot matrix
X1 = T1';  % Snapshot matrix with temperature fields as columns

% Step 2: Perform Singular Value Decomposition (SVD)
[U, S, V] = svd(X1, 'econ');

% Select the number of POD modes (r) to retain (can use energy criterion)
r = 5;  % Number of POD modes
Phi = U(:, 1:r);  % POD modes

% Step 3: Project the governing equations onto the POD basis
% Project Pv and T onto the reduced basis
Pv1_flat = reshape(Pv1, Nx*Ny, 1);
Pv2_flat = reshape(Pv2, Nx*Ny, 1);

Pv1_reduced = Phi' * repmat(Pv1_flat, 1, Nt);
T1_reduced = Phi' * T1';

Pv2_reduced = Phi' * repmat(Pv2_flat, 1, Nt);
T2_reduced = Phi' * T2';

% Initialize reduced-order model solution
a1 = zeros(r, Nt+1);
a1(:, 1) = T1_reduced(:, 1);  % Initial condition in reduced space

a2 = zeros(r, Nt+1);
a2(:, 1) = T2_reduced(:, 1);  % Initial condition in reduced space

% Step 4: Solve the reduced-order model
for n = 1:Nt
    % Update rule for the reduced-order model (Euler method for simplicity)
    a1(:, n+1) = a1(:, n) + dt * (Pv1_reduced(:, n) - k .* a1(:, n)) ./ (rho .* Cp);
    a2(:, n+1) = a2(:, n) + dt * (Pv2_reduced(:, n) - k .* a2(:, n)) ./ (rho .* Cp);
end

% Step 5: Reconstruct the temperature field from the reduced-order solution
T1_reconstructed = Phi * a1;
T2_reconstructed = Phi * a2;

% Reshape the reconstructed temperature fields
T1_reconstructed = reshape(T1_reconstructed', Nt+1, Nx, Ny);
T2_reconstructed = reshape(T2_reconstructed', Nt+1, Nx, Ny);

% Plot the results for the last time sample for both test cases
figure;
subplot(1,2,1);
imagesc(squeeze(T1_reconstructed(end, :, :)));
colorbar;
title('Test Case 1: Last Temperature Field');

subplot(1,2,2);
imagesc(squeeze(T2_reconstructed(end, :, :)));
colorbar;
title('Test Case 2: Last Temperature Field');

% Select a specific point in space (e.g., the center of the domain)
point_idx = sub2ind([Nx, Ny], round(Nx/2), round(Ny/2));

% Extract temperature at the selected point for all time points
T1_point_time = squeeze(T1_reconstructed(:, point_idx));
T2_point_time = squeeze(T2_reconstructed(:, point_idx));

% Plot the temperature at the selected point over all time points
figure;
plot(0:dt:(Nt)*dt, T1_point_time, 'b', 'DisplayName', 'Test Case 1');
hold on;
plot(0:dt:(Nt)*dt, T2_point_time, 'r', 'DisplayName', 'Test Case 2');
xlabel('Time');
ylabel('Temperature');
title('Temperature at Selected Point Over Time');
legend('show');
grid on;
