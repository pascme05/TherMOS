% Galerkin POD Implementation for Half Aluminum and Half Copper Plate with Time-Dependent Heat Generation

%% Clear Workspace and Command Window
clear; clc; close all;

%% 1. Define the Composite Plate Geometry and Material Properties

% Plate dimensions (meters)
Lx = 1.0; % Length in x-direction
Ly = 1.0; % Length in y-direction

% Number of grid points (interior points)
nx = 25; % Number of interior grid points in x
ny = 25; % Number of interior grid points in y
% Note: Increase nx and ny for higher resolution (e.g., 50), but ode45 may become slow

% Spatial discretization
dx = Lx / (nx + 1);
dy = Ly / (ny + 1);

% Create spatial grid including boundary points
x = linspace(0, Lx, nx+2); % x = [0, dx, 2dx, ..., Lx]
y = linspace(0, Ly, ny+2);
[X_grid, Y_grid] = meshgrid(x, y);

% Thermal diffusivities (m^2/s)
alpha_Al = 9.7e-5;  % Aluminum
alpha_Cu = 1.1e-4;  % Copper

% Initialize thermal diffusivity matrix (ny+2 x nx+2)
alpha = alpha_Al * ones(ny+2, nx+2); % Start with aluminum

% Assign copper diffusivity to the right half
half_x = ceil((nx+2)/2);
alpha(:, half_x:end) = alpha_Cu;

% Visualize thermal diffusivity distribution
figure;
imagesc(x, y, alpha);
colorbar;
title('Spatial Distribution of Thermal Diffusivity (\alpha)');
xlabel('x (m)');
ylabel('y (m)');
axis equal tight;

%% 2. Discretize the 2D Heat Equation Using Finite Differences

% Number of interior points
N = nx * ny;

% Initialize the thermal diffusivity for interior points (vector form)
alpha_vec = alpha(2:end-1, 2:end-1); % Exclude boundaries
alpha_vec = alpha_vec(:); % Flatten to vector

% Construct the Laplacian operator with variable diffusivity
% We use central differences and handle variable alpha

% Preallocate sparse matrix entries
row = [];
col = [];
val = [];

% Function to map 2D indices to 1D
idx = @(i, j) (j-1)*nx + i;

for j = 1:ny
    for i = 1:nx
        k = idx(i,j);
        % Current alpha
        alpha_ij = alpha_vec(k);
        
        % Center coefficient
        row(end+1) = k;
        col(end+1) = k;
        val(end+1) = -2*(alpha_ij / dx^2 + alpha_ij / dy^2);
        
        % East neighbor
        if i < nx
            row(end+1) = k;
            col(end+1) = idx(i+1, j);
            val(end+1) = alpha_ij / dx^2;
        end
        
        % West neighbor
        if i > 1
            row(end+1) = k;
            col(end+1) = idx(i-1, j);
            val(end+1) = alpha_ij / dx^2;
        end
        
        % North neighbor
        if j < ny
            row(end+1) = k;
            col(end+1) = idx(i, j+1);
            val(end+1) = alpha_ij / dy^2;
        end
        
        % South neighbor
        if j > 1
            row(end+1) = k;
            col(end+1) = idx(i, j-1);
            val(end+1) = alpha_ij / dy^2;
        end
    end
end

% Assemble the sparse Laplacian matrix
L = sparse(row, col, val, N, N);

%% 3. Initial Conditions and Boundary Conditions

% Initial temperature (K)
T_initial = 100; % Uniform initial temperature

% Initialize temperature matrix including boundaries
T0_matrix = T_initial * ones(ny+2, nx+2);

% Apply boundary conditions (constant temperature on all boundaries)
T0_matrix(:,1) = 300;        % Left boundary
T0_matrix(:,end) = 300;      % Right boundary
T0_matrix(1,:) = 300;        % Bottom boundary
T0_matrix(end,:) = 300;      % Top boundary

% Extract interior temperatures and flatten to vector
T0 = T0_matrix(2:end-1, 2:end-1);
T0 = T0(:);

%% 4. Define the Time-Dependent Internal Heat Generation Function

% Define the internal heat generation function (time-dependent)
heat_generation = @(t) 500 * sin(pi * t / 10); % Example: varies with time

%% 5. Define the ODE Function

% Define the ODE function as an anonymous function
% ODE function: dTdt = L * T + heat_generation(t)
heat_ode = @(t, T) L * T + heat_generation(t) * ones(N, 1); % Adjust for heat generation

%% 6. Solve the Full-Order Model Using ode45

% Time discretization for ODE
Tf = 10.0; % Final time (seconds)
dt = 0.1;  % Time step size (unused in ode45)
nt_output = 100; % Number of output time points
t_output = linspace(0, Tf, nt_output); % Define common output time points

% Options for ode45 to control accuracy and ensure outputs at t_output
options = odeset('RelTol',1e-4,'AbsTol',1e-6,'OutputFcn',[]);

fprintf('Solving the Full-Order Model (FOM) using ode45...\n');
tic;
[t_fom, T_fom] = ode45(heat_ode, t_output, T0, options);
toc;
fprintf('FOM solution completed.\n');

%% 7. Perform Proper Orthogonal Decomposition (POD)

% Collect snapshots (columns of temperature vectors)
snapshots = T_fom'; % Each column is a snapshot at a time step

% Compute Singular Value Decomposition (SVD)
fprintf('Performing Singular Value Decomposition (SVD)...\n');
[U, S, V] = svd(snapshots, 'econ');

% Compute cumulative energy
singular_values = diag(S);
energy = cumsum(singular_values.^2) / sum(singular_values.^2);

% Determine number of modes to retain (e.g., 99% energy)
energy_threshold = 0.99;
r = find(energy >= energy_threshold, 1, 'first');

fprintf('Number of POD modes retained: %d out of %d\n', r, N);

% Truncate to r modes
U_r = U(:,1:r);
S_r = S(1:r,1:r);
V_r = V(:,1:r);

%% 8. Construct the Galerkin Reduced-Order Model (ROM)

% Project the Laplacian matrix L onto the reduced basis
L_r = U_r' * L * U_r;

% Define the ODE function for ROM
heat_ode_rom = @(t, a) L_r * a + heat_generation(t) * ones(r, 1); % Adjusted for ROM

%% 9. Solve the ROM Using ode45

% Initial conditions for ROM
a0_rom = U_r' * T0;

% Time span for ROM (same as FOM)
time_span_rom = t_output; % Ensure same time points

% Solve the ROM using ode45 with specified output times
fprintf('Solving the Reduced-Order Model (ROM) using ode45...\n');
tic;
[t_rom, a_rom] = ode45(heat_ode_rom, time_span_rom, a0_rom, options);
toc;
fprintf('ROM solution completed.\n');

%% 10. Reconstruct the Full Solution from ROM

% Reconstruct temperature snapshots from ROM coefficients
T_rom = U_r * a_rom'; % Each column corresponds to a snapshot

% Add boundary conditions back to reconstructed snapshots
% Initialize with boundary temperatures
T_rom_full = 300 * ones(ny+2, nx+2, length(t_rom));

for k = 1:length(t_rom)
    % Reshape interior temperatures
    T_interior = reshape(T_rom(:,k), nx, ny);
    
    % Insert into full temperature matrix
    T_full = 300 * ones(ny+2, nx+2);
    T_full(2:end-1, 2:end-1) = T_interior;
    
    % Store the full temperature matrix
    T_rom_full(:,:,k) = T_full;
end

%% 11. Compute and Plot the Relative L2 Error Over Time

fprintf('Computing the Relative L2 Error at each time step...\n');

% Initialize error vector
relative_error = zeros(length(t_fom), 1);

% Compute relative L2 error at each snapshot
for k = 1:length(t_fom)
    % Full-order temperature at time k
    T_full_k = T_fom(k, :)'; % Column vector
    
    % Reduced-order reconstructed temperature at time k
    T_rom_k = T_rom(:,k); % Column vector
    
    % Compute L2 norm of the difference
    numerator = norm(T_full_k - T_rom_k, 2);
    
    % Compute L2 norm of the full-order solution
    denominator = norm(T_full_k, 2);
    
    % Compute relative error
    relative_error(k) = numerator / denominator;
end

% Plot Relative L2 Error Over Time
figure;
plot(t_fom, relative_error, 'b-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Relative L2 Error', 'FontSize', 12);
title('Relative L2 Error Between FOM and ROM Over Time', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Optionally, plot error on a logarithmic scale
figure;
semilogy(t_fom, relative_error, 'r-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Relative L2 Error (Log Scale)', 'FontSize', 12);
title('Relative L2 Error Between FOM and ROM Over Time (Log Scale)', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Display maximum error information
[max_error, max_idx] = max(relative_error);
fprintf('Maximum Relative L2 Error: %.4f at time t = %.2f seconds.\n', max_error, t_fom(max_idx));

%% 12. Visualize the Temperature Distribution at Final Time

% Full-order model temperature at final time
T_fom_final = T_fom(end, :)'; % Column vector

% Reduced-order model reconstructed temperature at final time
T_rom_final = T_rom(:,end); % Column vector

% Reshape to 2D for visualization
T_fom_final_matrix = 300 * ones(ny+2, nx+2); % Initialize with boundary temperatures
T_fom_final_matrix(2:end-1, 2:end-1) = reshape(T_fom_final, nx, ny);

T_rom_final_matrix = 300 * ones(ny+2, nx+2); % Initialize with boundary temperatures
T_rom_final_matrix(2:end-1, 2:end-1) = reshape(T_rom_final, nx, ny);

% Create spatial grid for plotting
[X_plot, Y_plot] = meshgrid(x, y);

% Plot Full-Order Model Solution
figure;
surf(X_plot, Y_plot, T_fom_final_matrix, 'EdgeColor', 'none');
xlabel('x (m)', 'FontSize', 12);
ylabel('y (m)', 'FontSize', 12);
zlabel('Temperature (K)', 'FontSize', 12);
title('Full-Order Model (FOM) Temperature Distribution at Final Time', 'FontSize', 14);
shading interp;
colorbar;
view(45, 30);

% Plot Reduced-Order Model Solution
figure;
surf(X_plot, Y_plot, T_rom_final_matrix, 'EdgeColor', 'none');
xlabel('x (m)', 'FontSize', 12);
ylabel('y (m)', 'FontSize', 12);
zlabel('Temperature (K)', 'FontSize', 12);
title('Reduced-Order Model (ROM) Temperature Distribution at Final Time', 'FontSize', 14);
shading interp;
colorbar;
view(45, 30);

%% 13. Plot Temporal Envelope for Selected Point

% Define the selected spatial point (e.g., at the center of the plate)
% Adjust these indices according to your spatial grid size
x_selected_index = ceil((nx+2)/2); % Middle point in x-direction
y_selected_index = ceil((ny+2)/2); % Middle point in y-direction

% Extract temperature data for the selected point
T_fom_selected = T_fom(:, idx(x_selected_index - 1, y_selected_index - 1)); % FOM
T_rom_selected = T_rom(idx(x_selected_index - 1, y_selected_index - 1), :); % ROM

% Plotting
figure;
plot(t_fom, T_fom_selected, 'b-', 'LineWidth', 2, 'DisplayName', 'Full-Order Model (FOM)');
hold on;
plot(t_rom, T_rom_selected, 'r--', 'LineWidth', 2, 'DisplayName', 'Reduced-Order Model (ROM)');
hold off;

xlabel('Time (s)', 'FontSize', 12);
ylabel('Temperature (K)', 'FontSize', 12);
title('Temporal Envelope at Selected Point', 'FontSize', 14);
legend('Location', 'Best');
grid on;
set(gca, 'FontSize', 12);