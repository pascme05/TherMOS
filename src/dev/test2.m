% Galerkin POD Implementation for Heated Aluminum Plate with Constant Temperature Boundaries

%% Step 1: Discretize the 2D Heat Equation Using Finite Differences

% Spatial discretization
nx = 50;        % Number of interior grid points in x
ny = 50;        % Number of interior grid points in y
Lx = 1.0;       % Length of the plate in x-direction (meters)
Ly = 1.0;       % Length of the plate in y-direction (meters)
dx = Lx / (nx + 1);
dy = Ly / (ny + 1);
x = linspace(0, Lx, nx+2);  % Including boundary points
y = linspace(0, Ly, ny+2);
[X_grid, Y_grid] = meshgrid(x, y);

% Temporal discretization
dt = 0.1;        % Time step (seconds)
Tf = 10.0;       % Final time (seconds)
nt = ceil(Tf/dt);% Number of time steps
time = linspace(0, Tf, nt+1);

% Initial condition: Uniform temperature
T_initial = 300;  % Kelvin
T0 = T_initial * ones(nx+2, ny+2);  % Including boundary points
T0(:,1) = 300;        % Left boundary
T0(:,end) = 300;      % Right boundary
T0(1,:) = 300;        % Bottom boundary
T0(end,:) = 300;      % Top boundary

% Reshape T into a column vector for computations (interior points only)
T = T0(2:end-1, 2:end-1); % Remove boundary points
T = T(:);                   % Vectorize

% Assemble the finite difference matrix
alpha = 9.7e-5;  % Thermal diffusivity of aluminum (m^2/s)

% Number of interior points
N = nx * ny;

% Create the 1D Laplacian in x-direction
e = ones(nx,1);
T_x = spdiags([e -2*e e], -1:1, nx, nx);
I_x = speye(nx);
L_x = kron(I_x, T_x) / dx^2;

% Create the 1D Laplacian in y-direction
T_y = spdiags([e -2*e e], -1:1, ny, ny);
I_y = speye(ny);
L_y = kron(T_y, I_y) / dy^2;

% Total Laplacian
L = L_x + L_y;

% Identity matrix
I = speye(N);

% System matrix for implicit Euler
A = I - dt * alpha * L;

%% Step 2: Generate Snapshots by Solving the Full-Order Model

% Preallocate snapshot matrix
snapshots = zeros(N, nt+1);
snapshots(:,1) = T;

% Time-stepping loop
fprintf('Solving the full-order model...\n');
for k = 1:nt
    % Solve A * T_new = T_old
    T = A \ T;
    
    % Since boundary conditions are constant and incorporated in A,
    % no additional steps are needed to enforce them
    
    % Store the snapshot
    snapshots(:,k+1) = T;
    
    % Optional: Display progress every 10%
    if mod(k, ceil(nt/10)) == 0
        fprintf('Completed %d%% of full-order simulation.\n', round(100*k/nt));
    end
end

% Transpose snapshots to have snapshots as columns
X = snapshots;

%% Step 3: Perform Proper Orthogonal Decomposition (POD)

fprintf('Performing Proper Orthogonal Decomposition (POD)...\n');

% Perform SVD
[U, S, V] = svd(X, 'econ');

% Determine the number of modes to retain (e.g., 99% energy)
singular_values = diag(S);
energy = cumsum(singular_values.^2) / sum(singular_values.^2);
r = find(energy >= 0.99, 1, 'first');  % Number of modes

fprintf('Number of POD modes retained: %d\n', r);

% Truncate to r modes
U_r = U(:,1:r);
S_r = S(1:r,1:r);
V_r = V(:,1:r);

%% Step 4: Construct the Galerkin Reduced-Order Model

fprintf('Constructing the Galerkin reduced-order model...\n');

% Project the system matrix A onto the reduced basis
A_r = U_r' * A * U_r;

% Precompute the inverse of A_r
A_r_inv = inv(A_r);

% Initialize reduced coefficients
a_r = U_r' * X(:,1);  % Initial condition in reduced space

% Preallocate for reduced snapshots
a_r_snapshots = zeros(r, nt+1);
a_r_snapshots(:,1) = a_r;

% Time-stepping loop for reduced-order model
fprintf('Simulating the reduced-order model...\n');
for k = 1:nt
    % Update reduced coefficients
    a_r = A_r_inv * a_r;
    
    % Store the reduced coefficients
    a_r_snapshots(:,k+1) = a_r;
end

%% Step 5: Reconstruct the Full Solution from Reduced-Order Model

fprintf('Reconstructing the full solution from the reduced-order model...\n');

% Reconstruct snapshots from reduced coefficients
X_reconstructed = U_r * a_r_snapshots;

%% Step 6: Compare Full-Order and Reduced-Order Models

% Compute the Relative L2 Error at Each Time Step
fprintf('Computing the Relative L2 Error at each time step...\n');

% Initialize error vector
relative_error = zeros(1, nt+1);

for k = 1:nt+1
    % Extract the full-order and reduced-order solutions at time step k
    T_full_k = X(:,k);
    T_reduced_k = X_reconstructed(:,k);
    
    % Compute the L2 norm of the difference
    numerator = norm(T_full_k - T_reduced_k, 2);
    
    % Compute the L2 norm of the full-order solution
    denominator = norm(T_full_k, 2);
    
    % Compute the relative error
    relative_error(k) = numerator / denominator;
end

% Plot the Relative L2 Error Over Time
figure;
plot(time, relative_error, 'b-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Relative L2 Error', 'FontSize', 12);
title('Relative L2 Error Between FOM and ROM Over Time', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Optionally, plot error on a logarithmic scale for better visibility
figure;
semilogy(time, relative_error, 'r-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Relative L2 Error (Log Scale)', 'FontSize', 12);
title('Relative L2 Error Between FOM and ROM Over Time (Log Scale)', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Additionally, display the maximum error and when it occurs
[max_error, max_idx] = max(relative_error);
fprintf('Maximum Relative L2 Error: %.4f at time t = %.2f seconds.\n', max_error, time(max_idx));

% Optionally, plot the error over time as a single plot with

%% Step 7: Optionally, plot the error over time as a single plot with
figure;
plot(T_full_k);
plot(T_reduced_k);