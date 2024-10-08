% Galerkin POD Implementation for 2D Heat Equation

%% Step 1: Discretize the 2D Heat Equation Using Finite Differences

% Spatial discretization
nx = 50;        % Number of grid points in x
ny = 50;        % Number of grid points in y
Lx = 1.0;       % Length of the domain in x
Ly = 1.0;       % Length of the domain in y
dx = Lx / (nx + 1);
dy = Ly / (ny + 1);
x = linspace(0, Lx, nx+2);  % Including boundary points
y = linspace(0, Ly, ny+2);
[X_grid, Y_grid] = meshgrid(x, y);

% Temporal discretization
dt = 0.001;     % Time step
T = 1.0;        % Final time
nt = ceil(T/dt);% Number of time steps
time = linspace(0, T, nt+1);

% Initial condition: Gaussian
u0 = exp(-100 * ((X_grid - Lx/2).^2 + (Y_grid - Ly/2).^2));
u = u0;

% Reshape u into a column vector for computations
u = u(2:end-1, 2:end-1); % Remove boundary points
u = u(:);                % Vectorize

% Assemble the finite difference matrix
alpha = 1.0;  % Thermal diffusivity

% Number of interior points
N = nx * ny;

% Create the 1D Laplacian in x
e = ones(nx,1);
T_x = spdiags([e -2*e e], -1:1, nx, nx);
I_x = speye(nx);
L_x = (kron(I_x, T_x)) / dx^2;

% Create the 1D Laplacian in y
T_y = spdiags([e -2*e e], -1:1, ny, ny);
I_y = speye(ny);
L_y = (kron(T_y, I_y)) / dy^2;

% Total Laplacian
L = L_x + L_y;

% Identity matrix
I = speye(N);

% System matrix for implicit Euler
A = I - dt * alpha * L;

%% Step 2: Generate Snapshots by Solving the Full-Order Model

% Preallocate snapshot matrix
snapshots = zeros(N, nt+1);
snapshots(:,1) = u;

% Time-stepping loop
for k = 1:nt
    % Solve A * u_new = u_old
    u = A \ u;
    
    % Store the snapshot
    snapshots(:,k+1) = u;
end

% Transpose snapshots to have snapshots as columns
X = snapshots;

%% Step 3: Perform Proper Orthogonal Decomposition (POD)

% Perform SVD
[U, S, V] = svd(X, 'econ');

% Determine the number of modes to retain (e.g., 99% energy)
singular_values = diag(S);
energy = cumsum(singular_values.^2) / sum(singular_values.^2);
r = find(energy >= 0.9999, 1, 'first');  % Number of modes

fprintf('Number of POD modes retained: %d\n', r);

% Truncate to r modes
U_r = U(:,1:r);
S_r = S(1:r,1:r);
V_r = V(:,1:r);

%% Step 4: Construct the Galerkin Reduced-Order Model

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
for k = 1:nt
    % Update reduced coefficients
    a_r = A_r_inv * a_r;
    
    % Store the reduced coefficients
    a_r_snapshots(:,k+1) = a_r;
end

%% Step 5: Reconstruct the Full Solution from Reduced-Order Model

% Reconstruct snapshots from reduced coefficients
X_reconstructed = U_r * a_r_snapshots;

%% Step 6: Compare Full-Order and Reduced-Order Models

% Reshape solutions for visualization
u_full = reshape(X(:,end), nx, ny);
u_reduced = reshape(X_reconstructed(:,end), nx, ny);

% Create a grid for interior points
x_interior = x(2:end-1);
y_interior = y(2:end-1);
[X_in, Y_in] = meshgrid(x_interior, y_interior);

% Plot full-order solution
figure;
surf(X_in, Y_in, u_full');
title('Full-Order Solution at Final Time');
xlabel('x');
ylabel('y');
zlabel('u');
shading interp;
colorbar;

% Plot reduced-order solution
figure;
surf(X_in, Y_in, u_reduced');
title(['Reduced-Order Solution at Final Time (r = ' num2str(r) ')']);
xlabel('x');
ylabel('y');
zlabel('u');
shading interp;
colorbar;

% Compute and display the error
error = norm(u_full - u_reduced, 2) / norm(u_full, 2);
disp(['Relative L2 Error at Final Time: ', num2str(error)]);
