%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Titel: Model Order Reduction (MOR)
% File: trainPOD
% Author: P. Schirmer
% Department: ES-641
% Version: v.1.2
% Date: 17.06.2024
% Copyright: BMW AG, Munich
% Comments: comparing different MORs for thermal modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This functions fits a proper orthogonal decomposed (POD) model with a 
% fixed model order.
% Input:
%       - data:     input training data
%       - setup:    all setup variables for the simulation
% Output:
%       - mdl:      trained pod model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FNC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = trainPOD3(data, setup)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG In
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Start training POD Model');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Parameter
% ----------------------------------------------------
[Nt, Ny, Nx] = size(data.y{1});
dx = data.dx;
dy = data.dy;
Kmax = setup.mdl.pod.Kmax;
Emax = setup.mdl.pod.Emax;
E = 0;
err = setup.mdl.hyp.err;
k = data.Mat2D.k;
rho = data.Mat2D.rho;
Cp = data.Mat2D.Cp;
alpha = k ./ (rho.*Cp);

% ----------------------------------------------------
% Variables
% ----------------------------------------------------
t = data.t{1};
U = data.y{1};
Q = data.X{1};
T = zeros(size(U));
T0 = squeeze(U(1, :, :));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Mean Centering
% ----------------------------------------------------
Tavg = squeeze(mean(U, 1));
for i = 1:Nt
    T(i, :, :) = U(i, :, :) - mean(U, 1);
end

% ----------------------------------------------------
% Snapshots 
% ----------------------------------------------------
T_snap = reshape(T,Nt,Nx*Ny);
T_snap_mean = mean(T_snap, 1);
T_snap = T_snap - T_snap_mean;

% ----------------------------------------------------
% Eigenvalue Decomposition 
% ----------------------------------------------------
% Snap Based
[~,S,Phi] = svd(T_snap/sqrt(Nt-0));
lam = diag(S);

% Number of Eigenvalues
fprintf('Overall numbers of eigenvalues: %i \n', length(lam));

% ----------------------------------------------------
% Cummulative Correlation Energy 
% ----------------------------------------------------
% Init
n = 1;

% Energy
if setup.mdl.pod.modes == 1
    while E < Emax && n < Kmax
        E = sum(abs(lam(1:n))) / sum(abs(lam));
        n = n + 1;
    end
    K = n;

% Opti
elseif setup.mdl.pod.modes == 2
    E = sum(abs(lam(1:setup.mdl.pod.K))) / sum(abs(lam));
    K = setup.mdl.pod.K;

% Default (fixed)
else
    E = sum(abs(lam(1:setup.mdl.pod.K))) / sum(abs(lam));
    K = setup.mdl.pod.K;
end

% Msg
fprintf('Number of used eigenvalues (model order) K: %i \n', K);
fprintf('Cumulative correlation energy Em: %f \n', E)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Init
% ----------------------------------------------------
Id = zeros(K, K);
dPhidx = zeros(Ny, Nx, K);
dPhidy = zeros(Ny, Nx, K);
theta = zeros(Nt, K);

% ----------------------------------------------------
% Spatial Modes
% ----------------------------------------------------
% POD modes
Phi = Phi(:,1:K);
phi = reshape(Phi, [Ny, Nx, K]);

% Orthogonal Matrix
for i = 1:K
    for ii = 1:K
        Id(i, ii) = trapz(dy, trapz(dx, phi(:, :, i) .* phi(:, :, ii), 1), 2) / dx / dy;
    end
end

if abs(norm(eye(K) - Id)) > err
    disp('ERROR: POD modes are not orthonormal');
end

% ----------------------------------------------------
% Temporal Modes
% ----------------------------------------------------
for i = 1:K
    for ii = 1:Nt
        theta(ii,i) = trapz(dx, trapz(dy, squeeze(phi(:, :, i)) .* squeeze(T(ii, :, :)), 1), 2) / dx / dy;
    end
end

% ----------------------------------------------------
% Spatial Gradients
% ----------------------------------------------------
for i = 1:K
    [dPhidx(:, :, i), dPhidy(:, :, i)] = gradient(phi(:, :, i), dx, dy);
end
[dT0dx, dT0dy] = gradient(squeeze(T(1, :, :)), dx, dy);

% ----------------------------------------------------
% System Matrices
% ----------------------------------------------------
% Init
Cth = zeros(K, K);
Gth = zeros(K, K);
Gx = zeros(K, K);
Gy = zeros(K, K);
c = zeros(1, K);
cy = zeros(1, K);
cx = zeros(1, K);

% System Matrices
for i = 1:K
    for ii = 1:K
        % Mass Matrix
        Cth(i, ii) = trapz(dx, trapz(dy, phi(:, :, ii) .* phi(:, :, i), 1), 2) / dx / dy;

        % Stiffness Matrix (interior contributions)
        Gth(i, ii) = trapz(dx, trapz(dy, alpha .* (dPhidx(:, :, ii) .* dPhidx(:, :, i) + dPhidy(:, :, ii) .* dPhidy(:, :, i)), 1), 2) / dx / dy;

        % Boundary Terms for Neumann conditions (Y-direction)
        tempY = alpha .* phi(:, :, ii) .* dPhidx(:, :, i);
        Gy(i, ii) = trapz(dy, tempY(:, end) - tempY(:, 1), 1) / dy;

        % Boundary Terms for Neumann conditions (X-direction)
        tempX = alpha .* phi(:, :, ii) .* dPhidy(:, :, i);
        Gx(i, ii) = trapz(dx, tempX(end, :) -  tempX(1, :), 2) / dx;
    end
end
Gth = Gth - (Gy + Gx);

% Boundary Conditions
for i = 1:K
    % Interior contributions
    c(i) = trapz(dx, trapz(dy, alpha .* (dPhidx(:, :, i) .* dT0dx + dPhidy(:, :, i) .* dT0dy), 1), 2) / dx / dy;
    
    % Boundary contributions in Y-direction
    tempY = alpha .* phi(:, :, i) .* dT0dx;
    cy(i) = trapz(dy, tempY(:, end) - tempY(:, 1), 1) / dy;
    
    % Boundary contributions in X-direction
    tempX = alpha .* phi(:, :, i) .* dT0dy;
    cx(i) = trapz(dx, tempX(end, :) - tempX(1, :), 2) / dx;
end
c = c - (cx + cy);

% Stiffness Matrices
GC = Gth / Cth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % ----------------------------------------------------
% % Optimize Cth and Gth
% % ----------------------------------------------------
% Heat Generation
Q = bsxfun(@times, Q, reshape(alpha ./ k, [1, Ny, Nx]));
Q = reshape(Q, [Ny*Nx, Nt]);
F = Phi' * Q + c';
d_theta = (theta(2:end, :) - theta(1:end-1, :)) / 1;
d_theta = [d_theta(1,:); d_theta];
test = Cth * d_theta' + Gth * theta' - F;

% Solve
% [Gth_opt, Cth_opt] = optiPINN(t, theta, F', Gth, Cth);
nx = 10;
theta_padded = [zeros(nx, size(theta, 2)); theta];
F_padded = [zeros(nx, size(F', 2)); F'];
ssData = iddata(theta_padded, F_padded, t(2)-t(1));
% sys = ssest(ssData);
sys = n4sid(ssData);
[y, t, x] = lsim(sys, F, t);

% ----------------------------------------------------
% Make Mass Orthogonal
% ----------------------------------------------------
% Decayrates
[beta,decay] = eig(Gth,Cth);
decay = diag(decay); 

% % Orthogonality
% R = chol(beta.'*Cth*beta);
% beta = beta/R;
% phi = Phi*beta;
% phi = reshape(phi, [Ny, Nx, K]);
% Cth = (Cth+Cth.')/2;
% Gth = (Gth+Gth.')/2;

% ----------------------------------------------------
% Model
% ----------------------------------------------------
mdl.phi = phi;
mdl.theta = theta;
mdl.dPhidx = dPhidx;
mdl.dPhidy = dPhidy;
mdl.Gth = Gth;
mdl.Cth = Cth;
mdl.lam = lam;
mdl.decay = decay;
mdl.beta = beta;
mdl.E = E;
mdl.GC = GC;
mdl.K = K;
mdl.Tavg = Tavg;
mdl.T0 = T0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG In
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Done training POD Model');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
