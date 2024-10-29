%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Titel: Model Order Reduction (MOR)
% File: testPOD
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
% This functions evaluates a proper orthogonal (POD) model with a fixed
% model order 
% Input:
%       - data:     input test data
%       - mdl:      pod model
%       - setup:    all setup variables for the simulation
% Output:
%       - out:      output structure including predictions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FNC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = testPOD3(data, mdl, setup)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG In
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Start testing POD Model');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Parameter
% ----------------------------------------------------
[Nt, Ny, Nx] = size(data.y);
dx = data.dx;
dy = data.dy;
k = data.Mat2D.k;
rho = data.Mat2D.rho;
Cp = data.Mat2D.Cp;
alpha = k ./ (rho.*Cp);
K = size(mdl.phi, 3);
dt = data.Ts;

% ----------------------------------------------------
% Variables
% ----------------------------------------------------
T_hat = zeros(Nt, Ny, Nx);
s = data.X;
c = zeros(1, K);
cy = zeros(1, K);
cx = zeros(1, K);
q = zeros(K, Nt);
Cth = mdl.Cth;
Gth = mdl.Gth;
g0 = zeros(1, K);
phi = mdl.phi;
dPhidx = mdl.dPhidx;
dPhidy = mdl.dPhidy;
theta_true = mdl.theta;
F = zeros(K, Nt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Mean Centering
% ----------------------------------------------------
Tavg = squeeze(mean(data.y, 1));
T0 = squeeze(data.y(1, :, :)) - Tavg;

% ----------------------------------------------------
% Source Terms
% ----------------------------------------------------
% Temperatures
[dT0dx, dT0dy] = gradient(T0, dx, dy);

% Initial Coniditions
for i = 1:K
    g0(1, i) = trapz(dx, trapz(dy, T0 .* phi(:, :, i), 1), 2) / ...
               trapz(dx, trapz(dy, phi(:, :, i) .* phi(:, :, i), 1), 2);
end
g0 = theta_true(1,:);

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

% Heat Generation
for i = 1:Nt
    for ii = 1:K
        q(ii, i) = trapz(dx, trapz(dy, alpha ./ k .* squeeze(s(i, :, :)) .* squeeze(phi(:, :, ii)), 1), 2) / dx / dy;
    end
end

% Source Term
for i = 1:Nt
    F(:, i) = (q(:, i) + c');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Init
% ----------------------------------------------------
% Variables
tlist = linspace(0, Nt*dt-dt, Nt)';

% Solver
odeoptions = odeset;
odeoptions = odeset(odeoptions, "Mass", Cth);
odeoptions = odeset(odeoptions, "JConstant","on");
odeoptions = odeset(odeoptions, 'RelTol',1e-12,'AbsTol',1e-13);

% Function
df = -Gth;
odeoptions = odeset(odeoptions, "Jacobian",df);

% ----------------------------------------------------
% Solve
% ----------------------------------------------------
sol = ode15s(@(t,y) odefnc2(t,y,Gth,F',tlist),tlist,g0,odeoptions);
theta = deval(sol,tlist)';
% theta = theta_true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Reconstruct
% ----------------------------------------------------
% Init
temp = zeros(Nt, Ny, Nx);

% Reconstruct
for i = 1:K
    for ii = 1:Nt
        temp(ii, :, :) = theta(ii, i) .* phi(:, :, i);
    end
    T_hat = T_hat + temp;
end

% Shift
for i = 1:Nt
    T_hat(i, :, :) = squeeze(T_hat(i, :, :)) + Tavg;
end

% ----------------------------------------------------
% Output
% ----------------------------------------------------
out = data;
out.y = T_hat;
out.theta = theta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG In
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Done testing POD Model');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%