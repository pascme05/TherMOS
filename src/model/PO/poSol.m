%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: ssSol                                                             %
% Date: 13.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculates the temperature response for a POD system
%
%                     T(x,t) ≈ Σ φ_i(x) * a_i(t)   
%
% where x is the spatial position, t is the time, φ are the spatial modes,
% and a(t) are the temporal modes based on a PDE system with matrices Gth 
% and Cth as extracted from the POD training.  
% -------------------------------------------------------------------------
% Inp:  1) mdl:     Fitted model parameters
%       2) data:    Testing input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Predicted temperature response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = poSol(mdl, data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Solving Proper Orthogonal Model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    %----------------------------------------
    % General
    %----------------------------------------
    [Nt, N] = size(data.y);                                                 % number of time steps (Nt) and snapshots (N)
    eps = para.Mdl.gen.eps;                                                 % numerical lower bound
    Ts = data.Ts;                                                           % sampling time (sec)

    %----------------------------------------
    % Data
    %----------------------------------------
    dy = data.Data.dy;                                                      % numerical step-width y direction (m)
    dx = data.Data.dx;                                                      % numerical step-width x direction
    Ly = data.Data.Ly;                                                      % length in y direction (m)
    Lx = data.Data.Lx;                                                      % length in x direction (m)
    k = data.Data.k;                                                        % thermal conductivity (W/mK)
    rho = data.Data.rho;                                                    % material density (kg/m³)
    Cp = data.Data.Cp;                                                      % specific heat capacity (J/KgK)
    alpha = k ./ (rho.*Cp);                                                 % Thermal diffusivity (m²/s)
    
    %----------------------------------------
    % Model
    %----------------------------------------
    Cth = mdl.Cth;
    Gth = mdl.Gth;
    dPhidx = mdl.dPhidx;
    dPhidy = mdl.dPhidy;
    sPhi = mdl.sPhi;
    rPhi = mdl.rPhi;
    K = mdl.K;

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Data
    %----------------------------------------
    xInp = data.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.Data.geo(:,2);                                              % sampled input values y (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    T = data.y;                                                             % temperature snapshots NtxN (°C)
    Q = data.X;                                                             % volumetric heat generation (W/m³)
    
    %----------------------------------------
    % Init
    %----------------------------------------
    g0 = zeros(1, K);
    c = zeros(1, K);
    cy = zeros(1, K);
    cx = zeros(1, K);
    q = zeros(K, Nt);
    F = zeros(K, Nt);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Mean Centering
    %===================================================
    Tavg = mean(T, 1);                                                      % average temperature over time steps (°C)
    T0 = T(1,:) - Tavg;                                                     % initial temperature (°C)
 
    %===================================================
    % Initial Conditions
    %===================================================
    %----------------------------------------
    % Temperatures
    %----------------------------------------
    % Reshape
    sT0 = map2D(T0', xInp, yInp, x, y);
    sT0 = reshape(sT0, [length(y), length(x)]);

    % Gradient
    [dT0dx, dT0dy] = gradient(sT0, dx, dy);

    %----------------------------------------
    % Start Values
    %----------------------------------------
    for i = 1:K
        g0(1, i) = trapz(dx, trapz(dy, T0 .* phi(:, :, i), 1), 2) / ...
                   trapz(dx, trapz(dy, phi(:, :, i) .* phi(:, :, i), 1), 2);
    end

    %===================================================
    % Source Terms
    %===================================================
    %----------------------------------------
    % Boundary Term
    %----------------------------------------
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

    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    for i = 1:Nt
        for ii = 1:K
            q(ii, i) = trapz(dx, trapz(dy, alpha ./ k .* squeeze(s(i, :, :)) .* squeeze(phi(:, :, ii)), 1), 2) / dx / dy;
        end
    end

    %----------------------------------------
    % Source Term
    %----------------------------------------
    for i = 1:Nt
        F(:, i) = (q(:, i) + c');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init
    %===================================================
    % Variables
    tlist = linspace(0, Nt*dt-dt, Nt)';
    
    % Solver
    odeoptions = odeset('Mass', Cth, 'JConstant', 'on', ...
                        'RelTol', 1e-3, 'AbsTol', 1e-5, ...
                        'Jacobian', -Gth);

    %===================================================
    % Solve
    %===================================================
    sol = ode15s(@(t,y) odefnc2(t,y,Gth,F',tlist),tlist,g0,odeoptions);
    theta = deval(sol,tlist)';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_est;
    out.X = Pv;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Solving Proper Orthogonal Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1