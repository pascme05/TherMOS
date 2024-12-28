%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: poSol3                                                            %
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
function out = poSol3(mdl, data, ~)
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
    Ts = data.Ts;                                                           % sampling time (sec)
    deg = 3;                                                                % degree for stencil solution

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
    % Boundary
    %----------------------------------------
    hc = data.Data.hc;                                                      % heat transfer coefficient (W/m²K)
    fl = data.Data.fl;                                                      % heat flux (W/m²)
    Ta = data.Data.Cp;                                                      % ambient temperature (°C)

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
    out = data;
    
    %----------------------------------------
    % BC Matrix
    %----------------------------------------
    dS = zeros(length(y), length(x));
    dS(1, :) = dy;      
    dS(end, :) = dy;     
    dS(:, 1) = dx;       
    dS(:, end) = dx; 

    %----------------------------------------
    % Init
    %----------------------------------------
    c = zeros(1, K);
    qBC = zeros(1, K);
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
    % 2D Reshaping
    %===================================================
    %----------------------------------------
    % Material Properties
    %----------------------------------------
    sAlpha = map2D(alpha', xInp, yInp, x, y, 1);
    sAlpha = reshape(sAlpha, [length(y), length(x)]);
    sK = map2D(k', xInp, yInp, x, y, 1);
    sK= reshape(sK, [length(y), length(x)]);
    
    %----------------------------------------
    % Boundary
    %----------------------------------------
    hc2D = squeeze(map2D(hc', xInp, yInp, x, y, 1));
    fl2D = squeeze(map2D(fl', xInp, yInp, x, y, 1));
    Ta2D = squeeze(map2D(Ta', xInp, yInp, x, y, 1));
    
    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    sQ = map2D(Q, xInp, yInp, x, y, 2);

    %----------------------------------------
    % Temperatures
    %----------------------------------------
    if sum(diff(alpha)) == 0
        sT0 = map2D(T0, xInp, yInp, x, y, 2);
    else
        sT0 = map2D(T0, xInp, yInp, x, y, 1);
    end
    sT0 = reshape(sT0, [length(y), length(x)]);
    
    %===================================================
    % Gradient
    %===================================================
    [dT0dx, dT0dy] = gradient(sT0, dx, dy);

    %===================================================
    % Init Values
    %===================================================
    g0 = T0 * rPhi;

    %===================================================
    % Source Terms
    %===================================================
    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    % % Solution using Trapz
    % for i = 1:Nt
    %     for ii = 1:K
    %         q2(ii, i) = trapz(dx, trapz(dy, sAlpha ./ sK .* squeeze(sQ(i, :, :)) .* squeeze(sPhi(:, :, ii)), 1), 2) / dx / dy;
    %     end
    % end
    
    % Solution using Stencil Int
    for i = 1:Nt
        for ii = 1:K
            temp = sAlpha ./ sK .* squeeze(sQ(i, :, :)) .* squeeze(sPhi(:, :, ii));
            q(ii, i) = intStencil(length(x), length(y), dx, dy, temp, deg) * dx * dy;
        end
    end


    %----------------------------------------
    % Interior contributions
    %----------------------------------------
    for i = 1:K
        % Interior
        % temp = sAlpha .* (dPhidx(:, :, i) .* dT0dx + dPhidy(:, :, i) .* dT0dy);
        % c2(i) = intStencil(length(x), length(y), dx, dy, temp, deg) * dx * dy;
        c(i) = trapz(dx, trapz(dy, sAlpha .* (dPhidx(:, :, i) .* dT0dx + dPhidy(:, :, i) .* dT0dy), 1), 2) / dx / dy;

        % Boundary Term Convection
        BC_q1 = sAlpha ./ sK .* hc2D .* sPhi(:,:,i) .* Ta2D;
        BC_q2 = -sAlpha ./ sK .* fl2D .* sPhi(:,:,i) / dx / dy; 
        qBC(i) = sum(sum(BC_q1 .* dS + BC_q2 .* dS));
    end

    %----------------------------------------
    % Source Term
    %----------------------------------------
    for i = 1:Nt
        F(:, i) = (q(:, i) + c' - qBC');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Normalisation Values
    %===================================================
    % u_scale = max(abs(g0));
    % q_scale = max(abs(F(:)));
    q_scale = 1;
    u_scale = 1;
    tau = max(Cth) / max(Gth); 
    
    %===================================================
    % Normalisation
    %===================================================
    Cth = Cth / tau;
    Gth = Gth * tau;
    F = F / q_scale * tau;
    g0 = g0 / u_scale;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init
    %===================================================
    % Variables
    tlist = linspace(0, Nt*Ts-Ts, Nt)' / tau / tau;
    
    % Choose solver dynamically
    % is_stiff = max(eig(Gth/Cth)) > tau; % Example stiffness criterion
    is_stiff = 1;
    
    %===================================================
    % Solve
    %===================================================
    % Solver options
    if is_stiff == 1
        odeoptions = odeset('Mass', Cth, 'JConstant', 'on', ...
                        'RelTol', 1e-5, 'AbsTol', 1e-7, ...
                        'Jacobian', -Gth);
        sol = ode23s(@(t,y) odefnc2(t,y,Gth,F',tlist),tlist,g0,odeoptions);
    else
        odeoptions = odeset('RelTol', 1e-3, 'AbsTol', 1e-5);
        sol = ode45(@(t, y) odefnc2(t, y, Gth/Cth, F', tlist), tlist, g0, odeoptions);
    end
    % u = rk4(Cth, Gth, F, g0, Nt, K, Ts / tau / tau);
    % [t, y] = rk6(tlist, g0, Ts/ tau / tau, Gth/Cth, F');
    
    % Evaluate solution
    theta_hat = deval(sol, tlist)';
    theta_hat = theta_hat * u_scale;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_est = theta_hat * rPhi' + Tavg.*ones(Nt,N);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_est;
    out.X = Q;

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