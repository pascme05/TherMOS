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
    deg = -1;                                                               % degree for stencil solution (-1 internal trapz approach)
    is_stiff = 1;                                                           % stiff vs. non-stiff solvers

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
    try
        hc = data.Data.hc;                                                  % heat transfer coefficient (W/m²K)
        fl = data.Data.fl;                                                  % heat flux (W/m²)
        Ta = data.Data.Ta;                                                  % ambient temperature (°C)
    catch
        hc = zeros(size(alpha));
        fl = zeros(size(alpha));
        Ta = zeros(size(alpha));
    end

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
    cy = zeros(1, K);
    cx = zeros(1, K);
    qBC = zeros(1, K);
    q = zeros(K, Nt);
    F = zeros(K, Nt);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Mean Centering
    %===================================================
    Tavg = mdl.Tavg;                                                      % average temperature over time steps (°C)
    % Tavg = 0;
    T0 = T(1,:);                                                     % initial temperature (°C)
    Ta = Ta - Tavg';

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
    
    % %----------------------------------------
    % % Sample BC
    % %----------------------------------------
    % dS(1, fl2D(1,:)==0 & hc2D(1,:)==0) = 0;
    % dS(end, fl2D(end,:)==0 & hc2D(end,:)==0) = 0;
    % dS(fl2D(:,1)==0 & hc2D(:,1)==0, 1) = 0;
    % dS(fl2D(:,end)==0 & hc2D(:,end)==0, end) = 0;

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
    %----------------------------------------
    % Extend
    %----------------------------------------
    sT02 = zeros(length(y)+2, length(x)+2);

    % Place the original matrix in the center
    sT02(2:end-1, 2:end-1) = sT0;
    
    % Add the vector to the edges
    sT02(1, 2:end-1) = Ta2D(1,:);   
    sT02(2:end-1, 1) = Ta2D(:,1);  
    sT02(end, 2:end-1) = Ta2D(end,:);  
    sT02(2:end-1, end) = Ta2D(:,end);   
    
    % Compute corner values as averages
    sT02(1,1) = Ta2D(1,1);  
    sT02(1,end) = Ta2D(1,end);
    sT02(end,1) = Ta2D(end,1);
    sT02(end,end) = Ta2D(end,end);

    %----------------------------------------
    % Calc
    %----------------------------------------
    [dT0dx2, dT0dy2] = gradient(sT02, dx, dy);
    [dT0dx, dT0dy] = gradient(sT0, dx, dy);

    % %----------------------------------------
    % % Limit
    % %----------------------------------------
    % dT0dx = dT0dx2(2:end-1,2:end-1);
    % dT0dy = dT0dy2(2:end-1,2:end-1);

    %===================================================
    % Init Values
    %===================================================
    % g0 = T0 * rPhi;
    g0 = zeros(K,1);

    %===================================================
    % Source Terms
    %===================================================
    % ratio = length(alpha) / ((length(x)-2) * (length(y)-2) + (length(y)-1) + (length(x)-1) + 1);
    % q = alpha(1) / k(1) * rPhi' * Q' * dx *dy / ratio;

    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    for i = 1:Nt
        for ii = 1:K
            % Solution using Trapz
            if deg == -1
                q(ii, i) = trapz(dx, trapz(dy, sAlpha ./ sK .* squeeze(sQ(i, :, :)) .* squeeze(sPhi(:, :, ii)), 1), 2);

            % Solution using Stencil Int
            else
                temp = sAlpha ./ sK .* squeeze(sQ(i, :, :)) .* squeeze(sPhi(:, :, ii));
                q(ii, i) = intStencil(length(x), length(y), dx, dy, temp, deg) * dx * dy;
            end
        end
    end

    %----------------------------------------
    % Interior contributions
    %----------------------------------------
    for i = 1:K
        % Interior
        if deg == -1
            c(i) = trapz(dx, trapz(dy, sAlpha .* (dPhidx(:, :, i) .* dT0dx + dPhidy(:, :, i) .* dT0dy), 1), 2);
        else
            temp = sAlpha .* (dPhidx(:, :, i) .* dT0dx + dPhidy(:, :, i) .* dT0dy);
            c(i) = intStencil(length(x), length(y), dx, dy, temp, deg) * dx * dy;
        end

        % % Convection
        % qBC(i) = qBC(i) +  sum(hc2D(:,1) .* sPhi(:,1,i) .* Ta2D(:,1) * dy);

        % Boundary contributions in Y-direction
        tempY = trapz(dy, sAlpha .* sPhi(:, :, i) .* dT0dx, 1);
        cy(i) = sum(tempY([1, end]));
        % cy(i) = trapz(dy, tempY([1, end]));

        % Boundary contributions in X-direction
        tempX = trapz(dx, sAlpha .* sPhi(:, :, i) .* dT0dy, 2);
        cx(i) = sum(tempX([1, end]));
        % cx(i) = trapz(dx, tempX([1, end]));

        % % Boundary
        % BC_q1 = sAlpha ./ sK .* hc2D .* sPhi(:,:,i) .* Ta2D;
        % BC_q2 = -sAlpha ./ sK .* fl2D .* sPhi(:,:,i); 
        % qBC(i) = trapz(dx, trapz(dy, (BC_q1 .* dS + BC_q2 .* dS), 1), 2);
        % c(i) = c(i) - qBC(i);
    end
    c = c - (cx + cy) + qBC;
    % c = 0;
    % c = c/dx/dy;

    %----------------------------------------
    % Source Term
    %----------------------------------------
    for i = 1:Nt
        F(:, i) = (q(:, i) + 0*c'); % 0.8912, 1.3
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Normalisation Values
    %===================================================
    % u_scale = max(abs(g0));
    % q_scale = max(abs(F(:)));
    % tau = max(Cth) / max(Gth); 
    q_scale = 1;
    u_scale = 1;
    tau = 1;
    
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
    
    %===================================================
    % Solve
    %===================================================
    % Solver options
    if is_stiff == 1
        odeoptions = odeset('Mass', Cth, 'JConstant', 'on', ...
                            'RelTol', 1e-5, 'AbsTol', 1e-7, ...
                            'Jacobian', -Gth);
        sol = ode23s(@(t,y) odefnc2(t,y,Gth,F',tlist),tlist,g0,odeoptions);
        theta_hat = deval(sol, tlist)';
    else
        odeoptions = odeset('RelTol', 1e-5, 'AbsTol', 1e-7);
        sol = ode45(@(t, y) odefnc2(t, y, Gth/Cth, F', tlist), tlist, g0, odeoptions);
        theta_hat = deval(sol, tlist)';
        theta_hat = theta_hat ./ diag(Cth)';
        % [~, theta_hat] = rk6(tlist, g0, Ts/ tau / tau, Gth/Cth, F');
        % theta_hat = theta_hat' ./ diag(Cth)';
    end
    % u = rk4(Cth, Gth, F, g0, Nt, K, Ts / tau / tau);
    % [t, y] = rk6(tlist, g0, Ts/ tau / tau, Gth/Cth, F');
    
    % Evaluate solution
    theta_hat = theta_hat * q_scale;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_est = theta_hat * rPhi' + T0.*ones(Nt,N);
    err = mean(abs(T_est - T),"all")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.y = T_est;
    out.X = Q;
    out.theta_hat = theta_hat;
    out.theta = (rPhi' * T')';

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