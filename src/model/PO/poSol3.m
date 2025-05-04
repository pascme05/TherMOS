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
    % Model
    %----------------------------------------
    Cth = mdl.Cth;
    Gth = mdl.Gth;
    sPhi = mdl.sPhi;
    rPhi = mdl.rPhi;
    K = mdl.K;
    scale = mdl.alpha;
    Jgrid = mdl.Jgrid;
    W = mdl.W;

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
    % Init
    %----------------------------------------
    q = zeros(K, Nt);
    F = zeros(K, Nt);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Shift Coordinate System
    %===================================================
    xInp = xInp - min(xInp);
    yInp = yInp - min(yInp);
    
    %===================================================
    % Mean Centering
    %===================================================
    T0 = T(1,:);                                                            % initial temperature (°C)
    
    %===================================================
    % Numerics
    %===================================================
    [x_n, ~] = gauss_legendre(length(x));
    [y_n, ~] = gauss_legendre(length(y));

    % map to [x_min,x_max] and [y_min,y_max]
    xq_1d = 0.5*(max(x)-min(x))*x_n + 0.5*(max(x)-min(x));
    yq_1d = 0.5*(max(y)-min(y))*y_n + 0.5*(max(y)-min(y));
    
    % form 2D arrays of points & weights
    [Xq, Yq] = meshgrid(xq_1d, yq_1d);   % Ny×Nx
    [Xinit, Yinit] = meshgrid(x, y); 

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
    % Integration 2D
    %----------------------------------------
    alpha2D_i = interp2(Xinit, Yinit, sAlpha, Xq, Yq, 'linear');
    k2D_i = interp2(Xinit, Yinit, sK, Xq, Yq, 'linear');

    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    sQ = map2D(Q, xInp, yInp, x, y, 2);

    %===================================================
    % Init Values
    %===================================================
    g0 = zeros(K,1);

    %===================================================
    % Source Terms
    %===================================================
    %----------------------------------------
    % Heat Generation
    %----------------------------------------
    for i = 1:Nt
        for ii = 1:K
            sPhi_ii = interp2(Xinit, Yinit, squeeze(sPhi(:,:,ii)), Xq, Yq, 'linear');
            sQ_i = interp2(Xinit, Yinit, squeeze(sQ(i, :, :)), Xq, Yq, 'linear');
            % q(ii, i) = sum(sum(Jgrid .* W .* sAlpha ./ sK .* sQ_i .* sPhi_ii));
            q(ii, i) = sum(sum(Jgrid .* W .* alpha2D_i ./ k2D_i .* sQ_i .* sPhi_ii));
        end
    end

    %----------------------------------------
    % Source Term
    %----------------------------------------
    for i = 1:Nt
        F(:, i) = scale.*q(:, i); 
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Normalisation Values
    %===================================================
    q_scale = max(abs(F(:)));
    tau = max(max(Cth/Gth)); 
    % q_scale = 1;
    % tau = 1;
    
    %===================================================
    % Normalisation
    %===================================================
    Cth = Cth / tau;
    Gth = Gth * tau;
    F = F / q_scale * tau;

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