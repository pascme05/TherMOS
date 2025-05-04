%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: poFit3                                                            %
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
% This function solves for the system matrices Gth and Cth as well as the 
% spatial modes of phi of the POD.
% -------------------------------------------------------------------------
% Inp:  1) data:    Training input data struct
%       2) val:     Validation input data struct
%       3) para:    All simulation parameters of the current simulation
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = poFit3(data, ~, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Fitting Proper Orthogonal Model")

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
    Kmax = para.Mdl.gen.Kmax;                                               % maximum number of modes
    Emax = para.Mdl.gen.Emax;                                               % maximum energy captured (%)
    E = 0;                                                                  % captured energy by POD
    eps = para.Mdl.gen.eps;                                                 % numerical lower bound
    deg = -1;                                                               % degree for stencil solution (-1 internal trapz approach)
    
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
    catch
        hc = zeros(size(alpha));
        fl = zeros(size(alpha));
    end

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % General
    %----------------------------------------
    xInp = data.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.Data.geo(:,2);                                              % sampled input values y (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    T = data.y;                                                             % temperature snapshots NtxN (°C)
    Q = data.X;                                                             % volumetric heat generation (W/m³)

    %----------------------------------------
    % BC Matrix
    %----------------------------------------
    dS = zeros(length(y), length(x));
    dS(1, :) = 1;      
    dS(end, :) = 1;     
    dS(:, 1) = 1;       
    dS(:, end) = 1; 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Shift Coordinate System
    %===================================================
    xInp = xInp - min(xInp);
    yInp = yInp - min(yInp);

    %===================================================
    % Boundary Conditions
    %===================================================
    %----------------------------------------
    % Converting 2D
    %----------------------------------------
    alpha2D = squeeze(map2D(alpha', xInp, yInp, x, y, 1));
    sQ = map2D(Q, xInp, yInp, x, y, 2);
    k2D = squeeze(map2D(k', xInp, yInp, x, y, 1));
    % hc2D = squeeze(map2D(hc', xInp, yInp, x, y, 1));
    % fl2D = squeeze(map2D(fl', xInp, yInp, x, y, 1));

    %----------------------------------------
    % Sample BC
    %----------------------------------------
    % dS(1, fl2D(1,:)==0 & hc2D(1,:)==0) = 0;
    % dS(end, fl2D(end,:)==0 & hc2D(end,:)==0) = 0;
    % dS(fl2D(:,1)==0 & hc2D(:,1)==0, 1) = 0;
    % dS(fl2D(:,end)==0 & hc2D(:,end)==0, end) = 0;

    %===================================================
    % Mean Centering
    %===================================================
    Tavg = mean(T, 1);                                                      % average temperature over time steps (°C)
    T = T - Tavg.*ones(Nt,N);                                               % mean centered observation matrix (°C)

    %===================================================
    % Eigenvalues
    %===================================================
    try
        [~,S,Phi] = svd(T/sqrt(Nt-1));
    catch
        [~,S,Phi] = svd(T/sqrt(Nt-1),"econ");
    end
    lam = diag(S);
    
    %===================================================
    % Cumulative Correlation Energy
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    n = 1;

    %----------------------------------------
    % Energy
    %----------------------------------------
    if para.Mdl.pod.sel == 2
        while E < Emax && n < Kmax
            E = sum(abs(lam(1:n))) / sum(abs(lam));
            n = n + 1;
        end
        K = n;

    %----------------------------------------
    % Opti
    %----------------------------------------
    elseif para.Mdl.pod.sel == 3
        E = sum(abs(lam(1:para.Mdl.pod.K))) / sum(abs(lam));
        K = setup.mdl.pod.K;

    %----------------------------------------
    % Fixed
    %----------------------------------------
    else
        E = sum(abs(lam(1:para.Mdl.pod.K))) / sum(abs(lam));
        K = para.Mdl.pod.K;
    end

    %----------------------------------------
    % Msg
    %----------------------------------------
    fprintf('INFO: Number of used eigenvalues (model order) K: %i \n', K);
    fprintf('INFO: Cumulative correlation energy Em: %f \n', E*100)
    
    %===================================================
    % Numerics
    %===================================================
    [x_n, w_x] = gauss_legendre(length(x));
    [y_n, w_y] = gauss_legendre(length(y));

    % map to [x_min,x_max] and [y_min,y_max]
    xq_1d = 0.5*(max(x)-min(x))*x_n + 0.5*(max(x)-min(x));
    wq_x  = 0.5*(max(x)-min(x))*w_x;
    yq_1d = 0.5*(max(y)-min(y))*y_n + 0.5*(max(y)-min(y));
    wq_y  = 0.5*(max(y)-min(y))*w_y;
    
    % form 2D arrays of points & weights
    [Xq, Yq] = meshgrid(xq_1d, yq_1d);   % Ny×Nx
    [Xinit, Yinit] = meshgrid(x, y); 
    W = wq_y(:) * wq_x(:)';

    Jgrid = jacobian2D(xInp, yInp, xq_1d, yq_1d, dx, dy);
    % Jgrid = ones(size(Jgrid));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init
    %===================================================
    dPhidx = zeros(length(y), length(x), K);
    dPhidy = zeros(length(y), length(x), K);
    Cth = zeros(K, K);
    Gth = zeros(K, K);
    Cth2 = zeros(K, K);
    Gth2 = zeros(K, K);
    Gx = zeros(K, K);
    Gy = zeros(K, K);
    Gx2 = zeros(K, K);
    Gy2 = zeros(K, K);

    %===================================================
    % Extract Modes
    %===================================================
    %----------------------------------------
    % Spatial Modes
    %----------------------------------------
    % Reduction
    rPhi = Phi(:,1:K);
    
    % Compute Orthogonal Matrix
    Id = rPhi'*rPhi;
    
    % Check Orthogonal Matrix
    if abs(norm(eye(K) - Id)) > eps
        disp('ERROR: POD modes are not orthonormal');
    end

    % Reshape
    sPhi = map2D(rPhi', xInp, yInp, x, y, 1);
    if K == 1
        sPhi = squeeze(sPhi);
    else
        sPhi = permute(sPhi, [2, 3, 1]);
    end

    %----------------------------------------
    % Temporal Modes
    %----------------------------------------
    theta = (rPhi' * T')';

    %===================================================
    % Gradients
    %===================================================
    if K == 1
        [dPhidx, dPhidy] = gradient(sPhi, dx, dy);
    else
        for i = 1:K
            [dPhidx(:, :, i), dPhidy(:, :, i)] = gradient(sPhi(:, :, i), dx, dy);
        end
    end

    %===================================================
    % System Matrices
    %===================================================
    %----------------------------------------
    % Thermal Capacitance
    %----------------------------------------
    if K == 1
        Cth = trapz(dx, trapz(dy, sPhi.*sPhi, 1), 2);
    else
        for i = 1:K
            for ii = 1:K
                % Trapasoidal
                if deg == -1
                    Cth(i, ii) = trapz(dx, trapz(dy, sPhi(:, :, ii) .* sPhi(:, :, i), 1), 2);
                
                % Stencil
                else
                    Cth(i, ii) = intStencil(length(x), length(y), dx, dy, sPhi(:, :, ii) .* sPhi(:, :, i), deg) * dx * dy;
                end
    
                % Symmetry
                Cth(i, ii) = Cth(ii, i);
                
                sPhi_i_q = interp2(Xinit, Yinit, squeeze(sPhi(:,:,i)), Xq, Yq, 'linear');
                sPhi_ii_q = interp2(Xinit, Yinit, squeeze(sPhi(:,:,ii)), Xq, Yq, 'linear');

                Cth2(i, ii) = sum(sum(Jgrid .* W .* sPhi_ii_q .* sPhi_i_q));
                % Cth2(i, ii) = Cth2(ii, i);
            end
        end
    end
    Cth = Cth2;


    %----------------------------------------
    % Thermal Conductance
    %----------------------------------------
    if K == 1
        % Boundary Free
        Gth = trapz(dx, trapz(dy, alpha2D .* (dPhidx.*dPhidx + dPhidy.*dPhidy), 1), 2);

        % Boundary Terms for Neumann conditions
        tempX = squeeze(alpha2D.*sPhi.*dPhidx.*dS);
        tempY = squeeze(alpha2D.*sPhi.*dPhidy.*dS);
        Gx = trapz(dy, tempX(:,1), 1) + trapz(dy, tempX(:,end), 1); 
        Gy = trapz(dx, tempY(1,:), 2) + trapz(dx, tempY(end,:), 2); 
    else
        for i = 1:K
            for ii = 1:K
                % Interpolation
                sPhi_i = interp2(Xinit, Yinit, squeeze(sPhi(:,:,i)), Xq, Yq, 'linear');
                sPhi_ii = interp2(Xinit, Yinit, squeeze(sPhi(:,:,ii)), Xq, Yq, 'linear');
                dPhidx_i = interp2(Xinit, Yinit, squeeze(dPhidx(:,:,i)), Xq, Yq, 'linear');
                dPhidx_ii = interp2(Xinit, Yinit, squeeze(dPhidx(:,:,ii)), Xq, Yq, 'linear');
                dPhidy_i = interp2(Xinit, Yinit, squeeze(dPhidy(:,:,i)), Xq, Yq, 'linear');
                dPhidy_ii = interp2(Xinit, Yinit, squeeze(dPhidy(:,:,ii)), Xq, Yq, 'linear');


                % Boundary Free
                if deg == -1
                    Gth(i, ii) = trapz(dx, trapz(dy, alpha2D .* (dPhidx(:, :, ii) .* dPhidx(:, :, i) + dPhidy(:, :, ii) .* dPhidy(:, :, i)), 1), 2);
                else
                    Gth(i, ii) = intStencil(length(x), length(y), dx, dy, alpha2D .* (dPhidx(:, :, ii) .* dPhidx(:, :, i) + dPhidy(:, :, ii) .* dPhidy(:, :, i)), deg) * dx * dy;
                end
                Gth2(i, ii) = sum(sum(Jgrid .* W .* squeeze(alpha2D .* (dPhidx_ii .* dPhidx_i + dPhidy_i .* dPhidy_ii))));
    
                % Boundary Terms for Neumann conditions
                tempX = squeeze(alpha2D .* sPhi(:, :, ii) .* dPhidx(:, :, i) .* dS);
                tempY = squeeze(alpha2D .* sPhi(:, :, ii) .* dPhidy(:, :, i) .* dS);
                tempX2 = alpha2D .* Jgrid .* sPhi_ii .* dPhidx_i;
                tempY2 = alpha2D .* Jgrid .* sPhi_ii .* dPhidy_i;
                Gx(i, ii) = trapz(dy, tempX(:,1), 1) + trapz(dy, tempX(:,end), 1); 
                Gy(i, ii) = trapz(dx, tempY(1,:), 2) + trapz(dx, tempY(end,:), 2); 
                Gx2(i, ii) = sum(wq_y .* tempX2(:,1)) + sum(wq_y .* tempX2(:,end)); 
                Gy2(i, ii) = sum(wq_x' .* tempY2(1,:)) + sum(wq_x' .* tempY2(end,:)); 
    
                % % Symmetry
                % Gth(i, ii) = Gth(ii, i);
                    
            end
        end
    end
    % Gth = Gth + (Gy + Gx);
    Gth = Gth2 + (Gy2 + Gx2);

    %----------------------------------------
    % Stiffness Matrices
    %----------------------------------------
    GC = Gth / Cth;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ----------------------------------------------------
    % Optimize Matrices
    % ----------------------------------------------------
    % Init
    q = zeros(K, Nt);
    g0 = T(1,:) * rPhi;
    tlist = linspace(0, Nt*Ts-Ts, Nt)';
    alpha_opt= ones(K,1);

    % Project Heat Losses
    for i = 1:Nt
        for ii = 1:K
            sPhi_ii = interp2(Xinit, Yinit, squeeze(sPhi(:,:,ii)), Xq, Yq, 'linear');
            sQ_i = interp2(Xinit, Yinit, squeeze(sQ(i, :, :)), Xq, Yq, 'linear');
            q(ii, i) = sum(sum(Jgrid .* W .* alpha2D ./ k2D .* sQ_i .* sPhi_ii));
        end
    end

    % Optimize
    scale = sqrt(lam(1:K));
    % [Cth, Gth, q_est] = optCthGth((theta-g0)', q, Ts, Cth, Gth, scale);
    % [alpha_opt, q_adj, residual] = optCthGth2((theta-g0)', q, Ts, Cth, Gth, scale);
    % [alpha_opt2, u_sim] = optCthGth3(Cth, Gth, rPhi, T, theta', q, tlist);
    % [C_opt, G_opt] = optCthGth2((theta-g0)', q, Ts, Cth, Gth);

    % % ----------------------------------------------------
    % % Reconstruct
    % % ----------------------------------------------------
    % % Init
    % temp = zeros(Nt, length(y), length(x));
    % T_hat = zeros(Nt, length(y), length(x));
    % T2D = map2D(T, xInp, yInp, x, y, 1);
    % 
    % % Reconstruct
    % for i = 1:K
    %     for ii = 1:Nt
    %         temp(ii, :, :) = theta(ii, i) .* sPhi(:, :, i);
    %     end
    %     T_hat = T_hat + temp;
    % end
    % T_pred2 = squeeze(T_hat(:,2,21));
    % T_true2 = squeeze(T2D(:,2,21));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mdl.rPhi = rPhi;
    mdl.sPhi = sPhi;
    mdl.theta = theta;
    mdl.dPhidx = dPhidx;
    mdl.dPhidy = dPhidy;
    mdl.Gth = Gth;
    mdl.Cth = Cth;
    mdl.lam = lam;
    mdl.E = E;
    mdl.GC = GC;
    mdl.K = K;
    mdl.Tavg = Tavg;
    mdl.lam = lam;
    mdl.Jgrid = Jgrid;
    mdl.alpha = alpha_opt;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting Proper Orthogonal Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1