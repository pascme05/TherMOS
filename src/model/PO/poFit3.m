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
    Kmax = para.Mdl.gen.Kmax;                                               % maximum number of modes
    Emax = para.Mdl.gen.Emax;                                               % maximum energy captured (%)
    E = 0;                                                                  % captured energy by POD
    eps = para.Mdl.gen.eps;                                                 % numerical lower bound
    deg = -1;                                                                % degree for stencil solution (-1 internal trapz approach)
    
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
    catch
        hc = zeros(size(alpha));
    end

    %===================================================
    % Variables
    %===================================================
    % General
    xInp = data.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.Data.geo(:,2);                                              % sampled input values y (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    T = data.y;                                                             % temperature snapshots NtxN (°C)
    
    % BC Matrix
    dS = zeros(length(y), length(x));
    dS(1, :) = dy;      
    dS(end, :) = dy;     
    dS(:, 1) = dx;       
    dS(:, end) = dx; 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Converting 2D
    %===================================================
    alpha2D = squeeze(map2D(alpha', xInp, yInp, x, y, 1));
    k2D = squeeze(map2D(k', xInp, yInp, x, y, 1));
    hc2D = squeeze(map2D(hc', xInp, yInp, x, y, 1));

    %===================================================
    % Mean Centering
    %===================================================
    Tavg = mean(T, 1);                                                      % average temperature over time steps (°C)
    T = T - Tavg.*ones(Nt,N);                                               % mean centered observation matrix (°C)

    %===================================================
    % Eigenvalues
    %===================================================
    [~,S,Phi] = svd(T/sqrt(Nt-1));
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
    Gx = zeros(K, K);
    Gy = zeros(K, K);
    BC_h = zeros(K, K);

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
    sPhi = permute(sPhi, [2, 3, 1]);

    %----------------------------------------
    % Temporal Modes
    %----------------------------------------
    theta = (rPhi' * T')';

    %===================================================
    % Gradients
    %===================================================
    if K == 1
        for i = 1:K
            [dPhidx(:, :, i), dPhidy(:, :, i)] = gradient(sPhi(:, :)', dx, dy);
        end
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
    for i = 1:K
        for ii = 1:K
            % Trapasoidal
            if deg == -1
                Cth(i, ii) = trapz(dx, trapz(dy, sPhi(:, :, ii) .* sPhi(:, :, i), 1), 2) / dx / dy;
            
            % Stencil
            else
                Cth(i, ii) = intStencil(length(x), length(y), dx, dy, sPhi(:, :, ii) .* sPhi(:, :, i), deg) * dx * dy;
            end
        end
    end

    %----------------------------------------
    % Thermal Conductance
    %----------------------------------------
    for i = 1:K
        for ii = 1:K
            % Boundary Free
            if deg == -1
                Gth(i, ii) = trapz(dx, trapz(dy, alpha2D .* (dPhidx(:, :, ii) .* dPhidx(:, :, i) + dPhidy(:, :, ii) .* dPhidy(:, :, i)), 1), 2) / dx / dy;
            else
                Gth(i, ii) = intStencil(length(x), length(y), dx, dy, alpha2D .* (dPhidx(:, :, ii) .* dPhidx(:, :, i) + dPhidy(:, :, ii) .* dPhidy(:, :, i)), deg) * dx * dy;
            end
            
            % Boundary Terms for Neumann conditions (Y-direction)
            tempY = trapz(dy, alpha2D .* sPhi(:, :, ii) .* dPhidx(:, :, i), 1) / dy;
            Gy(i, ii) = sum(tempY([1, end]));

            % Boundary Terms for Neumann conditions (X-direction)
            tempX = trapz(dx, alpha2D .* sPhi(:, :, ii) .* dPhidy(:, :, i), 2) / dx;
            Gx(i, ii) = sum(tempX([1, end]));

            % % Boundary Convection
            % tempBC = hc2D .* alpha2D ./ k2D .* sPhi(:, :, ii) .* sPhi(:, :, i);
            % BC_h(i,ii) = sum(sum(tempBC .* dS));

            % % Update Gth
            % Gth(i, ii) = Gth(i, ii) + BC_h(i,ii);
        end
    end
    Gth = Gth + (Gy/dy + Gx/dx);

    %----------------------------------------
    % Stiffness Matrices
    %----------------------------------------
    GC = Gth / Cth;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Fitting Proper Orthogonal Model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1