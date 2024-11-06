%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: poFit2                                                            %
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
function mdl = poFit2(data, ~, para)
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
    % alpha = k ./ (rho.*Cp);                                                 % Thermal diffusivity (m²/s)

    %===================================================
    % Variables
    %===================================================
    xInp = data.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.Data.geo(:,2);                                              % sampled input values y (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    T = data.y;                                                             % temperature snapshots NtxN (°C)
    % Q = data.X;                                                             % volumetric heat generation (W/m³)
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Mean Centering
    %===================================================
    k2D = squeeze(map2D(k', xInp, yInp, x, y));

    %===================================================
    % Mean Centering
    %===================================================
    Tavg = mean(T, 1);                                                      % average temperature over time steps (°C)
    T = T - Tavg.*ones(Nt,N);                                               % mean centered observation matrix (°C)

    %===================================================
    % Eigenvalues
    %===================================================
    [~,S,Phi] = svd(T/sqrt(Nt-1), 'econ');
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
    fprintf('Number of used eigenvalues (model order) K: %i \n', K);
    fprintf('Cumulative correlation energy Em: %f \n', E)

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
    sPhi = map2D(rPhi', xInp, yInp, x, y);
    sPhi = reshape(sPhi, [length(y), length(x), K]);

    %----------------------------------------
    % Temporal Modes
    %----------------------------------------
    theta = rPhi' * T';

    %===================================================
    % Gradients
    %===================================================
    for i = 1:K
        [dPhidx(:, :, i), dPhidy(:, :, i)] = gradient(sPhi(:, :, i), dx, dy);
    end

    %===================================================
    % System Matrices
    %===================================================
    % Compute Thermal Capacitance Matrix (Cth)
    for i = 1:K
        for j = 1:K
            Cth(i, j) = sum(sum(rho .* Cp .* rPhi(:, i) .* rPhi(:, j) * dx * dy));
        end
    end
    
    % Compute Thermal Conductance Matrix (Gth)
    for i = 1:K
        for j = 1:K
            Gx(i, j) = sum(sum(k2D .* dPhidx(:, :, i) .* dPhidx(:, :, j) * dx * dy));
            Gy(i, j) = sum(sum(k2D .* dPhidy(:, :, i) .* dPhidy(:, :, j) * dx * dy));
            Gth(i, j) = Gx(i, j) + Gy(i, j);
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
    mdl.decay = decay;
    mdl.beta = beta;
    mdl.E = E;
    mdl.GC = GC;
    mdl.K = K;
    mdl.Tavg = Tavg;
    mdl.T0 = T0;
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