%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: calcGrad3D                                                        %
% Date: 07.05.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculates the temperature gradient and the heat flux based
% on the spatial temperature distribution T(x,t).
% -------------------------------------------------------------------------
% Inp:  1) data:        All input data
%       2) Inp:         -1) last sample, 1) all data
% Out:  1) q:           Temperature gradient (K/m)
%       2) qk:          Heat flux (W/m2)
%       3) out:         All output data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [q, qk, out] = calcGrad3D(data, Inp)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [Nt, ~] = size(data.y);                                                 % number of output samples
    dx = data.Data.dx;                                                      % spacing x direction (m)
    dy = data.Data.dy;                                                      % spacing y direction (m)
    dz = data.Data.dz;                                                      % spacing z direction (m)
    Ly = data.Data.Ly;                                                      % length in y direction (m)
    Lx = data.Data.Lx;                                                      % length in x direction (m)
    Lz = data.Data.Lz;                                                      % length in z direction (m)

    %===================================================
    % Variables
    %===================================================
    T = data.y;
    xInp = data.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.Data.geo(:,2);                                              % sampled input values y (m)
    zInp = data.Data.geo(:,3);                                              % sampled input values z (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    z = 0:dz:Lz;                                                            % z vector (m)
    

    %===================================================
    % Model
    %===================================================
    k = data.Data.k;                                                        % thermal conductivity (W/mK)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Check Samples
    %===================================================
    if Inp == -1
        %----------------------------------------
        % Init
        %----------------------------------------
        qk = zeros(length(y), length(x), length(z));                        % init gradient
        q = zeros(length(y), length(x), length(z));                         % init heat flux

        %----------------------------------------
        % 3D Mapping
        %----------------------------------------
        k = squeeze(map3D(k', xInp, yInp, zInp, x, y, z, 1));
        T = squeeze(map3D(squeeze(T(Nt,:)), xInp, yInp, zInp, x, y, z, 1));
        Nt = 1;
    else
        %----------------------------------------
        % Init
        %----------------------------------------
        qk = zeros(Nt, length(y), length(x), length(z));                    % init gradient
        q = zeros(Nt, length(y), length(x), length(z));                     % init heat flux

        %----------------------------------------
        % 3D Mapping
        %----------------------------------------
        k = squeeze(map3D(k', xInp, yInp, zInp, x, y, z, 1));
        T = map3D(T, xInp, yInp, zInp, x, y, z, 1);
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:Nt
        if Nt == 1
            %----------------------------------------
            % Gradient
            %----------------------------------------
            [dTdy, dTdx, dTdz] = gradient(T, dy, dx, dz);
    
            %----------------------------------------
            % Flux
            %----------------------------------------
            qk = -(dTdx + dTdy + dTdz);
            q = k .* qk;
        else
            %----------------------------------------
            % Gradient
            %----------------------------------------
            [dTdy, dTdx, dTdz] = gradient(squeeze(T(i,:,:,:)), dy, dx, dz);
    
            %----------------------------------------
            % Flux
            %----------------------------------------
            qk(i, :, :, :) = -(dTdx + dTdy + dTdz);
            q(i, :, :, :) = k .* squeeze(qk(i,:,:,:));
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out = data;
    out.q = q;
    out.qk = qk;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1