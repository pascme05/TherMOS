%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: map2D                                                             %
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
% This function maps a snapshot quantity to a 2D field variable
% -------------------------------------------------------------------------
% Inp:  1) Tin:         Input snapshot matrix NtxN
%       2) xInp/yInp:   Spatial input variables
%       3) xOut/yOut:   Spatial output variables
%       4) Inter:       Interpolation method 1) nearest, 2) linear
% Out:  1) Tout:        Output 2D temperature field 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Tout = map2D(Tin, xInp, yInp, xOut, yOut, inter)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [Nt, ~] = size(Tin);                                                    % number of samples Nt and spatial points N
    x_min = min(xInp);                                                      % minimum input value x spacing (m)
    x_max = max(xInp);                                                      % maximum input value x spacing (m)
    y_min = min(yInp);                                                      % minimum input value y spacing (m)
    y_max = max(yInp);                                                      % maximum input value y spacing (m)
    Nx = length(xOut);                                                      % number of output samples x
    Ny = length(yOut);                                                      % number of output samples y
    Tout = zeros(Ny, Nx, Nt);                                               % Output temperature matrix

    %===================================================
    % Variables
    %===================================================
    x_grid = linspace(x_min, x_max, Nx);
    y_grid = linspace(y_min, y_max, Ny);
    
    %===================================================
    % Interpolation
    %===================================================
    if inter == 1
        method = 'nearest';
    elseif inter == 2
        method = 'linear';
    elseif inter == 3
        method = 'natural';
    else
        method = 'nearest';
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Create the new target grid
    %===================================================
    [Xq, Yq] = meshgrid(x_grid, y_grid);

    %===================================================
    % Interpolation
    %===================================================
    %----------------------------------------
    % Create interpolant once
    %----------------------------------------
    F = scatteredInterpolant(xInp, yInp, Tin(1,:)', method, 'nearest');
    
    %----------------------------------------
    % Process each time step
    %----------------------------------------
    for i = 1:Nt
        F.Values = Tin(i,:)';
        Tout(:,:,i) = F(Xq, Yq);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Tout = permute(Tout, [3 1 2]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1