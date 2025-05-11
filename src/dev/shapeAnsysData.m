%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Titel: Model Order Reduction (MOR)
% File: loadAnsys
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
% This functions loads the input data for training, testing and validation
% using ansys as an input file type. Furthmore, the data splitting can be 
% done on file level or on ID level. The following are the input and 
% output parameters:
% Input:
%       - path:     path variable to the data files
%       - setup:    all setup variables for the simulation
% Output:
%       - outTrain: output training data
%       - outTest:  output testing data
%       - outVal:   output validation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FNC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = shapeAnsysData(data, setup)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Reshaping Ansys Data')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Parameters
% ----------------------------------------------------
dx = setup.data.dx;
dy = setup.data.dy;
Nt = length(data.Time);
pos = setup.data.cutPos;
ax = setup.data.cut;

% ----------------------------------------------------
% Variables
% ----------------------------------------------------
geo = data.Geometry;
T = data.Temperature;
P = data.Load;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Get Axis
% ----------------------------------------------------
% X-Axis
if ax == 'x'
    selX = 2;
    selY = 3;
    selZ = 1;

% Y-Axis
elseif ax == 'y'
    selX = 1;
    selY = 3;
    selZ = 2;

% Z-Axis
else
    selX = 1;
    selY = 2;
    selZ = 3;
end

% ----------------------------------------------------
% Get Length
% ----------------------------------------------------
Nx = floor((max(geo(:, selX)) - min(geo(:, selX))) / dx) + 1;
Ny = floor((max(geo(:, selY)) - min(geo(:, selY))) / dy) + 1;
x = linspace(min(geo(:, selX)), max(geo(:, selX)), Nx);
y = linspace(min(geo(:, selY)), max(geo(:, selY)), Ny);
outTemp = zeros(Nt, Ny, Nx);
outHeat = zeros(Nt, Ny, Nx);
outRef = zeros(Nt, Ny, Nx);
outMat = zeros(Ny, Nx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reshape
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Find minimum in Data
% ----------------------------------------------------
for i = 1:Ny
    for ii = 1:Nx
        % Error Function
        err = sqrt(abs(geo(:, selX) - x(ii)).^2 + abs(geo(:, selY) - y(i)).^2 + abs(geo(:, selZ) - pos).^2);

        % Check for multiple minima
        min_val = min(err);
        min_id = find(err == min_val);
        
        % Extract
        if length(min_id) == 1
            [~, id1] = min(err);
        else
            id1 = min_id(end);
            % outMat(i, ii) = geo(id1, 4);
            % for iii = 1:length(min_id)
            %     outTemp(:, i, ii) = outTemp(:, i, ii) + T(min_id(iii), :)';
            %     outHeat(:, i, ii) = outHeat(:, i, ii) + P(:, geo(min_id(iii), 5));
            % end
            % outTemp(:, i, ii) = outTemp(:, i, ii) / length(min_id);
            % outHeat(:, i, ii) = outHeat(:, i, ii) / length(min_id);
        end
        outTemp(:, i, ii) = T(id1, :);
        outMat(i, ii) = geo(id1, 4);
        outHeat(:, i, ii) = P(:, geo(id1, 5));
    end
end

% ----------------------------------------------------
% Generate Reference Signal
% ----------------------------------------------------
for i = 1:Nt
    outRef(i, :, :) = squeeze(outTemp(1, :, :));
end

% ----------------------------------------------------
% Generate Output
% ----------------------------------------------------
out.X = outHeat;
out.y = outTemp;
out.r = outRef;
out.Lx = max(geo(:, selX)) - min(geo(:, selX));
out.Ly = max(geo(:, selY)) - min(geo(:, selY));
out.dx = out.Lx / (Nx - 1);
out.dy = out.Ly / (Ny - 1);
out.Mat = outMat;
out.Material = data.Material;
out.t = data.Time;
out.Ts = out.t(2) - out.t(1);

% ----------------------------------------------------
% Material Matrix
% ----------------------------------------------------
% Init
out.Mat2D.k = zeros(Ny, Nx);
out.Mat2D.rho = zeros(Ny, Nx);
out.Mat2D.Cp = zeros(Ny, Nx);

% 2D Matrices
for i = 1:Ny
    for ii = 1:Nx
        idx = floor(out.Mat(i,ii));
        out.Mat2D.k(i, ii) = out.Material(idx, 6);
        out.Mat2D.rho(i, ii) = out.Material(idx, 4);
        out.Mat2D.Cp(i, ii) = out.Material(idx, 3);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initial Value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Check Initial Value from Time
% ----------------------------------------------------
if data.Time(1) ~= 0
    out.t = [0; out.t];
end

% ----------------------------------------------------
% Add initial conditions
% ----------------------------------------------------
if data.Time(1) ~= 0
    % Temperatures
    out.y = [min(min(out.y(1,:,:)))*ones(1, Ny, Nx); out.y];

    % Load
    out.X = [out.X(1,:,:); out.X];

    % Reference
    out.r = [out.r(1,:,:); out.r];
end

end