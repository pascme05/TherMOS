%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: IMS3d                                                             %
% Date: 21.04.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright:                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear variables
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Parameters and Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimensions
M = 41;
N = 21;
K = 11;
l = 20e-3;
h = 5e-3;
b = 10e-3;

% Material layer thicknesses
h_Cu = 100e-6;
h_Di = 50e-6;
h_Al = 1.0e-3;
h_Ga = 4.0e-3;

% Material Properties
matK = [400, 2, 90, 4];
matRho = [8933, 2200, 2680, 3300];
matCp = [380, 800, 1000, 1500];

% Mesh
dx = 500e-6;

% Losses
Pv = 20;
Vol_Al = l*b*h_Al;
q = Pv / Vol_Al;

% Load case
Ta = 55;
hc = 1000;
Tinit = 55;
Tend = 50;
dt = 0.1;
tlist = 0:dt:Tend-dt;

% Settings
plotting = 1;
saving = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry Definition (3D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Z levels
z0 = 0;
z1 = z0 + h_Ga;

% Create Model
gm = multicuboid(l,b,[h_Ga h_Al],ZOffset=[z0 z1]);

% Combine geometry
thermalmodel = createpde('thermal', 'transient');
thermalmodel.Geometry = gm;
pdegplot(thermalmodel, "CellLabels", "on", "FaceLabels", "on", "FaceAlpha", 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Materials by face
% thermalProperties(thermalmodel, "ThermalConductivity", matK(1), ...
%                                  "MassDensity", matRho(1), ...
%                                  "SpecificHeat", matCp(1), ...
%                                  "Cell", 4);
% thermalProperties(thermalmodel, "ThermalConductivity", matK(2), ...
%                                  "MassDensity", matRho(2), ...
%                                  "SpecificHeat", matCp(2), ...
%                                  "Cell", 3);
thermalProperties(thermalmodel, "ThermalConductivity", matK(3), ...
                                 "MassDensity", matRho(3), ...
                                 "SpecificHeat", matCp(3), ...
                                 "Cell", 2);
thermalProperties(thermalmodel, "ThermalConductivity", matK(4), ...
                                 "MassDensity", matRho(4), ...
                                 "SpecificHeat", matCp(4), ...
                                 "Cell", 1);

% Initial Condition
thermalIC(thermalmodel, Tinit);

% Boundary Conditions
thermalBC(thermalmodel, "Face", 1, ...
                        "ConvectionCoefficient", hc, ...
                        "AmbientTemperature", Ta);

% Heat Source (only on Cu face 5 and 7 as example)
internalHeatSource(thermalmodel, q, 'Cell', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx);
pdemesh(thermalmodel);
results = solve(thermalmodel,tlist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Temperature 
%---------------------------------------------------
T = results.Temperature;
xyz = thermalmodel.Mesh.Nodes;
dx = (l)/(M-1);
dy = (b)/(N-1);
dz = (h)/(K-1);

%---------------------------------------------------
% Internal Heat 
%---------------------------------------------------
Q = zeros(size(T));
for i = 1:size(T,1)
    if results.Mesh.Nodes(3,i) >= z1 
        Q(i,:) = q;
    else
        % Q(i,:) = q/2;
    end
end

%---------------------------------------------------
% Spatial Parameters
%---------------------------------------------------
Cp = matCp(2) * ones(length(T),1);
k = matK(2) * ones(length(T),1);
rho = matRho(2) * ones(length(T),1);
for i = 1:size(T,1)
    % Aluminium
    if results.Mesh.Nodes(3,i) >= z1
        Cp(i,:) = matCp(3);
        k(i,:) = matK(3);
        rho(i,:) = matRho(3);
    elseif results.Mesh.Nodes(3,i) < z1
        Cp(i,:) = matCp(4);
        k(i,:) = matK(4);
        rho(i,:) = matRho(4);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define Variables 
%---------------------------------------------------
geo = results.Mesh.Nodes';
Lx = l;
Ly = b;
Lz = h;
r = Tinit * ones(size(T))';
t = tlist;
Ts = dt;
X = Q';
y = T';

%---------------------------------------------------
% Define Output Boundaries
%---------------------------------------------------
% Ambient Temperature
Ta = Ta * ones(length(T),1);

% Convection
hc = hc * zeros(length(T),1);

% Heat Flux
fl = zeros(length(T),1);

%---------------------------------------------------
% Save Variables 
%---------------------------------------------------
% Define Vars
vars_to_save = {'Cp', 'dx', 'dy', 'dz', 'geo', 'k', 'Lx', 'Ly', 'Lz', 'r', ...
                'rho', 't', 'Ts', 'X', 'y', 'Ta', 'hc', 'fl'};

% Save Vars
if saving == 1
    save('data.mat', vars_to_save{:});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotting == 1
    figure;
    for i = 1:floor(length(tlist)/10)
        pdeplot3D(thermalmodel, 'ColorMapData', results.Temperature(:, i*10));
        title(['Temperature at t = ', num2str(tlist(i*10)), ' s']);
        colorbar;
        drawnow;
    end
end
