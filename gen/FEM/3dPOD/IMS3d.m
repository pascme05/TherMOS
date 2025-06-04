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
M = 51;
N = 44;
K = 10;
l = 10e-3;
b = 2150e-6;
h = 10e-3;  % Now height is 50 mm in z-direction

% Material layer thicknesses
b_Cu = 100e-6;
b_Di = 50e-6;
b_Al = 1.5e-3;
b_Ga = 0.5e-3;

% Switch
l_Sw = 8e-3;
b_Sw = b_Cu;
h_Sw = 5.5e-3;
A_sw = l_Sw * b_Cu;
Vol_sw = h_Sw * A_sw;

% Material Properties
matK = [400, 2, 90, 4];
matRho = [8933, 2200, 2680, 3300];
matCp = [380, 800, 1000, 1500];

% Mesh
dx = 500e-6;

% Losses
Pv = 20;
q = Pv / (l*b * b_Cu * 8);

% Load case
Ta = 55;
hc = 1000;
Tinit = 55;
Tend = 50;
dt = 5;
tlist = 0:dt:Tend-dt;

% Settings
plotting = 1;
saving = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry Definition (3D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Z levels
z0 = 0;
z1 = z0 + b_Ga;
z2 = z1 + b_Al;
z3 = z2 + b_Di;
z4 = z3 + b_Cu;

% Create Model
gm = multicuboid(l,h,[b_Ga b_Al b_Di b_Cu],ZOffset=[z0 z1 z2 z3]);

% Combine geometry
thermalmodel = createpde('thermal', 'transient');
thermalmodel.Geometry = gm;
pdegplot(thermalmodel, "CellLabels", "on", "FaceLabels", "on", "FaceAlpha", 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Materials by face
thermalProperties(thermalmodel, "ThermalConductivity", matK(1), ...
                                 "MassDensity", matRho(1), ...
                                 "SpecificHeat", matCp(1), ...
                                 "Cell", 4);
thermalProperties(thermalmodel, "ThermalConductivity", matK(2), ...
                                 "MassDensity", matRho(2), ...
                                 "SpecificHeat", matCp(2), ...
                                 "Cell", 3);
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
internalHeatSource(thermalmodel, q, 'Cell', 4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
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
    if results.Mesh.Nodes(3,i) >= z3 
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
    % Copper
    if results.Mesh.Nodes(2,i) >= z3
        Cp(i,:) = matCp(1);
        k(i,:) = matK(1);
        rho(i,:) = matRho(1);
    % Dielectrica
    elseif results.Mesh.Nodes(2,i) >= z2
        Cp(i,:) = matCp(2);
        k(i,:) = matK(2);
        rho(i,:) = matRho(2);
    % Aluminium
    elseif results.Mesh.Nodes(2,i) >= z1
        Cp(i,:) = matCp(3);
        k(i,:) = matK(3);
        rho(i,:) = matRho(3);
    elseif results.Mesh.Nodes(2,i) < z1
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
vars_to_save = {'Cp', 'dx', 'dy', 'dz', 'geo', 'k', 'Lx', 'Ly', 'r', 'rho', 't', ...
                'Ts', 'X', 'y', 'Ta', 'hc', 'fl'};

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
