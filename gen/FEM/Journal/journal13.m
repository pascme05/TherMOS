%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: journal1                                                          %
% Date: 17.12.2024                                                        %
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
%---------------------------------------------------
% Dimension 
%---------------------------------------------------
M = 50;                                                                     % Number of X points
N = 8;                                                                      % Number of Y points
l = 0.05;                                                                   % x length (m)
b = 0.008;                                                                  % width (m)
h = 0.002;                                                                  % y length (m)
dx = 0.001;                                                                 % internal FEM resolution (m)
Vol = 8*l*b*h;                                                              % Busbar volume (m³)

%---------------------------------------------------
% Material 
%---------------------------------------------------
matK = 400;                                                                 % Thermal conductivity (W/mK)
matRho = 8900;                                                              % Material density (kg/m3)
matCp = 380;                                                                % Specific heat capacity (J/kgK)

%---------------------------------------------------
% losses 
%---------------------------------------------------
Idc = 500;                                                                  % Busbar current (A)
P_ohm = 0.01724 / (4*b*h*1e6) * 2*l * Idc^2;                                % Internal heat generation (W)
P_con = 5e-6 * Idc^2;                                                       % Heat flow screwing connection (W)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Ta = 55;                                                                    % ambient temperature (degC)
fl = 100;                                                                   % heat flux boundary (W/m2)
hc = 1000;                                                                  % heat transfer coefficient (W/m2K)
Tinit = 55;                                                                 % Initial temperature (degC)
Tend = 3000;                                                                 % end value time (sec)
q = 10*P_ohm/Vol;                                                              % Volumetric heat generation (W/m3)
dt = 10;                                                                     % sampling time (sec)
tlist = 0:dt:Tend-dt;                                                       % time vector (sec)

%---------------------------------------------------
% Settings
%---------------------------------------------------
plotting = 1;
saving = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thermalmodel = createpde("thermal","transient");
R1 = [3,4,-l,-0.5*l,-0.5*l,-l,b,b,-b,-b]'; % Left region (25 mm)
R2 = [3,4,-0.5*l,0.5*l,0.5*l,-0.5*l,b,b,-b,-b]';   % Middle region (50 mm)
R3 = [3,4,0.5*l,l,l,0.5*l,b,b,-b,-b]';     % Right region (25 mm)


% Combine the regions into a single geometry
gd = [R1, R2, R3];
sf = 'R1+R2+R3';
ns = char('R1','R2','R3');
ns = ns';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalmodel,g)
pdegplot(thermalmodel,"EdgeLabels","on","FaceAlpha",0.5)
% axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
thermalProperties(thermalmodel,"ThermalConductivity",matK,'Face',[1,2,3], ...
                               "MassDensity",matRho, ...
                               "SpecificHeat",matCp);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,Tinit);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% Temperature
thermalBC(thermalmodel,"Edge",[1,4,5,6],'Temperature',Ta);

% % Convection
% thermalBC(thermalmodel,"Edge",[1,4,5,6],'ConvectionCoefficient',hc, ...
%                                 'AmbientTemperature',Ta);

% thermalBC(thermalmodel,"Edge",[1,5,8,9],'ConvectionCoefficient',5, ...
%                                 'AmbientTemperature',Ta);

% % Heat Flux
% thermalBC(thermalmodel,"Edge",[2,6,9],'HeatFlux',P_con/(b+l));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
internalHeatSource(thermalmodel,q,'Face',[1, 2, 3]);
results = solve(thermalmodel,tlist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = results.Temperature;
xyz = thermalmodel.Mesh.Nodes;
dx = (l)/M;
dy = (b)/N;
for k = -M:1:M
    for kk = -N:1:N
        [minD, idx] = min(sum(abs(xyz-[dx*k; dy*kk].*ones(size(xyz))),1));
        xy(kk+N+1,k+M+1) = minD;
        T1(:,kk+N+1,k+M+1) = T(idx, :);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define Variables 
%---------------------------------------------------
Cp = matCp * ones(length(T),1);
geo = results.Mesh.Nodes';
k = matK * ones(length(T),1);
Lx = 2*l;
Ly = 2*b;
r = Tinit * ones(size(T))';
rho = matRho * ones(length(T),1);
t = tlist;
Ts = dt;
X = q * ones(size(T))';
y = T';

%---------------------------------------------------
% Define Output Boundaries
%---------------------------------------------------
% Mapping
x_coords = results.Mesh.Nodes(1, :);
y_coords = results.Mesh.Nodes(2, :);

% Define the boundary values
l = max(abs(x_coords));  % Half-length in x-direction
b = max(abs(y_coords));  % Half-length in y-direction

% Find indices of nodes that are on any edge
on_left_edge   = abs(x_coords + l) < 1e-4;   % x = -l
on_right_edge  = abs(x_coords - l) < 1e-4;   % x = l
on_bottom_edge = abs(y_coords + b) < 1e-4;   % y = -b
on_top_edge    = abs(y_coords - b) < 1e-3;   % y = b

on_edge = on_left_edge | on_right_edge | on_bottom_edge | on_top_edge;

% Apply the condition: x < -0.25 and node on an edge
mask = (x_coords < -0.025) & on_edge;
sum(mask)

% Set those nodes to zero
tempMesh = results.Mesh.Nodes;
tempMesh(:, mask) = 0;

% Ambient Temperature
Ta = Ta * ones(length(T),1);

% Convection
hc = hc * zeros(length(T),1);
% hc = hc * ones(length(T),1);
% for i = 1:numel(tempMesh(1,:))
%     if mask(i) == 0
%         hc(i) = 0;
%     end
% end

% Heat Flux
fl = zeros(length(T),1);

%---------------------------------------------------
% Save Variables 
%---------------------------------------------------
% Define Vars
vars_to_save = {'Cp', 'dx', 'dy', 'geo', 'k', 'Lx', 'Ly', 'r', 'rho', 't', ...
                'Ts', 'X', 'y', 'Ta', 'hc', 'fl'};

% Save Vars
if saving == 1
    save('data.mat', vars_to_save{:});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotting == 1
    %---------------------------------------------------
    % Fluxes
    %---------------------------------------------------
    figure
    [qx,qy] = evaluateHeatFlux(results);
    for i = 1:floor(length(tlist) / 10)
        subplot(2,1,1)
        pdeplot(thermalmodel,"XYData",results.Temperature(:,floor(i*10)))
        subplot(2,1,2)
        pdeplot(thermalmodel,'FlowData',[qx qy])
        pause(0.001)
    end
    
    %---------------------------------------------------
    % Temperatures
    %---------------------------------------------------
    figure
    for i = 1:floor(length(tlist) / 10)
        dT1 = gradient(squeeze(T1(floor(i*10),:,:)), dx, dy);
        subplot(2,1,1)
        contourf(squeeze(T1(floor(i*10),:,:)))
        subplot(2,1,2)
        contourf(dT1)
        pause(0.05)
    end
end