%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: exp1                                                              %
% Date: 10.06.2022                                                        %
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
M = 20;                                                                     % Number of X points
N = 20;                                                                     % Number of Y points
l = 0.20;                                                                   % x length (m)
h = 0.20;                                                                   % y length (m)
dx = 0.01;                                                                 % internal FEM resolution (m)

%---------------------------------------------------
% Material 
%---------------------------------------------------
matK = 29;                                                                  % Thermal conductivity (W/mK)
matRho = 1300;                                                              % Material density (kg/m3)
matCp = 800;                                                                % Specific heat capacity (J/kgK)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Ta = 20;                                                                    % ambient temperature (degC)
fl = 10000;                                                                   % heat flux boundary (W/m2)
hc = 5000;                                                                  % heat transfer coefficient (W/m2K)
Tinit = 20;                                                                 % Initial temperature (degC)
Tend = 5000;                                                                % end value time (sec)
q = 1000000;                                                                % Volumetric heat generation (W/m3)
dt = 10;                                                                    % sampling time (sec)
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
R1 = [3,4,-l,l,l,-l,h,h,-h,-h]';
sf = 'R1';
ns = char('R1');
ns = ns';
g = decsg(R1,sf,ns);
geometryFromEdges(thermalmodel,g)
% pdegplot(thermalmodel,"EdgeLabels","on","FaceAlpha",0.5)
% axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
thermalProperties(thermalmodel,"ThermalConductivity",matK,'Face',1, ...
                               "MassDensity",matRho, ...
                               "SpecificHeat",matCp);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,Tinit);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% General
thermalBC(thermalmodel,"Edge",[1,2,3],'Temperature',Ta);

% % Constant Temperature
% thermalBC(thermalmodel,"Edge",4,'Temperature',Ta);
% hc = 0;
% fl = 0;

% % Adiabatic 
% hc = 0;
% fl = 0;

% % Convection
% thermalBC(thermalmodel,"Edge",4,'ConvectionCoefficient',h, ...
%                                 'AmbientTemperature',Ta);
% fl = 0;

% Heat Flux
thermalBC(thermalmodel,"Edge",4,'HeatFlux',fl);
hc = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
internalHeatSource(thermalmodel,q,'Face',1);
results = solve(thermalmodel,tlist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = results.Temperature;
xyz = thermalmodel.Mesh.Nodes;
dx = (l)/M;
dy = (h)/N;
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
dx = dx;
dy = dy;
geo = results.Mesh.Nodes';
k = matK * ones(length(T),1);
Lx = 2*l;
Ly = 2*h;
r = Tinit * ones(size(T))';
rho = matRho * ones(length(T),1);
t = tlist;
Ts = dt;
X = q * ones(size(T))';
y = T';

%---------------------------------------------------
% Define Output Boundaries
%---------------------------------------------------
% Ambient Temperature
Ta = Ta * ones(length(T),1);

% Convection
hc = hc * ones(length(T),1);
for i = 1:numel(results.Mesh.Nodes(1,:))
    if results.Mesh.Nodes(1,i) ~= -l
        hc(i) = 0;
    end
end

% Heat Flux
fl = fl * ones(length(T),1);
for i = 1:numel(results.Mesh.Nodes(1,:))
    if results.Mesh.Nodes(1,i) ~= -l
        fl(i) = 0;
    end
end

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