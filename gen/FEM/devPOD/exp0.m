%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Model Order Reduction of spatial temporal signatures using KLE   %
% Topic: Battery Modelling                                                %
% File: main                                                              %
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
l = 1;                                                                      % x length (m)
h = 1;                                                                      % y length (m)
dx = 0.01;                                                                  % internal FEM resolution (m)

%---------------------------------------------------
% Material 
%---------------------------------------------------
matK = 400;                                                                 % Thermal conductivity (W/mK)
matRho = 1300;                                                              % Material density (kg/m3)
matCp = 600;                                                                % Specific heat capacity (J/kgK)

%---------------------------------------------------
% Loading 
%---------------------------------------------------
heatVariation = [10000 15000 20000];
snapShotIDs = [1:10 59 60 61];
dt = 10;
tlist = 0:dt:600;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thermalmodel = createpde("thermal","transient");
geometryFromEdges(thermalmodel,@squareg);
pdegplot(thermalmodel,"EdgeLabels","on","FaceAlpha",0.5)
axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
thermalProperties(thermalmodel,"ThermalConductivity",400, ...
                               "MassDensity",1300, ...
                               "SpecificHeat",600);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,0);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
thermalBC(thermalmodel,"Edge",2,"Temperature",100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
T = [];
T1 = [];
for q = heatVariation
    internalHeatSource(thermalmodel,q);
    results = solve(thermalmodel,tlist);
    T = [T, results.Temperature(:,snapShotIDs)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
[qx,qy] = evaluateHeatFlux(results);
for i = 1:floor(length(tlist) / 10)
    subplot(2,1,1)
    pdeplot(thermalmodel,"XYData",results.Temperature(:,floor(i*10)))
    subplot(2,1,2)
    pdeplot(thermalmodel,'FlowData',[qx qy])
    pause(0.001)
end

figure
for i = 1:floor(length(tlist) / 10)
    dT1 = gradient(squeeze(T1(floor(i*10),:,:)), dx, dy);
    subplot(2,1,1)
    contourf(squeeze(T1(floor(i*10),:,:)))
    subplot(2,1,2)
    contourf(dT1)
    pause(0.05)
end