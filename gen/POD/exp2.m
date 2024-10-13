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
%% Load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Parameters and Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 20;
N = 20;
l = 0.20;
h = 0.20;
q = 1e6;
dx = 0.001;
dt = 1;
tlist = 0:dt:500-dt;

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
pdegplot(thermalmodel,"EdgeLabels","on","FaceAlpha",0.5)
axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
thermalProperties(thermalmodel,"ThermalConductivity",29,'Face',1, ...
                               "MassDensity",1300, ...
                               "SpecificHeat",800);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,20);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% Adiabatic
thermalBC(thermalmodel,"Edge",[1,2,3],'Temperature',20);
        
% Constant Temperature
thermalBC(thermalmodel,"Edge",4,'Temperature',20); 

% % Convection
% thermalBC(thermalmodel,"Edge",4,'ConvectionCoefficient',1000, ...
%                                 'AmbientTemperature',20);

% % Heat Flux
% thermalBC(thermalmodel,"Edge",4,'HeatFlux',100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Init 
%---------------------------------------------------
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
thermalmodel.SolverOptions.RelativeTolerance = 1E-5;
thermalmodel.SolverOptions.AbsoluteTolerance = 1E-9;

%---------------------------------------------------
% Source 
%---------------------------------------------------
% % Step
% internalHeatSource(thermalmodel,@heatSourceStep,'Face',1);

% Sine
internalHeatSource(thermalmodel,@heatSourceSine,'Face',1);

%---------------------------------------------------
% Solve 
%---------------------------------------------------
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define the time-varying heat source (step)
%---------------------------------------------------
function Q = heatSourceStep(location,state)
    % Init
    q_dot_max = 1e6; % Maximum heat source value
    step_time = 100; % Time at which the step occurs
    Q = zeros(1,numel(location.x));
    
    % Check for NaNs
    if(isnan(state.time))
      Q(1,:) = NaN;
      return
    end

    % Step Function
    if state.time < step_time
        Q(1,:) = q_dot_max/10;
    else
        Q(1,:) = q_dot_max;
    end

    % Debug
    % disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(Q(1))]);
end

%---------------------------------------------------
% Define the time-varying heat source (sine)
%---------------------------------------------------
function Q = heatSourceSine(location,state)
    % Init
    q_dot_max = 1e6; % Maximum heat source value
    f = 0.01;
    Q = zeros(1,numel(location.x));
    
    % Check for NaNs
    if(isnan(state.time))
      Q(1,:) = NaN;
      return
    end

    % Step Function
    Q(1,:) = q_dot_max * abs(sin(2*pi*f*state.time));

    % Debug
    % disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(Q(1))]);
end