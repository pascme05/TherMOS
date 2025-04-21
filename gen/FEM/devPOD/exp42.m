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
% Dimensions in meters
width = 1e-3; % 1 mm
length = 1e-3; % 1 mm
thickness_silicon = 10e-6; % 10 µm
thickness_copper_substrate = 50e-6; % 50 µm
thickness_ceramic = 100e-6; % 100 µm
thickness_copper_baseplate = 200e-6; % 200 µm

% Total height
total_height = thickness_silicon + thickness_copper_substrate + thickness_ceramic + thickness_copper_baseplate;

%---------------------------------------------------
% Settings
%---------------------------------------------------
plotting = 1;
saving = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thermalmodel = createpde("thermal","transient");

% Define the coordinates of each layer
R1 = [3,4,0,width,width,0,0,0,thickness_silicon,thickness_silicon]'; % Silicon Die
R2 = [3,4,0,width,width,0,thickness_silicon,thickness_silicon,thickness_silicon+thickness_copper_substrate,thickness_silicon+thickness_copper_substrate]'; % Copper Substrate
R3 = [3,4,0,width,width,0,thickness_silicon+thickness_copper_substrate,thickness_silicon+thickness_copper_substrate,thickness_silicon+thickness_copper_substrate+thickness_ceramic,thickness_silicon+thickness_copper_substrate+thickness_ceramic]'; % Ceramic Isolation
R4 = [3,4,0,width,width,0,thickness_silicon+thickness_copper_substrate+thickness_ceramic,thickness_silicon+thickness_copper_substrate+thickness_ceramic,thickness_silicon+thickness_copper_substrate+thickness_ceramic+thickness_copper_baseplate,thickness_silicon+thickness_copper_substrate+thickness_ceramic+thickness_copper_baseplate]'; % Copper Baseplate

% Combine the regions into a single geometry
gd = [R1, R2, R3, R4];
sf = 'R1+R2+R3+R4';
ns = char('R1','R2','R3','R4');
ns = ns';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalmodel,g)

% Plot the geometry with edge and face labels
pdegplot(thermalmodel,"EdgeLabels","on","FaceLabels","on","FaceAlpha",0.5)
axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material Properties
%---------------------------------------------------
% Apply different material properties to each region
thermalProperties(thermalmodel,"ThermalConductivity",148,'Face',1); % Silicon Die
thermalProperties(thermalmodel,"ThermalConductivity",400,'Face',2); % Copper Substrate
thermalProperties(thermalmodel,"ThermalConductivity",30,'Face',3); % Ceramic Isolation
thermalProperties(thermalmodel,"ThermalConductivity",400,'Face',4); % Copper Baseplate

%---------------------------------------------------
% Initial Conditions
%---------------------------------------------------
thermalIC(thermalmodel,20);

%---------------------------------------------------
% Boundary Conditions
%---------------------------------------------------
% Adiabatic boundary conditions (no heat flux)
thermalBC(thermalmodel,"Edge",1:12,'HeatFlux',0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',width/100,'Hmin',width/100,'Hgrad',1.0);
internalHeatSource(thermalmodel,@timeVaryingHeatSource,'Face',1); % Apply heat source to the silicon die
tlist = 0:1:500; % Time steps
options = odeset('RelTol',1e-6,'AbsTol',1e-6,'MaxStep',1);
results = solve(thermalmodel,tlist,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = results.Temperature;
xyz = thermalmodel.Mesh.Nodes;
M = round(width / (width/100));
N = round(total_height / (width/100));
for k = -M:1:M
    for kk = -N:1:N
        [minD, idx] = min(sum(abs(xyz-[width/100*k; width/100*kk].*ones(size(xyz))),2));
        xy(kk+N+1,k+M+1) = minD;
        T1(:,kk+N+1,k+M+1) = T(idx, :);
    end
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the time-varying heat source function
function q = timeVaryingHeatSource(region, state)
    q_dot_max = 1e6; % Maximum heat source value
    step_time = 100; % Time at which the step occurs
    
    % Initialize the heat source vector
    q = zeros(1, numel(region.x));
    
    % Apply the heat source to the entire region
    if state.time >= step_time
        q(:) = q_dot_max; % Heat source is q_dot_max after the step time
    else
        q(:) = q_dot_max / 10;
    end

    disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(q(1))]);
end