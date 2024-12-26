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
M = 37;
N = 17;
l = 1.44e-6; % 1.44 um
h = 720e-9; % 720 nm
small_length = 360e-9; % 360 nm
small_height = 360e-9; % 360 nm
q = 2e13;
dx = 45e-9;
dt = 10e-9;
Tend = 20e-6;
tlist = 0:dt:Tend-dt;
T0 = 26.85;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Init 
%---------------------------------------------------
thermalmodel = createpde("thermal","transient");

%---------------------------------------------------
% Edges 
%---------------------------------------------------
% Define Region
R1 = [3,4,0,l,l,0,0,0,h,h]';
R2 = [3,4,0,small_length,small_length,0,h-small_height,h-small_height,h,h]';


% Combine the regions into a single geometry
gd = [R1, R2];
sf = 'R1+R2';
ns = char('R1','R2');
ns = ns';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalmodel,g)
pdegplot(thermalmodel,"EdgeLabels","on","FaceLabels","on","FaceAlpha",0.5)
axis equal

% %---------------------------------------------------
% % Edges 
% %---------------------------------------------------
% heat_source_region_x = [0, 0.2];
% heat_source_region_y = [0, 0.2];

%---------------------------------------------------
% Mesh 
%---------------------------------------------------
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
pdemesh(thermalmodel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
% Spatial Parameters
thermalProperties(thermalmodel,"ThermalConductivity",0.17,'Face',2, ...
                               "MassDensity",2200, ...
                               "SpecificHeat",1000);
thermalProperties(thermalmodel,"ThermalConductivity",400,'Face',1, ...
                               "MassDensity",8933, ...
                               "SpecificHeat",380);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,T0);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
thermalBC(thermalmodel,"Edge",[1,2,5,6,7,8],'HeatFlux',0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Init 
%---------------------------------------------------
thermalmodel.SolverOptions.RelativeTolerance = 1E-5;
thermalmodel.SolverOptions.AbsoluteTolerance = 1E-9;

%---------------------------------------------------
% Source 
%---------------------------------------------------
internalHeatSource(thermalmodel,q,'Face',2);

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
for k = 1:M
    for kk = 1:N
        [minD, idx] = min(sum(abs(xyz-[dx*k; dy*kk].*ones(size(xyz))),1));
        xy(kk,k) = minD;
        T1(:,kk,k) = T(idx, :);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% [qx,qy] = evaluateHeatFlux(results);
% for i = 1:floor(length(tlist) / 10)
%     subplot(2,1,1)
%     pdeplot(thermalmodel,"XYData",results.Temperature(:,floor(i*10)))
%     subplot(2,1,2)
%     pdeplot(thermalmodel,'FlowData',[qx qy])
%     pause(0.001)
% end

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
% %---------------------------------------------------
% % Define the spatial heat source
% %---------------------------------------------------
% function Q = heatSourceSpace(region,state)
%     % Init
%     q_dot_max = 1e6; % Maximum heat source value
%     Q = zeros(1,numel(region.x));
% 
%     % Check for NaNs
%     if(isnan(state.time))
%       Q(1,:) = NaN;
%       return
%     end
% 
%     % Define the region where you want to apply the heat source
%     heat_source_region = [0, 0.2, 0, 0.2];
% 
%     % Spatial Distribution
%     for i = 1:numel(region.x)
%         if region.x(i) >= heat_source_region(1) && region.x(i) <= heat_source_region(2) && ...
%            region.y(i) >= heat_source_region(3) && region.y(i) <= heat_source_region(4)
%             Q(1,i) = q_dot_max; % Heat source is q_dot_max after the step time
%         end
%     end
% 
%     % Debug
%     % disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(Q(1))]);
% end

