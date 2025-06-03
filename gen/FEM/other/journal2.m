%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: journal2                                                          %
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
%---------------------------------------------------
% Dimension 
%---------------------------------------------------
M = 33;                                                                     % Number of X points
N = 17;                                                                      % Number of Y points
l = 1.44e-6;                                                                   % x length (m)
b = 720e-9;                                                                  % width (m)
h = 360e-9;                                                                  % y length (m)
dx = 20e-9;                                                                 % internal FEM resolution (m)
Vol = 8*l*b*h;                                                              % Busbar volume (m³)

%---------------------------------------------------
% Material 
%---------------------------------------------------
matK = [400, 0.17];                                                          % Thermal conductivity (W/mK)
matRho = [8933, 2200];                                                      % Material density (kg/m3)
matCp = [380, 1000];                                                         % Specific heat capacity (J/kgK)

%---------------------------------------------------
% losses 
%---------------------------------------------------
q = 2e14;                                                                   % Volumetric heat generation (W/m3)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Ta = 26.85;                                                                    % ambient temperature (degC)
fl = 0;                                                                   % heat flux boundary (W/m2)
hc = 0;                                                                  % heat transfer coefficient (W/m2K)
Tinit = 26.85;                                                                 % Initial temperature (degC)
Tend = 20e-6;                                                                 % end value time (sec)
dt = 100e-9;                                                                     % sampling time (sec)
tlist = 0:dt:Tend-dt;                                                       % time vector (sec)

%---------------------------------------------------
% Settings
%---------------------------------------------------
plotting = 0;
saving = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thermalmodel = createpde("thermal","transient");
R1 = [3,4,0,b/2,b/2,0,b-b/2,b-b/2,b,b]';
R2 = [3,4, b/2, l, l, b/2,  b-b/2, b-b/2, b, b]';
R3 = [3,4, 0, l, l, 0,  0, 0, b-b/2, b-b/2]';

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
thermalProperties(thermalmodel,"ThermalConductivity",matK(1),'Face',1, ...
                               "MassDensity",matRho(1), ...
                               "SpecificHeat",matCp(1));
thermalProperties(thermalmodel,"ThermalConductivity",matK(2),'Face',[2,3], ...
                               "MassDensity",matRho(2), ...
                               "SpecificHeat",matCp(2));

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,Tinit);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% % Temperature
% thermalBC(thermalmodel,"Edge",[8],'Temperature',Ta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
internalHeatSource(thermalmodel,q,'Face',1);
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
for k = 1:M
    for kk = 1:N
        [minD, idx] = min(sum(abs(xyz-[dx*k; dy*kk].*ones(size(xyz))),1));
        xy(kk,k) = minD;
        T1(:,kk,k) = T(idx, :);
    end
end

%---------------------------------------------------
% Internal Heat 
%---------------------------------------------------
Q = zeros(size(T));
for i = 1:size(T,1)
    if results.Mesh.Nodes(2,i) >= b/2 && results.Mesh.Nodes(1,i) < b/2
        Q(i,:) = q;
    elseif results.Mesh.Nodes(1,i) == b/2 && results.Mesh.Nodes(2,i) > b/2
        % Q(i,:) = q/2;
    elseif results.Mesh.Nodes(2,i) == b/2 && results.Mesh.Nodes(1,i) < b/2
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
    if results.Mesh.Nodes(2,i) >= b/2 && results.Mesh.Nodes(1,i) < b/2
        Cp(i,:) = matCp(1);
        k(i,:) = matK(1);
        rho(i,:) = matRho(1);
    elseif results.Mesh.Nodes(1,i) == b/2 && results.Mesh.Nodes(2,i) > b/2
        % Cp(i,:) = (matCp(1) + matCp(2))/2;
        % k(i,:) = (matK(1) + matK(2))/2;
        % rho(i,:) = (matRho(1) + matRho(2))/2;
    elseif results.Mesh.Nodes(2,i) == b/2 && results.Mesh.Nodes(1,i) < b/2
        % Cp(i,:) = (matCp(1) + matCp(2))/2;
        % k(i,:) = (matK(1) + matK(2))/2;
        % rho(i,:) = (matRho(1) + matRho(2))/2;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define Variables 
%---------------------------------------------------
Cp = Cp;
dx = dx;
dy = dy;
geo = results.Mesh.Nodes';
k = k;
Lx = l;
Ly = b;
r = Tinit * ones(size(T))';
rho = rho;
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