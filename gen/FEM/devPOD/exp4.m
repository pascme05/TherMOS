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
M = 33;
N = 17;
l = 1.44e-6; % 1.44 um
h = 720e-9; % 720 nm
small_length = 360e-9; % 360 nm
small_height = 360e-9; % 360 nm
dx = 45e-9;

%---------------------------------------------------
% Material 
%---------------------------------------------------
matK = [0.17, 400];                                                          % Thermal conductivity (W/mK)
matRho = [2200, 8933];                                                      % Material density (kg/m3)
matCp = [1000, 380];                                                         % Specific heat capacity (J/kgK)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Ta = 26.85;                                                                    % ambient temperature (degC)
fl = 100;                                                                   % heat flux boundary (W/m2)
hc = 1000;                                                                  % heat transfer coefficient (W/m2K)
Tinit = 26.85;                                                                 % Initial temperature (degC)
Tend = 20e-6;                                                                % end value time (sec)
q = 2e13;                                                                    % Volumetric heat generation (W/m3)
dt = 10e-9;                                                                    % sampling time (sec)
tlist = 0:dt:Tend-dt;                                                       % time vector (sec)

%---------------------------------------------------
% Settings
%---------------------------------------------------
setting = 0;                                                                % 0) step, 1) sine
plotting = 1;
saving = 1;

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
% Define sub-regions
R1 = [3, 4, 0, small_length, small_length, 0, 0, 0, small_height, small_height]'; % Top-left square
R2 = [3, 4, small_length, l, l, small_length, 0, 0, small_height, small_height]'; % Top-right rectangle
R3 = [3, 4, 0, l, l, 0, small_height, small_height, h, h]';                       % Bottom rectangle

% Combine the regions into a single geometry
gd = [R3, R1, R2]; % Geometry description matrix
sf = 'R3+R1+R2';   % Set formula to union the regions
ns = char('R3', 'R1', 'R2'); % Names of the subregions
ns = ns';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalmodel,g)
pdegplot(thermalmodel,"EdgeLabels","on","FaceLabels","on","FaceAlpha",0.5)
axis equal

% Heat Source Region
heat_source_region_x = [0.0, 360e-9];
heat_source_region_y = [0.0, 360e-9];

%---------------------------------------------------
% Mesh 
%---------------------------------------------------
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);
% pdemesh(thermalmodel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
% Spatial Parameters
thermalProperties(thermalmodel,"ThermalConductivity",matK(1), ...
                               "MassDensity",matRho(1), ...
                               "SpecificHeat",matCp(1), "Face",[2,3]);
thermalProperties(thermalmodel,"ThermalConductivity",matK(2), ...
                               "MassDensity",matRho(2), ...
                               "SpecificHeat",matCp(2),'Face',1);

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,Tinit);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
thermalBC(thermalmodel,"Edge",[1,2,3,4,5,6,7],'HeatFlux',0);


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
% Step
if setting == 0
    internalHeatSource(thermalmodel,q,'Face',1);
    q = q * ones(length(tlist),1);
end

% Sine
if setting == 1
    internalHeatSource(thermalmodel,@heatSourceSine,'Face',1);
    q = 10 * q * (sin(pi/2*1e6*tlist) + 1)';
end

%---------------------------------------------------
% Solve 
%---------------------------------------------------
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
dy = (h)/(N-1);
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
    if results.Mesh.Nodes(1,i) >= heat_source_region_x(1) && results.Mesh.Nodes(1,i) < heat_source_region_x(2) && ...
       results.Mesh.Nodes(2,i) >= heat_source_region_y(1) && results.Mesh.Nodes(2,i) < heat_source_region_y(2)
        Q(i,:) = q;
    elseif results.Mesh.Nodes(1,i) <= heat_source_region_x(2) && results.Mesh.Nodes(2,i) <= heat_source_region_y(2)
        % Q(i,:) = q/2;
    end
end

%---------------------------------------------------
% Spatial Parameters
%---------------------------------------------------
Cp = matCp(1) * ones(length(T),1);
k = matK(1) * ones(length(T),1);
rho = matRho(1) * ones(length(T),1);
for i = 1:size(T,1)
    if results.Mesh.Nodes(1,i) >= heat_source_region_x(1) && results.Mesh.Nodes(1,i) < heat_source_region_x(2) && ...
       results.Mesh.Nodes(2,i) >= heat_source_region_y(1) && results.Mesh.Nodes(2,i) < heat_source_region_y(2)
        Cp(i,:) = matCp(2);
        k(i,:) = matK(2);
        rho(i,:) = matRho(2);
    elseif results.Mesh.Nodes(1,i) <= heat_source_region_x(2) && results.Mesh.Nodes(2,i) <= heat_source_region_y(2)
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
Ly = h;
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
        pdeplot(thermalmodel,"XYData",results.Temperature(:,floor(i*10))-Tinit)
        subplot(2,1,2)
        pdeplot(thermalmodel,'FlowData',[qx qy])
        pause(0.001)
    end
    
    %---------------------------------------------------
    % Temperatures
    %---------------------------------------------------
    figure
    for i = 1:floor(length(tlist) / 10)
        dT1 = gradient(squeeze(T1(floor(i*10),:,:))-Tinit, dx, dy);
        subplot(2,1,1)
        contourf(squeeze(T1(floor(i*10),:,:))-Tinit)
        subplot(2,1,2)
        contourf(dT1)
        pause(0.05)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define the time-varying heat source (sine)
%---------------------------------------------------
function Q = heatSourceSine(location,state)
    % Init
    q_dot_max = 2e13; % Maximum heat source value
    Q = zeros(1,numel(location.x));
    
    % Check for NaNs
    if(isnan(state.time))
      Q(1,:) = NaN;
      return
    end

    % Step Function
    Q(1,:) = q_dot_max * 10 * (sin(pi/2*1e6*state.time) + 1);

    % Debug
    % disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(Q(1))]);
end

