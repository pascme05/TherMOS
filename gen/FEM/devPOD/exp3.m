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
l = 0.20;                                                                   % x length (m)
h = 0.20;                                                                   % y length (m)
dx = 0.01;                                                                 % internal FEM resolution (m)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Tinit = 20;                                                                 % Initial temperature (degC)
Tend = 2000;                                                                % end value time (sec)
q = 1000000;                                                                % Volumetric heat generation (W/m3)
dt = 10;                                                                    % sampling time (sec)
tlist = 0:dt:Tend-dt;                                                       % time vector (sec)

%---------------------------------------------------
% Material 
%---------------------------------------------------
% Region-1
matK1 = 210;                                                                  % Thermal conductivity (W/mK)
matRho1 = 2710;                                                              % Material density (kg/m3)
matCp1 = 900;      

% Region-2
matK2 = 400;                                                                  % Thermal conductivity (W/mK)
matRho2 = 8933;                                                              % Material density (kg/m3)
matCp2 = 380;      

%---------------------------------------------------
% Settings
%---------------------------------------------------
space_Q = 1;                                                                % 1) spatial temperature distribution                                                          
space_M = 1;
plotting = 0;
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
% Define the coordinates of the two regions
R1 = [3,4,-l,0,0,-l,h,h,-h,-h]'; % Region 1
R2 = [3,4,0,l,l,0,h,h,-h,-h]';   % Region 2
heat_source_region = [-0.2, 0.0, -0.2, 0.2];

% Combine the regions into a single geometry
gd = [R1, R2];
sf = 'R1+R2';
ns = char('R1','R2');
ns = ns';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalmodel,g)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Material 
%---------------------------------------------------
% Single Area
if space_M == 0
    thermalProperties(thermalmodel,"ThermalConductivity",matK1,'Face',[1,2], ...
                                   "MassDensity",matRho1, ...
                                   "SpecificHeat",matCp1);
% Double Area
else
    thermalProperties(thermalmodel,"ThermalConductivity",matK1,'Face',1, ...
                                   "MassDensity",matRho1, ...
                                   "SpecificHeat",matCp1);
    thermalProperties(thermalmodel,"ThermalConductivity",matK2,'Face',2, ...
                                   "MassDensity",matRho2, ...
                                   "SpecificHeat",matCp2);
end

%---------------------------------------------------
% Initial Conditions 
%---------------------------------------------------
thermalIC(thermalmodel,20);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% Adiabatic
thermalBC(thermalmodel,"Edge",[2,3,4,5,6],'Temperature',20);
        
% Constant Temperature
thermalBC(thermalmodel,"Edge",1,'Temperature',20); 

% % Convection
% thermalBC(thermalmodel,"Edge",1,'ConvectionCoefficient',1000, ...
%                                 'AmbientTemperature',20);

% % Heat Flux
% thermalBC(thermalmodel,"Edge",1,'HeatFlux',100);


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
% Spatial Heat Source
if space_Q == 1
    internalHeatSource(thermalmodel,q,'Face',1);
    
% Constant Heat Source
else
    internalHeatSource(thermalmodel,q,'Face',[1,2]);
end

%---------------------------------------------------
% Mesh 
%---------------------------------------------------
generateMesh(thermalmodel,'Hmax',dx,'Hmin',dx,'Hgrad',1.0);

%---------------------------------------------------
% Solve 
%---------------------------------------------------
results = solve(thermalmodel,tlist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Temperatures
%---------------------------------------------------
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

%---------------------------------------------------
% Heat Generation
%---------------------------------------------------
Q = zeros(size(T))';
% Spatial Heat Source
if space_Q == 0
    Q = q * ones(size(Q));
    
% Spatial Parameters
else
    for i = 1:numel(results.Mesh.Nodes(1,:))
        if results.Mesh.Nodes(1,i) >= heat_source_region(1) && results.Mesh.Nodes(1,i) < heat_source_region(2) && ...
           results.Mesh.Nodes(2,i) >= heat_source_region(3) && results.Mesh.Nodes(2,i) <= heat_source_region(4)
            Q(:, i) = q;
        elseif results.Mesh.Nodes(1,i) >= heat_source_region(1) && results.Mesh.Nodes(1,i) <= heat_source_region(2) && ...
           results.Mesh.Nodes(2,i) >= heat_source_region(3) && results.Mesh.Nodes(2,i) <= heat_source_region(4)
            Q(:, i) = q/2;
        end
    end
    
end

%---------------------------------------------------
% Material Properties
%---------------------------------------------------
matCp = ones(length(T),1);
matK = ones(length(T),1);
matRho = ones(length(T),1);
if space_M == 0
    matCp = matCp1 * ones(length(T),1);
    matK = matK1 * ones(length(T),1);
    matRho = matRho1 * ones(length(T),1);
else
    for i = 1:numel(results.Mesh.Nodes(1,:))
        if results.Mesh.Nodes(1,i) >= heat_source_region(1) && results.Mesh.Nodes(1,i) < heat_source_region(2) && ...
           results.Mesh.Nodes(2,i) >= heat_source_region(3) && results.Mesh.Nodes(2,i) <= heat_source_region(4)
            matCp(i,1) = matCp1;
            matK(i,1) = matK1;
            matRho(i,1) = matRho1;
        elseif results.Mesh.Nodes(1,i) >= heat_source_region(1) && results.Mesh.Nodes(1,i) <= heat_source_region(2) && ...
           results.Mesh.Nodes(2,i) >= heat_source_region(3) && results.Mesh.Nodes(2,i) <= heat_source_region(4)
            matCp(i,1) = (matCp1 + matCp2)/2;
            matK(i,1) = (matK1 + matK2)/2;
            matRho(i,1) = (matRho1 + matRho2)/2;
            % matCp(i,1) = matCp1;
            % matK(i,1) = matK1;
            % matRho(i,1) = matRho1;
            matCp(i,1) = matCp2;
            matK(i,1) =  matK2;
            matRho(i,1) = matRho2;
        else
            matCp(i,1) = matCp2;
            matK(i,1) = matK2;
            matRho(i,1) = matRho2;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define Variables 
%---------------------------------------------------
Cp = matCp;
dx = dx;
dy = dy;
geo = results.Mesh.Nodes';
k = matK;
Lx = 2*l;
Ly = 2*h;
r = Tinit * ones(size(T))';
rho = matRho;
t = tlist;
Ts = dt;
X = Q;
y = T';

%---------------------------------------------------
% Save Variables 
%---------------------------------------------------
% Define Vars
vars_to_save = {'Cp', 'dx', 'dy', 'geo', 'k', 'Lx', 'Ly', 'r', 'rho', 't', ...
                'Ts', 'X', 'y'};

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

