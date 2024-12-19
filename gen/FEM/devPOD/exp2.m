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
% Material 
%---------------------------------------------------
matK = 29;                                                                  % Thermal conductivity (W/mK)
matRho = 1300;                                                              % Material density (kg/m3)
matCp = 800;                                                                % Specific heat capacity (J/kgK)

%---------------------------------------------------
% Load Case 
%---------------------------------------------------
Tinit = 20;                                                                 % Initial temperature (degC)
Tend = 2000;                                                                % end value time (sec)
q = 1000000;                                                                % volumetric heat generation (W/m3)
step_time = 100;                                                            % step time (sec)
f = 0.01;                                                                   % frequency sine (Hz)
dt = 10;                                                                    % sampling time (sec)
tlist = 0:dt:Tend-dt;                                                       % time vector (sec)

%---------------------------------------------------
% Settings
%---------------------------------------------------
waveform = 2;                                                               % 1) step, 2) sine
plotting = 0;
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
thermalIC(thermalmodel,Tinit);

%---------------------------------------------------
% Boundary Conditions 
%---------------------------------------------------
% Adiabatic
thermalBC(thermalmodel,"Edge",[1,2,3],'Temperature',20);
        
% % Constant Temperature
% thermalBC(thermalmodel,"Edge",4,'Temperature',20); 

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
% Step
if waveform == 1
    internalHeatSource(thermalmodel,@heatSourceStep,'Face',1);

% Sine
elseif waveform == 2
    internalHeatSource(thermalmodel,@heatSourceSine,'Face',1);
end

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
for i = 1:length(tlist)
    if tlist(i) < step_time
        if waveform == 1
            Q(i,:) = q/10;
        elseif waveform == 2
            Q(i,:) = q * abs(sin(2*pi*f*tlist(i)));
        end
    else
        if waveform == 1
            Q(i,:) = q;
        elseif waveform == 2
            Q(i,:) = q * abs(sin(2*pi*f*tlist(i)));
        end
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