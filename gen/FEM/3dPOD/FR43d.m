%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: FR43d                                                             %
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
% Dimensions
M = 51;
N = 21;
K = 16;
l = 50e-3;
h = 3e-3;
b = 20e-3;

% Define semiconductor dimensions
swWidth = 8e-3; % 5 mm
swLength = 5e-3; % 8 mm
swHeight = 0.4e-3; % 0.5 mm

% Material layer thicknesses
h_Fr = 1.6e-3;
h_Ga = 0.4e-3;

% Material Properties
matK = [10, 4, 50, 0.014];
matRho = [1800, 3300, 2500, 1.3];
matCp = [900, 1500, 750, 1];

% Mesh
dx = 200e-6;

% Losses
Pv = 10;
Vol_Sw = swWidth*swLength*swHeight;
q = Pv / Vol_Sw;

% Load case
Ta = 35;
hc = 1500;
Tinit = 35;
Tend = 50;
dt = 0.5;
tlist = 0:dt:Tend-dt;

% Settings
plotting = 1;
saving = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry Definition (3D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Z levels
z0 = 0;
z1 = z0 + h_Ga;
z2 = z1 + h_Fr;
z3 = z2 + swHeight;

% Create Model
gm = fegeometry(multicuboid(l,b,[h_Ga h_Fr],ZOffset=[z0 z1]));

% Add semiconductors next to each other on top of the copper surface
sw1 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z2));
sw1 = translate(sw1, [-15e-3, -4e-3, 0]);
gm = union(gm, sw1, "KeepBoundaries",true);

sw2 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z2));
sw2 = translate(sw2, [-6e-3, 4e-3, 0]);
gm = union(gm, sw2, "KeepBoundaries",true);

sw3 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z2));
sw3 = translate(sw3, [6e-3, 4e-3, 0]);
gm = union(gm, sw3, "KeepBoundaries",true);

sw4 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z2));
sw4 = translate(sw4, [15e-3, -4e-3, 0]);
gm = union(gm, sw4, "KeepBoundaries",true);

% Generate Air
air = fegeometry(multicuboid(l,b,2*swHeight, 'ZOffset', z2));
air = subtract(air, sw1);
air = subtract(air, sw2);
air = subtract(air, sw3);
air = subtract(air, sw4);
gm = union(gm, air, "KeepBoundaries",true);

% Generate Model
thermalmodel = femodel(AnalysisType="thermalTransient");
thermalmodel.Geometry = gm;
pdegplot(thermalmodel, "CellLabels", "on", "FaceLabels", "on", "FaceAlpha", 0.5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define materials by face using the materialProperties property
thermalmodel.MaterialProperties(7) = materialProperties( ...
    "ThermalConductivity", matK(1), ...
    "MassDensity", matRho(1), ...
    "SpecificHeat", matCp(1));

thermalmodel.MaterialProperties(6) = materialProperties( ...
    "ThermalConductivity", matK(2), ...
    "MassDensity", matRho(2), ...
    "SpecificHeat", matCp(2));

thermalmodel.MaterialProperties([2,3,4,5]) = materialProperties( ...
    "ThermalConductivity", matK(3), ...
    "MassDensity", matRho(3), ...
    "SpecificHeat", matCp(3));

thermalmodel.MaterialProperties(1) = materialProperties( ...
    "ThermalConductivity", matK(4), ...
    "MassDensity", matRho(4), ...
    "SpecificHeat", matCp(4));


% Initial Condition
thermalmodel.CellIC = cellIC(Temperature=Tinit);

% Boundary Conditions
thermalmodel.FaceLoad(31) = faceLoad(ConvectionCoefficient=hc, AmbientTemperature=Ta);

% Heat Source (only on Cu face 5 and 7 as example)
thermalmodel.CellLoad([2,3,4,5]) = cellLoad(Heat=q); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thermalmodel = generateMesh(thermalmodel,'Hmax',5*dx,'Hmin',dx);
pdemesh(thermalmodel);
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
dz = (h)/(K-1);

%---------------------------------------------------
% Internal Heat 
%---------------------------------------------------
Q = zeros(size(T));
for i = 1:size(T,1)
    if results.Mesh.Nodes(3,i) >= z2 && results.Mesh.Nodes(3,i) <= z3
        isAtSwitch = checkSwitchPosition(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));
        if any(isAtSwitch)
            Q(i,:) = q;
        end
    else
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
    % Position
    isAtSwitch = checkSwitchPosition(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));

    % Silicon
    if results.Mesh.Nodes(3,i) >= z2 && results.Mesh.Nodes(3,i) <= z3  && any(isAtSwitch)
        Cp(i,:) = matCp(3);
        k(i,:) = matK(3);
        rho(i,:) = matRho(3);

    % Air
    elseif results.Mesh.Nodes(3,i) >= z2
        Cp(i,:) = matCp(4);
        k(i,:) = matK(4);
        rho(i,:) = matRho(4);

    % FR4
    elseif results.Mesh.Nodes(3,i) >= z1
        Cp(i,:) = matCp(1);
        k(i,:) = matK(1);
        rho(i,:) = matRho(1);

    % TIM    
    elseif results.Mesh.Nodes(3,i) < z1
        Cp(i,:) = matCp(2);
        k(i,:) = matK(2);
        rho(i,:) = matRho(2);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Define Variables 
%---------------------------------------------------
geo = results.Mesh.Nodes';
mesh = results.Mesh;
Lx = l;
Ly = b;
Lz = h;
r = Tinit * ones(size(T))';
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
vars_to_save = {'Cp', 'dx', 'dy', 'dz', 'geo', 'k', 'Lx', 'Ly', 'Lz', 'r', ...
                'rho', 't', 'Ts', 'X', 'y', 'Ta', 'hc', 'fl', 'mesh'};

% Save Vars
if saving == 1
    save('data.mat', vars_to_save{:});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotting == 1
    figure;
    for i = 1:floor(length(tlist)/10)
        pdeplot3D(results.Mesh, 'ColorMapData', results.Temperature(:, i*10));
        title(['Temperature at t = ', num2str(tlist(i*10)), ' s']);
        colorbar;
        drawnow;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function isAtSwitch = checkSwitchPosition(x, y)
    % Define semiconductor dimensions
    swWidth = 8e-3;  % 8 mm
    swLength = 5e-3; % 5 mm

    % Define switch positions (centered at the origin for simplicity)
    switchPositions = [
        -15e-3, -4e-3;  % Position of switch 1
        -6e-3, 4e-3;    % Position of switch 2
        6e-3, 4e-3;     % Position of switch 3
        15e-3, -4e-3     % Position of switch 4
    ];

    % Initialize the output variable
    isAtSwitch = false(size(switchPositions, 1), 1);

    % Check if the point (x, y) is within the bounds of any switch
    for i = 1:size(switchPositions, 1)
        % Calculate the bounds for the current switch
        xMin = switchPositions(i, 1) - swWidth / 2;
        xMax = switchPositions(i, 1) + swWidth / 2;
        yMin = switchPositions(i, 2) - swLength / 2;
        yMax = switchPositions(i, 2) + swLength / 2;

        % Check if the point is within the bounds of the switch
        if (x >= xMin && x <= xMax) && ...
           (y >= yMin && y <= yMax)
            isAtSwitch(i) = true; % Mark the switch as present
        end
    end
end