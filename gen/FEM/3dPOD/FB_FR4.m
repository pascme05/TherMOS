%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: FB_FR4                                                            %
% Date: 21.50.2026                                                        %
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
N = 51;
K = 19;
l = 50e-3;
h = 3.6e-3;
b = 50e-3;

% Define semiconductor dimensions
swWidth = 12e-3; 
swLength = 10e-3; 
swHeight = 1.5e-3; 

% Material layer thicknesses
h_Fr = 1.6e-3;
h_Ga = 0.5e-3;

% Material Properties
matK = [0.3, 4, 20];
matRho = [1800, 3300, 2500];
matCp = [900, 1500, 750];

% Mesh
dx = 200e-6;

% Losses
Pv = 10;
Vol_Sw = swWidth*swLength*swHeight;
q = Pv / Vol_Sw;
q_asy11 = 1.3;
q_asy12 = 1.3;
q_asy21 = 1.0;
q_asy22 = 1.0;

% Load case
Ta = 35;
hc = 1500;
Tinit = 35;
Tend = 100;
dt = 1;
tlist = 0:dt:Tend-dt;

% Settings
plotting = 1;
saving = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry Definition (3D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Z levels (keep existing z0, z1, z2, z3)
z0 = 0;
z1 = z0 + h_Fr;      % top of FR4
z2 = z1 + swHeight;  % top of semiconductor
z3 = z2 + h_Ga;      % top of gapfiller / air

% Create single FR4 board layer (thickness h_Fr)
fr4 = fegeometry(multicuboid(l, b, h_Fr, 'ZOffset', z0));

% Create semiconductors positioned on top of FR4 (centered positions as before)
sw1 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z1));
sw1 = translate(sw1, [-10e-3, -10e-3, 0]);

sw2 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z1));
sw2 = translate(sw2, [10e-3, -10e-3, 0]);

sw3 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z1));
sw3 = translate(sw3, [-10e-3, 10e-3, 0]);

sw4 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z1));
sw4 = translate(sw4, [10e-3, 10e-3, 0]);

% Create gapfiller that starts at top of FR4 (z1) and goes up to z3
gapfiller_full = fegeometry(multicuboid(l, b, z3 - z1, 'ZOffset', z1));

% Subtract semiconductor volumes from gapfiller so switches are embedded inside
gapfiller = subtract(gapfiller_full, sw1);
gapfiller = subtract(gapfiller, sw2);
gapfiller = subtract(gapfiller, sw3);
gapfiller = subtract(gapfiller, sw4);

% Combine FR4, gapfiller and semiconductors into one geometry and keep boundaries
gm = union(fr4, gapfiller, "KeepBoundaries", true);
gm = union(gm, sw1, "KeepBoundaries", true);
gm = union(gm, sw2, "KeepBoundaries", true);
gm = union(gm, sw3, "KeepBoundaries", true);
gm = union(gm, sw4, "KeepBoundaries", true);

% Plot geometry labels for verification
thermalmodel = femodel(AnalysisType="thermalTransient");
thermalmodel.Geometry = gm;
pdegplot(thermalmodel, "CellLabels", "on", "FaceLabels", "off", "FaceAlpha", 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define materials by face using the materialProperties property
thermalmodel.MaterialProperties(6) = materialProperties( ...
    "ThermalConductivity", matK(1), ...
    "MassDensity", matRho(1), ...
    "SpecificHeat", matCp(1));

thermalmodel.MaterialProperties(5) = materialProperties( ...
    "ThermalConductivity", matK(2), ...
    "MassDensity", matRho(2), ...
    "SpecificHeat", matCp(2));

thermalmodel.MaterialProperties([1,2,3,4]) = materialProperties( ...
    "ThermalConductivity", matK(3), ...
    "MassDensity", matRho(3), ...
    "SpecificHeat", matCp(3));


% Initial Condition
thermalmodel.CellIC = cellIC(Temperature=Tinit);

% Boundary Conditions
thermalmodel.FaceLoad(26) = faceLoad(ConvectionCoefficient=hc, AmbientTemperature=Ta);

% Heat Source (only on Cu face 5 and 7 as example)
thermalmodel.CellLoad(1) = cellLoad(Heat=q_asy11*q);
thermalmodel.CellLoad(2) = cellLoad(Heat=q_asy12*q);
thermalmodel.CellLoad(3) = cellLoad(Heat=q_asy21*q);
thermalmodel.CellLoad(4) = cellLoad(Heat=q_asy22*q);


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
    if results.Mesh.Nodes(3,i) >= z1 && results.Mesh.Nodes(3,i) <= z2
        isAtSwitch = checkSwitchPosition11(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));
        if any(isAtSwitch)
            Q(i,:) = q_asy11*q;
        end

        isAtSwitch = checkSwitchPosition12(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));
        if any(isAtSwitch)
            Q(i,:) = q_asy12*q;
        end

        isAtSwitch = checkSwitchPosition21(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));
        if any(isAtSwitch)
            Q(i,:) = q_asy21*q;
        end

        isAtSwitch = checkSwitchPosition22(results.Mesh.Nodes(1,i), results.Mesh.Nodes(2,i));
        if any(isAtSwitch)
            Q(i,:) = q_asy22*q;
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
    if results.Mesh.Nodes(3,i) >= z1 && results.Mesh.Nodes(3,i) <= z2  && any(isAtSwitch)
        Cp(i,:) = matCp(3);
        k(i,:) = matK(3);
        rho(i,:) = matRho(3);

    % TIM
    elseif results.Mesh.Nodes(3,i) >= z1
        Cp(i,:) = matCp(2);
        k(i,:) = matK(2);
        rho(i,:) = matRho(2);

    % FR4    
    elseif results.Mesh.Nodes(3,i) < z1
        Cp(i,:) = matCp(1);
        k(i,:) = matK(1);
        rho(i,:) = matRho(1);
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
    swWidth = 12e-3;  
    swLength = 10e-3; 

    % Define switch positions (centered at the origin for simplicity)
    switchPositions = [
        -10e-3, -10e-3;  % Position of switch 1
        10e-3, -10e-3;    % Position of switch 2
        -10e-3, 10e-3;     % Position of switch 3
        10e-3, 10e-3     % Position of switch 4
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

function isAtSwitch = checkSwitchPosition11(x, y)
% Define semiconductor dimensions
swWidth = 12e-3;  
swLength = 10e-3; 

% Define switch positions (centered at the origin for simplicity)
switchPositions = [
    10e-3, 10e-3;     % Position of switch 11
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

function isAtSwitch = checkSwitchPosition12(x, y)
% Define semiconductor dimensions
swWidth = 12e-3;  
swLength = 10e-3; 

% Define switch positions (centered at the origin for simplicity)
switchPositions = [
    -10e-3, 10e-3     % Position of switch 12
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

function isAtSwitch = checkSwitchPosition21(x, y)
% Define semiconductor dimensions
swWidth = 12e-3;  
swLength = 10e-3; 

% Define switch positions (centered at the origin for simplicity)
switchPositions = [
    10e-3, -10e-3;  % Position of switch 21
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

function isAtSwitch = checkSwitchPosition22(x, y)
% Define semiconductor dimensions
swWidth = 12e-3;  
swLength = 10e-3; 

% Define switch positions (centered at the origin for simplicity)
switchPositions = [
    -10e-3, -10e-3;    % Position of switch 22
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