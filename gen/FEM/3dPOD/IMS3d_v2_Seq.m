%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: IMS3d                                                             %
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
K = 64;
l = 210e-3;
h = 3150e-6+0.5e-3;
b = 115e-3;

% Define semiconductor dimensions
swWidth = 10e-3; 
swLength = 10e-3; 
swHeight = 2.0e-3;

% Position Full bridges
l_FB = 75e-3;
b_FB = 30e-3;
dx_FB_Cu = 5e-3;
xPos_FB = 52.5e-3;
yPos_FB = 25e-3;

% Cooling
l_fix = sqrt(85e-6);

% Material layer thicknesses
h_Cu = 100e-6;
h_Di = 50e-6;
h_Al = 1.5e-3;
h_Ga = 0.5e-3;

% Material Properties
matK =   [400,  2,    90,   4,    30,   0.014];
matRho = [8933, 2200, 2680, 3300, 2000, 1.3];
matCp =  [380,  800,  1000, 1500, 500,  1];

% Mesh
dx = 2000e-6;

% Losses
Pv = [20, 30, 60, 20, 60, 30, 20];
Pv_Cu = [90, 135, 270, 90, 270, 135, 90];
Vol_Sw = swWidth*swLength*swHeight;
Vol_Cu = (l_FB+dx_FB_Cu)*(b_FB+dx_FB_Cu)*h_Cu * 2;
q_Cu = Pv_Cu / Vol_Cu;
q_sw = Pv / Vol_Sw;

% Cooling
hc = 2000;
hfix = 2700;
Ta = 35;
Tinit = 35;

% Load case
Tend = 900;
dt = 0.5;
dt0 = 0.001;
dt1 = 0.01;
dt2 = 0.1;
dt3 = 1;
dt4 = 10;
tlist = 0:dt:Tend-dt;
% tlist = [0:dt0:0.02-dt0, 0.02:dt1:0.2-dt1, 0.2:dt2:2-dt2, 2:dt3:60-dt3, 60:dt4:Tend];

% Settings
plotting = 1;
saving = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry Definition (3D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Z levels
z0 = 0;
z1 = z0 + h_Ga;
z2 = z1 + h_Al;
z3 = z2 + h_Di;
z4 = z3 + h_Cu;

% Create PCB Model
% gm = fegeometry(multicuboid(l,b,[h_Ga h_Al h_Di h_Cu],ZOffset=[z0 z1 z2 z3]));
gm = fegeometry(multicuboid(l,b,[h_Al h_Di],ZOffset=[z1 z2]));

% Generate Copper
cu1 = fegeometry(multicuboid(l_FB+dx_FB_Cu,b_FB+dx_FB_Cu,h_Cu, 'ZOffset', z3));
cu1 = translate(cu1, [-xPos_FB, -yPos_FB-5e-3/2, 0]);
gm = union(gm, cu1, "KeepBoundaries",true);

cu2 = fegeometry(multicuboid(l_FB+dx_FB_Cu,b_FB+dx_FB_Cu,h_Cu, 'ZOffset', z3));
cu2 = translate(cu2, [+xPos_FB, -yPos_FB-5e-3/2, 0]);
gm = union(gm, cu2, "KeepBoundaries",true);

cu3 = fegeometry(multicuboid(l,b,h_Cu,ZOffset=z3));
cu3 = subtract(cu3, cu1);
cu3 = subtract(cu3, cu2);

gm = union(gm, cu3, "KeepBoundaries",true);


% Generate Gapfiller
gap1 = fegeometry(multicuboid(l_FB,b_FB,h_Ga, 'ZOffset', z0));
gap1 = translate(gap1, [-xPos_FB-5e-3/2, -yPos_FB-5e-3/2, 0]);
gm = union(gm, gap1, "KeepBoundaries",true);

gap2 = fegeometry(multicuboid(l_FB,b_FB,h_Ga, 'ZOffset', z0));
gap2 = translate(gap2, [+xPos_FB+5e-3/2, -yPos_FB-5e-3/2, 0]);
gm = union(gm, gap2, "KeepBoundaries",true);

gap2a = fegeometry(multicuboid(l_fix,l_fix,h_Ga, 'ZOffset', z0));
gap2a = translate(gap2a, [-l/2+l_fix, -b/2+l_fix, 0]);
gm = union(gm, gap2a, "KeepBoundaries",true);

gap2b = fegeometry(multicuboid(l_fix,l_fix,h_Ga, 'ZOffset', z0));
gap2b = translate(gap2b, [l/2-l_fix, -b/2+l_fix, 0]);
gm = union(gm, gap2b, "KeepBoundaries",true);

gap2c = fegeometry(multicuboid(l_fix,l_fix,h_Ga, 'ZOffset', z0));
gap2c = translate(gap2c, [-l/2+l_fix, b/2-l_fix, 0]);
gm = union(gm, gap2c, "KeepBoundaries",true);

gap2d = fegeometry(multicuboid(l_fix,l_fix,h_Ga, 'ZOffset', z0));
gap2d = translate(gap2d, [l/2-l_fix, b/2-l_fix, 0]);
gm = union(gm, gap2d, "KeepBoundaries",true);

gap2e = fegeometry(multicuboid(l_fix,5*l_fix,h_Ga, 'ZOffset', z0));
gap2e = translate(gap2e, [0, -b/2+3*l_fix, 0]);
gm = union(gm, gap2e, "KeepBoundaries",true);

gap2f = fegeometry(multicuboid(l_fix,l_fix,h_Ga, 'ZOffset', z0));
gap2f = translate(gap2f, [0, b/2-l_fix, 0]);
gm = union(gm, gap2f, "KeepBoundaries",true);

gap3 = fegeometry(multicuboid(l,b,h_Ga,ZOffset=z0));
gap3 = subtract(gap3, gap1);
gap3 = subtract(gap3, gap2);
gap3 = subtract(gap3, gap2);
gap3 = subtract(gap3, gap2a);
gap3 = subtract(gap3, gap2b);
gap3 = subtract(gap3, gap2c);
gap3 = subtract(gap3, gap2d);
gap3 = subtract(gap3, gap2e);
gap3 = subtract(gap3, gap2f);
gm = union(gm, gap3, "KeepBoundaries",true);

% FB-1
sw1 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw1 = translate(sw1, [-55e-3/2-xPos_FB, -4e-3-yPos_FB, 0]);
gm = union(gm, sw1, "KeepBoundaries",true);

sw2 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw2 = translate(sw2, [-10e-3-xPos_FB, 4e-3-yPos_FB, 0]);
gm = union(gm, sw2, "KeepBoundaries",true);

sw3 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw3 = translate(sw3, [10e-3-xPos_FB, 4e-3-yPos_FB, 0]);
gm = union(gm, sw3, "KeepBoundaries",true);

sw4 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw4 = translate(sw4, [55e-3/2-xPos_FB, -4e-3-yPos_FB, 0]);
gm = union(gm, sw4, "KeepBoundaries",true);

% FB-2
sw5 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw5 = translate(sw5, [-55e-3/2+xPos_FB, -4e-3-yPos_FB, 0]);
gm = union(gm, sw5, "KeepBoundaries",true);

sw6 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw6 = translate(sw6, [-10e-3+xPos_FB, 4e-3-yPos_FB, 0]);
gm = union(gm, sw6, "KeepBoundaries",true);

sw7 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw7 = translate(sw7, [10e-3+xPos_FB, 4e-3-yPos_FB, 0]);
gm = union(gm, sw7, "KeepBoundaries",true);

sw8 = fegeometry(multicuboid(swWidth, swLength, swHeight, 'ZOffset', z4));
sw8 = translate(sw8, [55e-3/2+xPos_FB, -4e-3-yPos_FB, 0]);
gm = union(gm, sw8, "KeepBoundaries",true);


% Generate Air
air = fegeometry(multicuboid(l,b,2*swHeight, 'ZOffset', z4));
air = subtract(air, sw1);
air = subtract(air, sw2);
air = subtract(air, sw3);
air = subtract(air, sw4);
air = subtract(air, sw5);
air = subtract(air, sw6);
air = subtract(air, sw7);
air = subtract(air, sw8);
gm = union(gm, air, "KeepBoundaries",true);

% Generate Model
thermalmodel = femodel(AnalysisType="thermalTransient");
thermalmodel.Geometry = gm;
pdegplot(thermalmodel, "CellLabels", "off", "FaceLabels", "on", "FaceAlpha", 0.5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define materials by face using the materialProperties property
thermalmodel.MaterialProperties([19, 20, 21]) = materialProperties( ...
    "ThermalConductivity", matK(1), ...
    "MassDensity", matRho(1), ...
    "SpecificHeat", matCp(1));

thermalmodel.MaterialProperties(23) = materialProperties( ...
    "ThermalConductivity", matK(2), ...
    "MassDensity", matRho(2), ...
    "SpecificHeat", matCp(2));

thermalmodel.MaterialProperties(22) = materialProperties( ...
    "ThermalConductivity", matK(3), ...
    "MassDensity", matRho(3), ...
    "SpecificHeat", matCp(3));

thermalmodel.MaterialProperties([10, 11, 12, 13, 14, 15, 16, 17, 18]) = materialProperties( ...
    "ThermalConductivity", matK(4), ...
    "MassDensity", matRho(4), ...
    "SpecificHeat", matCp(4));

thermalmodel.MaterialProperties([2,3,4,5,6,7,8,9]) = materialProperties( ...
    "ThermalConductivity", matK(5), ...
    "MassDensity", matRho(5), ...
    "SpecificHeat", matCp(5));

thermalmodel.MaterialProperties(1) = materialProperties( ...
    "ThermalConductivity", matK(6), ...
    "MassDensity", matRho(6), ...
    "SpecificHeat", matCp(6));


% Initial Condition
thermalmodel.CellIC = cellIC(Temperature=Tinit);

% Boundary Conditions
% thermalmodel.FaceLoad(42) = faceLoad(ConvectionCoefficient=7.5, AmbientTemperature=22);
thermalmodel.FaceLoad([97, 107, 109]) = faceLoad(ConvectionCoefficient=hc, AmbientTemperature=Ta);
thermalmodel.FaceLoad([95, 99, 101, 103, 105]) = faceLoad(ConvectionCoefficient=hfix, AmbientTemperature=Ta);

% Heat Source
thermalmodel.CellLoad([20, 21]) = cellLoad(Heat=q_Cu); 
thermalmodel.CellLoad([2,3,4,5,6,7,8,9]) = cellLoad(Heat=q_sw); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Meshing 
%---------------------------------------------------
cellsToRefine = [2:9,20,21];
facesToRefine = cellFaces(thermalmodel.Geometry,cellsToRefine);
thermalmodel = generateMesh(thermalmodel,'Hmax',2*dx,'Hmin',dx,'Hface',{facesToRefine, dx/2},Hgrad=2);
pdemesh(thermalmodel);

%---------------------------------------------------
% Solving 
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
dy = (b)/(N-1);
dz = (h)/(K-1);

%---------------------------------------------------
% Internal Heat 
%---------------------------------------------------
% Init
Q = zeros(size(T));

% Switches and Copper Region
for i = 1:size(T,1)
    xNode = results.Mesh.Nodes(1,i);
    yNode = results.Mesh.Nodes(2,i);
    zNode = results.Mesh.Nodes(3,i);

    % Switch region
    if zNode >= z4 && zNode <= (z4 + swHeight)
        isAtSwitch = checkSwitchPosition(xNode, yNode);
        if any(isAtSwitch)
            Q(i,:) = q_sw;   % switch heat density
        end

    % Copper pad region
    elseif zNode >= z3 && zNode <= (z3 + h_Cu)
        isAtCuPad = checkCuPadPosition(xNode, yNode);
        if any(isAtCuPad)
            Q(i,:) = q_Cu;   % copper pad heat density
        end
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
    if results.Mesh.Nodes(3,i) >= z4 && results.Mesh.Nodes(3,i) <= (z4 + swHeight) &&  any(isAtSwitch)
        Cp(i,:) = matCp(5);
        k(i,:) = matK(5);
        rho(i,:) = matRho(5);

    % Air
    elseif results.Mesh.Nodes(3,i) >= z4
        Cp(i,:) = matCp(6);
        k(i,:) = matK(6);
        rho(i,:) = matRho(6);

    % Copper
    elseif results.Mesh.Nodes(3,i) >= z3
        Cp(i,:) = matCp(1);
        k(i,:) = matK(1);
        rho(i,:) = matRho(1);

    % Dielectrica
    elseif results.Mesh.Nodes(3,i) >= z2
        Cp(i,:) = matCp(2);
        k(i,:) = matK(2);
        rho(i,:) = matRho(2);

    % Aluminium
    elseif results.Mesh.Nodes(3,i) >= z1
        Cp(i,:) = matCp(3);
        k(i,:) = matK(3);
        rho(i,:) = matRho(3);

    % TIM    
    elseif results.Mesh.Nodes(3,i) < z1
        Cp(i,:) = matCp(4);
        k(i,:) = matK(4);
        rho(i,:) = matRho(4);
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
    % 3D plot
    figure;
    for i = 1:floor(length(tlist)/10)
        pdeplot3D(results.Mesh, 'ColorMapData', results.Temperature(:, i*10));
        title(['Temperature at t = ', num2str(tlist(i*10)), ' s']);
        colorbar;
        drawnow;
    end

    % Material values
    figure
    subplot(1,3,1);
    pdeplot3D(results.Mesh, 'ColorMapData', Cp, 'FaceAlpha',0.5);
    subplot(1,3,2);
    pdeplot3D(results.Mesh, 'ColorMapData', k, 'FaceAlpha',0.5);
    subplot(1,3,3);
    pdeplot3D(results.Mesh, 'ColorMapData', rho, 'FaceAlpha',0.5);
    
    % 2D Projection
    figure;
    xx = -Lx/2:1e-3:Lx/2;
    yy = -Ly/2:1e-3:Ly/2;
    [XX,YY] = meshgrid(xx,yy);
    ZZ = Lz*ones(size(XX));
    Tintrp = interpolateTemperature(results,XX,YY,ZZ,1:length(tlist));
    Tsol = Tintrp(:,end);
    Tsol = reshape(Tsol,size(XX));
    contourf((XX+Lx/2)*1000,(YY+Ly/2)*1000,Tsol,20,'LineStyle','none')
    colormap("parula")
    colorbar
    caxis([44 62]); 
    axis equal   

    % 1D Heat-up
    Tsw = zeros(1,length(tlist));
    for i = 1:length(tlist)
        temp = Tintrp(:,i);
        temp = reshape(temp,size(XX));
        Tsw(i) = mean(mean(temp(31:40,37:46)));
        Tsw(i) = max(results.Temperature(:,i));
    end
    figure;
    Zth = (Tsw - Tinit) / (Pv + Pv_Cu/8);
    Zth2 = (Tsw - Tinit) / (Pv);
    semilogx(tlist, Zth);
    hold on
    semilogx(tlist, Zth2);
    grid on
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------
% Spatial Variables Switch
%---------------------------------------------------
function isAtSwitch = checkSwitchPosition(x, y)
    % Switch dimensions
    swWidth  = 10e-3;  % 10 mm
    swLength = 10e-3;  % 10 mm

    % Full bridge offsets
    xPos_FB = 52.5e-3;
    yPos_FB = 25e-3;

    % Define switch positions (centers)
    switchPositions = [
        -55e-3/2 - xPos_FB,   -4e-3 - yPos_FB;   % sw1
        -10e-3   - xPos_FB,    4e-3 - yPos_FB;   % sw2
         10e-3   - xPos_FB,    4e-3 - yPos_FB;   % sw3
         55e-3/2 - xPos_FB,   -4e-3 - yPos_FB;   % sw4
        -55e-3/2 + xPos_FB,   -4e-3 - yPos_FB;   % sw5
        -10e-3   + xPos_FB,    4e-3 - yPos_FB;   % sw6
         10e-3   + xPos_FB,    4e-3 - yPos_FB;   % sw7
         55e-3/2 + xPos_FB,   -4e-3 - yPos_FB    % sw8
    ];

    % Initialize
    isAtSwitch = false(size(switchPositions, 1), 1);

    % Check each switch region
    for i = 1:size(switchPositions, 1)
        xCenter = switchPositions(i, 1);
        yCenter = switchPositions(i, 2);

        xMin = xCenter - swWidth/2;
        xMax = xCenter + swWidth/2;
        yMin = yCenter - swLength/2;
        yMax = yCenter + swLength/2;

        if (x >= xMin && x <= xMax) && (y >= yMin && y <= yMax)
            isAtSwitch(i) = true;
        end
    end
end

%---------------------------------------------------
% Spatial Variables Copper
%---------------------------------------------------
function isAtCuPad = checkCuPadPosition(x, y)
    % Copper pad dimensions
    l_FB     = 75e-3;
    b_FB     = 30e-3;
    dx_FB_Cu = 5e-3;
    xPos_FB  = 52.5e-3;
    yPos_FB  = 25e-3;

    % Dimensions of pads (in x and y)
    cu1_w = l_FB + dx_FB_Cu;
    cu1_h = b_FB + dx_FB_Cu;

    cu2_w = l_FB + 10e-3;
    cu2_h = b_FB + 10e-3;

    % Pad center positions (in x,y)
    cuCenters = [
        -xPos_FB,  -yPos_FB - 5e-3/2;   % cu1
        +xPos_FB,  -yPos_FB - 5e-3/2    % cu2
    ];

    cuSizes = [
        cu1_w, cu1_h;   % cu1 size
        cu2_w, cu2_h    % cu2 size
    ];

    % Initialize
    isAtCuPad = false(size(cuCenters, 1), 1);

    % Check each pad
    for i = 1:size(cuCenters, 1)
        xCenter = cuCenters(i, 1);
        yCenter = cuCenters(i, 2);

        w = cuSizes(i, 1);
        h = cuSizes(i, 2);

        xMin = xCenter - w/2;
        xMax = xCenter + w/2;
        yMin = yCenter - h/2;
        yMax = yCenter + h/2;

        if (x >= xMin && x <= xMax) && (y >= yMin && y <= yMax)
            isAtCuPad(i) = true;
        end
    end
end

%---------------------------------------------------
% Temporal Variables Switch
%---------------------------------------------------
function Q = heatSourceTimeSw(location, state)
    % Geometry
    swWidth = 10e-3; 
    swLength = 10e-3; 
    swHeight = 2.0e-3;
    
    % Inputs
    t = [60, 70, 72];
    Pv1 = [20, 30, 60];

    % Parameters
    Vol_Sw = swWidth*swLength*swHeight;
    Q = zeros(1,numel(location.x));
    
    % Check for NaNs
    if(isnan(state.time))
      Q(1,:) = NaN;
      return
    end
    
    % Time-dependent power
    for i = 1:length(t)
        if state.time < t(i)
            Pv = Pv1(i);
            break;
        else
            Pv = Pv1(end);
        end
    end
    
    % Heat generation rate
    q_val = Pv / Vol_sw;
 
    % Compute heat generation
    for i = 1:length(numel(location.x))
        if location.y(i) >= (b_Ga + b_Al + b_Di) && location.x(i) > 15e-3 && location.x(i) <= 23e-3
            Q(i,:) = q_val;
        elseif location.y(i) >= (b_Ga + b_Al + b_Di) && location.x(i) > 27e-3 && location.x(i) <= 35e-3
            Q(i,:) = q_val;
        else
            % Q(i,:) = q/2;
        end
    end
 
    % Debug
    disp(['Time: ', num2str(state.time), ', Heat Source: ', num2str(Q(1))]);
end