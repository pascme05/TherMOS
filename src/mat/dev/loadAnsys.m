%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Titel: Model Order Reduction (MOR)
% File: loadAnsys
% Author: P. Schirmer
% Department: ES-641
% Version: v.1.2
% Date: 17.06.2024
% Copyright: BMW AG, Munich
% Comments: comparing different MORs for thermal modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This functions loads the input data for training, testing and validation
% using ansys as an input file type. Furthmore, the data splitting can be 
% done on file level or on ID level. The following are the input and 
% output parameters:
% Input:
%       - path:     path variable to the data files
%       - setup:    all setup variables for the simulation
% Output:
%       - outTrain: output training data
%       - outTest:  output testing data
%       - outVal:   output validation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FNC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outTrain, outTest, outVal] = loadAnsys(path, setup)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Loading Ansys Data')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = setup.data.file;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Selecting Train/Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% File Based
% ----------------------------------------------------
if setup.data.sep == 0
    % Training
    train = load(path.data + file, "-mat");
    
    % Testing
    test = load(path.data + file, "-mat");
    
    % Validation
    val = load(path.data + file, "-mat");

% ----------------------------------------------------
% ID Based
% ----------------------------------------------------
elseif setup.data.sep == 1
    % Load file

    % Splitting


% ----------------------------------------------------
% Ratio Based
% ----------------------------------------------------
elseif setup.data.sep == 2
    % Load file

    % Splitting
    

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------
% Reshaping
% ----------------------------------------------------
[outTrainTemp] = shapeAnsysData(train, setup);
[outTest] = shapeAnsysData(test, setup);
[outVal] = shapeAnsysData(val, setup);

% ----------------------------------------------------
% Training Structure
% ----------------------------------------------------
outTrain.X{1} = outTrainTemp.X;
outTrain.y{1} = outTrainTemp.y;
outTrain.r{1} = outTrainTemp.r;
outTrain.Lx = outTrainTemp.Lx;
outTrain.Ly = outTrainTemp.Ly;
outTrain.dx = outTrainTemp.dx;
outTrain.dy = outTrainTemp.dy;
outTrain.Mat = outTrainTemp.Mat;
outTrain.Material = outTrainTemp.Material;
outTrain.t{1} = outTrainTemp.t;
outTrain.Ts{1} = outTrainTemp.Ts;
outTrain.Mat2D.k = outTrainTemp.Mat2D.k;
outTrain.Mat2D.rho = outTrainTemp.Mat2D.rho;
outTrain.Mat2D.Cp = outTrainTemp.Mat2D.Cp;

end