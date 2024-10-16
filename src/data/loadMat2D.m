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
% using mat2D as an input file type. Furthmore, the data splitting can be 
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
function [outTrain, test, val] = loadMat2D(path, setup)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('INFO: Loading Matlab 2D Data')

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
% Training Structure
% ----------------------------------------------------
outTrain.X{1} = train.X;
outTrain.y{1} = train.y;
outTrain.r{1} = train.r;
outTrain.Lx = train.Lx;
outTrain.Ly = train.Ly;
outTrain.dx = train.dx;
outTrain.dy = train.dy;
outTrain.Mat = train.Mat;
outTrain.Material = train.Material;
outTrain.t{1} = train.t;
outTrain.Ts{1} = train.Ts;
outTrain.Mat2D.k = train.Mat2D.k;
outTrain.Mat2D.rho = train.Mat2D.rho;
outTrain.Mat2D.Cp = train.Mat2D.Cp;

end