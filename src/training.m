%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: training                                                          %
% Date: 11.10.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function trains a prediction model.
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) setup:   All setup values of the current simulation
%       3) para:    All simulation parameters of the current simulation
%       4) path:    Structure of all path variables
% Out:  1) mdl:     Trained model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mdl = training(data, setup, para, path)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Training model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Foster-Model
    %===================================================
    if setup.selRC == 1
        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        mdl = rcFit(data.tr, data.vl, para);
        mdl.timeTrain = toc(timeStart);
        
        %----------------------------------------
        % Model Size 
        %----------------------------------------
        mdl.size = numel(mdl.Rth) + numel(mdl.Cth);

        %----------------------------------------
        % Saving
        %----------------------------------------
        mdlName = 'mdl_rc_' + setup.name + '.mat';
        filename = fullfile(path.mdl, mdlName);
        save(filename, 'mdl');
    end
    
    %===================================================
    % State-Space Model
    %===================================================
    if setup.selSS == 1
        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        mdl.sys = ssFit(data.tr, data.vl, para);
        mdl.timeTrain = toc(timeStart);

        %----------------------------------------
        % Model Size 
        %----------------------------------------
        mdl.size = numel(mdl.sys.A) + numel(mdl.sys.B) + ...
                   numel(mdl.sys.C) + numel(mdl.sys.D);

        %----------------------------------------
        % Saving
        %----------------------------------------
        mdlName = 'mdl_ss_' + setup.name + '.mat';
        filename = fullfile(path.mdl, mdlName);
        save(filename, 'mdl');
    end

    %===================================================
    % Structure Function
    %===================================================
    if setup.selSF == 1
        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        mdl.sys = sfFit(data.tr, data.vl, para);
        mdl.timeTrain = toc(timeStart);

        %----------------------------------------
        % Model Size 
        %----------------------------------------
        mdl.size = numel(mdl.sys.Rth) + numel(mdl.sys.Cth);

        %----------------------------------------
        % Saving
        %----------------------------------------
        mdlName = 'mdl_sf_' + setup.name + '.mat';
        filename = fullfile(path.mdl, mdlName);
        save(filename, 'mdl');
    end
    
    %===================================================
    % POD Model
    %===================================================

    %===================================================
    % Machine Learning
    %===================================================
    if setup.selML ~= 0
        %----------------------------------------
        % KNN 
        %----------------------------------------
        if setup.selML == 1
            % Fitting
            timeStart = tic;
            mdl.sys = knnFit(data.tr, data.vl, para);
            mdl.timeTrain = toc(timeStart);

            % Model Size
            mdl.size = mdl.sys.NumNeighbors;

            % Saving
            mdlName = 'mdl_ml_knn_' + setup.name + '.mat';
            filename = fullfile(path.mdl, mdlName);
            save(filename, 'mdl');
        end

        %----------------------------------------
        % RF 
        %----------------------------------------
        if setup.selML == 2
            % Fitting
            timeStart = tic;
            mdl.sys = dtFit(data.tr, data.vl, para);
            mdl.timeTrain = toc(timeStart);

            % Model Size
            mdl.size = mdl.sys.ModelParameters.MinLeaf;

            % Saving
            mdlName = 'mdl_ml_dt_' + setup.name + '.mat';
            filename = fullfile(path.mdl, mdlName);
            save(filename, 'mdl');
        end

        %----------------------------------------
        % SVR 
        %----------------------------------------
        if setup.selML == 3
            % Fitting
            timeStart = tic;
            mdl.sys = svrFit(data.tr, data.vl, para);
            mdl.timeTrain = toc(timeStart);

            % Model Size
            mdl.size = numel(mdl.sys.SupportVectors);

            % Saving
            mdlName = 'mdl_ml_svr_' + setup.name + '.mat';
            filename = fullfile(path.mdl, mdlName);
            save(filename, 'mdl');
        end

    end

    %===================================================
    % Deep Learning
    %===================================================
    if setup.selDL ~= 0
        % Fitting
        timeStart = tic;
        mdl.sys = dlFit(data.tr, data.vl, para);
        mdl.timeTrain = toc(timeStart);

        % Model Size
        mdl.size = 1;

        % Saving
        mdlName = 'mdl_dl_' + setup.name + '.mat';
        filename = fullfile(path.mdl, mdlName);
        save(filename, 'mdl');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Training model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1