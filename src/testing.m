%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: testing                                                             %
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
% This function tests a prediction model.
% -------------------------------------------------------------------------
% Inp:  1) mdl:     Input model data from training
%       2) data:    Input data struct including tr, te, and vl
%       3) setup:   All setup values of the current simulation
%       4) para:    All simulation parameters of the current simulation
%       5) path:    Structure of all path variables
% Out:  1) pred:    Predicted output
%       2) grt:     Updated grt data
%       3) mdl:     Loaded model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pred, grt, mdl] = testing(mdl, data, setup, para, path)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Testing model")
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grt = data.te;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Foster-Model
    %===================================================
    if setup.selRC == 1
        %----------------------------------------
        % Loading
        %----------------------------------------
        if isempty(mdl)
            try
                mdlName = 'mdl_rc_' + setup.name + '.mat';
                filename = fullfile(path.mdl, mdlName);
                load(filename, 'mdl');
                disp('INFO: Model loaded successfully.');
            catch ME
                disp('WARN: Failed to load the model.');
                disp(['Error: ', ME.message]);
            end
        end

        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        pred = rcSol(mdl, data.te, para);
        pred.testTime = toc(timeStart);

        %----------------------------------------
        % Formatting 
        %----------------------------------------
        % Prediction
        pred.y = pred.y(:,1);
        pred.X = pred.X(:,1);
        pred.r = pred.r(:,1);
        pred.off = pred.off(:,1);

        % Testing
        grt.y = data.te.y(:,1);
        grt.X = data.te.X(:,1);
        grt.r = data.te.r(:,1);
        grt.off = data.te.off(:,1);
    end
    
    %===================================================
    % State-Space Model
    %===================================================
    if setup.selSS == 1
        %----------------------------------------
        % Loading
        %----------------------------------------
        if isempty(mdl)
            try
                mdlName = 'mdl_ss_' + setup.name + '.mat';
                filename = fullfile(path.mdl, mdlName);
                load(filename, 'mdl');
                disp('INFO: Model loaded successfully.');
            catch ME
                disp('WARN: Failed to load the model.');
                disp(['Error: ', ME.message]);
            end
        end

        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        pred = ssSol(mdl, data.te, para);
        pred.testTime = toc(timeStart);

        %----------------------------------------
        % Formatting 
        %----------------------------------------
        % Prediction
        pred.y = pred.y;
        pred.X = pred.X;
        pred.r = pred.r;
        pred.off = pred.off;

        % Testing
        grt.y = data.te.y;
        grt.X = data.te.X;
        grt.r = data.te.r;
        grt.off = data.te.off;
    end

    %===================================================
    % SF Model
    %===================================================

    %===================================================
    % POD Model
    %===================================================

    %===================================================
    % ML Model
    %===================================================
    if setup.selML ~= 0
        %----------------------------------------
        % KNN 
        %----------------------------------------
        if setup.selML == 1
            % Loading
            if isempty(mdl)
                try
                    mdlName = 'mdl_ml_knn_' + setup.name + '.mat';
                    filename = fullfile(path.mdl, mdlName);
                    load(filename, 'mdl');
                    disp('INFO: Model loaded successfully.');
                catch ME
                    disp('WARN: Failed to load the model.');
                    disp(['Error: ', ME.message]);
                end
            end
    
            % Fitting 
            timeStart = tic;
            pred = knnSol(mdl, data.te, para);
            pred.testTime = toc(timeStart);
    
            % Formatting 
            pred.y = pred.y(:,1);
            pred.X = pred.X(:,1);
            pred.r = pred.r(:,1);
            pred.off = pred.off(:,1);
            grt.y = data.te.y(:,1);
            grt.X = data.te.X(:,1);
            grt.r = data.te.r(:,1);
            grt.off = data.te.off(:,1);
        end

        %----------------------------------------
        % DT 
        %----------------------------------------
        if setup.selML == 2
            % Loading
            if isempty(mdl)
                try
                    mdlName = 'mdl_ml_dt_' + setup.name + '.mat';
                    filename = fullfile(path.mdl, mdlName);
                    load(filename, 'mdl');
                    disp('INFO: Model loaded successfully.');
                catch ME
                    disp('WARN: Failed to load the model.');
                    disp(['Error: ', ME.message]);
                end
            end
    
            % Fitting 
            timeStart = tic;
            pred = dtSol(mdl, data.te, para);
            pred.testTime = toc(timeStart);
    
            % Formatting 
            pred.y = pred.y(:,1);
            pred.X = pred.X(:,1);
            pred.r = pred.r(:,1);
            pred.off = pred.off(:,1);
            grt.y = data.te.y(:,1);
            grt.X = data.te.X(:,1);
            grt.r = data.te.r(:,1);
            grt.off = data.te.off(:,1);
        end

        %----------------------------------------
        % SVR 
        %----------------------------------------
        if setup.selML == 3
            % Loading
            if isempty(mdl)
                try
                    mdlName = 'mdl_ml_svm_' + setup.name + '.mat';
                    filename = fullfile(path.mdl, mdlName);
                    load(filename, 'mdl');
                    disp('INFO: Model loaded successfully.');
                catch ME
                    disp('WARN: Failed to load the model.');
                    disp(['Error: ', ME.message]);
                end
            end
    
            % Fitting 
            timeStart = tic;
            pred = svrSol(mdl, data.te, para);
            pred.testTime = toc(timeStart);
    
            % Formatting 
            pred.y = pred.y(:,1);
            pred.X = pred.X(:,1);
            pred.r = pred.r(:,1);
            pred.off = pred.off(:,1);
            grt.y = data.te.y(:,1);
            grt.X = data.te.X(:,1);
            grt.r = data.te.r(:,1);
            grt.off = data.te.off(:,1);
        end
    end

    %===================================================
    % DL Model
    %===================================================
    if setup.selDL ~= 0
        %----------------------------------------
        % Loading
        %----------------------------------------
        if isempty(mdl)
            try
                mdlName = 'mdl_dl_' + setup.name + '.mat';
                filename = fullfile(path.mdl, mdlName);
                load(filename, 'mdl');
                disp('INFO: Model loaded successfully.');
            catch ME
                disp('WARN: Failed to load the model.');
                disp(['Error: ', ME.message]);
            end
        end

        %----------------------------------------
        % Fitting 
        %----------------------------------------
        timeStart = tic;
        pred = dlSol(mdl, data.te, para);
        pred.testTime = toc(timeStart);

        %----------------------------------------
        % Formatting 
        %----------------------------------------
        % Prediction
        pred.y = pred.y;
        pred.X = pred.X;
        pred.r = pred.r;
        pred.off = pred.off;

        % Testing
        grt.y = data.te.y;
        grt.X = data.te.X;
        grt.r = data.te.r;
        grt.off = data.te.off;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Testing model")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1