%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: performance                                                       %
% Date: 13.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here goes the description of the function.
% -------------------------------------------------------------------------
% Inp:  1) dataTest:    Input testing data
%       2) dataPred:    Input prediction data
%       3) para:        All simulation parameters
%       4) mdl:         Model input
% Out:  1) out:         Output performance metrics

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = performance(dataTest, dataPred, para, mdl)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Calculating performance")
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dim = para.Dat.gen.dOut;                                                % dimension of the dataset
    th = para.Par.gen.th;                                                   % transient threshold (%)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Performance 1D
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init
    %===================================================
    [~, N] = size(dataTest.y);
    yTrue = dataTest.y;
    yPred = dataPred.y;

    %===================================================
    % Pre-processing
    %===================================================
    dT = [zeros(1,length(yTrue(1,:))); diff(yTrue)];
    dT_tr = zeros(size(dT));
    dT_ss = zeros(size(dT));
    dT_tr(dT > th*max(dT)) = 1;
    dT_ss(dT < th*max(dT)) = 1;

    %===================================================
    % Error Calcualtion
    %===================================================
    %----------------------------------------
    % Total
    %----------------------------------------
    for i = 1:N
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue(:,i), yPred(:,i));
        out.err.tot.MAE(i) = MAE;
        out.err.tot.RMSE(i) = RMSE;
        out.err.tot.SAE(i) = SAE;
        out.err.tot.NAE(i) = NAE;
        out.err.tot.MAX(i) = MAX;
        out.err.tot.NRSME(i) = NRSME;
    end

    %----------------------------------------
    % Transient
    %----------------------------------------
    for i = 1:N
        % Grt Signals
        yTrueT = yTrue(:,i);
        yTrueT(dT_tr(:,i) == 0) = [];

        % Pred Signals
        yPredT = yPred(:,i);
        yPredT(dT_tr(:,i) == 0) = [];

        % Check if empty
        if isempty(yTrueT) == 1
            yTrueT = 0;
            yPredT = 0;
        end

        % Calc
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueT, yPredT);
        out.err.tr.MAE(i) = MAE;
        out.err.tr.RMSE(i) = RMSE;
        out.err.tr.SAE(i) = SAE;
        out.err.tr.NAE(i) = NAE;
        out.err.tr.MAX(i) = MAX;
        out.err.tr.NRSME(i) = NRSME;
    end

    %----------------------------------------
    % Steady-State
    %----------------------------------------
    for i = 1:N
        % Grt Signals
        yTrueS = yTrue(:,i);
        yTrueS(dT_ss(:,i) == 0) = [];

        % Pred Signals
        yPredS = yPred(:,i);
        yPredS(dT_ss(:,i) == 0) = [];
        
        % Check if empty
        if isempty(yTrueS) == 1
            yTrueS = 0;
            yPredS = 0;
        end

        % Calc
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueS, yPredS);
        out.err.ss.MAE(i) = MAE;
        out.err.ss.RMSE(i) = RMSE;
        out.err.ss.SAE(i) = SAE;
        out.err.ss.NAE(i) = NAE;
        out.err.ss.MAX(i) = MAX;
        out.err.ss.NRSME(i) = NRSME;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Performance 2D
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if dim == 2
        %===================================================
        % Total
        %===================================================
        out.err.tot.MAE = mean(out.err.tot.MAE);
        out.err.tot.RMSE = mean(out.err.tot.RMSE);
        out.err.tot.SAE = mean(out.err.tot.SAE);
        out.err.tot.NAE = mean(out.err.tot.NAE);
        out.err.tot.MAX = max(out.err.tot.MAX);
        out.err.tot.NRSME = mean(out.err.tot.NRSME);

        %===================================================
        % Steady-State
        %===================================================
        out.err.ss.MAE = mean(out.err.ss.MAE);
        out.err.ss.RMSE = mean(out.err.ss.RMSE);
        out.err.ss.SAE = mean(out.err.ss.SAE);
        out.err.ss.NAE = mean(out.err.ss.NAE);
        out.err.ss.MAX = max(out.err.ss.MAX);
        out.err.ss.NRSME = mean(out.err.ss.NRSME);

        %===================================================
        % Transient
        %===================================================
        out.err.tr.MAE = mean(out.err.tr.MAE);
        out.err.tr.RMSE = mean(out.err.tr.RMSE);
        out.err.tr.SAE = mean(out.err.tr.SAE);
        out.err.tr.NAE = mean(out.err.tr.NAE);
        out.err.tr.MAX = max(out.err.tr.MAX);
        out.err.tr.NRSME = mean(out.err.tr.NRSME);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Times
    %===================================================
    %----------------------------------------
    % Training
    %----------------------------------------
    if isempty(mdl)
        out.time.train = 0;
    else
        out.time.train = mdl.timeTrain;
    end

    %----------------------------------------
    % Testing
    %----------------------------------------
    out.time.test = dataPred.testTime;

    %===================================================
    % Memory
    %===================================================
    if isempty(mdl)
        out.mem = 0;
    else
        out.mem = mdl.size;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Calculating performance")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1