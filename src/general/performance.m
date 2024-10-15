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
    dim = size(dataTest.y);                                                 % dimension of the dataset
    th = para.Par.gen.th;                                                   % transient threshold (%)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Performance 1D
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if length(dim) == 2
        %===================================================
        % Init
        %===================================================
        yTrue = dataTest.y;
        yPred = dataPred.y;

        %===================================================
        % Pre-processing
        %===================================================
        %----------------------------------------
        % Status
        %----------------------------------------
        dT = [zeros(1,length(yTrue(1,:))); diff(yTrue)];
        dT_tr = zeros(size(dT));
        dT_ss = zeros(size(dT));
        dT_tr(dT > th*max(dT)) = 1;
        dT_ss(dT < th*max(dT)) = 1;

        %----------------------------------------
        % Signals
        %----------------------------------------
        % Grt Signals
        yTrueS = yTrue;
        yTrueS(dT_ss == 0) = [];
        yTrueT = yTrue;
        yTrueT(dT_tr == 0) = [];

        % Pred Signals
        yPredS = yPred;
        yPredS(dT_ss == 0) = [];
        yPredT = yPred;
        yPredT(dT_tr == 0) = [];

        %===================================================
        % Error Calcualtion
        %===================================================
        %----------------------------------------
        % Total
        %----------------------------------------
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue, yPred);
        out.err.tot.MAE = MAE;
        out.err.tot.RMSE = RMSE;
        out.err.tot.SAE = SAE;
        out.err.tot.NAE = NAE;
        out.err.tot.MAX = MAX;
        out.err.tot.NRSME = NRSME;

        %----------------------------------------
        % Transient
        %----------------------------------------
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueT, yPredT);
        out.err.tr.MAE = MAE;
        out.err.tr.RMSE = RMSE;
        out.err.tr.SAE = SAE;
        out.err.tr.NAE = NAE;
        out.err.tr.MAX = MAX;
        out.err.tr.NRSME = NRSME;

        %----------------------------------------
        % Steady-State
        %----------------------------------------
        [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueS, yPredS);
        out.err.ss.MAE = MAE;
        out.err.ss.RMSE = RMSE;
        out.err.ss.SAE = SAE;
        out.err.ss.NAE = NAE;
        out.err.ss.MAX = MAX;
        out.err.ss.NRSME = NRSME;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Performance 2D
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Times
    %===================================================
    out.time.train = mdl.timeTrain;
    out.time.test = dataPred.testTime;

    %===================================================
    % Memory
    %===================================================
    out.mem = mdl.size;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Calculating performance")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1