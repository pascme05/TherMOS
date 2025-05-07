%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: performance                                                       %
% Date: 07.05.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
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
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yTrue = dataTest.y;
    yPred = dataPred.y;
    [~, N] = size(yTrue);
    th = para.Par.gen.th;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Derivatives and thresholds
    %===================================================
    dT = [zeros(1, N); diff(yTrue)];
    maxDT = max(dT);
    dT_tr = dT > th * maxDT;
    dT_ss = dT < th * maxDT;

    %===================================================
    % Preallocate error containers
    %===================================================
    metricNames = {'MAE', 'RMSE', 'SAE', 'NAE', 'MAX', 'NRSME'};
    for name = metricNames
        [out.err.tot.(name{1}), out.err.tr.(name{1}), out.err.ss.(name{1})] = deal(zeros(1, N));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:N
        %===================================================
        % Raw
        %===================================================
        [out.err.tot.MAE(i), out.err.tot.RMSE(i), out.err.tot.SAE(i), ...
         out.err.tot.NAE(i), out.err.tot.MAX(i), out.err.tot.NRSME(i)] = ...
            calcerror(yTrue(:,i), yPred(:,i));
        
        %===================================================
        % Transient
        %===================================================
        mask_tr = dT_tr(:,i);
        yTrueT = yTrue(mask_tr, i);
        yPredT = yPred(mask_tr, i);
        if isempty(yTrueT), yTrueT = 0; yPredT = 0; end
        [out.err.tr.MAE(i), out.err.tr.RMSE(i), out.err.tr.SAE(i), ...
         out.err.tr.NAE(i), out.err.tr.MAX(i), out.err.tr.NRSME(i)] = ...
            calcerror(yTrueT, yPredT);
        
        %===================================================
        % Steady-state
        %===================================================
        mask_ss = dT_ss(:,i);
        yTrueS = yTrue(mask_ss, i);
        yPredS = yPred(mask_ss, i);
        if isempty(yTrueS), yTrueS = 0; yPredS = 0; end
        [out.err.ss.MAE(i), out.err.ss.RMSE(i), out.err.ss.SAE(i), ...
         out.err.ss.NAE(i), out.err.ss.MAX(i), out.err.ss.NRSME(i)] = ...
            calcerror(yTrueS, yPredS);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out.time.train = ifelse(isempty(mdl), 0, mdl.timeTrain);
    out.time.test = dataPred.testTime;
    out.mem = ifelse(isempty(mdl), 0, mdl.size);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Help Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = ifelse(cond, valTrue, valFalse)
    if cond
        result = valTrue;
    else
        result = valFalse;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1