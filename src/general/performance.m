%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: performance                                                       %
% Date: 19.12.2024                                                        %
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
% function out = performance(dataTest, dataPred, para, mdl)
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% Init
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     dim = para.Dat.gen.dOut;                                                % dimension of the dataset
%     th = para.Par.gen.th;                                                   % transient threshold (%)
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% Performance 1D
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %===================================================
%     % Init
%     %===================================================
%     [~, N] = size(dataTest.y);
%     yTrue = dataTest.y;
%     yPred = dataPred.y;
% 
%     %===================================================
%     % Pre-processing
%     %===================================================
%     dT = [zeros(1,length(yTrue(1,:))); diff(yTrue)];
%     dT_tr = zeros(size(dT));
%     dT_ss = zeros(size(dT));
%     dT_tr(dT > th*max(dT)) = 1;
%     dT_ss(dT < th*max(dT)) = 1;
% 
%     %===================================================
%     % Error Calcualtion
%     %===================================================
%     %----------------------------------------
%     % Total
%     %----------------------------------------
%     for i = 1:N
%         [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue(:,i), yPred(:,i));
%         out.err.tot.MAE(i) = MAE;
%         out.err.tot.RMSE(i) = RMSE;
%         out.err.tot.SAE(i) = SAE;
%         out.err.tot.NAE(i) = NAE;
%         out.err.tot.MAX(i) = MAX;
%         out.err.tot.NRSME(i) = NRSME;
%     end
% 
%     %----------------------------------------
%     % Transient
%     %----------------------------------------
%     for i = 1:N
%         % Grt Signals
%         yTrueT = yTrue(:,i);
%         yTrueT(dT_tr(:,i) == 0) = [];
% 
%         % Pred Signals
%         yPredT = yPred(:,i);
%         yPredT(dT_tr(:,i) == 0) = [];
% 
%         % Check if empty
%         if isempty(yTrueT) == 1
%             yTrueT = 0;
%             yPredT = 0;
%         end
% 
%         % Calc
%         [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueT, yPredT);
%         out.err.tr.MAE(i) = MAE;
%         out.err.tr.RMSE(i) = RMSE;
%         out.err.tr.SAE(i) = SAE;
%         out.err.tr.NAE(i) = NAE;
%         out.err.tr.MAX(i) = MAX;
%         out.err.tr.NRSME(i) = NRSME;
%     end
% 
%     %----------------------------------------
%     % Steady-State
%     %----------------------------------------
%     for i = 1:N
%         % Grt Signals
%         yTrueS = yTrue(:,i);
%         yTrueS(dT_ss(:,i) == 0) = [];
% 
%         % Pred Signals
%         yPredS = yPred(:,i);
%         yPredS(dT_ss(:,i) == 0) = [];
% 
%         % Check if empty
%         if isempty(yTrueS) == 1
%             yTrueS = 0;
%             yPredS = 0;
%         end
% 
%         % Calc
%         [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrueS, yPredS);
%         out.err.ss.MAE(i) = MAE;
%         out.err.ss.RMSE(i) = RMSE;
%         out.err.ss.SAE(i) = SAE;
%         out.err.ss.NAE(i) = NAE;
%         out.err.ss.MAX(i) = MAX;
%         out.err.ss.NRSME(i) = NRSME;
%     end
% 
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% Post-Processing
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %===================================================
%     % Times
%     %===================================================
%     %----------------------------------------
%     % Training
%     %----------------------------------------
%     if isempty(mdl)
%         out.time.train = 0;
%     else
%         out.time.train = mdl.timeTrain;
%     end
% 
%     %----------------------------------------
%     % Testing
%     %----------------------------------------
%     out.time.test = dataPred.testTime;
% 
%     %===================================================
%     % Memory
%     %===================================================
%     if isempty(mdl)
%         out.mem = 0;
%     else
%         out.mem = mdl.size;
%     end
% end

function out = performance(dataTest, dataPred, para, mdl)
    % Initialization
    yTrue = dataTest.y;
    yPred = dataPred.y;
    [~, N] = size(yTrue);
    th = para.Par.gen.th;

    % Derivatives and thresholds
    dT = [zeros(1, N); diff(yTrue)];
    maxDT = max(dT);
    dT_tr = dT > th * maxDT;
    dT_ss = dT < th * maxDT;

    % Preallocate error containers
    metricNames = {'MAE', 'RMSE', 'SAE', 'NAE', 'MAX', 'NRSME'};
    for name = metricNames
        [out.err.tot.(name{1}), out.err.tr.(name{1}), out.err.ss.(name{1})] = deal(zeros(1, N));
    end

    % Loop once, calculate all three error modes per signal
    for i = 1:N
        % --- Raw
        [out.err.tot.MAE(i), out.err.tot.RMSE(i), out.err.tot.SAE(i), ...
         out.err.tot.NAE(i), out.err.tot.MAX(i), out.err.tot.NRSME(i)] = ...
            calcerror(yTrue(:,i), yPred(:,i));

        % --- Transient
        mask_tr = dT_tr(:,i);
        yTrueT = yTrue(mask_tr, i);
        yPredT = yPred(mask_tr, i);
        if isempty(yTrueT), yTrueT = 0; yPredT = 0; end
        [out.err.tr.MAE(i), out.err.tr.RMSE(i), out.err.tr.SAE(i), ...
         out.err.tr.NAE(i), out.err.tr.MAX(i), out.err.tr.NRSME(i)] = ...
            calcerror(yTrueT, yPredT);

        % --- Steady-state
        mask_ss = dT_ss(:,i);
        yTrueS = yTrue(mask_ss, i);
        yPredS = yPred(mask_ss, i);
        if isempty(yTrueS), yTrueS = 0; yPredS = 0; end
        [out.err.ss.MAE(i), out.err.ss.RMSE(i), out.err.ss.SAE(i), ...
         out.err.ss.NAE(i), out.err.ss.MAX(i), out.err.ss.NRSME(i)] = ...
            calcerror(yTrueS, yPredS);
    end

    % Timing and memory
    out.time.train = ifelse(isempty(mdl), 0, mdl.timeTrain);
    out.time.test = dataPred.testTime;
    out.mem = ifelse(isempty(mdl), 0, mdl.size);
end

% Helper inline function (ternary-like)
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