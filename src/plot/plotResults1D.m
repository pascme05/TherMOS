%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: plotResults1D                                                     %
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
% Inp:  1) data:    All simulation input data as well as prediction
%       2) results: All obtained accuracy values and results
%       3) setup:   All simulation setup parameters
% Out:  1) None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plotResults1D(data, results, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Plotting results")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [~, nTr] = size(data.tr.X2);                                            % number of training instances
    [~, nVl] = size(data.vl.X2);                                            % number of validation instances
    [~, Ny] = size(data.te.y);                                              % number of output samples

    %===================================================
    % Variables
    %===================================================
    time = data.te.t;
    yPred = data.pr.y;
    yTest = data.te.y;
    XPred = data.pr.X;
    XTest = data.te.X;
    rTest = data.te.r;
    
    %===================================================
    % Names
    %===================================================
    namesInp = setup.inp;
    namesOut = setup.out;
    names = [namesInp, namesOut];
    metrics = ["MAE", "RMSE", "MAX"];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Feature Ranking
    %===================================================
    [~, weights] = relieff(data.tr.X,data.tr.y,10);

    %===================================================
    % Correlation Analysis
    %===================================================
    vaCor = corr([data.tr.X, data.tr.y]);

    %===================================================
    % Residuals
    %===================================================
    err = yTest - yPred;

    %===================================================
    % Error Metric
    %===================================================
    errTot = [results.err.tot.MAE, results.err.tot.RMSE, results.err.tot.MAX];
    errSS = [results.err.ss.MAE, results.err.ss.RMSE, results.err.ss.MAX];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Input Features Analysis
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    figure;
    sgtitle('Input Feature Analysis using Distribution, Heatmap, and Feature Ranking');

    %----------------------------------------
    % Boxplot
    %----------------------------------------
    subplot(2,2, [1, 2]);
    boxplot(data.tr.X);
    xlabel(namesInp')
    ylabel('Input Features')
    title('Boxplot of Training Input Features')
    grid on;

    %----------------------------------------
    % Correlation
    %----------------------------------------
    subplot(2,2,3);
    h = heatmap(vaCor,'MissingDataColor','w');
    h.XDisplayLabels = names;
    h.YDisplayLabels = names;
    title('Pearson Correlation Input/Output')
    grid on;

    %----------------------------------------
    % Feature Ranking
    %----------------------------------------
    subplot(2,2,4);
    bar(namesInp,weights);
    ylabel('Input Features')
    title('Feature Ranking using ReliefF')
    grid on;

    %===================================================
    % Plotting Input
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    figure;
    sgtitle('Input Training/Validation/Testing Data (First Dataset only)');

    %----------------------------------------
    % Training Data
    %----------------------------------------
    subplot(1,3,1);
    yyaxis left
    plot(data.tr.t2{1,1}, data.tr.X2{1,1});
    xlabel('time (sec)');
    ylabel('losses (W)');
    yyaxis right
    plot(data.tr.t2{1,1}, data.tr.y2{1,1});
    title('Train Data-1');
    ylabel('T (°C)');
    grid on;

    %----------------------------------------
    % Validation Data
    %----------------------------------------
    subplot(1,3,2);
    yyaxis left
    plot(data.vl.t2{1,1}, data.vl.X2{1,1});
    xlabel('time (sec)');
    ylabel('losses (W)');
    yyaxis right
    plot(data.vl.t2{1,1}, data.vl.y2{1,1});
    title('Validation Data-1');
    ylabel('T (°C)');
    grid on;

    %----------------------------------------
    % Testing Data
    %----------------------------------------
    subplot(1,3,3);
    plot(time, data.te.X);
    xlabel('time (sec)');
    ylabel('losses (W)');
    yyaxis right
    plot(time, data.te.y);
    title('Testing Data-1');
    ylabel('T (°C)');
    grid on;

    %===================================================
    % Average Accuracies
    %===================================================
    %----------------------------------------
    % Correlation
    %----------------------------------------
    figure;
    subplot(2,2,1);
    hold on;
    title('Scattering Prediction and Residuals');
    for i = 1:Ny
        scatter(yTest, yPred);
        xlabel('True Values');
        ylabel('Pred Values');
    end
    grid on;

    %----------------------------------------
    % Error Distribution
    %----------------------------------------
    subplot(2,2,2);
    hold on;
    title('Residual Distribution');
    for i = 1:Ny
        histogram(err,'Normalization','probability');
        xlabel('Error');
        ylabel('Data Samples (%)');
    end
    grid on;

    %----------------------------------------
    % Accuracy Metric
    %----------------------------------------
    % Total Error
    subplot(2,2,3);
    bar(metrics,errTot);
    ylabel('Error (K)')
    title('Total Error Values')
    grid on;

    % Steady Error
    subplot(2,2,4);
    bar(metrics,errSS);
    ylabel('Error (K)')
    title('Steady State Error Values')
    grid on;

    %===================================================
    % Plotting Predictions
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    figure;
    sgtitle('Predicted Temperature and Error for Testing Data');

    %----------------------------------------
    % Predictions
    %----------------------------------------
    % Init
    subplot(2,1,1);
    hold on

    % Prediction
    set(gca,'ColorOrderIndex',1);
    for i = 1:Ny
        set(gca,'ColorOrderIndex',i);
        plot(time, yTest(:,i));
        plot(time, yPred(:,i),'--');
    end
    
    % Reference
    plot(time,rTest,'k--');

    % Labels
    xlabel('time (sec)');
    ylabel('temperature (°C)');
    title('Temperature Prediction');
    grid on;
    
    %----------------------------------------
    % Error
    %----------------------------------------
    % Init
    subplot(2,1,2);
    hold on

    % Prediction
    set(gca,'ColorOrderIndex',1);
    for i = 1:Ny
        set(gca,'ColorOrderIndex',i);
        yErr = yTest(:,i) - yPred(:,i);
        plot(time, yErr);
    end

    % Labels
    xlabel('time (sec)');
    ylabel('error (K)');
    title('Temperature Prediction Error');
    grid on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Plotting results")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1