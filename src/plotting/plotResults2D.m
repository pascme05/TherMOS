%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: plotResults2D                                                     %
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
% This function plots the results from a 1D model.
% -------------------------------------------------------------------------
% Inp:  1) data:    All simulation input data as well as prediction
%       2) results: All obtained accuracy values and results
%       3) setup:   All simulation setup parameters
% Out:  1) None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plotResults2D(data, results, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Plotting results 2D")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [~, Ny] = size(data.te.y);                                              % number of output samples

    %===================================================
    % Variables
    %===================================================
    time = data.te.t;
    yPred = data.pr.y;
    yTest = data.te.y;
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
    % Mapping 2D Space
    %===================================================


    %===================================================
    % Feature Ranking
    %===================================================
    weights = zeros(length(namesInp), length(namesOut));
    if setup.featRank == 1
        for i = 1:Ny
            [~, weights(:,i)] = relieff(data.tr.X,data.tr.y(:,i),10);
        end
        weights = mean(weights,2);
    end

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
    errTot = [results.err.tot.MAE; results.err.tot.RMSE; results.err.tot.MAX];
    errSS = [results.err.ss.MAE; results.err.ss.RMSE; results.err.ss.MAX];

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
    boxplot(data.tr.X, namesInp);
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
    yyaxis left
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
        scatter(yTest(:,i)/max(yTest(:,i)), yPred(:,i)/max(yPred(:,i)));
        xlabel('True Values (p.u.)');
        ylabel('Pred Values (p.u.)');
    end
    xlim([0 1])
    ylim([0 1])
    grid on;
    legend(namesOut, 'Location','northwest');

    %----------------------------------------
    % Error Distribution
    %----------------------------------------
    subplot(2,2,2);
    hold on;
    title('Residual Distribution');
    for i = 1:Ny
        histogram(err(:,i),'Normalization','probability');
        xlabel('Error (K)');
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
    % Plotting Predictions (One Figure)
    %===================================================
    if setup.plotOut == 1
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
            set(gca,'ColorOrderIndex',i);
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
        legend(namesOut, 'Location','southeast', 'NumColumns', Ny);
    end
    
    %===================================================
    % Plotting Predictions (Multi Figure)
    %===================================================
    if setup.plotOut == 2
        for i = 1:Ny
            %----------------------------------------
            % Init
            %----------------------------------------
            figure;
            txt = 'Predicted Temperature and Error for Testing Data of Signal ' + namesOut(i);
            sgtitle(txt);
        
            %----------------------------------------
            % Predictions
            %----------------------------------------
            % Init
            subplot(2,1,1);
            hold on
        
            % Prediction
            set(gca,'ColorOrderIndex',1);
            plot(time, yTest(:,i));
            set(gca,'ColorOrderIndex',1);
            plot(time, yPred(:,i),'--');
            
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
            yErr = yTest(:,i) - yPred(:,i);
            plot(time, yErr);
    
            % Labels
            xlabel('time (sec)');
            ylabel('error (K)');
            title('Temperature Prediction Error');
            grid on;
            legend(namesOut, 'Location','southeast', 'NumColumns', length(namesOut));
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Plotting results 2D")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1