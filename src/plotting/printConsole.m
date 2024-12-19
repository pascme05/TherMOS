%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: printConsole                                                      %
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
% This function prints the results on the console.
% -------------------------------------------------------------------------
% Inp:  1) data:    All simulation input data as well as prediction
%       2) results: All obtained accuracy values and results
%       3) setup:   All simulation setup parameters
%       4) mdl:     All model parameters
% Out:  1) None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = printConsole(data, results, setup, mdl)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [Nt, ~] = size(data.pr.y);
    nY = length(results.err.tot.MAE);
    namesOut = setup.out;
 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Performance Values
    %===================================================
    %----------------------------------------
    % Total
    %----------------------------------------
    MAE_tot = results.err.tot.MAE;
    MSE_tot = (results.err.tot.RMSE).^2;
    MAX_tot = results.err.tot.MAX;
    NRS_tot = results.err.tot.NRSME;

    %----------------------------------------
    % Steady State
    %----------------------------------------
    MAE_ss = results.err.ss.MAE;
    MSE_ss = (results.err.ss.RMSE).^2;
    MAX_ss = results.err.ss.MAX;
    NRS_ss = results.err.ss.NRSME;

    %----------------------------------------
    % Transient
    %----------------------------------------
    MAE_tr = results.err.tr.MAE;
    MSE_tr = (results.err.tr.RMSE).^2;
    MAX_tr = results.err.tr.MAX;
    NRS_tr = results.err.tr.NRSME;

    
    %===================================================
    % Average Performance Values
    %===================================================
    %----------------------------------------
    % Total
    %----------------------------------------
    MAE_tot_avg = mean(results.err.tot.MAE);
    MSE_tot_avg = mean((results.err.tot.RMSE).^2);
    MAX_tot_avg = max(results.err.tot.MAX);
    NRS_tot_avg = mean(results.err.tot.NRSME);

    %----------------------------------------
    % Steady State
    %----------------------------------------
    MAE_ss_avg = mean(results.err.ss.MAE);
    MSE_ss_avg = mean((results.err.ss.RMSE).^2);
    MAX_ss_avg = max(results.err.ss.MAX);
    NRS_ss_avg = mean(results.err.ss.NRSME);

    %----------------------------------------
    % Transient
    %----------------------------------------
    MAE_tr_avg = mean(results.err.tr.MAE);
    MSE_tr_avg = mean((results.err.tr.RMSE).^2);
    MAX_tr_avg = max(results.err.tr.MAX);
    NRS_tr_avg = mean(results.err.tr.NRSME);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Init
    %===================================================
    disp('================================================================================================================')
    disp('================================================================================================================')
    disp('|          |         MAE (K)        |         MSE (K²)       |         MAX (K)        |        NRMSE (%)       |')
    disp('|   Node   | Total | Steady | Trans | Total | Steady | Trans | Total | Steady | Trans | Total | Steady | Trans |')
    disp('----------------------------------------------------------------------------------------------------------------')

    %===================================================
    % Looping through nodes
    %===================================================
    for i = 1:nY
        fprintf('|  %-7s | %5.2f | %5.2f | %6.2f | %5.2f | %6.2f | %5.2f | %5.2f | %6.2f | %5.2f | %5.2f | %6.2f | %5.2f |\n', ...
            namesOut(i), ... 
            MAE_tot(i), MAE_ss(i), MAE_tr(i), ...  % MAE values
            MSE_tot(i), MSE_ss(i), MSE_tr(i), ...  % MSE values
            MAX_tot(i), MAX_ss(i), MAX_tr(i), ...  % MAX values
            NRS_tot(i), NRS_ss(i), NRS_tr(i));     % NRMSE values
    end
    
    %===================================================
    % Averages row
    %===================================================
    disp('----------------------------------------------------------------------------------------------------------------')
    disp('----------------------------------------------------------------------------------------------------------------')
    fprintf('|  Avg     | %5.2f | %5.2f | %6.2f | %5.2f | %6.2f | %5.2f | %5.2f | %6.2f | %5.2f | %5.2f | %6.2f | %5.2f |\n', ...
        MAE_tot_avg, MAE_ss_avg, MAE_tr_avg, ...  % Average MAE values
        MSE_tot_avg, MSE_ss_avg, MSE_tr_avg, ...  % Average MSE values
        MAX_tot_avg, MAX_ss_avg, MAX_tr_avg, ...  % Average MAX values
        NRS_tot_avg, NRS_ss_avg, NRS_tr_avg);     % Average NRMSE values
    
    %===================================================
    % End of table
    %===================================================
    disp('================================================================================================================')
    disp('================================================================================================================')
    
    %===================================================
    % Training and Testing Times
    %===================================================
    fprintf('INFO: Training time (sec): %5.2f \n', mdl.timeTrain);
    fprintf('INFO: Inference time (us/sample): %5.2f \n', data.pr.testTime*1e6/Nt);

    %===================================================
    % Memory Requirements
    %===================================================
    fprintf('INFO: Number Model Parameters: %-5d \n', mdl.size);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1