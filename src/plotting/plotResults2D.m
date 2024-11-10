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
%       3) mdl:     All model parameters
%       4) setup:   All simulation setup parameters
%       5) para:    All parameters
% Out:  1) None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plotResults2D(data, results, mdl, setup, para)
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
    [Nt, Ny] = size(data.te.y);                                              % number of output samples
    selX = para.Dat.gen.inpX;                                               % x-position for 1D temporal plots                           
    selY = para.Dat.gen.inpY;                                               % y-position for 1D temporal plots
    selT = -1;                                                              % temporal instant for 1D/2D spatial plots (if -1 last sample)
    Ts = data.tr.Ts;
    dx = data.tr.Data.dx;
    dy = data.tr.Data.dy;
    Ly = data.tr.Data.Ly;                                                      % length in y direction (m)
    Lx = data.tr.Data.Lx;                                                      % length in x direction (m)

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Split Data
    %----------------------------------------
    train = data.tr;
    test = data.te;
    val = data.vl;
    pred = data.pr;

    %----------------------------------------
    % Time
    %----------------------------------------
    t_train = train.t;
    t_test = test.t;
    t_val = val.t;
    
    %----------------------------------------
    % Derived
    %----------------------------------------
    xInp = data.tr.Data.geo(:,1);                                              % sampled input values x (m)
    yInp = data.tr.Data.geo(:,2);                                              % sampled input values y (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    
    %===================================================
    % Model
    %===================================================
    k = data.tr.Data.k;
    rho = data.tr.Data.rho;
    Cp = data.tr.Data.Cp;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Mapping Position
    %===================================================
    %----------------------------------------
    % Spatial
    %----------------------------------------
    [~, selX] = min(abs(x - selX));
    [~, selY] = min(abs(x - selY));
    
    %----------------------------------------
    % Temporal
    %----------------------------------------
    if selT == -1
        selT = length(t_test);
    end

    %===================================================
    % Mapping 2D Space
    %===================================================
    %----------------------------------------
    % Material
    %----------------------------------------
    k = squeeze(map2D(k', xInp, yInp, x, y));
    rho = squeeze(map2D(rho', xInp, yInp, x, y));
    Cp = squeeze(map2D(Cp', xInp, yInp, x, y));

    %----------------------------------------
    % Input/Output
    %----------------------------------------
    % Train
    train.X = squeeze(map2D(train.X, xInp, yInp, x, y));
    train.y = squeeze(map2D(train.y, xInp, yInp, x, y));

    % Test
    test.X = squeeze(map2D(test.X, xInp, yInp, x, y));
    test.y = squeeze(map2D(test.y, xInp, yInp, x, y));

    % Val
    val.X = squeeze(map2D(val.X, xInp, yInp, x, y));
    val.y = squeeze(map2D(val.y, xInp, yInp, x, y));

    % Pred
    pred.X = squeeze(map2D(pred.X, xInp, yInp, x, y));
    pred.y = squeeze(map2D(pred.y, xInp, yInp, x, y));

    %----------------------------------------
    % Output
    %----------------------------------------


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Material Properties
    %===================================================
    %----------------------------------------
    % Init
    %----------------------------------------
    figure;
    txt = 'Spatially Distributed Material data';
    sgtitle(txt);
    
    %----------------------------------------
    % Thermal Conductivity
    %----------------------------------------
    subplot(1,3,1);
    pcolor(x,y,k);
    title("Thermal Conductivity k (W/mK)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("k (W/mK)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    
    %----------------------------------------
    % Specific Heat Capacity
    %----------------------------------------
    subplot(1,3,2);
    pcolor(x,y,Cp);
    title("Specific Heat Capacity Cp (J/kgK)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("Cp (J/kgK)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    
    %----------------------------------------
    % Mass Density
    %----------------------------------------
    subplot(1,3,3);
    pcolor(x,y,rho);
    title("Mass Density Rho (kg/m³)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("Rho (kg/m³)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar

    %===================================================
    % Inputs
    %===================================================
    %----------------------------------------
    % Spatial Inputs
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Input Data for Training (1), Testing, and Validation for t=' + string(selT*Ts - Ts) + 'sec';
    sgtitle(txt);
    
    % Training
    subplot(2,3,1);
    pcolor(x,y,squeeze(train.X(end, :, :)));
    title("Heat Generation (Train)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("q (W/m³)");
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    subplot(2,3,4);
    pcolor(x,y,squeeze(train.y(end, :, :)));
    title("Temperatures (Train)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("T (°C)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    
    % Testing 
    subplot(2,3,2);
    pcolor(x,y,squeeze(test.X(end, :, :)));
    title("Heat Generation (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("q (W/m³)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    subplot(2,3,5);
    pcolor(x,y,squeeze(test.y(end, :, :)));
    title("Temperatures (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("T (°C)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    
    % Validaton 
    subplot(2,3,3);
    pcolor(x,y,squeeze(val.X(end, :, :)));
    title("Heat Generation (Val)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("q (W/m³)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    subplot(2,3,6);
    pcolor(x,y,squeeze(val.y(end, :, :)));
    title("Temperatures (Val)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("T (°C)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    colorbar
    
    
    %----------------------------------------
    % Temporal Inputs
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Input Data for Training (1), Testing, and Validation for x=' + string(selX*dx-dx) + 'm and y=' + string(selY*dy-dy) + 'm';
    sgtitle(txt);
    
    % Training
    subplot(2,3,1);
    plot(t_train, squeeze(train.X(:, selX, selY)));
    title("Heat Generation (Train)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,4);
    plot(t_train, squeeze(train.y(:, selX, selY)));
    title("Temperatures (Train)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on
    
    % Testing 
    subplot(2,3,2);
    plot(t_test, squeeze(test.X(:, selX, selY)));
    title("Heat Generation (Test)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,5);
    plot(t_test, squeeze(test.y(:, selX, selY)));
    title("Temperatures (Test)")
    xlabel("t (sec)");
    zlabel("T (°C)");
    grid on
    
    % Validaton 
    subplot(2,3,3);
    plot(t_val, squeeze(val.X(:, selX, selY)));
    title("Heat Generation (Val)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,6);
    plot(t_val, squeeze(val.y(:, selX, selY)));
    title("Temperatures (Val)")
    xlabel("t (sec)");
    zlabel("T (°C)");
    grid on

    %===================================================
    % Predictions
    %===================================================
    %----------------------------------------
    % Plot Spatial Prediction X
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction error for y=' + string(selY*dy-dy) + 'm and t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Model Prediction X Temperatures
    subplot(2,3,1);
    plot(x, squeeze(test.y(selT,selY,:)));
    hold on;
    plot(x, squeeze(pred.y(selT,selY,:)));
    title("Temperatures (X)")
    xlabel("x (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,3,4);
    err = squeeze(test.y(selT,selY,:)) - squeeze(pred.y(selT,selY,:));
    plot(x, err);
    title("Temperatures Error (X)")
    xlabel("x (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction X Gradients
    subplot(2,3,2);
    plot(x, squeeze(test.q(selT,selY,:)));
    hold on;
    plot(x, squeeze(pred.q(selT,selY,:)));
    title("Gradient (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(selT,selY,:)) - squeeze(pred.q(selT,selY,:));
    plot(x, err);
    title("Gradient Error (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction X Fluxes
    subplot(2,3,3);
    plot(x, squeeze(test.qk(selT,selY,:)));
    hold on;
    plot(x, squeeze(pred.qk(selT,selY,:)));
    title("Flux (X)")
    xlabel("x (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(selT,selY,:)) - squeeze(pred.qk(selT,selY,:));
    plot(x, err);
    title("Flux Error (W/m²)")
    xlabel("x (m)");
    ylabel("q (W/m²)");
    grid on;
    
    %----------------------------------------
    % Plot Spatial Prediction Y
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction error for x=' + string(selX*dx-dx) + 'm and t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Model Prediction Y
    subplot(2,3,1);
    plot(y, squeeze(test.y(selT,:,selX)));
    hold on;
    plot(y, squeeze(pred.y(selT,:,selX)));
    title("Temperatures (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,3,4);
    err = squeeze(test.y(selT,:,selX)) - squeeze(pred.y(selT,:,selX));
    plot(y, err);
    title("Temperatures Error (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction Y Gradients
    subplot(2,3,2);
    plot(y, squeeze(test.q(selT,:,selX)));
    hold on;
    plot(y, squeeze(pred.q(selT,:,selX)));
    title("Gradient (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(selT,:,selX)) - squeeze(pred.q(selT,:,selX));
    plot(y, err);
    title("Gradient Error (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction Y Fluxes
    subplot(2,3,3);
    plot(y, squeeze(test.qk(selT,:,selX)));
    hold on;
    plot(y, squeeze(pred.qk(selT,:,selX)));
    title("Flux (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(selT,:,selX)) - squeeze(pred.qk(selT,:,selX));
    plot(y, err);
    title("Flux Error (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    
    %----------------------------------------
    % Plot Spatial Prediction X-Y
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction Error for t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Temperatures
    subplot(2,3,1);
    pcolor(x,y,squeeze(pred.y(selT,:,:)));
    title("Temperatures (X-Y)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    subplot(2,3,4);
    err = squeeze(test.y(selT,:,:)) - squeeze(pred.y(selT,:,:));
    pcolor(x,y,err);
    title("Temperatures Error (X)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    
    % Gradients
    subplot(2,3,2);
    pcolor(x,y,squeeze(pred.q(selT,:,:)));
    title("Gradient (X-Y)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    subplot(2,3,5);
    err = squeeze(test.q(selT,:,:)) - squeeze(pred.q(selT,:,:));
    pcolor(x,y,err);
    title("Gradient Error (X)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    
    % Fluxes
    subplot(2,3,3);
    pcolor(x,y,squeeze(pred.qk(selT,:,:)));
    title("Flux (X-Y)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    subplot(2,3,6);
    err = squeeze(test.qk(selT,:,:)) - squeeze(pred.qk(selT,:,:));
    pcolor(x,y,err);
    title("Flux Error (X)")
    xlabel("x (m)");
    ylabel("y (m)");
    colorbar
    grid on;
    
    %----------------------------------------
    % Plot Temporal Prediction
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Prediction and Prediction error for x=' + string(selX*dx-dx) + 'm and y=' + string(selY*dy-dy) + 'm';
    sgtitle(txt);
    
    % Model Prediction
    subplot(2,1,1);
    plot(t_test, squeeze(test.y(:,selY,selX)));
    hold on;
    plot(t_test, squeeze(pred.y(:,selY,selX)));
    title("Absolute Temperatures (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,1,2);
    err = squeeze(test.y(:,selY,selX)) - squeeze(pred.y(:,selY,selX));
    plot(t_test, err);
    title("Temperatures Error (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Plotting results 2D")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1