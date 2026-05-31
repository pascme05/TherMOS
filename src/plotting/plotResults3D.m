%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: plotResults3D                                                     %
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
function [] = plotResults3D(data, results, mdl, setup, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("INFO: Plotting results 3D")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    K = mdl.K;                                                              % number of modes
    selX = para.Dat.gen.inpX;                                               % x-position for 1D temporal plots                           
    selY = para.Dat.gen.inpY;                                               % y-position for 1D temporal plots
    selZ = para.Dat.gen.inpZ;                                               % y-position for 1D temporal plots
    selT = -1;                                                              % temporal instant for 1D/2D spatial plots (if -1 last sample)
    Ts = data.tr.Ts;
    dx = data.tr.Data.dx;
    dy = data.tr.Data.dy;
    dz = data.tr.Data.dz;
    Ly = data.tr.Data.Ly;                                                   % length in y direction (m)
    Lx = data.tr.Data.Lx;                                                   % length in x direction (m)
    Lz = data.tr.Data.Lz;                                                   % length in x direction (m)

    %===================================================
    % Variables
    %===================================================
    %----------------------------------------
    % Split Data
    %----------------------------------------
    test = data.te;
    pred = data.pr;

    %----------------------------------------
    % Time
    %----------------------------------------
    t_test = test.t;
    
    %----------------------------------------
    % Derived
    %----------------------------------------
    xInp = data.tr.Data.geo(:,1);                                           % sampled input values x (m)
    yInp = data.tr.Data.geo(:,2);                                           % sampled input values y (m)
    zInp = data.tr.Data.geo(:,3);                                           % sampled input values z (m)
    x = 0:dx:Lx;                                                            % x vector (m)
    y = 0:dy:Ly;                                                            % y vector (m)
    z = 0:dz:Lz;                                                            % z vector (m)
    
    %===================================================
    % Model
    %===================================================
    %----------------------------------------
    % Material
    %----------------------------------------
    k = data.tr.Data.k;
    rho = data.tr.Data.rho;
    Cp = data.tr.Data.Cp;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Shift Coordinate System
    %===================================================
    xInp = xInp - min(xInp);                                                % normalised coordinate system with x=0
    yInp = yInp - min(yInp);                                                % normalised coordinate system with y=0
    zInp = zInp - min(zInp);                                                % normalised coordinate system with z=0
    
    %===================================================
    % Mapping Position
    %===================================================
    %----------------------------------------
    % Spatial
    %----------------------------------------
    % Initial
    [~, selX] = min(abs(x - selX));
    [~, selY] = min(abs(y - selY));
    [~, selZ] = min(abs(z - selZ));

    % Limit
    if selX > length(x)
        selX = length(x);
    end

    if selY > length(y)
        selY = length(y);
    end

    if selZ > length(z)
        selZ = length(z);
    end
    
    %----------------------------------------
    % Temporal
    %----------------------------------------
    if selT == -1
        selT = length(t_test);
    end
    
    %----------------------------------------
    % Slicing
    %----------------------------------------
    [X,Y,Z] = meshgrid(x,y,z);
    xslice = para.Dat.gen.outX;                                             % location of y-z planes
    yslice = para.Dat.gen.inpY;                                             % location of x-z plane
    zslice = para.Dat.gen.inpZ;                                             % location of x-y planes

    %===================================================
    % Mapping 2D Space
    %===================================================
    %----------------------------------------
    % Material
    %----------------------------------------
    k = squeeze(map3D(k', xInp, yInp, zInp, x, y, z, 1));
    rho = squeeze(map3D(rho', xInp, yInp, zInp, x, y, z, 1));
    Cp = squeeze(map3D(Cp', xInp, yInp, zInp, x, y, z, 1));

    %----------------------------------------
    % Input/Output
    %----------------------------------------
    % Test
    test.X = squeeze(map3D(test.X, xInp, yInp, zInp, x, y, z, 1));
    test.y = squeeze(map3D(test.y, xInp, yInp, zInp, x, y, z, 1));

    % Pred
    pred.X = squeeze(map3D(pred.X, xInp, yInp, zInp, x, y, z, 1));
    pred.y = squeeze(map3D(pred.y, xInp, yInp, zInp, x, y, z, 1));


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
    slice(X,Y,Z,k,xslice,yslice,zslice);
    title("Thermal Conductivity k (W/mK)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ztick',z)
    colorbar
    
    %----------------------------------------
    % Specific Heat Capacity
    %----------------------------------------
    subplot(1,3,2);
    slice(X,Y,Z,Cp,xslice,yslice,zslice);
    title("Specific Heat Capacity Cp (J/kgK)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ztick',z)
    colorbar
    
    %----------------------------------------
    % Mass Density
    %----------------------------------------
    subplot(1,3,3);
    slice(X,Y,Z,rho,xslice,yslice,zslice);
    title("Mass Density Rho (kg/m³)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ztick',z)
    colorbar


    %===================================================
    % Inputs
    %===================================================
    %----------------------------------------
    % Spatial Inputs
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Input Data for Testing for t=' + string(selT*Ts - Ts) + 'sec';
    sgtitle(txt);

    % Testing 
    subplot(1,2,1);
    slice(X,Y,Z,squeeze(test.X(end, :, :, :)),xslice,yslice,zslice);
    title("Heat Generation (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ztick',z)
    colorbar

    subplot(1,2,2);
    slice(X,Y,Z,squeeze(test.y(end, :, :, :)),xslice,yslice,zslice);
    title("Temperatures (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ztick',z)
    colorbar
    
    %----------------------------------------
    % Temporal Inputs
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Input Data for Testing for x=' + ...
           string(selX*dx-dx) + 'm and y=' + string(selY*dy-dy) + 'm and z=' + string(selZ*dz-dz);
    sgtitle(txt);
    
    % Testing 
    subplot(1,2,1);
    plot(t_test, squeeze(test.X(:, selY, selX, selZ)));
    title("Heat Generation (Test)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(1,2,2);
    plot(t_test, squeeze(test.y(:, selY, selX, selZ)));
    title("Temperatures (Test)")
    xlabel("t (sec)");
    zlabel("T (°C)");
    grid on
    

    %===================================================
    % Predictions
    %===================================================
    %----------------------------------------
    % Temporal Modes
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Prediction Modes';
    sgtitle(txt);

    % Mode Prediction
    K_max = min([K, 5]);
    for i = 1:K_max
        subplot(2, K_max, i);
        plot(t_test, pred.theta_hat(:,i));
        hold on;
        plot(t_test, pred.theta(:,i)-pred.theta(1,i));
        ylabel("Mag (p.u.)");
        xlabel("t (sec)");
        txt = "Temporal Mode Theta-" + num2str(i);
        title(txt);
        grid on;
    end

    % Mode Error
    for i = 1:K_max
        subplot(2, K_max, i+K_max);
        err = (pred.theta(:,i)-pred.theta(1,i)-pred.theta_hat(:,i))/(pred.theta(end,i)-pred.theta(1,i))*100;
        plot(t_test, err);
        ylabel("Mag Error (%)");
        xlabel("t (sec)");
        txt = "Temporal Mode Error Theta-" + num2str(i);
        title(txt);
        grid on;
    end

    %----------------------------------------
    % Plot Spatial Prediction X
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction error for y=' + string(selY*dy-dy) + ...
          'm and z=' + string(selZ*dz-dz) + 'm and t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Model Prediction X Temperatures
    subplot(2,3,1);
    plot(x, squeeze(test.y(selT,selY,:,selY)));
    hold on;
    plot(x, squeeze(pred.y(selT,selY,:,selY)));
    title("Temperatures (X)")
    xlabel("x (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,3,4);
    err = squeeze(test.y(selT,selY,:,selY)) - squeeze(pred.y(selT,selY,:,selY));
    plot(x, err);
    title("Temperatures Error (X)")
    xlabel("x (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction X Gradients
    subplot(2,3,2);
    plot(x, squeeze(test.q(selY,:,selY)));
    hold on;
    plot(x, squeeze(pred.q(selY,:,selY)));
    title("Gradient (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(selY,:,selY)) - squeeze(pred.q(selY,:,selY));
    plot(x, err);
    title("Gradient Error (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction X Fluxes
    subplot(2,3,3);
    plot(x, squeeze(test.qk(selY,:,selY)));
    hold on;
    plot(x, squeeze(pred.qk(selY,:,selY)));
    title("Flux (X)")
    xlabel("x (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(selY,:,selY)) - squeeze(pred.qk(selY,:,selY));
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
    txt = 'Spatial Prediction and Prediction error for x=' + string(selX*dx-dx) + ...
          'm and z=' + string(selZ*dz-dz) + 'm and t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Model Prediction Y
    subplot(2,3,1);
    plot(y, squeeze(test.y(selT,:,selX,selZ)));
    hold on;
    plot(y, squeeze(pred.y(selT,:,selX,selZ)));
    title("Temperatures (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,3,4);
    err = squeeze(test.y(selT,:,selX,selZ)) - squeeze(pred.y(selT,:,selX,selZ));
    plot(y, err);
    title("Temperatures Error (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction Y Gradients
    subplot(2,3,2);
    plot(y, squeeze(test.q(:,selX,selZ)));
    hold on;
    plot(y, squeeze(pred.q(:,selX,selZ)));
    title("Gradient (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(:,selX,selZ)) - squeeze(pred.q(:,selX,selZ));
    plot(y, err);
    title("Gradient Error (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction Y Fluxes
    subplot(2,3,3);
    plot(y, squeeze(test.qk(:,selX,selZ)));
    hold on;
    plot(y, squeeze(pred.qk(:,selX,selZ)));
    title("Flux (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(:,selX,selZ)) - squeeze(pred.qk(:,selX,selZ));
    plot(y, err);
    title("Flux Error (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    
    %----------------------------------------
    % Plot Spatial Prediction Z
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction error for x=' + string(selX*dx-dx) + ...
          'm and z=' + string(selZ*dz-dz) + 'm and t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Model Prediction Z
    subplot(2,3,1);
    plot(z, squeeze(test.y(selT,selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.y(selT,selY,selX,:)));
    title("Temperatures (Z)")
    xlabel("z (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');

    subplot(2,3,4);
    err = squeeze(test.y(selT,selY,selX,:)) - squeeze(pred.y(selT,selY,selX,:));
    plot(z, err);
    title("Temperatures Error (Z)")
    xlabel("z (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction Z Gradients
    subplot(2,3,2);
    plot(z, squeeze(test.q(selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.q(selY,selX,:)));
    title("Gradient (Z)")
    xlabel("z (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');

    subplot(2,3,5);
    err = squeeze(test.q(selY,selX,:)) - squeeze(pred.q(selY,selX,:));
    plot(z, err);
    title("Gradient Error (Z)")
    xlabel("z (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction Z Fluxes
    subplot(2,3,3);
    plot(z, squeeze(test.qk(selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.qk(selY,selX,:)));
    title("Flux (Z)")
    xlabel("z (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');

    subplot(2,3,6);
    err = squeeze(test.qk(selY,selX,:)) - squeeze(pred.qk(selY,selX,:));
    plot(z, err);
    title("Flux Error (Z)")
    xlabel("z (m)");
    ylabel("q (W/m²)");
    grid on;

    %----------------------------------------
    % Plot Spatial Prediction (Plane)
    %----------------------------------------
    % Init
    figure;
    txt = 'Spatial Prediction and Prediction Error for t=' + string(selT*Ts-Ts) + 'sec';
    sgtitle(txt);
    
    % Temperatures
    subplot(2,3,1);
    slice(X,Y,Z,squeeze(pred.y(selT,:,:,:)),xslice,yslice,zslice);
    title("Temperatures")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;

    subplot(2,3,4);
    err = squeeze(test.y(selT,:,:,:)) - squeeze(pred.y(selT,:,:,:));
    slice(X,Y,Z,err,xslice,yslice,zslice);
    title("Temperatures Error")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;
    
    % Gradients
    subplot(2,3,2);
    slice(X,Y,Z,pred.q,xslice,yslice,zslice);
    title("Gradient")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;

    subplot(2,3,5);
    err = test.q - pred.q;
    slice(X,Y,Z,err,xslice,yslice,zslice);
    title("Gradient Error")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;
    
    % Fluxes
    subplot(2,3,3);
    slice(X,Y,Z,pred.qk,xslice,yslice,zslice);
    title("Flux")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;

    subplot(2,3,6);
    err = test.qk - pred.qk;
    slice(X,Y,Z,err,xslice,yslice,zslice);
    title("Flux Error")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;
    
    %----------------------------------------
    % Plot Temporal Prediction
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Prediction and Prediction error for x=' + ... 
           string(selX*dx-dx) + 'm and y=' + string(selY*dy-dy) + ... 
           'm and z=' + string(selZ*dz-dz) + 'm';
    sgtitle(txt);
    
    % Model Prediction
    subplot(2,1,1);
    plot(t_test, squeeze(test.y(:,selY,selX,selZ)));
    hold on;
    plot(t_test, squeeze(pred.y(:,selY,selX,selZ)));
    title("Absolute Temperatures (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');

    subplot(2,1,2);
    err = squeeze(test.y(:,selY,selX,selZ)) - squeeze(pred.y(:,selY,selX,selZ));
    plot(t_test, err);
    title("Temperatures Error (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;

    %----------------------------------------
    % Plot Temporal Prediction (Average)
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Prediction and Prediction error for averaged over the total geometry';
    sgtitle(txt);
    
    % Model Prediction
    subplot(2,1,1);
    plot(t_test, mean(squeeze(test.y),[2,3,4]));
    hold on;
    plot(t_test, mean(squeeze(pred.y),[2,3,4]));
    title("Absolute Temperatures (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');
    subplot(2,1,2);
    err = mean(squeeze(test.y) - squeeze(pred.y),[2,3,4]);
    plot(t_test, err);
    title("Temperatures Error (T)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on;

    %----------------------------------------
    % Plot 3D
    %----------------------------------------
    try
        % Init
        figure;
        txt = 'Spatial 3D Prediction and Prediction error for t=' + string(selT*Ts-Ts) + 'sec';
        sgtitle(txt);
    
        % Grt
        subplot(1,3,1);
        pdeplot3D(data.te.Data.mesh, 'ColorMapData', data.te.Data.y(end, :));
        title("Grt Temperatures (T)")
        colorbar;
    
        % Pred
        subplot(1,3,2);
        pdeplot3D(data.te.Data.mesh, 'ColorMapData', data.pr.y(end,:));
        title("Est Temperatures (T)")
        colorbar;
    
        % Error
        subplot(1,3,3);
        err = data.pr.y(end,:) - data.te.y(end,:);
        pdeplot3D(data.te.Data.mesh, 'ColorMapData', err);
        title("Err Temperatures (T)")
        colorbar;
    catch
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1