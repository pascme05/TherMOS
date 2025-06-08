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
    [Nt, Ny] = size(data.te.y);                                             % number of output samples
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

    % %----------------------------------------
    % % Boundary
    % %----------------------------------------
    % try
    %     hc = data.tr.Data.hc;                                                  % heat transfer coefficient (W/m²K)
    %     fl = data.tr.Data.fl;                                                  % heat flux (W/m²)
    %     Ta = data.tr.Data.Ta;                                                  % ambient temperature (°C)
    % catch
    %     hc = zeros(size(k));
    %     fl = zeros(size(k));
    %     Ta = zeros(size(k));
    % end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    % Train
    train.X = squeeze(map3D(train.X, xInp, yInp, zInp, x, y, z, 1));
    train.y = squeeze(map3D(train.y, xInp, yInp, zInp, x, y, z, 1));

    % Test
    test.X = squeeze(map3D(test.X, xInp, yInp, zInp, x, y, z, 1));
    test.y = squeeze(map3D(test.y, xInp, yInp, zInp, x, y, z, 1));

    % Val
    val.X = squeeze(map3D(val.X, xInp, yInp, zInp, x, y, z, 1));
    val.y = squeeze(map3D(val.y, xInp, yInp, zInp, x, y, z, 1));

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
    set(gca,'ytick',z)
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
    set(gca,'ytick',z)
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
    set(gca,'ytick',z)
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
    slice(X,Y,Z,squeeze(train.X(end, :, :, :)),xslice,yslice,zslice);
    title("Heat Generation (Train)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar

    subplot(2,3,4);
    slice(X,Y,Z,squeeze(train.y(end, :, :, :)),xslice,yslice,zslice);
    title("Temperatures (Train)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar
    
    % Testing 
    subplot(2,3,2);
    slice(X,Y,Z,squeeze(test.X(end, :, :, :)),xslice,yslice,zslice);
    title("Heat Generation (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar

    subplot(2,3,5);
    slice(X,Y,Z,squeeze(test.y(end, :, :, :)),xslice,yslice,zslice);
    title("Temperatures (Test)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar
    
    % Validaton 
    subplot(2,3,3);
    slice(X,Y,Z,squeeze(val.X(end, :, :, :)),xslice,yslice,zslice);
    title("Heat Generation (Val)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar

    subplot(2,3,6);
    slice(X,Y,Z,squeeze(val.y(end, :, :, :)),xslice,yslice,zslice);
    title("Temperatures (Val)")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    grid on
    set(gca,'xtick',x)
    set(gca,'ytick',y)
    set(gca,'ytick',z)
    colorbar
    
    %----------------------------------------
    % Temporal Inputs
    %----------------------------------------
    % Init
    figure;
    txt = 'Temporal Input Data for Training (1), Testing, and Validation for x=' + ...
           string(selX*dx-dx) + 'm and y=' + string(selY*dy-dy) + 'm and z=' + string(selZ*dz-dz);
    sgtitle(txt);
    
    % Training
    subplot(2,3,1);
    plot(t_train, squeeze(train.X(:, selY, selX, selZ)));
    title("Heat Generation (Train)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,4);
    plot(t_train, squeeze(train.y(:, selY, selX, selZ)));
    title("Temperatures (Train)")
    xlabel("t (sec)");
    ylabel("T (°C)");
    grid on
    
    % Testing 
    subplot(2,3,2);
    plot(t_test, squeeze(test.X(:, selY, selX, selZ)));
    title("Heat Generation (Test)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,5);
    plot(t_test, squeeze(test.y(:, selY, selX, selZ)));
    title("Temperatures (Test)")
    xlabel("t (sec)");
    zlabel("T (°C)");
    grid on
    
    % Validaton 
    subplot(2,3,3);
    plot(t_val, squeeze(val.X(:, selY, selX, selZ)));
    title("Heat Generation (Val)")
    xlabel("t (sec)");
    ylabel("q (W/m³)");
    grid on
    subplot(2,3,6);
    plot(t_val, squeeze(val.y(:, selY, selX, selZ)));
    title("Temperatures (Val)")
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
    plot(x, squeeze(test.q(selT,selY,:,selY)));
    hold on;
    plot(x, squeeze(pred.q(selT,selY,:,selY)));
    title("Gradient (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(selT,selY,:,selY)) - squeeze(pred.q(selT,selY,:,selY));
    plot(x, err);
    title("Gradient Error (X)")
    xlabel("x (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction X Fluxes
    subplot(2,3,3);
    plot(x, squeeze(test.qk(selT,selY,:,selY)));
    hold on;
    plot(x, squeeze(pred.qk(selT,selY,:,selY)));
    title("Flux (X)")
    xlabel("x (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(selT,selY,:,selY)) - squeeze(pred.qk(selT,selY,:,selY));
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
    plot(y, squeeze(test.q(selT,:,selX,selZ)));
    hold on;
    plot(y, squeeze(pred.q(selT,:,selX,selZ)));
    title("Gradient (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');
    subplot(2,3,5);
    err = squeeze(test.q(selT,:,selX,selZ)) - squeeze(pred.q(selT,:,selX,selZ));
    plot(y, err);
    title("Gradient Error (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction Y Fluxes
    subplot(2,3,3);
    plot(y, squeeze(test.qk(selT,:,selX,selZ)));
    hold on;
    plot(y, squeeze(pred.qk(selT,:,selX,selZ)));
    title("Flux (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');
    subplot(2,3,6);
    err = squeeze(test.qk(selT,:,selX,selZ)) - squeeze(pred.qk(selT,:,selX,selZ));
    plot(y, err);
    title("Flux Error (Y)")
    xlabel("y (m)");
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
    plot(z, squeeze(test.y(selT,selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.y(selT,selY,selX,:)));
    title("Temperatures (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    legend('True','Pred');

    subplot(2,3,4);
    err = squeeze(test.y(selT,selY,selX,:)) - squeeze(pred.y(selT,selY,selX,:));
    plot(z, err);
    title("Temperatures Error (Y)")
    xlabel("y (m)");
    ylabel("T (°C)");
    grid on;
    
    % Model Prediction Y Gradients
    subplot(2,3,2);
    plot(z, squeeze(test.q(selT,selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.q(selT,selY,selX,:)));
    title("Gradient (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    legend('True','Pred');

    subplot(2,3,5);
    err = squeeze(test.q(selT,selY,selX,:)) - squeeze(pred.q(selT,selY,selX,:));
    plot(z, err);
    title("Gradient Error (Y)")
    xlabel("y (m)");
    ylabel("dT (K/m)");
    grid on;
    
    % Model Prediction Y Fluxes
    subplot(2,3,3);
    plot(z, squeeze(test.qk(selT,selY,selX,:)));
    hold on;
    plot(z, squeeze(pred.qk(selT,selY,selX,:)));
    title("Flux (Y)")
    xlabel("y (m)");
    ylabel("q (W/m²)");
    grid on;
    legend('True','Pred');

    subplot(2,3,6);
    err = squeeze(test.qk(selT,selY,selX,:)) - squeeze(pred.qk(selT,selY,selX,:));
    plot(z, err);
    title("Flux Error (Y)")
    xlabel("y (m)");
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
    slice(X,Y,Z,squeeze(pred.q(selT,:,:,:)),xslice,yslice,zslice);
    title("Gradient")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;

    subplot(2,3,5);
    err = squeeze(test.q(selT,:,:,:)) - squeeze(pred.q(selT,:,:,:));
    slice(X,Y,Z,err,xslice,yslice,zslice);
    title("Gradient Error")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;
    
    % Fluxes
    subplot(2,3,3);
    slice(X,Y,Z,squeeze(pred.qk(selT,:,:,:)),xslice,yslice,zslice);
    title("Flux")
    xlabel("x (m)");
    ylabel("y (m)");
    zlabel("z (m)");
    colorbar
    grid on;

    subplot(2,3,6);
    err = squeeze(test.qk(selT,:,:,:)) - squeeze(pred.qk(selT,:,:,:));
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

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1