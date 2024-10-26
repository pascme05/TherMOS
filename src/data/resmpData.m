%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: resmpData                                                         %
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
% This function resamples the temporal axis of the data using different
% resampling techniques.
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) para:    All simulation parameters of the current simulation
% Out:  1) outX:    Adapted input simulation data
%       2) outY:    Adapted output simulation data
%       3) outR:    Adapted reference simulation data
%       4) outT:    Adapted time simulation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xOut, yOut, rOut, tRef]  = resmpData(data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Resampling (temporal) data")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ts = para.Exp.gen.Ts;                                                   % target sampling time (sec)
    Ts_data = data.t(2) - data.t(1);                                        % current samplin time (sec)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outDim = size(data.y);                                                  % output data dimension
    inpDim = size(data.X);                                                  % input data dimension
    refDim = size(data.r);                                                  % reference data dimension

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Time Vector
    %===================================================
    tRef = 0:Ts:outDim(1)*Ts_data - Ts_data;
    tDat = 0:Ts_data:outDim(1)*Ts_data - Ts_data;
    
    %===================================================
    % Init Output
    %===================================================
    yOut = zeros(length(tRef), outDim(2:end));
    xOut = zeros(length(tRef), inpDim(2:end));
    rOut = zeros(length(tRef), refDim(2:end));

    %===================================================
    % Interpolation Method
    %===================================================
    if para.Exp.gen.samp == 1
        intp = 'previous';
        disp("INFO: Using ZoH for interpolation")
    elseif para.Exp.gen.samp == 2
        intp = 'linear';
        disp("INFO: Using linear interpolation")
    elseif para.Exp.gen.samp == 3
        intp = 'spline';
        disp("INFO: Using spline interpolation")
    else
        disp("WARN: Invalid interpolation method, ZOH is used.")
        intp = 'previous';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Resampling Output (y)
    %===================================================
    %----------------------------------------
    % Spatial 1D
    %----------------------------------------
    if (length(outDim) - 1) == 1
        for i = 1:outDim(2)
            yOut(:,i) = interp1(tDat, data.y(:,i), tRef, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(outDim) - 1) == 2
        for i = 1:outDim(2)
            for ii = 1:outDim(3)
                yOut(:,i,ii) = interp1(tDat, data.y(:,i,ii), tRef, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(outDim) - 1) == 3
        for i = 1:outDim(2)
            for ii = 1:outDim(3)
                for iii = 1:outDim(4)
                    yOut(:,i,ii) = interp1(tDat, data.y(:,i,ii,iii), tRef, intp);
                end
            end
        end

    %----------------------------------------
    % Incorrect Dim
    %----------------------------------------
    else
        disp("WARN: Incorrect output dimension, no resampling")
    end

    %===================================================
    % Resampling Input (X)
    %===================================================
    %----------------------------------------
    % Spatial 1D
    %----------------------------------------
    if (length(inpDim) - 1) == 1
        for i = 1:inpDim(2)
            xOut(:,i) = interp1(tDat, data.X(:,i), tRef, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(inpDim) - 1) == 2
        for i = 1:inpDim(2)
            for ii = 1:inpDim(3)
                xOut(:,i,ii) = interp1(tDat, data.X(:,i,ii), tRef, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(inpDim) - 1) == 3
        for i = 1:inpDim(2)
            for ii = 1:inpDim(3)
                for iii = 1:inpDim(4)
                    xOut(:,i,ii) = interp1(tDat, data.X(:,i,ii,iii), tRef, intp);
                end
            end
        end

    %----------------------------------------
    % Incorrect Dim
    %----------------------------------------
    else
        disp("WARN: Incorrect input dimension, no resampling")
    end

    %===================================================
    % Resampling Reference (r)
    %===================================================
    %----------------------------------------
    % Spatial 1D
    %----------------------------------------
    if (length(refDim) - 1) == 1
        for i = 1:refDim(2)
            rOut(:,i) = interp1(tDat, data.r(:,i), tRef, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(refDim) - 1) == 2
        for i = 1:refDim(2)
            for ii = 1:refDim(3)
                rOut(:,i,ii) = interp1(tDat, data.r(:,i,ii), tRef, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(refDim) - 1) == 3
        for i = 1:refDim(2)
            for ii = 1:refDim(3)
                for iii = 1:refDim(4)
                    rOut(:,i,ii) = interp1(tDat, data.r(:,i,ii,iii), tRef, intp);
                end
            end
        end

    %----------------------------------------
    % Incorrect Dim
    %----------------------------------------
    else
        disp("WARN: Incorrect reference dimension, no resampling")
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Resampling (temporal) data")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1