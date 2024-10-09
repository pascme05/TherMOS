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
%       2) setup:   All setup values of the current simulation
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Adapted input simulation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = resmpData(data, setup, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Resampling (temporal) data")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ts = para.Exp.gen.Ts;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    outDim = size(data.y{1,1});
    inpDim = size(data.X{1,1});
    refDim = size(data.r{1,1});

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Time Vector
    %===================================================
    tInp = data.X.time(1):Ts:data.X.time(end);
    tOut = data.y.time(1):Ts:data.y.time(end);
    tRef = data.r.time(1):Ts:data.r.time(end);
    
    %===================================================
    % Init Output
    %===================================================
    yOut = zeros(length(tOut), outDim(2:end));
    xOut = zeros(length(tInp), inpDim(2:end));
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
            yOut(:,i) = interp1(data.y.time, data.y(:,i), tOut, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(outDim) - 1) == 2
        for i = 1:outDim(2)
            for ii = 1:outDim(3)
                yOut(:,i,ii) = interp1(data.y.time, data.y(:,i,ii), tOut, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(outDim) - 1) == 3
        for i = 1:outDim(2)
            for ii = 1:outDim(3)
                for iii = 1:outDim(4)
                    yOut(:,i,ii) = interp1(data.y.time, data.y(:,i,ii,iii), tOut, intp);
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
            xOut(:,i) = interp1(data.X.time, data.X(:,i), tInp, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(inpDim) - 1) == 2
        for i = 1:inpDim(2)
            for ii = 1:inpDim(3)
                xOut(:,i,ii) = interp1(data.X.time, data.X(:,i,ii), tInp, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(inpDim) - 1) == 3
        for i = 1:inpDim(2)
            for ii = 1:inpDim(3)
                for iii = 1:inpDim(4)
                    xOut(:,i,ii) = interp1(data.X.time, data.X(:,i,ii,iii), tInp, intp);
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
            rOut(:,i) = interp1(data.r.time, data.r(:,i), tRef, intp);
        end
    
    %----------------------------------------
    % Spatial 2D
    %----------------------------------------
    elseif (length(refDim) - 1) == 2
        for i = 1:refDim(2)
            for ii = 1:refDim(3)
                rOut(:,i,ii) = interp1(data.r.time, data.r(:,i,ii), tRef, intp);
            end
        end

    %----------------------------------------
    % Spatial 3D
    %----------------------------------------
    elseif (length(refDim) - 1) == 3
        for i = 1:refDim(2)
            for ii = 1:refDim(3)
                for iii = 1:refDim(4)
                    rOut(:,i,ii) = interp1(data.r.time, data.r(:,i,ii,iii), tRef, intp);
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
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out = data;
    out.X = xOut;
    out.y = yOut;
    out.r = rOut;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Resampling (temporal) data")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1