%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: normData                                                          %
% Date: 26.09.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function normalises the data.
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) para:    All simulation parameters of the current simulation
%       3) sel:     Selector: 1) apply and get norm, 2) applying norm
%                             3) applying inv norm
% Out:  1) data:    Normalised input simulation data
%       2) para:    Includes the normalisation parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data, para] = normData(data, para, sel)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Data normalisation")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Input
    %===================================================
    minX = min(data.y);
    maxX = max(data.y);
    avgX = mean(data.y);
    stdX = std(data.y);

    %===================================================
    % Extract Output
    %===================================================
    minY = min(data.X);
    maxY = max(data.X);
    avgY = mean(data.X);
    stdY = std(data.X);

    %===================================================
    % Extract Reference
    %===================================================
    minR = min(data.r);
    maxR = max(data.r);
    avgR = mean(data.r);
    stdR = std(data.r);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Input (X)
    %===================================================
    %----------------------------------------
    % None
    %----------------------------------------
    if para.Par.gen.normX == 1
        disp("INFO: No normalisation applied to input")
        
    %----------------------------------------
    % 0/1
    %----------------------------------------
    elseif para.Par.gen.normX == 2
        % Inv Norm
        if sel == 3
            data.X = (data.X - minX) / (maxX - minX);

        % Norm
        else
            data.X = data.X * (maxX - minX) + minX;
        end

        % Msg
        disp("INFO: 0/1 normalisation applied to input")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normX == 3
        % Inv Norm
        if sel == 3
            data.X = (data.X - avgX) / (maxX - minX);

        % Norm
        else
            data.X = data.X * (maxX - minX) + avgX;
        end

        % Msg
        disp("INFO: Min/Max normalisation applied to input")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normX == 4
        % Inv Norm
        if sel == 3
            data.X = (data.X - avgX) / sigX;

        % Norm
        else
            data.X = data.X * sigX + avgX;
        end

        % Msg
        disp("INFO: Avg/Std normalisation applied to input")

    %----------------------------------------
    % Invalid
    %----------------------------------------
    else
        disp("INFO: Invalid normalisation approach")
    end

    %===================================================
    % Output/Reference (y/r)
    %===================================================
    %----------------------------------------
    % None
    %----------------------------------------
    if para.Par.gen.normY == 1
        disp("INFO: No normalisation applied to output")
        
    %----------------------------------------
    % 0/1
    %----------------------------------------
    elseif para.Par.gen.normY == 2
        % Inv Norm
        if sel == 3
            data.y = (data.y - minY) / (maxY - minY);

        % Norm
        else
            data.y = data.y * (maxY - minY) + minY;
        end

        % Msg
        disp("INFO: 0/1 normalisation applied to output")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normY == 3
        % Inv Norm
        if sel == 3
            data.y = (data.y - avgY) / (maxY - minY);

        % Norm
        else
            data.y = data.y * (maxY - minY) + avgY;
        end

        % Msg
        disp("INFO: Min/Max normalisation applied to output")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normY == 4
        % Inv Norm
        if sel == 3
            data.y = (data.y - avgY) / sigY;

        % Norm
        else
            data.y = data.y * sigY + avgY;
        end

        % Msg
        disp("INFO: Avg/Std normalisation applied to output")

    %----------------------------------------
    % Invalid
    %----------------------------------------
    else
        disp("INFO: Invalid normalisation approach")
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Update Parameters
    %===================================================
    if sel == 1
        %----------------------------------------
        % Output
        %----------------------------------------
        para.Dat.normVal.y.max = maxX;
        para.Dat.normVal.y.min = minX;
        para.Dat.normVal.y.avg = avgX;
        para.Dat.normVal.y.std = stdX;

        %----------------------------------------
        % Input
        %----------------------------------------
        para.Dat.normVal.X.max = maxY;
        para.Dat.normVal.X.min = minY;
        para.Dat.normVal.X.avg = avgY; 
        para.Dat.normVal.X.std = stdY;

        %----------------------------------------
        % Reference
        %----------------------------------------
        para.Dat.normVal.r.max = maxR;
        para.Dat.normVal.r.min = minR;
        para.Dat.normVal.r.avg = avgR; 
        para.Dat.normVal.r.std = stdR;

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Normalisation values have been updated")
    else
        disp("INFO: Normalisation values have not been updated")
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Data normalisation")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1