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
    Tref = para.Par.loss.Tref;                                              % reference temperature (°C)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Extract Input
    %===================================================
    if sel == 3
        maxX = para.Dat.normVal.X.max;
        minX = para.Dat.normVal.X.min;
        avgX = para.Dat.normVal.X.avg;
        stdX = para.Dat.normVal.X.std;
    else
        minX = min(data.X);
        maxX = max(data.X);
        avgX = mean(data.X);
        stdX = std(data.X);
    end

    %===================================================
    % Extract Output
    %===================================================
    if sel == 3
        maxY = para.Dat.normVal.y.max;
        minY = para.Dat.normVal.y.min;
        avgY = para.Dat.normVal.y.avg;
        stdY = para.Dat.normVal.y.std;
    else
        minY = min(data.y);
        maxY = max(data.y);
        avgY = mean(data.y);
        stdY = std(data.y);
    end

    %===================================================
    % Extract Reference
    %===================================================
    if sel == 3
        maxR = para.Dat.normVal.r.max;
        minR = para.Dat.normVal.r.min;
        avgR = para.Dat.normVal.r.avg;
        stdR = para.Dat.normVal.r.std;
    else
        minR = min(data.r);
        maxR = max(data.r);
        avgR = mean(data.r);
        stdR = std(data.r);
    end
    
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
            data.X = data.X .* (maxX - minX) + minX;

        % Norm
        else
            data.X = (data.X - minX) ./ (maxX - minX);
        end

        % Msg
        disp("INFO: 0/1 normalisation applied to input")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normX == 3
        % Inv Norm
        if sel == 3
            data.X = data.X .* (maxX - minX) + avgX;

        % Norm
        else
            data.X = (data.X - avgX) ./ (maxX - minX);
        end

        % Msg
        disp("INFO: Min/Max normalisation applied to input")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normX == 4
        % Inv Norm
        if sel == 3
            data.X = data.X .* stdX + avgX;

        % Norm
        else
            data.X = (data.X - avgX) ./ stdX;
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
            data.y = data.y .* (maxY - minY) + minY;

        % Norm
        else
            data.y = (data.y - minY) ./ (maxY - minY);
        end

        % Msg
        disp("INFO: 0/1 normalisation applied to output")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normY == 3
        % Inv Norm
        if sel == 3
            data.y = data.y .* (maxY - minY) + avgY;

        % Norm
        else
            data.y = (data.y - avgY) ./ (maxY - minY);
        end

        % Msg
        disp("INFO: Min/Max normalisation applied to output")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normY == 4
        % Inv Norm
        if sel == 3
            data.y = data.y .* stdY + avgY;

        % Norm
        else
            data.y = (data.y - avgY) ./ stdY;
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
    %===================================================
    % Temperature Reference
    %===================================================
    %----------------------------------------
    % None
    %----------------------------------------
    if para.Par.gen.normY == 1
        disp("INFO: No normalisation applied to Tref")
        
    %----------------------------------------
    % 0/1
    %----------------------------------------
    elseif para.Par.gen.normY == 2
        % Inv Norm
        if sel == 3
            Tref = Tref .* (maxY - minY) + minY;

        % Norm
        else
            Tref = (Tref - minY) ./ (maxY - minY);
        end

        % Msg
        disp("INFO: 0/1 normalisation applied to Tref")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normY == 3
        % Inv Norm
        if sel == 3
            Tref = Tref .* (maxY - minY) + avgY;

        % Norm
        else
            Tref = (Tref - avgY) ./ (maxY - minY);
        end

        % Msg
        disp("INFO: Min/Max normalisation applied to Tref")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normY == 4
        % Inv Norm
        if sel == 3
            Tref = Tref .* stdY + avgY;

        % Norm
        else
            Tref = (Tref - avgY) ./ stdY;
        end

        % Msg
        disp("INFO: Avg/Std normalisation applied to Tref")

    %----------------------------------------
    % Invalid
    %----------------------------------------
    else
        disp("INFO: Invalid normalisation approach")
    end

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
        para.Dat.normVal.y.max = maxY;
        para.Dat.normVal.y.min = minY;
        para.Dat.normVal.y.avg = avgY;
        para.Dat.normVal.y.std = stdY;

        %----------------------------------------
        % Input
        %----------------------------------------
        para.Dat.normVal.X.max = maxX;
        para.Dat.normVal.X.min = minX;
        para.Dat.normVal.X.avg = avgX; 
        para.Dat.normVal.X.std = stdX;

        %----------------------------------------
        % Reference
        %----------------------------------------
        para.Dat.normVal.r.max = maxR;
        para.Dat.normVal.r.min = minR;
        para.Dat.normVal.r.avg = avgR; 
        para.Dat.normVal.r.std = stdR;
        
        %----------------------------------------
        % Loss Update
        %----------------------------------------
        para.Par.loss.Tref = Tref;

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