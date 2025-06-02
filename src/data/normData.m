%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: normData                                                          %
% Date: 07.05.2025                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
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
%                             3) applying inverse norm
% Out:  1) data:    Normalised input simulation data
%       2) para:    Includes the normalisation parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data, para] = normData(data, para, sel)
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
    if sel ~= 1
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
    if sel ~= 1
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
    if sel ~= 1
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
    
    %===================================================
    % Correct Input
    %===================================================
    if sel == 1
        if maxY == minY
            minY = 0;
        end
        if maxX == minX
            minX = 0;
        end
        if maxR == minR
            minR = 0;
        end
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
            for i = 1:numel(data.X2)
                data.X2{i} = data.X2{i} .* (maxX - minX) + minX;
            end

        % Norm
        else
            data.X = (data.X - minX) ./ (maxX - minX);
            for i = 1:numel(data.X2)
                data.X2{i} = (data.X2{i} - minX) ./ (maxX - minX);
            end
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
            for i = 1:numel(data.X2)
                data.X2{i} = data.X2{i} .* (maxX - minX) + avgX;
            end

        % Norm
        else
            data.X = (data.X - avgX) ./ (maxX - minX);
            for i = 1:numel(data.X2)
                data.X2{i} = (data.X2{i} - avgX) ./ (maxX - minX);
            end
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
            for i = 1:numel(data.X2)
                data.X2{i} = data.X2{i} .* stdX + avgX;
            end

        % Norm
        else
            data.X = (data.X - avgX) ./ stdX;
            for i = 1:numel(data.X2)
                data.X2{i} = (data.X2{i} - avgX) ./ stdX;
            end
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
            % data.r = data.r .* (maxY - minY) + minY;
            for i = 1:numel(data.y2)
                data.y2{i} = data.y2{i} .* (maxY - minY) + minY;
                % data.r2{i} = data.r2{i} .* (maxY - minY) + minY;
            end

        % Norm
        else
            data.y = (data.y - minY) ./ (maxY - minY);
            % data.r = (data.r - minY) ./ (maxY - minY);
            for i = 1:numel(data.y2)
                data.y2{i} = (data.y2{i} - minY) ./ (maxY - minY);
                % data.r2{i} = (data.r2{i} - minY) ./ (maxY - minY);
            end
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
            % data.r = data.r .* (maxY - minY) + avgY;
            for i = 1:numel(data.y2)
                data.y2{i} = data.y2{i} .* (maxY - minY) + avgY;
                % data.r2{i} = data.r2{i} .* (maxY - minY) + avgY;
            end

        % Norm
        else
            data.y = (data.y - avgY) ./ (maxY - minY);
            % data.r = (data.r - avgY) ./ (maxY - minY);
            for i = 1:numel(data.y2)
                data.y2{i} = (data.y2{i} - minY) .* (maxY - minY) + avgY;
                % data.r2{i} = (data.r2{i} - minY) .* (maxY - minY) + avgY;
            end
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
            % data.r = data.r .* stdY + avgY;
            for i = 1:numel(data.y2)
                data.y2{i} = data.y2{i} .* stdY + avgY;
                % data.r2{i} = data.r2{i} .* stdY + avgY;
            end

        % Norm
        else
            data.y = (data.y - avgY) ./ stdY;
            % data.r = (data.r - avgY) ./ stdY;
            for i = 1:numel(data.y2)
                data.y2{i} = (data.y2{i} - avgY) ./ stdY;
                % data.r2{i} = (data.r2{i} - avgY) ./ stdY;
            end
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1