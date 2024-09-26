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
%       2) setup:   All setup values of the current simulation
%       3) para:    All simulation parameters of the current simulation
%       4) sel:     Selector: 1) extracting norm values, 2) applying norm
% Out:  1) out:     Adapted input simulation data
%       2) para:    Includes the normalisation parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out, para] = normData(data, setup, para, sel)
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
    minInp = min(data.y);
    maxInp = max(data.y);
    avgInp = mean(data.y);
    stdInp = std(data.y);

    %===================================================
    % Extract Output
    %===================================================
    minOut = min(data.X);
    maxOut = max(data.X);
    avgOut = mean(data.X);
    stdOut = std(data.X);

    %===================================================
    % Extract Reference
    %===================================================
    minRef = min(data.r);
    maxRef = max(data.r);
    avgRef = mean(data.r);
    stdRef = std(data.r);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Input
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
        disp("INFO: 0/1 normalisation applied to input")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normX == 3
        disp("INFO: Min/Max normalisation applied to input")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normX == 4
        disp("INFO: Avg/Std normalisation applied to input")

    %----------------------------------------
    % Invalid
    %----------------------------------------
    else
        disp("INFO: Invalid normalisation approach")
    end

    %===================================================
    % Output/Reference
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
        disp("INFO: 0/1 normalisation applied to output")

    %----------------------------------------
    % Min/Max
    %----------------------------------------
    elseif para.Par.gen.normY == 3
        disp("INFO: Min/Max normalisation applied to output")

    %----------------------------------------
    % Avg/Std
    %----------------------------------------
    elseif para.Par.gen.normY == 4
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
        para.Dat.normVal.y.max =
        para.Dat.normVal.y.min = 
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