%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: offRemove                                                         %
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
% This function removes the offset from the output data or set the intial
% value of the output data.
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) para:    All simulation parameters of the current simulation
% Out:  1) out:     Adapted input simulation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = offRemove(data, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Offset removal output data")
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ts = data.Ts;                                                           % Sampling time (sec)
    N = length(data.y2);                                                    % Number of structured data element
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % No removal
    %===================================================
    if para.Exp.gen.init == 1
        %----------------------------------------
        % Define Offset
        %----------------------------------------
        % Unstructured
        off = zeros(size(data.y));

        % Structured
        off2 = cell(1,N);
        for i = 1:N
            off2{1,i} = zeros(size(data.y2{1,i}));
        end

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Output data unchanged")

    %===================================================
    % Init to zero
    %===================================================
    elseif para.Exp.gen.init == 2
        %----------------------------------------
        % Define Offset
        %----------------------------------------
        % Unstructured
        initVal = data.y(1,:);
        off = ones(size(data.y)) .* initVal;

        % Structured
        off2 = cell(1,N);
        for i = 1:N
            off2{1,i} = ones(size(data.y2{1,i})) .* data.y2{1,i}(1,:);
        end

        %----------------------------------------
        % Removal
        %----------------------------------------
        % Unstructured
        data.y = data.y - off;

        % Structured
        for i = 1:N
            data.y2{1,i} = data.y2{1,i} - off2{1,i};
        end

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Initial value of zero is used")

    %===================================================
    % Remove reference
    %===================================================
    elseif para.Exp.gen.init == 3
        %----------------------------------------
        % Define Offset
        %----------------------------------------
        off = data.r;
        off2 = data.r2;

        %----------------------------------------
        % Removal
        %----------------------------------------
        % Unstructured
        data.y = data.y - off;

        % Structured
        for i = 1:N
            data.y2{1,i} = data.y2{1,i} - data.r2{1,i};
        end

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Reference signal is subtracted")

    %===================================================
    % Invalid Input
    %===================================================
    else
        disp("WARN: Invalid input for offset removal")
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % for i = 1:N
    %     if i == 1
    %         data.idData = iddata(data.y2{1,i},data.X2{1,i},Ts);
    %     else
    %         temp = iddata(data.y2{1,i},data.X2{1,i},Ts);
    %         data.idData = merge(data.idData,temp);
    %     end
    % end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out = data;
    out.off = off;
    out.off2 = off2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Offset removal output data")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1