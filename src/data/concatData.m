%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: concatData                                                        %
% Date: 18.12.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: reviewed                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function concatenates multiple input data files
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) setup:   All setup values of the current simulation
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Concatenated data including iddata

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = concatData(data, setup, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("INFO: Concatenating data")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Parameters
    %===================================================
    [~, N] = size(data.y);                                                  % number of profiles in data
    [Nt,~] = size(data.y{1,1});                                             % number of timesteps
    Ts = data.Ts;                                                           % sampling time data (sec)
     
    %===================================================
    % Variables
    %===================================================
    out.X = [];                                                             % empty input data
    out.y = [];                                                             % empty output data
    out.r = [];                                                             % empty reference data
    out.t2 = [];                                                            % total time vector including all profiles
    out.id = [];                                                            % id vector for each profile

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Reformat General Info
    %===================================================
    out.Data = data.Data;
    out.Type = data.Type;
    out.Dim = data.Dim;
    out.Split = data.Split;
    out.Intp = data.Intp;
    out.Inp = data.Inp;
    out.Out = data.Out;
    out.Ts = data.Ts;

    %===================================================
    % Keep Structured Data
    %===================================================
    out.X2 = data.X;
    out.y2 = data.y;
    out.r2 = data.r;
    out.t2 = data.t;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:N
        %===================================================
        % Input X
        %===================================================
        out.X = [out.X; data.X{1,i}];
    
        %===================================================
        % Output y
        %===================================================
        out.y = [out.y; data.y{1,i}];

        %===================================================
        % Reference r
        %===================================================
        out.r = [out.r; data.r{1,i}];

        %===================================================
        % ID value
        %===================================================
        out.id = [out.id; i*ones(length(data.t{1,i}),1)];

        %===================================================
        % ID value
        %===================================================
        out.Nt(i) = length(data.t{1,i}); 
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Total Time
    %===================================================
    out.t = 0:Ts:Nt*Ts-Ts;
    out.t = out.t';
    
    %===================================================
    % Reshape
    %===================================================
    out.t2 = out.t2';

    %===================================================
    % ID Data
    %===================================================
    %----------------------------------------
    % 1D Data
    %----------------------------------------
    if setup.datDim == 1
        for i = 1:N
            if i == 1
                out.idData = iddata(data.y{1,i},data.X{1,i},Ts);
            else
                temp = iddata(data.y{1,i},data.X{1,i},Ts);
                out.idData = merge(out.idData,temp);
            end
        end

    %----------------------------------------
    % 2D Data
    %----------------------------------------
    else
        % Find Minimum
        if data.Dim == 2
            [~,idInp] = min(sum(abs(data.Data.geo - [para.Dat.gen.inpX, para.Dat.gen.inpY]),2));
            [~,idOut] = min(sum(abs(data.Data.geo - [para.Dat.gen.outX, para.Dat.gen.outY]),2));
        else
            [~,idInp] = min(sum(abs(data.Data.geo - [para.Dat.gen.inpX, para.Dat.gen.inpY, para.Dat.gen.inpZ]),2));
            [~,idOut] = min(sum(abs(data.Data.geo - [para.Dat.gen.outX, para.Dat.gen.outY, para.Dat.gen.outZ]),2));
        end

        % Extract Data
        for i = 1:N
            if i == 1
                out.idData = iddata(data.y{1,i}(:,idOut),data.X{1,i}(:,idInp),Ts);
            else
                temp = iddata(data.y{1,i}(:,idOut),data.X{1,i}(:,idInp),Ts);
                out.idData = merge(out.idData,temp);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1