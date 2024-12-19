%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: dimData                                                           %
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
% This function adapts the dimension if the input data.
% -------------------------------------------------------------------------
% Inp:  1) data:    Input data struct including tr, te, and vl
%       2) setup:   All setup values of the current simulation
%       3) para:    All simulation parameters of the current simulation
% Out:  1) out:     Adapted input simulation data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = dimData(data, setup, para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ts = data.Ts;                                                           % sampling time (sec)
    [~, N] = size(data.y2);                                                 % number of profiles in data
    dim = size(data.Data.geo,2);                                            % dimension of raw data input
    pos = para.Dat.gen.cutVal;                                              % position for reducing from 3D -> 2D
    ax = para.Dat.gen.cut;                                                  % axis which is being removed

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Pre-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Check dimension
    %===================================================
    if dim ~= setup.datDim
        disp("WARN: Expected and actual data dimension is different")
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Reduction 3D -> 2D
    %===================================================
    if setup.datDim == 3 && para.Dat.gen.dOut == 2
        %----------------------------------------
        % Get positions
        %----------------------------------------

        %----------------------------------------
        % Reduce data
        %----------------------------------------

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Data reduced from 3D -> 2D");

    %===================================================
    % Reduction 3D -> 1D
    %===================================================
    elseif setup.datDim == 3 && para.Dat.gen.dOut == 1
        %----------------------------------------
        % Init
        %----------------------------------------
        out = data;

        %----------------------------------------
        % Get positions
        %----------------------------------------
        [~,idInp] = min(sum(abs(data.Data.geo - [para.Dat.gen.inpX, para.Dat.gen.inpY, para.Dat.gen.inpZ]),2));
        [~,idOut] = min(sum(abs(data.Data.geo - [para.Dat.gen.outX, para.Dat.gen.outY, para.Dat.gen.outZ]),2));

        %----------------------------------------
        % Reduce Unstructured Data
        %----------------------------------------
        out.X = data.X(:,idInp);
        out.y = data.y(:,idOut);
        out.r = data.r(:,idOut);
        out.off = data.off(:,idOut);

        %----------------------------------------
        % Reduce Structured Data
        %----------------------------------------
        for i = 1:N
            out.X2{1,i} = data.X2{1,i}(:,idInp);
            out.y2{1,i} = data.y2{1,i}(:,idOut);
            out.r2{1,i} = data.r2{1,i}(:,idOut);
            out.off2{1,i} = data.off2{1,i}(:,idOut);
        end

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Data reduced from 3D -> 1D");

    %===================================================
    % Reduction 2D -> 1D
    %===================================================
    elseif setup.datDim == 2 && para.Dat.gen.dOut == 1
        %----------------------------------------
        % Init
        %----------------------------------------
        out = data;

        %----------------------------------------
        % Get positions
        %----------------------------------------
        [~,idInp] = min(sum(abs(data.Data.geo - [para.Dat.gen.inpX, para.Dat.gen.inpY]),2));
        [~,idOut] = min(sum(abs(data.Data.geo - [para.Dat.gen.outX, para.Dat.gen.outY]),2));

        %----------------------------------------
        % Reduce Unstructured Data
        %----------------------------------------
        out.X = data.X(:,idInp);
        out.y = data.y(:,idOut);
        out.r = data.r(:,idOut);
        out.off = data.off(:,idOut);

        %----------------------------------------
        % Reduce Structured Data
        %----------------------------------------
        for i = 1:N
            out.X2{1,i} = data.X2{1,i}(:,idInp);
            out.y2{1,i} = data.y2{1,i}(:,idOut);
            out.r2{1,i} = data.r2{1,i}(:,idOut);
            out.off2{1,i} = data.off2{1,i}(:,idOut);
        end

        %----------------------------------------
        % Msg
        %----------------------------------------
        disp("INFO: Data reduced from 2D -> 1D");
    
    %===================================================
    % Correct dimension
    %===================================================
    elseif setup.datDim == para.Dat.gen.dOut
        disp("INFO: Data dimension are matching");
        out = data;

    %===================================================
    % Invalid Reduction
    %===================================================
    else
        disp("WARN: Reduction invalid, check data dimensions");
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % ID Data 1D 
    %===================================================
    if para.Dat.gen.dOut == 1
        for i = 1:N
            if i == 1
                out.idData = iddata(data.y2{1,i},data.X2{1,i},Ts);
            else
                temp = iddata(data.y2{1,i},data.X2{1,i},Ts);
                out.idData = merge(out.idData,temp);
            end
        end

    %===================================================
    % ID Data 2D 
    %===================================================
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
                out.idData = iddata(data.y2{1,i}(:,idOut),data.X2{1,i}(:,idInp),Ts);
            else
                temp = iddata(data.y2{1,i}(:,idOut),data.X2{1,i}(:,idInp),Ts);
                out.idData = merge(out.idData,temp);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1