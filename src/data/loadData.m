%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: loadData                                                          %
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
% This function loads the input data from \data.
% -------------------------------------------------------------------------
% Inp:  1) para:    input parameter file
%       2) setup:   input setup file
% Out:  1) data:    output data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = loadData(para, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Output
    %===================================================
    data = [];

    %===================================================
    % Data Loader
    %===================================================
    loader = DataLoader(para.Exp.gen.Ts, setup.format, setup.datDim, ...
                        setup.datSep, para.Exp.gen.samp, setup.inp, ...
                        setup.out);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Training
    %===================================================
    %----------------------------------------
    % Msg
    %----------------------------------------
    disp("----------------------------------------");
    disp("Loading Training Data")
    disp("----------------------------------------");

    %----------------------------------------
    % General Info
    %----------------------------------------
    % Status
    stat = 1;                                                           

    % Load
    data.tr = loader.loadData(setup.trFile(1), setup.trSheet(1), stat, ...
                              setup);

    % Clear data
    data.tr.X = [];
    data.tr.y = [];
    data.tr.r = [];
    data.tr.t = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if setup.datSep == 1
        for i = 1:length(setup.trFile)
            % Data
            temp = loader.loadData(setup.trFile(i), setup.trSheet(1), ...
                                   stat, setup);
            
            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.tr.X{i} = xOut;
            data.tr.y{i} = yOut;
            data.tr.r{i} = rOut;
            data.tr.t{i} = tOut;
            data.tr.Ts = tOut(2) - tOut(1);
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif setup.datSep == 2
        for i = 1:length(setup.trSheet)
            % Data
            temp = loader.loadData(setup.trFile(1), setup.trSheet(i), ...
                                   stat, setup);

            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.tr.X{i} = xOut;
            data.tr.y{i} = yOut;
            data.tr.r{i} = rOut;
            data.tr.t{i} = tOut;
            data.tr.Ts = tOut(2) - tOut(1);
        end
    
    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        % Data
        temp = loader.loadData(setup.trFile(1), setup.trSheet(1), ...
                               stat, setup);

        % Resample
        [xOut, yOut, rOut, tOut] = resmpData(temp, para);

        % Adapt
        data.tr.X{1,1} = xOut;
        data.tr.y{1,1} = yOut;
        data.tr.r{1,1} = rOut;
        data.tr.t{1,1} = tOut;
        data.tr.Ts = tOut(2) - tOut(1);
    end

    %===================================================
    % Testing
    %===================================================
    %----------------------------------------
    % Msg
    %----------------------------------------
    disp("----------------------------------------");
    disp("Loading Testing Data")
    disp("----------------------------------------");

    %----------------------------------------
    % General Info
    %----------------------------------------
    % Status
    stat = 2;

    % Load
    data.te = loader.loadData(setup.teFile(1), setup.teSheet(1), ...
                              stat, setup);

    % Clear data
    data.te.X = [];
    data.te.y = [];
    data.te.r = [];
    data.te.t = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if setup.datSep == 1
        for i = 1:length(setup.teFile)
            % Data
            temp = loader.loadData(setup.teFile(i), setup.teSheet(1), ...
                                   stat, setup);
            
            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.te.X{i} = xOut;
            data.te.y{i} = yOut;
            data.te.r{i} = rOut;
            data.te.t{i} = tOut;
            data.te.Ts = tOut(2) - tOut(1);
        end

    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif setup.datSep == 2
        for i = 1:length(setup.teSheet)
            % Data
            temp = loader.loadData(setup.teFile(1), setup.teSheet(i), ...
                                   stat, setup);
            
            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.te.X{i} = xOut;
            data.te.y{i} = yOut;
            data.te.r{i} = rOut;
            data.te.t{i} = tOut;
            data.te.Ts = tOut(2) - tOut(1);
        end

    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        % Data
        temp = loader.loadData(setup.teFile(1), setup.teSheet(1), ...
                               stat, setup);

        % Resample
        [xOut, yOut, rOut, tOut] = resmpData(temp, para);

        % Adapt
        data.te.X{1,1} = xOut;
        data.te.y{1,1} = yOut;
        data.te.r{1,1} = rOut;
        data.te.t{1,1} = tOut;
        data.te.Ts = tOut(2) - tOut(1);
    end

    %===================================================
    % Validation
    %===================================================
    %----------------------------------------
    % Msg
    %----------------------------------------
    disp("----------------------------------------");
    disp("Loading Validation Data")
    disp("----------------------------------------");

    %----------------------------------------
    % General Info
    %----------------------------------------
    % Status
    stat = 3;

    % Load
    data.vl = loader.loadData(setup.vlFile(1), setup.vlSheet(1), ...
                              stat, setup);

    % Clear data
    data.vl.X = [];
    data.vl.y = [];
    data.vl.r = [];
    data.vl.t = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if setup.datSep == 1
        for i = 1:length(setup.vlFile)
            % Data
            temp = loader.loadData(setup.vlFile(i), setup.vlSheet(1), ...
                                   stat, setup);
            
            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.vl.X{i} = xOut;
            data.vl.y{i} = yOut;
            data.vl.r{i} = rOut;
            data.vl.t{i} = tOut;
            data.vl.Ts = tOut(2) - tOut(1);
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif setup.datSep == 2
        for i = 1:length(setup.vlSheet)
            % Data
            temp = loader.loadData(setup.vlFile(1), setup.vlSheet(i), ...
                                   stat, setup);

            % Resample
            [xOut, yOut, rOut, tOut] = resmpData(temp, para);

            % Adapt
            data.vl.X{i} = xOut;
            data.vl.y{i} = yOut;
            data.vl.r{i} = rOut;
            data.vl.t{i} = tOut;
            data.vl.Ts = tOut(2) - tOut(1);
        end
    
    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        % Data
        temp = loader.loadData(setup.vlFile(1), setup.vlSheet(1), ...
                               stat, setup);

        % Resample
        [xOut, yOut, rOut, tOut] = resmpData(temp, para);

        % Adapt
        data.vl.X{1,1} = xOut;
        data.vl.y{1,1} = yOut;
        data.vl.r{1,1} = rOut;
        data.vl.t{1,1} = tOut;
        data.vl.Ts = tOut(2) - tOut(1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Post-Processing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Msg
    %===================================================
    disp("----------------------------------------");
    disp("Concatenating Data")
    disp("----------------------------------------");
    
    %===================================================
    % Calc
    %===================================================
    data.tr = concatData(data.tr, setup, para);
    data.te = concatData(data.te, setup, para);
    data.vl = concatData(data.vl, setup, para);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Loading input data")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1