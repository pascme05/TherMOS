%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: loadData                                                          %
% Date: 14.09.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
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
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("START: Loading input data")

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
                        para.Dat.gen.sep, para.Exp.gen.samp, setup.inp, ...
                        setup.out);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Training
    %===================================================
    %----------------------------------------
    % General Info
    %----------------------------------------
    % Load
    data.tr = loader.loadData(setup.trFile(1), setup.trSheet(1), ...
                              setup.trID, para, setup);

    % Clear data
    data.tr.X = [];
    data.tr.y = [];
    data.tr.r = [];
    data.tr.time = [];
    data.tr.Ts = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.trFile)
            % Data
            temp = loader.loadData(setup.trFile(i), setup.trSheet(1), ...
                                   setup.trID, para, setup);

            % Adapt
            data.tr.X{i} = temp.X;
            data.tr.y{i} = temp.y;
            data.tr.r{i} = temp.r;
            data.tr.time{i} = temp.time;
            data.tr.Ts{i} = temp.Ts;
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.trSheet)
            % Data
            temp = loader.loadData(setup.trFile(1), setup.trSheet(i), ...
                                   setup.trID, para, setup);

            % Adapt
            data.tr.X{i} = temp.X;
            data.tr.y{i} = temp.y;
            data.tr.r{i} = temp.r;
            data.tr.time{i} = temp.time;
            data.tr.Ts{i} = temp.Ts;
        end
    
    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        data.tr = loader.loadData(setup.trFile(1), setup.trSheet(1), ...
                                  setup.trID, para, setup);
    end

    %===================================================
    % Testing
    %===================================================
    %----------------------------------------
    % General Info
    %----------------------------------------
    % Load
    data.te = loader.loadData(setup.teFile(1), setup.teSheet(1), ...
                              setup.teID, para, setup);

    % Clear data
    data.te.X = [];
    data.te.y = [];
    data.te.r = [];
    data.te.time = [];
    data.te.Ts = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.teFile)
            % Data
            temp = loader.loadData(setup.teFile(i), setup.teSheet(1), ...
                                   setup.teID, para, setup);

            % Adapt
            data.te.X{i} = temp.X;
            data.te.y{i} = temp.y;
            data.te.r{i} = temp.r;
            data.te.time{i} = temp.time;
            data.te.Ts = temp.Ts;
        end

    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.teSheet)
            % Data
            temp = loader.loadData(setup.teFile(1), setup.teSheet(i), ...
                                   setup.teID, para, setup);

            % Adapt
            data.te.X{i} = temp.X;
            data.te.y{i} = temp.y;
            data.te.r{i} = temp.r;
            data.te.time{i} = temp.time;
            data.te.Ts = temp.Ts;
        end

    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        data.te = loader.loadData(setup.teFile(1), setup.teSheet(1), ...
                                  setup.teID, para, setup);
    end

    %===================================================
    % Validation
    %===================================================
    %----------------------------------------
    % General Info
    %----------------------------------------
    % Load
    data.vl = loader.loadData(setup.vlFile(1), setup.vlSheet(1), ...
                              setup.vlID, para, setup);

    % Clear data
    data.vl.X = [];
    data.vl.y = [];
    data.vl.r = [];
    data.vl.time = [];
    data.vl.Ts = [];

    %----------------------------------------
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.vlFile)
            % Data
            temp = loader.loadData(setup.vlFile(i), setup.vlSheet(1), ...
                                   setup.vlID, para, setup);

            % Adapt
            data.vl.X{i} = temp.X;
            data.vl.y{i} = temp.y;
            data.vl.r{i} = temp.r;
            data.vl.time{i} = temp.time;
            data.vl.Ts = temp.Ts;
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.vlSheet)
            % Data
            temp = loader.loadData(setup.vlFile(1), setup.vlSheet(i), ...
                                   setup.vlID, para, setup);

            % Adapt
            data.vl.X{i} = temp.X;
            data.vl.y{i} = temp.y;
            data.vl.r{i} = temp.r;
            data.vl.time{i} = temp.time;
            data.vl.Ts = temp.Ts;
        end
    
    %----------------------------------------
    % ID and Split based
    %----------------------------------------
    else
        data.vl = loader.loadData(setup.vlFile(1), setup.vlSheet(1), ...
                                  setup.vlID, para, setup);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Loading input data")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1