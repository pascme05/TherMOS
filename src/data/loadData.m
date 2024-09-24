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
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.trFile)
            data.tr = loader.loadData(setup.trFile(i), setup.trSheet(1), ...
                                      setup.trID, para, setup);
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.trSheet)
            data.tr = loader.loadData(setup.trFile(1), setup.trSheet(i), ...
                                      setup.trID, para, setup);
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
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.teFile)
            data.te = loader.loadData(setup.teFile(i), setup.teSheet(1), ...
                                      setup.teID, para, setup);
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.teSheet)
            data.te = loader.loadData(setup.teFile(1), setup.teSheet(i), ...
                                        setup.teID, para, setup);
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
    % File based
    %----------------------------------------
    if para.Dat.gen.sep == 1
        for i = 1:length(setup.vlFile)
            data.vl = loader.loadData(setup.vlFile(i), setup.vlSheet(1), ...
                                      setup.vlID, para, setup);
        end
    
    %----------------------------------------
    % Sheet based
    %----------------------------------------
    elseif para.Dat.gen.sep == 2
        for i = 1:length(setup.vlSheet)
            data.vl = loader.loadData(setup.vlFile(1), setup.vlSheet(i), ...
                                        setup.vlID, para, setup);
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