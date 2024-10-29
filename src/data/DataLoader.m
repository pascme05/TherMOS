%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: DataLoader                                                        %
% Date: 13.08.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments:                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class defines all loading operations for the data
% -------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class Defintion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef DataLoader
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Properties and Variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        Data                                                                % Table to store the loaded data
        Type                                                                % Input data type
        Dim                                                                 % Data dimension
        Split                                                               % Splitting method
        Intp                                                                % Interpolation method
        Inp                                                                 % List of input features
        Out                                                                 % List of output features
        t                                                                   % Time vector (sec)
        X                                                                   % Input vector
        y                                                                   % Output vector
        r                                                                   % Reference vector
        Ts                                                                  % sampling time (sec)
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Methods and Functions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        %===================================================
        % Constructor
        %===================================================
        function obj = DataLoader(Ts, Type, Dim, Split, Intp, Inp, Out)
            obj.Ts = Ts;
            obj.Type = Type;
            obj.Dim = Dim;
            obj.Split = Split;
            obj.Intp = Intp;
            obj.Inp = Inp;
            obj.Out = Out;
        end
        
        %===================================================
        % General Loading
        %===================================================
        function data = loadData(obj, FileName, SheetName, stat, setup)
            %----------------------------------------
            % 1D Xlsx
            %----------------------------------------
            if obj.Type == "xlsx"
                % Loading
                data = loadXlsx(obj, FileName, SheetName, stat, setup);

            %----------------------------------------
            % 1D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim == 1
                % Loading
                data = loadMat1D(obj, FileName, stat, setup);

            %----------------------------------------
            % 2D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim >= 2
                % Loading
                data = loadMat2D(obj, FileName);

            %----------------------------------------
            % Error Handling
            %----------------------------------------
            else
                error('ERROR: Wrong data type or dimension.');
            end
        end

        %===================================================
        % Load xlsx Data
        %===================================================
        function obj = loadXlsx(obj, FileName, SheetName, stat, setup)
            %----------------------------------------
            % Loading Data
            %----------------------------------------
            % Check file name
            if isempty(FileName)
                error('ERROR: FileName is not set. Please set the FileName property');
            end

            % Loading
            try
                obj.Data = readtable(FileName, 'Sheet', SheetName);
                disp('INFO: Xlsx data successfully loaded');
            catch ME
                disp('INFO: Failed to load data from the specified worksheet');
                rethrow(ME);
            end

            %----------------------------------------
            % Selecting Data
            %----------------------------------------
            % File and Sheet based
            if setup.datSep == 1 || setup.datSep == 2
                disp('INFO: File and sheet based data');

            % ID based
            elseif setup.datSep == 3
                if stat == 1
                    ID = unique(obj.Data.id);
                    ID = setdiff(ID, setup.teID);
                    ID = setdiff(ID, setup.vlID);
                elseif stat == 2
                    ID = setup.teID;
                else
                    ID = setup.vlID;
                end
                rowsToKeep = ismember(obj.Data.id, ID);
                obj.Data = obj.Data(rowsToKeep, :);
                disp('INFO: ID based data');

            % Split based
            else
                num_samples = height(obj.Data);
                tr_idx = floor(num_samples * setup.rTr);
                te_idx = floor(num_samples * setup.rTr);

                if stat == 1
                    obj.Data = obj.Data(1:tr_idx, :);
                elseif stat == 2
                    obj.Data = obj.Data(te_idx:end, :);
                else
                    obj.Data = obj.Data(1:tr_idx, :);
                    num_samples = height(obj.Data);
                    vl_idx = floor(num_samples * setup.rVl);
                    obj.Data = obj.Data(1:vl_idx, :);
                end
                disp('INFO: Split based data');
            end

            %----------------------------------------
            % Selecting Input and Ouput
            %----------------------------------------
            try
                obj = selectInpOut(obj, setup);
            catch ME
                disp('INFO: Failed to select input and output');
                rethrow(ME);
            end

        end

        %===================================================
        % Load Matlab 1D Data
        %===================================================
        function obj = loadMat1D(obj, FileName, stat, setup)
            %----------------------------------------
            % Loading Data
            %----------------------------------------
            % Check file name
            if isempty(FileName)
                error('ERROR: FileName is not set. Please set the FileName property');
            end

            % Loading
            try
                obj.Data = load(FileName, "-mat");
                obj.Data = obj.Data.data;
                disp('INFO: 1D Matlab data successfully loaded');
            catch ME
                disp('INFO: Failed to load data');
                rethrow(ME);
            end

            %----------------------------------------
            % Selecting Data
            %----------------------------------------
            % File and Sheet based
            if setup.datSep == 1 || setup.datSep == 2
                disp('INFO: File and sheet based data');

            % ID based
            elseif setup.datSep == 3
                if stat == 1
                    ID = unique(obj.Data.id);
                    ID = setdiff(ID, setup.teID);
                    ID = setdiff(ID, setup.vlID);
                elseif stat == 2
                    ID = setup.teID;
                else
                    ID = setup.vlID;
                end
                rowsToKeep = ismember(obj.Data.id, ID);
                obj.Data = obj.Data(rowsToKeep, :);
                disp('INFO: ID based data');

            % Split based
            else
                num_samples = height(obj.Data);
                tr_idx = floor(num_samples * setup.rTr);
                te_idx = floor(num_samples * setup.rTr);

                if stat == 1
                    obj.Data = obj.Data(1:tr_idx, :);
                elseif stat == 2
                    obj.Data = obj.Data(te_idx:end, :);
                else
                    obj.Data = obj.Data(1:tr_idx, :);
                    num_samples = height(obj.Data);
                    vl_idx = floor(num_samples * setup.rVl);
                    obj.Data = obj.Data(1:vl_idx, :);
                end
                disp('INFO: Split based data');
            end

            %----------------------------------------
            % Selecting Input and Ouput
            %----------------------------------------
            % Time
            obj.t = obj.Data.time;

            % Input
            obj.X = zeros(length(obj.t), length(setup.inp));
            for i = 1:length(setup.inp)
                obj.X(:, i) = obj.Data.(setup.inp(i));
            end

            % Output
            obj.y = zeros(length(obj.t), length(setup.out));
            for i = 1:length(setup.out)
                obj.y(:, i) = obj.Data.(setup.out(i));
            end

            % Reference
            obj.r = zeros(length(obj.t), length(setup.out));
            for i = 1:length(setup.ref)
                obj.r(:, i) = obj.Data.(setup.ref(i));
            end

            % Data
            obj.Data.time = [];
            obj.Data.id = [];

        end

        %===================================================
        % Load Matlab 2D Data
        %===================================================
        function obj = loadMat2D(obj, FileName)
            %----------------------------------------
            % Loading Data
            %----------------------------------------
            if isempty(FileName)
                error('ERROR: FileName is not set. Please set the FileName property');
            end
            
            %----------------------------------------
            % Loading
            %----------------------------------------
            try
                obj.Data = load(FileName, "-mat");
                obj.r = obj.Data.r;
                obj.t = obj.Data.t;
                obj.Ts = obj.Data.Ts;
                obj.X = obj.Data.X;
                obj.y = obj.Data.y;
                disp('INFO: 2D Matlab data successfully loaded');
            catch ME
                disp('INFO: Failed to load data');
                rethrow(ME);
            end
        end

        %===================================================
        % Select inputs and outputs
        %===================================================
        function obj = selectInpOut(obj, setup)
            %----------------------------------------
            % Validate setup inputs (setup.inp, setup.out)
            %----------------------------------------
            if ~isfield(setup, 'inp') || ~isfield(setup, 'out')
                error('ERROR: The setup structure must contain "inp", "out" and "ref" fields');
            end
            
            %----------------------------------------
            % Select input columns if specified
            %----------------------------------------
            if ~isempty(setup.inp)
                inputCols = ismember(setup.inp, obj.Data.Properties.VariableNames);
                if ~all(inputCols)
                    error('ERROR: Some columns in setup.inp do not exist in the data');
                end
                selectedInpCols = setup.inp;
            else
                selectedInpCols = obj.Data.Properties.VariableNames;
            end
            
            %----------------------------------------
            % Select output columns if specified
            %----------------------------------------
            if ~isempty(setup.out)
                outputCols = ismember(setup.out, obj.Data.Properties.VariableNames);
                if ~all(outputCols)
                    error('ERROR: Some columns in setup.out do not exist in the data');
                end
                selectedOutCols = setup.out;
            else
                selectedOutCols = obj.Data.Properties.VariableNames;
            end

            %----------------------------------------
            % Select reference columns if specified
            %----------------------------------------
            if ~isempty(setup.ref)
                outputCols = ismember(setup.ref, obj.Data.Properties.VariableNames);
                if ~all(outputCols)
                    error('ERROR: Some columns in setup.ref do not exist in the data');
                end
                selectedRefCols = setup.ref;
            else
                selectedRefCols = obj.Data.Properties.VariableNames;
            end
            
            %----------------------------------------
            % Combine selected input and output columns
            %----------------------------------------
            selectedCols = [selectedInpCols, selectedOutCols, selectedRefCols];
            selectedCols = unique(selectedCols, 'stable');
            obj.t = obj.Data.time;
            obj.X = table2array(obj.Data(:, selectedInpCols));
            obj.y = table2array(obj.Data(:, selectedOutCols));
            obj.r = table2array(obj.Data(:, selectedRefCols));

            %----------------------------------------
            % Filter the Data
            %----------------------------------------
            obj.Data = obj.Data(:, selectedCols);
            disp('INFO: Data filtered including input and output columns');
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1