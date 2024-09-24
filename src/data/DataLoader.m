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
% Here goes the description of the class.
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
        time                                                                % Time vector (sec)
        X                                                                   % Input vector
        y                                                                   % Output vector
        r                                                                   % Reference vector
        Ts {mustBeNumeric}                                                  % sampling time (sec)
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
        function data = loadData(obj, FileName, SheetName, ID, para, setup)
            %----------------------------------------
            % 1D Xlsx
            %----------------------------------------
            if obj.Type == "xlsx"
                % Loading
                obj = loadXlsx(obj, FileName, SheetName, ID, para, setup);

                % Output
                data.Raw = obj.Data;

            %----------------------------------------
            % 1D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim == 1
                % Loading
                obj = loadXlsx(obj, FileName, SheetName, ID, para, setup);

            %----------------------------------------
            % 2D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim == 2
                % Loading
                obj = loadXlsx(obj, FileName, SheetName, ID, para, setup);

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
        function obj = loadXlsx(obj, FileName, SheetName, ID, para, setup)
            %----------------------------------------
            % File based and Sheet based
            %----------------------------------------
            if para.Dat.gen.sep == 1 || para.Dat.gen.sep == 2
                % Check if FileName is set
                if isempty(FileName)
                    error('ERROR: FileName is not set. Please set the FileName property');
                end
                
                % Try loading the data from the specified worksheet
                try
                    obj.Data = readtable(FileName, 'Sheet', SheetName);
                    disp('INFO: Xlsx data successfully loaded');
                catch ME
                    disp('INFO: Failed to load data from the specified worksheet');
                    rethrow(ME);
                end

                % Resampling Data
                try
                    obj.Data = resampleData1D(obj, obj.Data.time);
                catch ME
                    disp('INFO: Failed to resample data');
                    rethrow(ME);
                end

                % Selecting Input and Ouput
                try
                    obj = selectInpOut(obj, setup);
                catch ME
                    disp('INFO: Failed to select input and output');
                    rethrow(ME);
                end

            %----------------------------------------
            % ID based
            %----------------------------------------
            elseif para.Dat.gen.sep == 3

            %----------------------------------------
            % Split based
            %----------------------------------------
            else
            end

            %----------------------------------------
            % Output
            %----------------------------------------

        end

        %===================================================
        % Load Matlab 1D Data
        %===================================================
        function r = loadMat1D(obj,n)
            r = [obj.Value] * n;
        end

        %===================================================
        % Load Matlab 2D Data
        %===================================================
        function r = loadMat2D(obj,n)
            r = [obj.Value] * n;
        end

        %===================================================
        % Resample the Data
        %===================================================
        function out = resampleData1D(obj, tref)
            %----------------------------------------
            % Check if Data exists
            %----------------------------------------
            if isempty(obj.Data)
                error('ERROR: No data to resample. Load the data first');
            end
            
            %----------------------------------------
            % Original time vector 
            %----------------------------------------
            if nargin < 2
                error('INFO: You must provide the original time vector');
            end
            
            %----------------------------------------
            % Create new time vector
            %----------------------------------------
            startTime = tref(1);
            endTime = tref(end);
            newTime = (startTime:obj.Ts:endTime)';
            
            %----------------------------------------
            % New Output
            %----------------------------------------
            names = obj.Data.Properties.VariableNames;
            out = zeros(length(newTime), width(obj.Data));

            %----------------------------------------
            % Interpolation
            %----------------------------------------
            for col = 1:width(obj.Data)
                if obj.Intp == 1
                    out(:, col) = interp1(tref, obj.Data{:, col}, newTime, 'previous');
                elseif obj.Intp == 3
                    out(:, col) = interp1(tref, obj.Data{:, col}, newTime, 'spline');
                else
                    out(:, col) = interp1(tref, obj.Data{:, col}, newTime, 'linear');
                end
            end

            %----------------------------------------
            % Update the Data 
            %----------------------------------------
            out = array2table(out, 'VariableNames', names);
            disp('INFO: Data resampled');
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
            obj.time = obj.Data.time;
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