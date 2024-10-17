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
        function data = loadData(obj, FileName, SheetName, ID, stat, para, setup)
            %----------------------------------------
            % 1D Xlsx
            %----------------------------------------
            if obj.Type == "xlsx"
                % Loading
                data = loadXlsx(obj, FileName, SheetName, ID, stat, para, setup);

            %----------------------------------------
            % 1D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim == 1
                % Loading
                obj = loadXlsx(obj, FileName, SheetName, ID, stat, para, setup);

            %----------------------------------------
            % 2D Mat
            %----------------------------------------
            elseif obj.Type == "mat" && obj.Dim == 2
                % Loading
                obj = loadXlsx(obj, FileName, SheetName, ID, stat, para, setup);

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
        function obj = loadXlsx(obj, FileName, SheetName, ID, stat, para, setup)
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
            if para.Dat.gen.sep == 1 || para.Dat.gen.sep == 2
                disp('INFO: File and sheet based data');

            % ID based
            elseif para.Dat.gen.sep == 3
                obj.Data = obj.Data(obj.Data.id == ID, :);
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