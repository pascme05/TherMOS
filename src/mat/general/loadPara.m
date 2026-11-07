%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: loadPara                                                          %
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
% Load parameters from the specified xlsx file and store them in a struct.
% -------------------------------------------------------------------------
% Inputs:
%   - fileName - The name of the Excel file (e.g., 'example.xlsx')
%   - filePath - The path to the directory (e.g., 'C:\data\')
% Outputs:
%   - config - A struct containing the parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function config = loadPara(fileName, filePath)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("----------------------------------------");
    disp("Loading Parameters")
    disp("----------------------------------------");
    disp("START: Loading Config Parameters")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fullFileName = fullfile(filePath.config, fileName);                     % full path name including file name
    fullFileName = fullFileName + '.xlsx';                                  % file name including extension
    sheets = {'Exp', 'Dat', 'Mdl', 'Par'};                                  % sheets names from parameter file
    config = struct();                                                      % empty structure for config

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % Load the complete xlsx file
    %===================================================
    sheetNames = sheetnames(fullFileName);                                  % sheetnames from xlsx file
    data = struct();                                                        % empty data struct
    
    %===================================================
    % Access the sheets
    %===================================================
    for i = 1:numel(sheetNames)
        sheetName = sheetNames{i};
        data.(sheetName) = readtable(fullFileName, 'Sheet', sheetName);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:numel(sheets)
        %===================================================
        % Init Sheet Name
        %===================================================
        sheetName = sheets{i};
        
        %===================================================
        % Access the data from the pre-loaded data struct
        %===================================================
        if isfield(data, sheetName)
            rawData = data.(sheetName);
        else
            error(['Sheet "' sheetName '" not found in the file.']);
        end
        
        %===================================================
        % Check that the necessary columns exist
        %===================================================
        if ~all(ismember({'Category', 'Variable', 'Value'}, rawData.Properties.VariableNames))
            error(['The sheet "' sheetName '" must contain the columns "Category", "Variable", and "Value".']);
        end
        
        %===================================================
        % Loop through each row in the sheet
        %===================================================
        for j = 1:height(rawData)
            %----------------------------------------
            % Extract Values
            %----------------------------------------
            category = rawData.Category{j};
            variable = rawData.Variable{j};
            value = rawData.Value(j);
            
            %----------------------------------------
            % Store the value in the struct
            %----------------------------------------
            % No Category
            if isnan(category)
                if isnan(value)
                    config.(sheetName).(variable) = 0;
                elseif isnumeric(value)
                    config.(sheetName).(variable) = value;
                else
                    config.(sheetName).(variable) = char(value);
                end

            % Category
            else
                if isnan(value)
                    config.(sheetName).(category).(variable) = 0;
                elseif isnumeric(value)
                    config.(sheetName).(category).(variable) = value;
                else
                    config.(sheetName).(category).(variable) = char(value);
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Output
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("DONE: Loading Config Parameters")
    fprintf('\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1