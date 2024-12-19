%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: plotMdl                                                           %
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
% Here goes the description of the function.
% -------------------------------------------------------------------------
% Inp:  1) data:    All simulation input data as well as prediction
%       2) mdl:     Model data
%       3) setup:   All simulation setup parameters
% Out:  1) None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plotMdl(~, mdl, setup)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Message Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("INFO: Plotting Model")

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Models
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %===================================================
    % RC-Model
    %===================================================
    if setup.selRC == 1
        %----------------------------------------
        % Init
        %----------------------------------------
        % Parameters
        Rth = mdl.Rth;                                                      % Thermal resistances in °C/W
        Cth = mdl.Cth;                                                      % Thermal capacitances in J/°C
        tau = Rth.*Cth;                                                     % Thermal time constants in sec
        num_nodes = length(Rth);                                            % number of RC nodes

        % Variables
        t = logspace(log10(min(tau)/10), log10(max(tau)*10), 1000);         % Log time vector
        Z_t = zeros(size(t));                                               % Initialize Z(t)
        
        %----------------------------------------
        % Calculation
        %----------------------------------------
        for i = 1:length(Rth)
            Z_t = Z_t + Rth(i) * (1 - exp(-t / tau(i)));
        end

        %----------------------------------------
        % Plotting
        %----------------------------------------
        % Create figure
        figure;
        
        % Set figure size
        set(gcf, 'Position', [100, 100, 800, 600]);
    
        % Load and display the image (RC network schematic)
        subplot(2, 1, 1); % Top subplot for image and table
        axis off; % No axis for the image and table
        
        % Load the image (image_file is the path to your image file)
        img = imread("rcMdl.png");

        % Display the image in the top half of the figure
        axes('Position', [0.25, 0.7, 0.5, 0.2]);
        imshow(img);
        axis off;
        
        % Insert the table of RC parameters below the image
        uitable('Data', [Rth(:), Cth(:)], 'ColumnName', {'Rth (°C/W)', 'Cth (J/°C)'}, ...
            'RowName', arrayfun(@(x) ['Node-', num2str(x)], 1:num_nodes, 'UniformOutput', false), ...
            'Units', 'normalized', 'Position', [0.25, 0.51, 0.5, 0.12]);
    
        % Plot the transient thermal impedance in the lower half of the figure
        subplot(2, 1, 2);
        semilogx(t, Z_t, 'LineWidth', 2); 
        grid on;
        xlabel('Time (s)');
        ylabel('Z(t) (°C/W)');
        title('Transient Thermal Impedance');
        
    end
    
    %===================================================
    % SS-Model
    %===================================================
    if setup.selSS == 1
        %----------------------------------------
        % Init
        %----------------------------------------
        A = mdl.sys.A;
        B = mdl.sys.B;
        C = mdl.sys.C;
        D = mdl.sys.D;

        %----------------------------------------
        % Plotting
        %----------------------------------------
        % Create figure
        figure;
        
        % Set figure size
        set(gcf, 'Position', [100, 100, 800, 600]);
    
        % Load and display the image (RC network schematic)
        subplot(2, 1, 1); % Top subplot for image and table
        axis off; % No axis for the image and table
        
        % Load the image (image_file is the path to your image file)
        img = imread("ssMdl.png");

        % Display the image in the top half of the figure
        axes('Position', [0.25, 0.7, 0.5, 0.2]);
        imshow(img);
        axis off;
        
        % Determine the number of matrices to be displayed
        num_tables = 4;
        
        % Define the width each table should take up (assuming equal width for each)
        table_width = 1 / num_tables;
        
        % Define the vertical position and height for all tables (they are aligned horizontally)
        table_height = 0.1;
        table_vert_pos = 0.55;
        
        % Convert the state-space matrices to display in uitables side by side
        uitable('Data', A, 'ColumnName', arrayfun(@(x) ['A', num2str(x)], 1:size(A, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['A', num2str(x)], 1:size(A, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [0, table_vert_pos, table_width, table_height]);
        
        uitable('Data', B, 'ColumnName', arrayfun(@(x) ['B', num2str(x)], 1:size(B, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['B', num2str(x)], 1:size(B, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [table_width, table_vert_pos, table_width, table_height]);
        
        uitable('Data', C, 'ColumnName', arrayfun(@(x) ['C', num2str(x)], 1:size(C, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['C', num2str(x)], 1:size(C, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [2 * table_width, table_vert_pos, table_width, table_height]);
        
        uitable('Data', D, 'ColumnName', arrayfun(@(x) ['D', num2str(x)], 1:size(D, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['D', num2str(x)], 1:size(D, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [3 * table_width, table_vert_pos, table_width, table_height]);
    
        % Plot the transient thermal impedance in the lower half of the figure
        subplot(2, 1, 2);
        step(mdl.sys);
    end

    %===================================================
    % SF-Model
    %===================================================
    if setup.selSF == 1
    end

    %===================================================
    % PO-Model
    %===================================================
    if setup.selPO == 1
        %----------------------------------------
        % Init
        %----------------------------------------
        Cth = mdl.Cth;
        Gth = mdl.Gth;
        K = mdl.K;
        Phi = mdl.sPhi;
        
        %----------------------------------------
        % Pre-Processing
        %----------------------------------------
        if K > 3
            K = 3;
        end

        %----------------------------------------
        % Plotting
        %----------------------------------------
        % Create figure
        figure;
        
        % Set figure size
        set(gcf, 'Position', [100, 100, 800, 600]);
    
        % Load and display the image (RC network schematic)
        subplot(2, K, 1:K); % Top subplot for image and table
        axis off; % No axis for the image and table
        
        % Load the image (image_file is the path to your image file)
        img = imread("ssMdl.png");

        % Display the image in the top half of the figure
        axes('Position', [0.25, 0.7, 0.5, 0.2]);
        imshow(img);
        axis off;
        
        % Determine the number of matrices to be displayed
        num_tables = 2;
        
        % Define the width each table should take up (assuming equal width for each)
        table_width = 1 / num_tables;
        
        % Define the vertical position and height for all tables (they are aligned horizontally)
        table_height = 0.1;
        table_vert_pos = 0.55;
        
        % Convert the state-space matrices to display in uitables side by side
        uitable('Data', Gth, 'ColumnName', arrayfun(@(x) ['Gth', num2str(x)], 1:size(Gth, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['Gth', num2str(x)], 1:size(Gth, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [0, table_vert_pos, table_width, table_height]);
        
        uitable('Data', Cth, 'ColumnName', arrayfun(@(x) ['Cth', num2str(x)], 1:size(Cth, 2), 'UniformOutput', false), ...
                'RowName', arrayfun(@(x) ['Cth', num2str(x)], 1:size(Cth, 1), 'UniformOutput', false), ...
                'Units', 'normalized', 'Position', [table_width, table_vert_pos, table_width, table_height]);
    
        % Plot the transient thermal impedance in the lower half of the figure
        for i = 1:K
            subplot(2, K, i+K);
            surf(squeeze(Phi(:,:,i)));
            ylabel("y (m)");
            xlabel("x (m)");
            txt = "Spatial Mode Phi-" + num2str(i);
            title(txt);
        end
    end

    %===================================================
    % ML-Model
    %===================================================
    if setup.selML == 1
    end
    
    %===================================================
    % DL-Model
    %===================================================
    if setup.selDL == 1
        figure;
        plot(mdl.sys);
        title('Deep Learning Network Structure');
        grid on;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] REF-1