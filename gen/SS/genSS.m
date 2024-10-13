% #########################################################################
% #                        Example Step Responses                         #
% #                                                                       #
% # This script generates step responses for a thermal system using a     #
% # capacitance matrix (C), a conductance matrix (G), and constant heat   #
% # inputs applied to each of the 5 heat inputs. The system has 5 heat    #
% # inputs and 5 temperature outputs. The results are plotted to visualize#
% # the temperature response over time for each input.                    #
% #########################################################################

% Define the parameters
n = 5; % Number of temperature outputs (nodes) and heat inputs

% Example capacitance matrix (C)
C = diag([10, 15, 20, 25, 30]); % Diagonal matrix representing thermal capacitance

% Example conductance matrix (G)
G = [8 -2 0 0 0; -2 6 -2 0 0; 0 -2 5 -1 0; 0 0 -1 4 -1; 0 0 0 -1 3]; % Symmetric conductance matrix

% Time vector
t = linspace(0, 100, 501); % Time from 0 to 100 seconds, with 500 points

% Initial temperature (assuming all temperatures start from 0)
T0 = zeros(n, 1);

% Pre-allocate for temperature response storage
T_responses = zeros(length(t), n, 3);

% Define constant heat inputs for each node
P_tr = [10, 15, 20, 25, 30];
P_te = [20, 50, 10, 15, 60];
P_vl = [30, 35, 50, 10, 5];
P_values = [P_tr; P_te; P_vl];

% Calculate step responses for each heat input
for i = 1:3
    % Create constant input vector for the i-th heat input
    P = P_values(i, :)';

    % Define the system of ODEs: C*dT/dt = P(t) - G*T
    % Rearrange to form dT/dt = C^(-1)*(P(t) - G*T)
    dTdt = @(t, T) C \ (P - G*T);
    
    % Solve the system of ODEs using ode45
    [~, T] = ode15s(dTdt, t, T0);
    
    % Store the temperature response for this input
    T_responses(:, :, i) = T;
end

% Plot the results
figure;
for i = 1:3
    subplot(3, 1, i);
    hold on;
    for j = 1:n
        plot(t, T_responses(:, j, i), 'DisplayName', ['Output ' num2str(j)]);
    end
    title(['Step Response to Heat Input ' num2str(i)]);
    xlabel('Time (s)');
    ylabel('Temperature (°C)');
    legend show;
    grid on;
end

% Output
T_tr = squeeze(T_responses(:,:,1)) + 20;
T_te = squeeze(T_responses(:,:,2)) + 50;
T_vl = squeeze(T_responses(:,:,3)) + 35;


% #########################################################################
% #                         End of Script                                 #
% #########################################################################