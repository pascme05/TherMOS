%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Thermal Model Order Reduction and Simulation (TherMOS)           %
% Topic: Power Electronics, Model Order Reduction                         %
% File: rcSol                                                             %
% Date: 31.10.2024                                                        %
% Author: Dr. Pascal A. Schirmer                                          %
% Version: V.0.1                                                          %
% Copyright: Pascal Schirmer                                              %
% Comments: Update for core losses                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculates the temperature response for a 1D Foster network
% using trapazoidal rule:
%
%  dT = (2 * tau - Ts) / (2 * tau + Ts) * Tc + (Rth * dt) / (tau + dt)* Pv
%
% where tau=Rth*Cth is the time constant, dT is the sampling time and Pv 
% are the power losses.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Parameter
%===================================================
Ts = 1;                                                                     % Sampling time (sec)
Rth = [17.85, 24.95, 0.34, -38.86];                                         % Thermal resistances (K/W)
Cth = [12.13, 12.23, 25.30, -6.67];                                         % Thermal capacitances (Ws/K)
Weight = [1.00, 0.16, 0.05];                                                % Weights losses                                                
Tend = 3600;                                                                % final time (sec)
K = length(Rth);                                                            % Order of the RC network

%===================================================
% Variables
%===================================================
t = 0:Ts:Tend;
Test = zeros(length(t), K);
Tj = zeros(length(t),1);

%===================================================
% Power Losses
%===================================================
Pv_Pri = 1.90;
Pv_Sec = 5.06;
Pv_Cor = 1.84;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%===================================================
% Init
%===================================================
Pv = sum([Pv_Pri, Pv_Sec, Pv_Cor] .* Weight) .* ones(length(t), 1);
tau = Rth.*Cth;

%===================================================
% Solve Network
%===================================================
for i = 2:length(t)
    %----------------------------------------
    % Solve Nodes
    %----------------------------------------
    for ii = 1:K
        Test(i,ii) = ((2*tau(ii) - Ts)/(2*tau(ii) + Ts)) * Test(i-1,ii) + ... 
                      ((Rth(ii)*Ts)/(2*tau(ii) + Ts)) * (Pv(i)+Pv(i-1));
    end

    %----------------------------------------
    % Solve Junction
    %----------------------------------------
    Tj(i) = sum(Test(i,:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plot(t, Tj);
xlabel('Time (s)');
ylabel('Temperature (°C)');
title('Step Response of an RC Network');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1] Touzelbaev, Maxat N., et al. "High-efficiency transient temperature 
% calculations for applications in dynamic thermal management of electronic 
% devices." Journal of Electronic Packaging 135.3 (2013): 031001.