function [t, y] = rk6(odefun, tspan, y0, h, varargin)
    % Sixth-Order Runge-Kutta method for solving ODEs
    % odefun: function handle for the ODE (dy/dt = f(t, y, ...))
    % tspan: [t0, tf] interval of integration
    % y0: initial value of y at t0
    % h: step size
    % varargin: additional parameters for the ODE function

    % Extract the start and end times
    t0 = tspan(1);
    tf = tspan(2);
    
    % Number of steps
    N = ceil((tf - t0) / h);
    
    % Initialize arrays to store the results
    t = linspace(t0, tf, N + 1);
    y = zeros(length(y0), N + 1);
    
    % Set the initial condition
    y(:, 1) = y0;
    
    % Coefficients for the sixth-order Runge-Kutta method
    b1 = 1/6;
    b2 = 0;
    b3 = 2/3;
    b4 = 1/6;
    b5 = 0;
    b6 = 0;
    
    c2 = 1/3;
    c3 = 1/3;
    c4 = 1/2;
    c5 = 1;
    c6 = 1;
    
    a21 = 1/3;
    a31 = -1/3; a32 = 1;
    a41 = 1/2; a42 = -1; a43 = 1;
    a51 = 1; a52 = -1; a53 = 1; a54 = 0;
    a61 = 0; a62 = 0; a63 = 0; a64 = 1;
    
    % Main integration loop
    for n = 1:N
        % Current time
        tn = t(n);
        
        % Current value of y
        yn = y(:, n);
        
        % Compute the Runge-Kutta stages
        k1 = h * odefun(tn, yn, varargin{:});
        k2 = h * odefun(tn + c2 * h, yn + a21 * k1, varargin{:});
        k3 = h * odefun(tn + c3 * h, yn + a31 * k1 + a32 * k2, varargin{:});
        k4 = h * odefun(tn + c4 * h, yn + a41 * k1 + a42 * k2 + a43 * k3, varargin{:});
        k5 = h * odefun(tn + c5 * h, yn + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, varargin{:});
        k6 = h * odefun(tn + c6 * h, yn + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + k5, varargin{:});
        
        % Update the solution
        y(:, n + 1) = yn + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6;
    end
end
