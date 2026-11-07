function [x, w] = gauss_legendre(N)
% Returns N-point Gauss-Legendre quadrature nodes x and weights w on [-1, 1]

    beta = 0.5 ./ sqrt(1 - (2*(1:N-1)).^(-2));     % Recurrence coefficients
    T = diag(beta,1) + diag(beta,-1);              % Tridiagonal Jacobi matrix
    [V, D] = eig(T);                               
    x = diag(D);                                   % Eigenvalues = nodes
    [x, i] = sort(x);                              % Sort nodes
    w = 2 * (V(1,i)').^2;                          % Weights
end