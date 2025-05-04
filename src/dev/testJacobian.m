% Create a simple uniform grid
dx = 0.01;
dy = 0.01;
[xInp, yInp] = meshgrid(0:dx:0.4, 0:dy:0.4);
xInp = xInp(:);
yInp = yInp(:);

% Define output grid (same resolution for interpolation)
xOut = 0:dx:0.4;
yOut = 0:dy:0.4;

% Call your function
Jgrid = jacobian2D(xInp, yInp, xOut, yOut, dx, dy);

% Visualize the result
figure;
imagesc(xOut, yOut, Jgrid);
axis equal tight;
colorbar;
title('Jacobian Determinant Grid');
xlabel('x');
ylabel('y');

% Check statistics
fprintf('Jacobian statistics:\n');
fprintf('Min: %.4f\n', min(Jgrid(:)));
fprintf('Max: %.4f\n', max(Jgrid(:)));
fprintf('Mean: %.4f\n', mean(Jgrid(:)));
fprintf('Std:  %.4f\n', std(Jgrid(:)));