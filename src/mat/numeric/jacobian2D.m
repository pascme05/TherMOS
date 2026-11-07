function Jgrid = jacobian2D(xInp, yInp, xOut, yOut, th)
    % Create triangulation of input mesh
    dt = delaunayTriangulation(xInp(:), yInp(:));
    tri = dt.ConnectivityList;
    pts = dt.Points;
    
    % Get vertices for all triangles
    v1 = pts(tri(:,1),:);
    v2 = pts(tri(:,2),:);
    v3 = pts(tri(:,3),:);
    
    % Calculate Jacobian matrix for each triangle [dx/dξ dy/dξ; dx/dη dy/dη]
    % For standard reference triangle (0,0)-(1,0)-(0,1) with area = 0.5
    J11 = v2(:,1) - v1(:,1);  % dx/dξ
    J12 = v2(:,2) - v1(:,2);  % dy/dξ
    J21 = v3(:,1) - v1(:,1);  % dx/dη
    J22 = v3(:,2) - v1(:,2);  % dy/dη
    
    % Jacobian determinant (area scaling factor)
    detJ = J11.*J22 - J12.*J21;
    
    % Reference area for standard triangle = 0.5
    refArea = median(abs(detJ));
    
    % Normalized Jacobian (should be ~1 for perfect mapping)
    jacobian = abs(detJ)/refArea;
    
    % Interpolate to output grid using triangle centers
    xc = mean([v1(:,1), v2(:,1), v3(:,1)], 2);
    yc = mean([v1(:,2), v2(:,2), v3(:,2)], 2);
    
    [Xq, Yq] = meshgrid(xOut, yOut);
    F = scatteredInterpolant(xc, yc, jacobian, 'linear', 'nearest');
    Jgrid = F(Xq, Yq);
    
    % Handle points outside convex hull
    Jgrid(isnan(Jgrid)) = 1;
    Jgrid(Jgrid < 1+th & Jgrid > 1-th) = 1;
    
    % Ensure positive values
    Jgrid = abs(Jgrid);
end