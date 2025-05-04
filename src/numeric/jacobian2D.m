function Jgrid = jacobian2D(xInp, yInp, xOut, yOut, dx, dy)
    dt = delaunayTriangulation(xInp(:), yInp(:));
    tri = dt.ConnectivityList;
    pts = dt.Points;

    x1 = pts(tri(:,1),1); y1 = pts(tri(:,1),2);
    x2 = pts(tri(:,2),1); y2 = pts(tri(:,2),2);
    x3 = pts(tri(:,3),1); y3 = pts(tri(:,3),2);

    detJ = (x2 - x1).*(y3 - y1) - (x3 - x1).*(y2 - y1);
    area = 0.5 * abs(detJ);  % triangle area


    % Normalize by expected uniform triangle area
    refArea = mean(area);
    jacobianCorrection = area / refArea;  % should be ~1 if mesh is uniform

    % Interpolate triangle centers
    xc = (x1 + x2 + x3) / 3;
    yc = (y1 + y2 + y3) / 3;

    [Xq, Yq] = meshgrid(xOut, yOut);
    F = scatteredInterpolant(xc, yc, jacobianCorrection, 'linear', 'none');
    % Jgrid = F(Xq, Yq)/max(xOut)/max(yOut);
    % Jgrid = F(Xq, Yq)/max(xOut);
    Jgrid = F(Xq, Yq);

    Jgrid(isnan(Jgrid)) = 1;  % default to 1 if outside convex hull
end