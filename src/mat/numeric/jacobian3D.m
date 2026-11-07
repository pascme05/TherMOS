function Jgrid = jacobian3D(xInp, yInp, zInp, xOut, yOut, zOut, th)
    % Create tetrahedral mesh of input points
    dt = delaunayTriangulation(xInp(:), yInp(:), zInp(:));
    tetra = dt.ConnectivityList;
    pts = dt.Points;
    
    % Get vertices of each tetrahedron
    v1 = pts(tetra(:,1), :);
    v2 = pts(tetra(:,2), :);
    v3 = pts(tetra(:,3), :);
    v4 = pts(tetra(:,4), :);
    
    % Compute edge vectors from v1 to the other vertices
    e1 = v2 - v1;  % ξ direction
    e2 = v3 - v1;  % η direction
    e3 = v4 - v1;  % ζ direction
    
    % Form the Jacobian matrices for each tetrahedron (3x3 matrix per tetrahedron)
    % Each row corresponds to partial derivatives of x, y, z with respect to ξ, η, ζ
    J = zeros(size(tetra,1), 1);
    for i = 1:size(tetra,1)
        Jmat = [e1(i,:); e2(i,:); e3(i,:)]';  % 3x3 matrix
        J(i) = abs(det(Jmat));  % Volume scaling factor
    end
    
    % Reference tetrahedron volume = 1/6
    refVol = median(J);
    jacobian = J / refVol;
    
    % Compute centroid of each tetrahedron for interpolation
    xc = mean([v1(:,1), v2(:,1), v3(:,1), v4(:,1)], 2);
    yc = mean([v1(:,2), v2(:,2), v3(:,2), v4(:,2)], 2);
    zc = mean([v1(:,3), v2(:,3), v3(:,3), v4(:,3)], 2);
    
    % Interpolate onto output grid
    [Xq, Yq, Zq] = ndgrid(xOut, yOut, zOut);
    F = scatteredInterpolant(xc, yc, zc, jacobian, 'linear', 'nearest');
    Jgrid = F(Xq, Yq, Zq);
    
    % Handle points outside convex hull
    Jgrid(isnan(Jgrid)) = 1;
    Jgrid(Jgrid < 1+th & Jgrid > 1-th) = 1;
    
    % Ensure positive values
    Jgrid = abs(Jgrid);
end