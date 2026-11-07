% Function to perform stencil integration
function result = intStencil(Nx, Ny, dx, dy, Inp, deg)
    % Get the stencil weights for the specified degree
    weights = get_weights(deg);
    
    % Determine the grid offsets based on the stencil degree
    % Example for a 3x3 stencil, the offsets would be {-1, 0, 1} for both x and y directions
    % The stencil degree defines how many neighboring points will be considered
    [x_offset, y_offset] = get_stencil_offsets(deg);
    
    % Initialize the result to zero
    result = 0;
    
    % Loop through the grid to apply the stencil
    for ix = 1:Nx
        for iy = 1:Ny
            % Loop over all stencil weights and apply them
            for w_idx = 1:numel(weights)
                % Calculate the corresponding neighbor indices
                x_idx = ix + x_offset(w_idx);
                y_idx = iy + y_offset(w_idx);
                
                % Check if the neighbor is within bounds
                if x_idx >= 1 && x_idx <= Nx && y_idx >= 1 && y_idx <= Ny
                    % Apply the stencil weight to the Phi value at the neighboring point
                    result = result + weights(w_idx) * Inp(y_idx, x_idx);
                end
            end
        end
    end
    
    % Normalize by dx and dy
    result = result / (dx * dy);
end

function w = get_weights(deg)
    % Returns the stencil weights for the given degree
    switch deg
        case 1
            % Single central weight
            w = 0.25;
            
        case 2
            % 4-point stencil (corners of a square)
            w = [0.25, 0.25, 0.25, 0.25];
            
        case 3
            % 5-point stencil
            w = [-0.8, 0.45, 0.45, 0.45, 0.45];
            
        case 4
            % 14-point stencil
            w = [0.0190476190476190, 0.0190476190476190, 0.0190476190476190, 0.0190476190476190, ...
                 0.0190476190476190, 0.0190476190476190, 0.0885898247429807, 0.0885898247429807, ...
                 0.0885898247429807, 0.0885898247429807, 0.1328387466855907, 0.1328387466855907, ...
                 0.1328387466855907, 0.1328387466855907];
             
        case 5
            % 15-point stencil
            w = [0.1817020685825351, ...
                 0.0361607142857143, 0.0361607142857143, 0.0361607142857143, 0.0361607142857143, ...
                 0.0698714945161738, 0.0698714945161738, 0.0698714945161738, 0.0698714945161738, ...
                 0.0656948493683187, 0.0656948493683187, 0.0656948493683187, 0.0656948493683187, ...
                 0.0656948493683187, 0.0656948493683187];
             
        case 6
            % 24-point stencil
            w = [0.0399227502581679, 0.0399227502581679, 0.0399227502581679, 0.0399227502581679, ...
                 0.0100772110553207, 0.0100772110553207, 0.0100772110553207, 0.0100772110553207, ...
                 0.0553571815436544, 0.0553571815436544, 0.0553571815436544, 0.0553571815436544, ...
                 0.0482142857142857, 0.0482142857142857, 0.0482142857142857, 0.0482142857142857, ...
                 0.0482142857142857, 0.0482142857142857, 0.0482142857142857, 0.0482142857142857, ...
                 0.0482142857142857, 0.0482142857142857, 0.0482142857142857, 0.0482142857142857];
             
        otherwise
            error('Degree not implemented');
    end
end

function [x_offset, y_offset] = get_stencil_offsets(deg)
    % This generates the x and y offsets for each stencil weight
    switch deg
        case 1
            % Single central point
            x_offset = 0;
            y_offset = 0;

        case 2
            % 4-point stencil (corners of a square)
            x_offset = [-1, 1, 0, 0];
            y_offset = [0, 0, -1, 1];

        case 3
            % 5-point stencil
            x_offset = [0, -1, 1, 0, 0];
            y_offset = [0, 0, 0, -1, 1];

        case 4
            % 14-point stencil
            x_offset = [ 0, -1,  1,  0, -1,  1, ...
                        -2, -2,  2,  2, -1,  1,  0, 0];
            y_offset = [ 0,  0,  0, -1,  1,  1, ...
                         -1,  1, -1,  1, -2, -2, -2, 2];

        case 5
            % 15-point stencil
            x_offset = [ 0, -1,  1,  0, -1,  1, ...
                        -2, -2,  2,  2, -1,  1,  0,  0,  0];
            y_offset = [ 0,  0,  0, -1,  1,  1, ...
                         -1,  1, -1,  1, -2, -2,  2,  2,  0];

        case 6
            % 24-point stencil
            x_offset = [-1,  0,  1, -1,  0,  1, ...
                        -2, -1,  1,  2, -2, -1,  1,  2, ...
                        -3, -2, -1,  1,  2,  3, -3, -2,  2,  3];
            y_offset = [ 0, -1,  0,  1,  0,  1, ...
                        -1, -1, -1, -1,  1,  1,  1,  1, ...
                         0, -1, -2, -2, -2,  0,  0,  2,  2,  2];

        otherwise
            error('Stencil degree not supported.');
    end
end