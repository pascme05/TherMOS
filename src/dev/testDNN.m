% Given input data X of size [3, 10, 17511]
C = 3; % Number of features (channels)
W = 10; % Sequence length (window length)
Nt = 17511; % Number of time steps (samples)
N = 5; % Number of output nodes (assuming)

% Randomly generate example input and output data
X = rand(C, W, Nt);
Y = rand(N, Nt); % Make sure Y has size [N, Nt]

% Convert X to a cell array for sequence input (each cell is a sequence)
X = squeeze(mat2cell(X, C, W, ones(1, Nt))); % Cell array of size [1, Nt]

% Define LSTM Network Architecture
inputSize = C; % Number of features (channels)
numHiddenUnits = 100; % Number of hidden units in LSTM
numClasses = N; % Number of output nodes

layers = [ ...
    sequenceInputLayer(inputSize) % Input feature dimension
    lstmLayer(numHiddenUnits, 'OutputMode', 'last') % 'last' for sequence-to-point
    fullyConnectedLayer(numClasses)
    regressionLayer]; % Use regression layer for continuous output

% Specify Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', 0);

% Train the Network
net = trainNetwork(X, Y', layers, options); % Transpose Y to match the expected format

% Evaluate the Network (uncomment and modify if you have test data)
% Xt = ... % Prepare test input data
% Yt = ... % Prepare test output data

% Convert test data to cell arrays (similar to training data preparation)
% Xt = squeeze(mat2cell(Xt, C, W, ones(1, Nt_test))); % Cell array of size [1, Nt_test]

% Predict using the trained network
% Yt_pred = predict(net, Xt);

% Evaluate the performance
% performance = mean((cell2mat(Yt_pred) - Yt').^2); % Mean Squared Error
% disp(['Mean Squared Error: ', num2str(performance)]);
