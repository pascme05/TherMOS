function [A_opt, B_opt] = optiPINN(t_data, a_data, F_data, A_init, B_init)
    % Input

    % Define the neural network
    layers = [
        sequenceInputLayer(length(A_init), 'Normalization', 'none')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(1)
        fullyConnectedLayer(length(A_init))
    ];
    
    % Define the dlnetwork object
    model = dlnetwork(layers);
    
    % Training options
    numEpochs = 100000;
    learningRate = 1e-4;
    Ts = t_data(2) - t_data(1); % Sample time
    
    % Input
    da_data = (a_data(2:end, :) - a_data(1:end-1, :)) / Ts;
    da_data = [da_data(1,:); da_data];

    % Training loop
    A = dlarray(A_init);
    B = dlarray(B_init);
    
    % Initialize the Adam optimizer parameters for A and B
    averageGrad_A = zeros(size(A_init), 'like', A_init);
    averageSqGrad_A = zeros(size(A_init), 'like', A_init);
    averageGrad_B = zeros(size(B_init), 'like', B_init);
    averageSqGrad_B = zeros(size(B_init), 'like', B_init);
    
    % Initialize the Adam optimizer parameters for the model
    averageGrad = [];
    averageSqGrad = [];
    
    for epoch = 1:numEpochs
        % Convert data to dlarray and specify the dimensions
        t_dl = dlarray(t_data', 'CT');  % 'CT' indicates channel (feature) and time (sequence)
        a_dl = dlarray(a_data, 'TC');  
        F_dl = dlarray(F_data, 'TC');  
    
        % Evaluate the loss and gradients
        [loss, gradients, ~, gradA, gradB] = dlfeval(@lossFunction, model, t_dl, a_dl, F_dl, A, B, Ts);
        
        % Update the learnable parameters using Adam optimizer
        [model.Learnables, averageGrad, averageSqGrad] = adamupdate(model.Learnables, gradients, ...
            averageGrad, averageSqGrad, epoch, learningRate);
        
        % Update A and B using the Adam optimizer
        [A, averageGrad_A, averageSqGrad_A] = adamupdate(A, gradA, averageGrad_A, averageSqGrad_A, epoch, learningRate);
        [B, averageGrad_B, averageSqGrad_B] = adamupdate(B, gradB, averageGrad_B, averageSqGrad_B, epoch, learningRate);
        
        % Display loss
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    % Extract optimized system matrices
    A_opt = extractdata(A);
    B_opt = extractdata(B);
    
    % Loss function
    function [loss, gradients, a_pred, gradA, gradB] = lossFunction(model, t, a_true, F, A, B, Ts)
        a_pred = forward(model, F);
        a_t = (a_pred(:, 2:end) - a_pred(:, 1:end-1)) / Ts;
        a_t = [a_t(:,1), a_t];
        a_pred = extractdata(a_pred);
        a_true = extractdata(a_true);
        a_t = extractdata(a_t);
        physics_residual = B * a_t + A * a_pred - F;
        data_loss = mean((a_true - a_pred).^2, 'all') / 100000;
        physics_loss = mean(physics_residual.^2, 'all');
        loss = data_loss + physics_loss;
        [gradients, gradA, gradB] = dlgradient(loss, model.Learnables, A, B);
    end
end

