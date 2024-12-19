function [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue, yPred)
    err = yTrue - yPred; 
    MAE = mean(abs(err));
    RMSE = sqrt(mean(err.^2));
    SAE = (mean(abs(yTrue)) - mean(abs(yPred))) / mean(abs(yTrue));
    NAE = mean(abs(err)./mean(yTrue));
    MAX = max(abs(err));
    if norm(yTrue - mean(yTrue)) == 0
        NRSME = 100 * (1 - norm(err));
    else
        NRSME = 100 * (1 - norm(err) / norm(yTrue - mean(yTrue)));
    end
end

