function [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue, yPred)
    err = yTrue - yPred; 
    MAE = mean(abs(err));
    RMSE = sqrt(mean(err.^2));
    SAE = (mean(abs(yTrue)) - mean(abs(yPred))) / mean(abs(yTrue));
    NAE = mean(abs(err)./mean(yTrue));
    MAX = max(abs(err));
    NRSME = 100*(1-(sum(abs(err)))/sum(abs(yTrue - mean(yTrue))));
end

