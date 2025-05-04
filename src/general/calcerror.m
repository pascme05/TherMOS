function [MAE, RMSE, SAE, NAE, MAX, NRSME] = calcerror(yTrue, yPred)
    err = yTrue - yPred;
    absErr = abs(err);
    absYTrue = abs(yTrue);

    MAE = mean(absErr);
    RMSE = norm(err) / sqrt(length(err));
    SAE = (mean(absYTrue) - mean(abs(yPred))) / mean(absYTrue);
    NAE = sum(absErr) / sum(absYTrue);
    MAX = max(absErr);

    yTrueCentered = yTrue - mean(yTrue);
    normDenom = norm(yTrueCentered);
    
    if normDenom == 0
        NRSME = 100 * (1 - norm(err));
    else
        NRSME = 100 * (1 - norm(err) / normDenom);
    end
end

