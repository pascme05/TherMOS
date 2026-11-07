#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         mlFit
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function fits a static machine learning regression (scikit-learn) from the input features of one sample to the
temperature of the first node: T(k) = f(X(k)). Dynamic behaviour is only captured through the EWMA/EWMS features
(setup['featRoll']). Models (setup['selML']): (1) linear regression, (2) regression tree, (3) support vector
regression, (4) Gaussian process regression, (5) ensemble (LSBoost or bagging), (6) neural network (MLP). If
para['Mdl']['gen']['opt'] = 1 the hyperparameters of models 2, 3, 5 and 6 are optimised with a random search with
3-fold cross validation on the validation data before the model is trained on the training data.
Inputs:     1) data:    training data
            2) val:     validation data
            3) para:    all simulation parameters
            4) setup:   includes all simulation variables
            5) legacy:  (0): corrected behaviour, (1): MATLAB behaviour
Outputs:    1) mdl:     model {'sys': fitted estimator}

V1.0: 07.11.2026, Pascal Schirmer
- Python port of mlFit.m; the target is the concatenated temperature of all profiles (MATLAB used the profile mean
  with a different length than the inputs, F-L1), the stray 'MPG' argument of the ensemble optimisation is removed
  (F-L2), the GPR basis function is approximated by a linear/constant mean (normalize_y)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.py.general.calcFeat import calcFeat

# ==============================================================================
# External
# ==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _feat(d, setup, legacy):
    if setup.get('featRoll', 0) == 0:
        return np.asarray(d['X'], dtype=float)
    return np.vstack([calcFeat(x, setup['EWMA'], setup['EWMS'], None, legacy) for x in d['X2']])


def _model(sel, p):
    if sel == 1:
        return LinearRegression(), {}
    elif sel == 2:
        dt = p['dt']
        return (DecisionTreeRegressor(min_samples_leaf=int(dt['leaf']), max_leaf_nodes=max(int(dt['split']) + 1, 2),
                                      min_samples_split=max(int(dt['parent']), 2), random_state=42),
                {'min_samples_leaf': [1, 2, 5, 10, 20, 50], 'max_leaf_nodes': [8, 16, 32, 64, 128, 256]})
    elif sel == 3:
        svm = p['svm']
        kernel = {1: 'linear', 2: 'poly', 3: 'rbf'}.get(int(svm['kernel']), 'linear')
        return (make_pipeline(StandardScaler(), SVR(kernel=kernel, C=svm['C'], epsilon=svm['eps'], gamma='scale')),
                {'svr__C': np.logspace(-2, 3, 12), 'svr__epsilon': np.logspace(-3, 0, 8)})
    elif sel == 4:
        gpr = p['gpr']
        base = Matern(nu=0.5) if int(gpr['kernel']) == 1 else RBF()
        return (GaussianProcessRegressor(ConstantKernel() * base + WhiteKernel(), normalize_y=True,
                                         n_restarts_optimizer=0, random_state=42), {})
    elif sel == 5:
        en = p['en']
        if int(en['method']) == 2:
            return (BaggingRegressor(DecisionTreeRegressor(), n_estimators=int(en['cycle']), random_state=42),
                    {'n_estimators': [10, 50, 100, 200]})
        return (GradientBoostingRegressor(n_estimators=int(en['cycle']), max_leaf_nodes=max(int(en['split']) + 1, 2),
                                          random_state=42),
                {'n_estimators': [10, 50, 100, 200, 500], 'max_leaf_nodes': [2, 4, 8, 16, 32]})
    elif sel == 6:
        nn = p['nn']
        layers = tuple([int(nn['nodes'])] * {1: 1, 2: 2, 3: 3}.get(int(nn['layer']), 2))
        act = {1: 'relu', 2: 'tanh', 3: 'logistic'}.get(int(nn['act']), 'relu')
        return (make_pipeline(StandardScaler(), MLPRegressor(layers, activation=act, max_iter=int(nn['iterMax']),
                                                             random_state=42)),
                {'mlpregressor__alpha': np.logspace(-6, -1, 6)})
    else:
        raise ValueError('Invalid model for machine learning')


#######################################################################################################################
# Main Function
#######################################################################################################################
def mlFit(data, val, para, setup, legacy=0):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("------------------------------------------")
    print("START: Fitting Machine Learning model")
    print("------------------------------------------")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    sel = int(setup['selML'])
    X, T = _feat(data, setup, legacy), np.asarray(data['y'], dtype=float)[:, 0]
    Xvl, Tvl = _feat(val, setup, legacy), np.asarray(val['y'], dtype=float)[:, 0]
    [mdl, grid] = _model(sel, para['Mdl'])

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    if para['Mdl']['gen']['opt'] == 1:
        if grid:
            search = RandomizedSearchCV(mdl, grid, n_iter=20, cv=3, random_state=42, n_jobs=-1)
            search.fit(Xvl, Tvl)
            mdl.set_params(**search.best_params_)
            print("INFO: Optimised hyperparameters: " + str(search.best_params_))
        else:
            print("ERROR: Invalid model for optimisation")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    mdl.fit(X, T)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Fitting Machine Learning model")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return {'sys': mdl}


#######################################################################################################################
# References
#######################################################################################################################
