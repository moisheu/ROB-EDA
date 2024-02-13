from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import math

def SVR_train(config, X, y):
    if config.getboolean('SVR', 'bayesian_search'):
        search_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'epsilon': Real(1e-6, 1e+1, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform')
        }

        model = SVR()
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
        result = bayes_search.fit(X, y)
        best_model = bayes_search.best_estimator_
        return best_model
    else:
        model = SVR()
        model.fit(X, y)
        return model

def regression_performance(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return mse, mae, rmse

def shap_vis(model, X_train, y_str):
    # Note: SHAP with SVR requires using KernelExplainer, which can be computationally expensive and less accurate
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results\SVR results\{y_str}.png')
    plt.close()

def svr_complete(config, X, y, y_str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVR_train(config, X_train, y_train)
    y_pred = model.predict(X_test)
    mse, mae, rmse = regression_performance(y_test, y_pred)
    print(f'SUPPORT VECTOR REGRESSION: MSE: {mse}, MAE: {mae}, RMSE: {rmse}')
    
    #shap_vis(model, X_train, y_str)

    return model, mse, mae, rmse
