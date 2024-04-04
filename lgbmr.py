from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import math

def LGBM_train(config, X, y):
    if config.getboolean('LGBMR', 'bayesian_search'):
        search_space = {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(-1, 50),
            'learning_rate': Real(0.01, 0.3),
            'num_leaves': Integer(20, 300),
        }

        model = LGBMRegressor(random_state=42)
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
        result = bayes_search.fit(X, y)
        best_model = bayes_search.best_estimator_
        return best_model
    else:
        model = LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

def regression_performance(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return mse, mae, rmse

def shap_vis(model, X_train, y_str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/LGBMR results/{y_str}.png')
    plt.close()

def lgbmr_complete(config, X, y, y_str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBM_train(config, X_train, y_train)
    y_pred = model.predict(X_test)
    mse, mae, rmse = regression_performance(y_test, y_pred)
    print(f'LIGHTGBM REGRESSION: MSE: {mse}, MAE: {mae}, RMSE: {rmse}')
    shap_vis(model, X_train, y_str)
    return model, mse, mae, rmse
