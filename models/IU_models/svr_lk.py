from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

def SVR_train(config, X_train, y_train, X_test, y_test):
    if config.getboolean('SVRLK', 'bayesian_search'):
        search_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'epsilon': Real(1e-6, 1e+1, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform')
        }

        model = SVR(kernel='linear')
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        return best_model
    else:
        model = SVR(kernel='linear')
        model.fit(X_train, y_train)
        return model

def regression_performance(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mse, mae, rmse, mape

def shap_viz(model, X_train, y_str):
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_train)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/SVR figures/SHAP/{y_str}.png')
    plt.close()

def svr_complete(config, train_df, test_df, y_str):
    if config.getboolean('SVRLK', 'Regressor'):
        y_train = train_df[y_str]
        X_train = train_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[y_str]
        X_test = test_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)

        model = SVR_train(config, X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)

        mse, mae, rmse, mape = regression_performance(y_test, y_pred)
        print(f'SUPPORT VECTOR REGRESSION: MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

        # plt.figure(figsize=(8, 8))
        # plt.scatter(y_test, y_pred, alpha=0.5, s=5)
        # plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')
        # plt.title('Actual vs Predicted Values')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.savefig(fr'results/SVR figures/accuracy vis/{y_str}_accuracy_visual.png')
        # plt.close()

        # shap_viz(model, X_train, y_str)
        
        return model, mse, mae, rmse, mape
    else:
        return None

def kfolds_svr_lk(k, config, y_str):
    rmse_list = []
    mse_list = []
    mape_list = []
    mae_list = []
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, mse, mae, rmse, mape = svr_complete(config, train_df, test_df, y_str)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mape_list.append(mape)
        mae_list.append(mae)

    metrics = ['RMSE', 'MSE', 'MAPE', 'MAE']
    means = [np.mean(rmse_list), np.mean(mse_list), np.mean(mape_list), np.mean(mae_list)]
    stds = [np.std(rmse_list), np.std(mse_list), np.std(mape_list), np.std(mae_list)]

    df = pd.DataFrame({
        'Name': [config['general']['exp_name']] * len(metrics),
        'Model': ['svr_lk'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df
