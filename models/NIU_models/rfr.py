from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import math
import utils
import os

def RFR_train(config, X_train, y_train, X_test, y_test):
    if config.getboolean('RFR', 'bayesian_search'):
        search_space = {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 10),
        }

        model = RandomForestRegressor(random_state=42)
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
        result = bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        return best_model
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

def get_rfr_importance(model, feature_str):
    try:
        importance_dict = model.feature_importances_
        importance_df = pd.DataFrame(importance_dict, index=X_train.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
        importance_df.to_csv(fr'results/RFR figures/feature importance/{feature_str}_feature_importance.csv')
    except Exception as e:
        print(f"Error in getting feature importance: {e}")

def aggregate_feature_importance(directory_path):
    aggregated_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        aggregated_data = pd.concat([aggregated_data, df])

    aggregated_data = aggregated_data.fillna(0)
    aggregated_data.reset_index(inplace=True)
    
    return aggregated_data

def shap_viz(model, X_train, y_str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    fig = shap.summary_plot(shap_values, X_train, feature_names=X_train.columns.tolist(), show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/RFR figures/SHAP/{y_str}.png')
    plt.close()

def rfr_complete(config, train_df, test_df, y_str):
    if config.getboolean('RFR', 'Regressor'):
        y_train = train_df[y_str]
        X_train = train_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[y_str]
        X_test = test_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)

        model = RFR_train(config, X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)

        RMSE, MSE, MAPE, MAE = utils.tree_performance(y_test, y_pred)
        print(f'MSE: {MSE}, RMSE: {RMSE}, MAPE: {MAPE}, MAE: {MAE}')

        # plt.figure(figsize=(8, 8))
        # plt.scatter(y_test, y_pred, alpha=0.5, s=5)
        # plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')
        # plt.title('Actual vs Predicted Values')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.savefig(fr'results/RFR results/accuracy vis/{y_str}_accuracy_visual.png')
        # plt.close()

        # get_rfr_importance(model, y_str)
        # shap_viz(model, X_train, y_str)
        
        return model, RMSE, MSE, MAPE, MAE
    else:
        return None

def kfolds_rfr(k, config, y_str):
    rmse_list = []
    mse_list = []
    mape_list = [] 
    mae_list = []
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, rmse, mse, mape, mae = rfr_complete(config, train_df, test_df, y_str)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mape_list.append(mape)
        mae_list.append(mae)

    metrics = ['RMSE', 'MSE', 'MAPE', 'MAE']
    means = [np.mean(rmse_list), np.mean(mse_list), np.mean(mape_list), np.mean(mae_list)]
    stds = [np.std(rmse_list), np.std(mse_list), np.std(mape_list), np.std(mae_list)]

    df = pd.DataFrame({
        'Name': [config['general']['exp_name']] * len(metrics),
        'Model': ['rfr'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df
