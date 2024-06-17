from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import seaborn as sns
from sklearn.decomposition import PCA
from skopt import BayesSearchCV 
from skopt.space import Real, Integer

# Train the LinearRegression model
def LinearRegression_train(config, X_train, y_train, X_test, y_test): 
    # If bayesian search is enabled in configs, search will start
    if config.getboolean('LINREG','bayesian_search'):
        search_spaces = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }

        # Initialize the LinearRegression model
        lin_reg = LinearRegression()

        # Setup the BayesSearchCV optimizer
        bayes_cv_tuner = BayesSearchCV(
            estimator=lin_reg,
            search_spaces=search_spaces,
            scoring='neg_mean_squared_error',
            cv=3,
            n_iter=10,
            n_jobs=-1,
            return_train_score=True,
            refit=True,
            random_state=42
        )

        # Fit the model 
        bayes_cv_tuner.fit(X_train, y_train)
        best_model = bayes_cv_tuner.best_estimator_
        return best_model 
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

def tree_performance(y_test, y_pred_df):
    RMSE = mean_squared_error(y_test, y_pred_df, squared=False)
    MSE = mean_squared_error(y_test, y_pred_df, squared=True)
    MAPE = mean_absolute_percentage_error(y_test, y_pred_df)
    return RMSE, MSE, MAPE

def get_linear_regression_importance(model, X_train, feature_str):
    try:
        importance_df = pd.DataFrame(model.coef_, X_train.columns, columns=['Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index()
        importance_df.rename(columns={'index': 'Feature'}, inplace=True)
        importance_df.to_csv(fr'results/LinearRegression figures/feature importance/{feature_str}_feature_importance')
    except Exception as e:
        print(e)

def aggregate_feature_importance(directory_path):
    aggregated_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        # Check if the file is a CSV file
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        aggregated_data = pd.concat([aggregated_data, df])

    aggregated_data = aggregated_data.fillna(0)
    aggregated_data.reset_index(inplace=True)
    
    return aggregated_data

def shap_viz(model, X_train, y_str):
    features = X_train.columns.tolist()
    # Since SHAP is primarily designed for tree models, it's more complex for linear models.
    # Plotting the coefficients as feature importance
    importance_df = pd.DataFrame(model.coef_, features, columns=['Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    importance_df.plot(kind='barh', figsize=(10, 6))
    plt.title(f'{y_str} Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.savefig(fr'results/LinearRegression figures/SHAP/{y_str}.png')
    plt.close()

def linear_regression_complete(config, train_df, test_df, y_str):
    # If config enables LinearRegression, then start process of training 
    if config.getboolean('LINREG', 'Regressor'):
        y_train = train_df[y_str]
        X_train = train_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[y_str]
        X_test = test_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        
        # Train the model 
        model = LinearRegression_train(config, X_train, y_train, X_test, y_test)
        
        # Make predictions and save as pred and test variables 
        y_pred = model.predict(X_test)

        # Evaluate performance
        RMSE, MSE, MAPE = tree_performance(y_test, y_pred)
        print(f'MSE: {MSE}, RMSE: {RMSE}, MAPE: {MAPE}')

        # # Plot actual vs predicted values
        # plt.figure(figsize=(8, 8))
        # plt.scatter(y_test, y_pred, alpha=0.5, s=5)
        # plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')  # Line for perfect predictions
        # plt.title('Actual vs Predicted Values')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.savefig(fr'results/LinearRegression figures/accuracy vis/{y_str}_accuracy_visual.png')
        # plt.close()
        
        # get_linear_regression_importance(model, X_train, y_str)
        # shap_viz(model, X_train, y_str)
        return model, RMSE, MSE, MAPE
    else:
        return None

def kfolds_linear_regression(k, config, y_str):
    rmse_list = []
    mse_list = []
    mape_list = []
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, rmse, mse, mape = linear_regression_complete(config, train_df, test_df, y_str)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mape_list.append(mape)

    metrics = ['RMSE', 'MSE', 'MAPE']
    means = [np.mean(rmse_list), np.mean(mse_list), np.mean(mape_list)]
    stds = [np.std(rmse_list), np.std(mse_list), np.std(mape_list)]

    df = pd.DataFrame({
        'Name': [config['general']['exp_name']]*len(metrics),
        'Model': ['LINREG']*len(metrics),
        'Target': [y_str]*len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df
