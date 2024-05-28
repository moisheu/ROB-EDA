import xgboost as xgb
from xgboost import XGBRegressor as xgbr
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgbfir
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import os
import seaborn as sns
from sklearn.decomposition import PCA
import shap 
from skopt import BayesSearchCV 
from skopt.space import Real,  Integer

#Train the XGBRegressor if config settings are enabled 
def XGBR_train(config, X_train, y_train, X_test,y_test): 
  #If bayesian search is enabled in configs, search will start
  if config.getboolean('XGBR','bayesian_search'):
    search_spaces = {
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 10),
    'min_child_weight': Integer(1, 6),
    'subsample': Real(0.5, 1.0, 'uniform'),
    'colsample_bytree': Real(0.5, 1.0, 'uniform'),
    'gamma': Real(0.1, 5.0, 'uniform'),
    'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
    'reg_alpha': Real(1e-9, 1.0, 'log-uniform')
    }

    #Initialize the XGBRegressor
    xgb_regressor = xgbr(objective='reg:squarederror')

    #Setup the BayesSearchCV optimizer
    bayes_cv_tuner = BayesSearchCV(
        estimator=xgb_regressor,
        search_spaces=search_spaces,
        scoring='neg_mean_squared_error',
        cv=3,
        n_iter=30,
        n_jobs=-1,
        return_train_score=True,
        refit=True,
        random_state=42
        )
    #resolve integer conflict with bayes tuner 
    np.int = int
    #fit the model 
    bayes_cv_tuner.fit(X_train, y_train)
    best_model = bayes_cv_tuner.best_estimator_
    xgb.plot_importance(best_model)
    plt.close()
    return best_model 
  #if bayesian search is not toggled, then normal XGBR instance will be called 
  else:
    model = xgbr( n_estimators = 150, objective='reg:squarederror', verbose = False)
    model.fit(X_train,y_train,eval_set=[(X_test, y_test)], eval_metric='rmse')
    #visualize importances 
    xgb.plot_importance(model)
    plt.close()
    return model

def tree_performance(y_test, y_pred_df):
    RMSE = mean_squared_error(y_test, y_pred_df, squared=False)
    MSE = mean_squared_error(y_test, y_pred_df, squared=True)
    MAPE = mean_absolute_percentage_error(y_test, y_pred_df)
    return RMSE, MSE, MAPE

def get_xgb_importance(model, feature_str):
    try:
      importance_dict = model.get_score(importance_type='weight')  
      importance_df = pd.DataFrame(importance_dict.items(), columns=['Feature', 'Importance'])
      importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
      importance_df = importance_df.set_index('Feature').T
      importance_df.reset_index(drop=True, inplace=True)
      importance_df.to_csv(fr'results/XGB figures/feature importance/{feature_str}_feature_importance')
    except Exception as e:
      importance_dict = xgbr.feature_importances_

def aggregate_feature_importance(directory_path):
    aggregated_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        
        #check if the file is a CSV file
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
                
        aggregated_data = pd.concat([aggregated_data, df])

    aggregated_data = aggregated_data.fillna(0)
    
    aggregated_data.reset_index(inplace=True)
    
    return aggregated_data

def shap_viz(model, X_train, y_str):
      features = X_train.columns.tolist()
      #Calculate SHAP values
      try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_train)
      except Exception as e:
         explainer = shap.TreeExplainer(model)
         shap_values = explainer.shap_values(X_train)

        #Use your training data
      #Plot summary plot
      fig = shap.summary_plot(shap_values, X_train, feature_names=features,show=False)
      plt.title(f'{y_str} SHAP scores')
      plt.savefig(fr'results/XGB figures/SHAP/{y_str}.png')
      plt.close()

def xgb_complete(config, train_df, test_df, y_str):
    #if config enables XGBRegressor, then start process of training 
    if config.getboolean('XGBR', 'Regressor'):
      y_train = train_df[y_str]
      X_train = train_df.drop(columns =[f'{y_str}.encoded', y_str], axis=1)
      y_test = test_df[y_str]
      X_test = test_df.drop(columns = [f'{y_str}.encoded', y_str], axis=1)
      #Train the model 
      model = XGBR_train(config, X_train, y_train, X_test, y_test)
      #Make predictions and save as pred and test variables 
      y_pred = model.predict(X_test)

      #Evaluate performance
      RMSE, MSE, MAPE = tree_performance(y_test, y_pred)
      print(f'MSE: {MSE}, RMSE: {RMSE}, MAPE: {MAPE}')

      #Plot actual vs predicted values
      plt.figure(figsize=(8, 8))
      plt.scatter(y_test, y_pred, alpha=0.5, s = 5)
      plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')  #Line for perfect predictions
      plt.title('Actual vs Predicted Values')
      plt.xlabel('Actual Values')
      plt.ylabel('Predicted Values')
      plt.savefig(fr'results/XGB figures/accuracy vis/{y_str}_accuracy_visual.png')
      plt.close()
      get_xgb_importance(model, y_str)
      xgbfir.saveXgbFI(model, OutputXlsxFile=fr'results/XGB figures/xgbfir/{y_str}_xgbfir_results.xlsx')
      shap_viz(model, X_train, y_str)
      return model, RMSE, MSE, MAPE
    else:
       return None

def kfolds_xgbr(k, config, y_str):
    rmse_list = []
    mse_list = []
    mape_list = [] 
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, rmse, mse, mape = xgb_complete(config, train_df, test_df, y_str)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mape_list.append(mape)

    metrics = ['RMSE', 'MSE', 'MAPE']
    means = [np.mean(rmse_list), np.mean(mse_list), np.mean(mape_list)]
    stds = [np.std(rmse_list), np.std(mse_list), np.std(mape_list)]

    df = pd.DataFrame({
        'Name': [config['general']['exp_name']]* len(metrics),
        'Model': ['xgbr'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df
