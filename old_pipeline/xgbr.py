<<<<<<< Updated upstream:xgbr.py
import xgboost as xgb
from xgboost import XGBRegressor as xgbr
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

#Convert data into dmatrix format for XGB 
def convert_to_dmatrix(df, y_str):
    y = df[y_str]
    X= df.drop(y_str, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_test, label = y_test)
    return dtrain, dtest, X_train

#Train the XGBRegressor if config settings are enabled 
def XGBR_train(config, X_train, y_train, X_test,y_test): 
  #If bayesian search is enabled in configs, search will start
  if config.getboolean('XGB','bayesian_search'):
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
  MAE = mean_absolute_error(y_test, y_pred_df)
  return RMSE, MSE, MAE

def get_xgb_importance(model, feature_str):
    try:
      importance_dict = model.get_score(importance_type='weight')  
      importance_df = pd.DataFrame(importance_dict.items(), columns=['Feature', 'Importance'])
      importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
      importance_df = importance_df.set_index('Feature').T
      importance_df.reset_index(drop=True, inplace=True)
      importance_df.to_csv(fr'results\XGB figures\feature importance\{feature_str}_feature_importance')
    except Exception as e:
      importance_dict = xgbr.feature_importances_

def xgb_complete(config, X, y, y_str):
    #if config enables XGBRegressor, then start process of training 
    if config.getboolean('XGB', 'Regressor'):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = XGBR_train(config, X_train, y_train, X_test, y_test)
      #Make predictions and save as pred and test variables 
      y_pred = model.predict(X_test)

      #Evaluate performance
      RMSE, MSE, MAE = tree_performance(y_test, y_pred)
      print(f'MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}')

      #Plot actual vs predicted values
      plt.figure(figsize=(8, 8))
      plt.scatter(y_test, y_pred, alpha=0.5, s = 5)
      plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')  #Line for perfect predictions
      plt.title('Actual vs Predicted Values')
      plt.xlabel('Actual Values')
      plt.ylabel('Predicted Values')
      plt.savefig(fr'results\XGB figures\accuracy vis\{y_str}_accuracy_visual.png')
      plt.close()
      get_xgb_importance(model, y_str)
      xgbfir.saveXgbFI(model, OutputXlsxFile=fr'C:\Users\takab\Desktop\ROB EDA\results\XGB figures\xgbfir\{y_str}_xgbfir_results.xlsx')
      shap_viz(model, X_train, y_str)
      return model, RMSE, MSE, MAE
    else:
       return None

def aggregate_feature_importance(directory_path):
    aggregated_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        
        #Check if the file is a CSV file
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
      plt.savefig(fr'results\XGB figures\SHAP\{y_str}.png')
      plt.close()


def perform_kmeans_and_visualize(data, max_clusters=10):
    #Normalize the data --> Data is already z-score normalized
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data.drop(columns = ['index','Unnamed: 0'])) 
    #Determine the optimal number of clusters using the elbow method
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    #Plot SSE for each k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
    plt.close()


    #Prompt the user to input the optimal number of clusters
    optimal_clusters = int(input("Enter the optimal number of clusters: "))

    #Run K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    #Add cluster labels to the data
    data['cluster'] = cluster_labels

    #Perform PCA for visualization purposes
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    data['pca1'] = principal_components[:, 0]
    data['pca2'] = principal_components[:, 1]

    #Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis', legend='full', alpha=0.8)
    plt.title('Cluster Visualization with PCA')
    plt.show()
    plt.savefig(fr'results\XGB figures\kmeans.png')

    return data

=======
import xgboost as xgb
from xgboost import XGBRegressor as xgbr
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

#Convert data into dmatrix format for XGB 
def convert_to_dmatrix(df, y_str):
    y = df[y_str]
    X= df.drop(y_str, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_test, label = y_test)
    return dtrain, dtest, X_train

#Train the XGBRegressor if config settings are enabled 
def XGBR_train(config, X_train, y_train, X_test,y_test): 
  #If bayesian search is enabled in configs, search will start
  if config.getboolean('XGB','bayesian_search'):
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
  MAE = mean_absolute_error(y_test, y_pred_df)
  return RMSE, MSE, MAE

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

def xgb_complete(config, X, y, y_str):
    #if config enables XGBRegressor, then start process of training 
    if config.getboolean('XGB', 'Regressor'):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = XGBR_train(config, X_train, y_train, X_test, y_test)
      #Make predictions and save as pred and test variables 
      y_pred = model.predict(X_test)

      #Evaluate performance
      RMSE, MSE, MAE = tree_performance(y_test, y_pred)
      print(f'MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}')

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
      return model, RMSE, MSE, MAE
    else:
       return None

def aggregate_feature_importance(directory_path):
    aggregated_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        
        #Check if the file is a CSV file
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


def perform_kmeans_and_visualize(data, max_clusters=10):
    #Normalize the data --> Data is already z-score normalized
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data.drop(columns = ['index','Unnamed: 0'])) 
    #Determine the optimal number of clusters using the elbow method
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    #Plot SSE for each k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
    plt.close()


    #Prompt the user to input the optimal number of clusters
    optimal_clusters = int(input("Enter the optimal number of clusters: "))

    #Run K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    #Add cluster labels to the data
    data['cluster'] = cluster_labels

    #Perform PCA for visualization purposes
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    data['pca1'] = principal_components[:, 0]
    data['pca2'] = principal_components[:, 1]

    #Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis', legend='full', alpha=0.8)
    plt.title('Cluster Visualization with PCA')
    plt.show()
    plt.savefig(fr'results/XGB figures/kmeans.png')

    return data

>>>>>>> Stashed changes:old_pipeline/xgbr.py
