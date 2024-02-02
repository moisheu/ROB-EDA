from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import shap 
import matplotlib.pyplot as plt 
import numpy as np


def RFC_train(config, X, y, X_test, y_test, col = None):
   #If bayesian search is enabled in configs, search will start
    if config.getboolean('XGBClassifier','bayesian_search'):
        search_space = {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 10),
        }

        model = RandomForestClassifier(random_state=42)
        np.int = int
        #Initialize Bayesian Search
        bayes_search = BayesSearchCV(rf, search_space, n_iter=32, random_state=42, cv=3)
        #Perform the search on the training data
        result = bayes_search.fit(X, y)
        best_score = result.best_score_
        #best_params = result.best_params_
        best_model = bayes_search.best_estimator_
        #predictions = best_model.predict(X_test)
        #score = best_model.score(X_test, y_test)
        return best_model
    else:
        #Creating a RandomForest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        #Fitting the model
        model  = rf.fit(X, y)
        return model 

def shap_vis(model, X_train, y_str):
    #Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    #Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_train)

    #Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results\RF results\{y_str}.png')
    plt.close()

def tree_performance_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def rfc_complete(config, X, y, y_str):
    if (config.getboolean('RandomForest', 'classifier')) and (config.getboolean('RandomForest', 'bayesian_search') == False):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = RFC_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      #Evaluate performance
      accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
      print(f'RANDOMFOREST BASE: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
      #evalate feature importance
      #export_and_plot_feature_importances(model, X, y_str)
      shap_vis(model, X_train, y_str)
      #return model, accuracy, precision, recall, f1
    if config.getboolean('RandomForest', 'classifier') and (config.getboolean('RandomForest', 'bayesian_search') == True):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = RFC_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      #Evaluate performance
      accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
      print(f'RANDOMFOREST HP: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
      #evalate feature importance
      #export_and_plot_feature_importances(model, X, y_str)
      shap_vis(model, X_train, y_str)
      #return model, accuracy, precision, recall, f1
    else:
       return None
    return None 
