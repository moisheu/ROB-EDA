from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import shap 
import matplotlib.pyplot as plt 
import numpy as np

def LGBM_train(config, X, y, X_test, y_test, col=None):
    if config.getboolean('LightGBM','bayesian_search'):
        search_space = {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(-1, 50),
            'learning_rate': Real(0.01, 0.3),
            'num_leaves': Integer(20, 300),
        }

        model = LGBMClassifier(random_state=42)
        np.int = int
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
        result = bayes_search.fit(X, y)
        best_model = bayes_search.best_estimator_
        return best_model
    else:
        model = LGBMClassifier(n_estimators=100, random_state=42)
        model = model.fit(X, y)
        return model 

def shap_vis(model, X_train, y_str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results\LGBM results\{y_str}.png')
    plt.close()

def tree_performance_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def lgbm_complete(config, X, y, y_str):
    if (config.getboolean('LightGBM', 'classifier')) and (config.getboolean('LightGBM', 'bayesian_search') == False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LGBM_train(config, X_train, y_train, X_test, y_test, y_str)
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
        print(f'LIGHTGBM BASE: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
        shap_vis(model, X_train, y_str)
    elif config.getboolean('LightGBM', 'classifier') and (config.getboolean('LightGBM', 'bayesian_search') == True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LGBM_train(config, X_train, y_train, X_test, y_test, y_str)
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
        print(f'LIGHTGBM HP: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
        shap_vis(model, X_train, y_str)
    else:
       return None
    return None
