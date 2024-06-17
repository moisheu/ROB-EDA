from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import shap 
import matplotlib.pyplot as plt 
import numpy as np


def RFC_train(config, X, y, X_test, y_test, col=None):
    #If bayesian search is enabled in configs, search will start
    if config.getboolean('RFC','bayesian_search'):
        search_space = {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 10),
        }

        model = RandomForestClassifier(random_state=42)
        np.int = int
        #Initialize Bayesian Search
        bayes_search = BayesSearchCV(model, search_space, n_iter=32, random_state=42, cv=3)
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
    #shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results\RF results\{y_str}.png')
    plt.close()

def tree_performance_classification(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_proba)
    return accuracy, precision, recall, balanced_acc, f1, auc_roc


def rfc_complete(config, train_df, test_df, y_str):
    if config.getboolean('RFC', 'classifier'):
        y_train = train_df[f'{y_str}.encoded']
        X_train = train_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[f'{y_str}.encoded']
        X_test = test_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        #Train the model 
        model = RFC_train(config, X_train, y_train, X_test, y_test, y_str)
        #Make predictions and save as pred
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        #Evaluate performance
        accuracy, precision, recall, balanced_acc, f1, auc_roc = tree_performance_classification(y_test, y_pred, y_proba)
        print(f'RFC: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')
       
        #Evaluate feature importance
        #shap_vis(model, X_train, y_str)
        return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
        return None, None, None, None, None, None, None

def kfolds_rfc(k, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = rfc_complete(config, train_df, test_df, y_str)
        if model is not None:
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            bal_acc_list.append(bal_acc)
            f1_list.append(f1)
            auc_roc_list.append(auc_roc)

    metrics = ['ACC', 'PREC', 'REC', 'BAL_ACC', 'F1', 'AUC_ROC']
    means = [np.mean(acc_list), np.mean(prec_list), np.mean(rec_list), np.mean(bal_acc_list), np.mean(f1_list), np.mean(auc_roc_list)]
    stds = [np.std(acc_list), np.std(prec_list), np.std(rec_list),  np.std(bal_acc_list), np.std(f1_list), np.std(auc_roc_list)]

    df = pd.DataFrame({
        'Name': [experiment_name] * len(metrics),
        'Model': ['rfc'] * len(metrics),
        'Target': [y_str] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df
