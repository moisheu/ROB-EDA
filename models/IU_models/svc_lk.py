from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import GridSearchCV
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

def SVC_train(config, X_train, y_train, X_test, y_test):
    if config.getboolean('SVCRK', 'bayesian_search'):
        param_grid = {
            'C': [1e-6, 1e-4, 1e-2, 1, 1e+2, 1e+3],
            'gamma': [1e-6, 1e-4, 1e-2, 0.1, 1, 10]
        }

        model = SVC(kernel='linear', probability=True)
        grid_search = GridSearchCV(model, param_grid, cv=5, verbose=3)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model
    else:
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        return model

def classification_performance(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_proba)
    return accuracy, precision, recall, balanced_acc, f1, auc_roc

def shap_viz(model, X_train, y_str):
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_train)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/SVCLK figures/SHAP/{y_str}.png')
    plt.close()

def svc_complete(config, train_df, test_df, y_str):
    if config.getboolean('SVCLK', 'Classifier'):
        y_train = train_df[f'{y_str}.encoded']
        X_train = train_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[f'{y_str}.encoded']
        X_test = test_df.drop(columns=[f'{y_str}.encoded', y_str], axis=1)

        model = SVC_train(config, X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy, precision, recall, balanced_acc, f1, auc_roc = classification_performance(y_test, y_pred, y_proba)
        print(f'SVC LINEAR: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')

        # plt.figure(figsize=(8, 8))
        # plt.scatter(y_test, y_pred, alpha=0.5, s=5)
        # plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red')
        # plt.title('Actual vs Predicted Values')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.savefig(fr'results/SVCLK figures/accuracy vis/{y_str}_accuracy_visual.png')
        # plt.close()

        # shap_viz(model, X_train, y_str)
        
        return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
        return None

def kfolds_svc_lk(k, config, y_str):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = svc_complete(config, train_df, test_df, y_str)
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
        'Name': [config['general']['exp_name']] * len(metrics),
        'Model': ['svc_lk'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df

def full_df_svclk(df, config, y_str):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    train_df, test_df = train_test_split(df, test_size=0.2)
    model, acc, prec, rec, bal_acc, f1, auc_roc = svc_complete(config, train_df, test_df, y_str)
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
        'Name': [config['general']['exp_name']]* len(metrics),
        'Model': ['svc_lk'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df
