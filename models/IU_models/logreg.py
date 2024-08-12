from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import matplotlib.pyplot as plt
import numpy as np

def logreg_train(config, X, y, X_test, y_test, col=None):
    if config.getboolean('LogisticRegression', 'bayesian_search'):
        #define the space of hyperparameters to search
        search_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'penalty': Categorical(['l2', 'none'])
        }

        #initialize the logistic regression model
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        #initialize bayesian search
        bayes_search = BayesSearchCV(logreg, search_space, n_iter=32, cv=3, random_state=42)

        #perform the search
        bayes_search.fit(X, y)

        #best hyperparameters
        print("best parameters found: ", bayes_search.best_params_)
        return bayes_search.best_estimator_
    else:
        #create a logistic regression model
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        #fitting the model
        logreg.fit(X, y)

        #making predictions
        y_pred = logreg.predict(X_test)

        #evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        #print(f'accuracy: {accuracy}')
        return logreg

def tree_performance_classification(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_proba)
    return accuracy, precision, recall, balanced_acc, f1, auc_roc

def logreg_complete(config, train_df, test_df, y_str, fold_num, experiment_name):
    if config.getboolean('LogisticRegression', 'classifier'):
        y_train = train_df[f'{y_str}.encoded']
        X_train = train_df.drop(columns =[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[f'{y_str}.encoded']
        X_test = test_df.drop(columns = [f'{y_str}.encoded', y_str], axis=1)
        
        #train the model
        model = logreg_train(config, X_train, y_train, X_test, y_test, y_str)
        
        #make predictions and save as pred
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        #evaluate performance
        accuracy, precision, recall, balanced_acc, f1, auc_roc = tree_performance_classification(y_test, y_pred, y_proba)
        print(f'LOGREG: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')
        
        return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
        return None

def kfolds_logreg(k, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = logreg_complete(config, train_df, test_df, y_str, i, experiment_name)
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
        'Model': ['logreg'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })

    return df

def full_df_logreg(df, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    train_df, test_df = train_test_split(df, test_size=0.2)
    model, acc, prec, rec, bal_acc, f1, auc_roc = logreg_complete(config, train_df, test_df, y_str, 0, experiment_name)
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
        'Model': ['logreg'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df