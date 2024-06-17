import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def NB_train(X, y):
    #initialize the Naive Bayes classifier
    model = GaussianNB()
    #fit the model
    model.fit(X, y)
    return model

def performance_classification(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_proba)
    return accuracy, precision, recall, balanced_acc, f1, auc_roc


def plot_confusion_matrix(y_test, y_pred, fold_num, experiment_name, class_names=['Class 0', 'Class 1']):
    #compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #print unique values to debug
    print("Unique values in y_test:", np.unique(y_test, return_counts=True))
    print("Unique values in y_pred:", np.unique(y_pred, return_counts=True))
    
    #print confusion matrix for debugging
    print("Confusion Matrix:\n", cm)
    
    #create the plot
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    #fix the annotations in the heatmap
    for text in ax.texts:
        text.set_size(12)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plot_filename = f'results/NB results/{experiment_name}_{fold_num}_cfm.png'
    plt.savefig(plot_filename)
    plt.close()

def nb_complete(config, train_df, test_df, y_str, fold_num, experiment_name):
    if config.getboolean('NaiveBayes', 'classifier'):
        y_train = train_df[f'{y_str}.encoded']
        X_train = train_df.drop(columns =[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[f'{y_str}.encoded']
        X_test = test_df.drop(columns = [f'{y_str}.encoded', y_str], axis=1)
        model = NB_train(X_train, y_train)
        #make predictions and save as pred
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        #evaluate performance
        accuracy, precision, recall, balanced_acc, f1, auc_roc = performance_classification(y_test, y_pred, y_proba)
        print(f'NAIVE BAYES: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')
        plot_confusion_matrix(y_test, y_pred, fold_num, experiment_name)
        return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
        return None

def kfolds_nb(k, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = nb_complete(config, train_df, test_df, y_str, i, experiment_name)
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
        'Model': ['naive_bayes'] * len(metrics),
        'Target': [y_str] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df
