import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import shap 
import warnings 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.filterwarnings("ignore")

def DTC_train(config, X, y, X_test, y_test, col=None):
    # If Bayesian search is enabled in configs, search will start
    if config.getboolean('DTC','bayesian_search'):
        search_spaces = {
            'max_depth': Integer(1, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 20),
            'max_features': ['auto', 'sqrt', 'log2', None]
        }

        bayes_search = BayesSearchCV(
            estimator=DecisionTreeClassifier(),
            search_spaces=search_spaces,
            scoring='accuracy',
            n_iter=32,
            cv=3,
            n_jobs=-1,
            return_train_score=True,
            refit=True
        )

        result = bayes_search.fit(X, y)
        best_score = result.best_score_
        best_model = bayes_search.best_estimator_
        predictions = best_model.predict(X_test)
        score = best_model.score(X_test, y_test)
        return best_model
    else: 
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model

def tree_performance_classification(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(y_test, y_proba)
    return accuracy, precision, recall, balanced_acc, f1, auc_roc

def shap_vis(model, X_train, y_str):
    # Create the SHAP Explainer
    explainer = shap.Explainer(model)
    # Calculate SHAP values for the test set
    shap_values = explainer(X_train)
    # Summary plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/DTC results/SHAP/{y_str}.png')
    plt.close()

def export_and_plot_feature_importances(best_model, X, y):
    # Determine the identifier from y, if available
    if y is not None:
        try:
            identifier = y
        except Exception as e:
            print(f"Error occurred while extracting identifier: {e}")

    # Calculate feature importances
    feature_importances = best_model.feature_importances_

    # Create a DataFrame with the feature names and their importance scores
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Export to CSV
    csv_filename = fr'results/DTC results/{identifier}_feature_importances.csv'
    feature_importances_df.to_csv(csv_filename, index=False)
    # print(f"Feature importances exported to {csv_filename}")

    # Plot feature importances
    plt.figure(figsize=(20, 15))
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plot_tree(best_model, max_depth=3, fontsize=10, filled=True, feature_names=X.columns)
    plot_filename = fr'results/DTC results/{identifier}_accuracy_visual.png'
    plt.savefig(plot_filename)
    plt.close()
    # print(f"Feature importance plot saved to {plot_filename}")

def plot_confusion_matrix(y_test, y_pred, fold_num, experiment_name, class_names=['Class 0', 'Class 1']):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print unique values to debug
    print("Unique values in y_test:", np.unique(y_test, return_counts=True))
    print("Unique values in y_pred:", np.unique(y_pred, return_counts=True))
    
    # Print confusion matrix for debugging
    print("Confusion Matrix:\n", cm)
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Fix the annotations in the heatmap
    for text in ax.texts:
        text.set_size(12)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plot_filename = f'results/DTC results/{experiment_name}_{fold_num}_cfm.png'
    plt.savefig(plot_filename)
    plt.close()

def dtc_complete(config, train_df, test_df, y_str, fold_num, experiment_name):
    if config.getboolean('DTC', 'classifier'):
        y_train = train_df[f'{y_str}.encoded']
        X_train = train_df.drop(columns =[f'{y_str}.encoded', y_str], axis=1)
        y_test = test_df[f'{y_str}.encoded']
        X_test = test_df.drop(columns = [f'{y_str}.encoded', y_str], axis=1)
        model = DTC_train(config, X_train, y_train, X_test, y_test, y_str)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy, precision, recall, balanced_acc, f1, auc_roc = tree_performance_classification(y_test, y_pred, y_proba)
        print(f'DTC: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')
        #export_and_plot_feature_importances(model, X_train, y_str)
        # shap_vis(model, X_train, y_str)
        plot_confusion_matrix(y_test, y_pred, fold_num, experiment_name)
        return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
       return None

def kfolds_dtc(k, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = dtc_complete(config, train_df, test_df, y_str, i, experiment_name)
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
        'Model': ['dtc'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df

def full_df_dtc(df, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    train_df, test_df = train_test_split(df, test_size=0.2)
    model, acc, prec, rec, bal_acc, f1, auc_roc = dtc_complete(config, train_df, test_df, y_str, 0, experiment_name)
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
        'Model': ['dtc'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df