import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
import shap 
import warnings 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real,  Integer
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

warnings.filterwarnings("ignore")

def XGBC_train(config, X, y, X_test, y_test, col = None):
    #If bayesian search is enabled in configs, search will start
    if config.getboolean('XGBClassifier','bayesian_search'):
        search_spaces = {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'max_depth': Integer(3, 10),
            #'min_child_weight': Integer(1, 6),
            #'subsample': Real(0.7, 1.0),
            #'gamma': Real(0.5, 1.0),
            #'colsample_bytree': Real(0.5, 1.0),
        }

        

        model = xgb.XGBClassifier(objective = 'multi:softmax' ,  num_class=3, eval_set=[(X_test, y_test)] ,eval_metric='mlogloss', verbosity=0, early_stopping_rounds = 10)
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            scoring='accuracy',
            n_iter=32,
            cv=3,
            n_jobs=-1,
            return_train_score=True,
            refit=True
        )
        np.int = int
        result = bayes_search.fit(X, y, eval_set=[(X_test, y_test)], verbose = False)
        best_score = result.best_score_
        #best_params = result.best_params_
        best_model = bayes_search.best_estimator_
        predictions = best_model.predict(X_test)
        score = best_model.score(X_test, y_test)
        return best_model
    else: 
        try: 
           model = xgb.XGBClassifier(objective = 'multi:softmax' ,  num_class=3, eval_set=[(X_test, y_test)] ,eval_metric='mlogloss', verbosity=0, early_stopping_rounds = 10)
           model.fit(X,y,eval_set=[(X_test, y_test)], verbose = False)
        except ValueError as e:
            model = xgb.XGBClassifier(objective = 'binary:logistic' ,eval_metric='mlogloss', verbosity=0)
            model.fit(X,y,verbose = False)
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
    #Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    #Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_train)
    #Summary plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/XGBC results/SHAP/{y_str}.png')
    plt.close()

def export_and_plot_feature_importances(best_model, X, y):
    #Determine the identifier from y, if available
    if y is not None:
        try:
            identifier = y
        except Exception as e:
            print(f"Error occurred while extracting identifier: {e}")

    #Calculate feature importances
    feature_importances = best_model.feature_importances_

    #Create a DataFrame with the feature names and their importance scores
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    #Export to CSV
    csv_filename = fr'results/XGBC results/feature importance/feature_importance_csv/{identifier}_feature_importances.csv'
    feature_importances_df.to_csv(csv_filename, index=False)
    #print(f"Feature importances exported to {csv_filename}")

    # Plot feature importances
    plt.figure(figsize=(20, 15))
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plot_importance(best_model, max_num_features=15, importance_type='weight')
    plot_filename = fr'results/XGBC results/feature importance/feature_importance_vis/{identifier}_accuracy_visual.png'
    plt.savefig(plot_filename)
    plt.close()
    #print(f"Feature importance plot saved to {plot_filename}")

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
    plot_filename = f'results/XGBC results/{experiment_name}_{fold_num}_cfm.png'
    plt.savefig(plot_filename)
    plt.close()


def xgbc_complete(config, train_df, test_df, y_str, fold_num, experiment_name):
    #if config enables XGBRegressor, then start process of training 
    if config.getboolean('XGBClassifier', 'classifier'):      #Train the model 
      y_train = train_df[f'{y_str}.encoded']
      X_train = train_df.drop(columns =[f'{y_str}.encoded', y_str], axis=1)
      y_test = test_df[f'{y_str}.encoded']
      X_test = test_df.drop(columns = [f'{y_str}.encoded', y_str], axis=1)
      model = XGBC_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      y_proba = model.predict_proba(X_test)[:, 1]
      #Evaluate performance
      accuracy, precision, recall, balanced_acc, f1, auc_roc = tree_performance_classification(y_test, y_pred, y_proba)
      print(f'XGB: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, BALANCED ACC: {balanced_acc}, F1: {f1} , AUC ROC: {auc_roc}')
      #evalate feature importance
      #export_and_plot_feature_importances(model, X_train, y_str)
      #shap_vis(model, X_train, y_str)
      plot_confusion_matrix(y_test, y_pred, fold_num, experiment_name)
      return model, accuracy, precision, recall, balanced_acc, f1, auc_roc
    else:
       return None

def kfolds_xgbc(k, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        model, acc, prec, rec, bal_acc, f1, auc_roc = xgbc_complete(config, train_df, test_df, y_str, i, experiment_name)
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
        'Model': ['xgbc'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df


def full_df_xgbc(df, config, y_str, experiment_name):
    acc_list = []
    prec_list = []
    rec_list = [] 
    bal_acc_list = []
    f1_list = []
    auc_roc_list = []

    train_df, test_df = train_test_split(df, test_size=0.2)
    model, acc, prec, rec, bal_acc, f1, auc_roc = xgbc_complete(config, train_df, test_df, y_str, 0, experiment_name)
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
        'Model': ['xgbc'] * len(metrics),
        'Target': ['Composite.Trust.Narrow.Combined'] * len(metrics),
        'Metric': metrics,
        'Mean': means,
        'Std': stds
    })
    
    return df









