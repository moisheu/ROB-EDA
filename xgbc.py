import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
import shap 
import warnings 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Real,  Integer
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def XGBC_train(config, X, y, X_test, y_test, col = None):
    #If bayesian search is enabled in configs, search will start
    if config.getboolean('XGBClassifier','bayesian_search'):
        search_spaces = {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'min_child_weight': Integer(1, 6),
            'subsample': Real(0.7, 1.0),
            'gamma': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
        }

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

        model = xgb.XGBClassifier(objective = 'multi:softmax' ,  num_class=3, eval_set=[(X_test, y_test)] ,eval_metric='mlogloss', verbosity=0, early_stopping_rounds = 10)
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
            model = xgb.XGBClassifier(objective = 'multi:softmax' ,  num_class=2, eval_set=[(X_test, y_test)] ,eval_metric='mlogloss', verbosity=0, early_stopping_rounds = 10)
            model.fit(X,y,eval_set=[(X_test, y_test)], verbose = False)
        return model

def tree_performance_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def shap_vis(model, X_train, y_str):
    #Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    #Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_train, check_additivity=False)

    #Summary plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results\xgbclassifier_results\SHAP\{y_str}.png')
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
    csv_filename = fr'results\xgbclassifier_results\feature importance\feature_importance_csv\{identifier}_feature_importances.csv'
    feature_importances_df.to_csv(csv_filename, index=False)
    #print(f"Feature importances exported to {csv_filename}")

    # Plot feature importances
    plt.figure(figsize=(20, 15))
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plot_importance(best_model, max_num_features=15, importance_type='weight')
    plot_filename = fr'results\xgbclassifier_results\feature importance\feature_importance_vis\{identifier}_accuracy_visual.png'
    plt.savefig(plot_filename)
    plt.close()
    #print(f"Feature importance plot saved to {plot_filename}")

def xgbc_complete(config, X, y, y_str):
    #if config enables XGBRegressor, then start process of training 
    if config.getboolean('XGBClassifier', 'classifier'):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = XGBC_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      #Evaluate performance
      accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
      print(f'XGB: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
      #evalate feature importance
      export_and_plot_feature_importances(model, X, y_str)
      shap_vis(model, X_train, y_str)
      return model, accuracy, precision, recall, f1
    else:
       return None









