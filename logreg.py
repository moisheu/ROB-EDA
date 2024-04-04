from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import shap 
import matplotlib.pyplot as plt 

def logreg_train(config, X, y, X_test, y_test, col = None):
    if config.getboolean('LogisticRegression','bayesian_search'):
        #Define the space of hyperparameters to search
        search_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'penalty': Categorical(['l2', 'none'])
        }

        #Initialize the Logistic Regression model
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        #Initialize Bayesian Search
        bayes_search = BayesSearchCV(logreg, search_space, n_iter=32, cv=3, random_state=42)

        #Perform the search
        bayes_search.fit(X, y)

        #Best hyperparameters
        print("Best parameters found: ", bayes_search.best_params_)
        return bayes_search.best_model
    else: 
        #Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Creating a logistic regression model
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        #Fitting the model
        logreg.fit(X_train, y_train)

        #Making predictions
        y_pred = logreg.predict(X_test)

        #Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        #print(f'Accuracy: {accuracy}')
        return logreg
    
def shap_vis(model, X_train, y_str):
    #Create the SHAP Explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train)

    #Calculate SHAP values for the train set
    shap_values = explainer.shap_values(X_train)

    #Summary plot for each class (assuming multi-class classification)
    for i in range(len(shap_values)):
        plt.figure()
        shap.summary_plot(shap_values[i], X_train, show=False)
        plt.title(f'{y_str} Class {i} SHAP Scores')
        plt.savefig(f'results/LogReg results/SHAP/{y_str}_class_{i}.png')
        plt.close() 

def tree_performance_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def logreg_complete(config, X, y, y_str):
    if config.getboolean('LogisticRegression', 'classifier') and (config.getboolean('LogisticRegression', 'bayesian_search') == False):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = logreg_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      #Evaluate performance
      accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
      print(f'LOGREG BASE: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
      #evalate feature importance
      #export_and_plot_feature_importances(model, X, y_str)
      #shap_vis(model, X_train, y_str)
      #return model, accuracy, precision, recall, f1
    if config.getboolean('LogisticRegression', 'classifier') and (config.getboolean('LogisticRegression', 'bayesian_search') == True):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      #Train the model 
      model = logreg_train(config, X_train, y_train, X_test, y_test, y_str)
      #Make predictions and save as pred
      y_pred = model.predict(X_test)
      #Evaluate performance
      accuracy, precision, recall, f1 = tree_performance_classification(y_test, y_pred)
      print(f'LOGREG HP: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
      #evalate feature importance
      #export_and_plot_feature_importances(model, X, y_str)
      #shap_vis(model, X_train, y_str)
      #return model, accuracy, precision, recall, f1
    else:
       return None
    return None 