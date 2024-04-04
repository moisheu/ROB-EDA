from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import shap 
import matplotlib.pyplot as plt 

def NB_train(X, y):
    #Initialize the Naive Bayes classifier
    model = GaussianNB()
    #Fitting the model
    model.fit(X, y)
    return model 

def shap_vis(model, X_train, y_str):
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    
    #Calculate SHAP values for the training set
    shap_values = explainer.shap_values(X_train)

    #Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f'{y_str} SHAP scores')
    plt.savefig(fr'results/NB results/{y_str}.png')
    plt.close()

def performance_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def nb_complete(config, X, y, y_str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Train the model 
    model = NB_train(X_train, y_train)
    #Make predictions and save as pred
    y_pred = model.predict(X_test)
    #Evaluate performance
    accuracy, precision, recall, f1 = performance_classification(y_test, y_pred)
    print(f'NAIVE BAYES: Acc: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}')
    #Evaluate feature importance
    #shap_vis(model, X_train, y_str)

    return model, accuracy, precision, recall, f1
