import utils
import pandas as pd
from sklearn.svm import SVR
import ast
import warnings 
import xgbr
import xgbc
import rf
import logreg
import lgbm
import nb
import mlr 
import svr
import rfr 
import lgbmr

warnings.filterwarnings("ignore")

#Setting up configuration settings 
config = utils.config_setup()
df, ns_df = utils.load_dataframe(config)
#apply binary encoding 
#merged_df = utils.merge_dataframes_for_encoding(df,ns_df,config)
#apply multiclass encoding 
df = utils.apply_encoding(config , df)
#drop necessary columns 
utils.drop_columns(df, config)

trust_list = ast.literal_eval(config['general']['trust_list'])
xgb_evaluation_df = pd.DataFrame(columns = ['Component','RMSE','MSE','MAE'])
for trust in trust_list:
    print(f'BEGINNING TRAINING FOR {trust} -------------------------------------------------------------------------------------------')
    df = pd.read_csv(fr'data\post_processing\driver\{trust}_driver.csv')

    #Start Kprototypes --> Not using kprototypes for now, commented out
    #kprototypes.kprototypes(df)
    #Retrieve target variable and training data 

    #REGRESSORS----------------------------------------------
    if config.getboolean('XGB', 'regressor'):
        y = df[f'{trust}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XBG Regressor 
        xgbr.xgb_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('MLR', 'regressor'):
        y = df[f'{trust}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XBG Regressor 
        mlr.mlr_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('SVR', 'regressor'):
        y = df[f'{trust}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XBG Regressor 
        svr.svr_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('RFR', 'regressor'):
        y = df[f'{trust}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XBG Regressor 
        rfr.rfr_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('LGBMR', 'regressor'):
        y = df[f'{trust}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XBG Regressor 
        lgbmr.lgbmr_complete(config, X, y, trust)
    else:
        pass
    #CLASSIFIERS----------------------------------------------
    if config.getboolean('XGBClassifier', 'classifier'): 
        #temporary fix--- 
        y = df[f'{trust}_encoded']
        if 'Composite.Trust.Human_encoded' == f'{trust}_encoded':
            y =  y.replace(2, 1)
        else:
            pass
        #Drop encoded and nonencoded version of the target variable
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        if 'Composite.Trust.Human_encoded' in X.columns.tolist():
            X['Composite.Trust.Human_encoded'] =  X['Composite.Trust.Human_encoded'].replace(2, 1)
        else:
            pass
        #XGB Classifier
        xgbc.xgbc_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('RandomForest', 'classifier'): 
        y = df[f'{trust}_encoded']
        #Drop encoded and nonencoded version of the target variable
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XGB Classifier
        rf.rfc_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('LogisticRegression', 'classifier'): 
        y = df[f'{trust}_encoded']
        #Drop encoded and nonencoded version of the target variable
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XGB Classifier
        logreg.logreg_complete(config, X, y, trust)
    else:
        pass
    
    if config.getboolean('LightGBM', 'classifier'): 
        y = df[f'{trust}_encoded']
        #Drop encoded and nonencoded version of the target variable
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XGB Classifier
        lgbm.lgbm_complete(config, X, y, trust)
    else:
        pass

    if config.getboolean('NativeBayes', 'classifier'): 
        y = df[f'{trust}_encoded']
        #Drop encoded and nonencoded version of the target variable
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
        #XGB Classifier
        nb.nb_complete(config, X, y, trust)
    else:
        pass
