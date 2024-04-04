import pandas as pd
import numpy as np 
import utils 
import os 
import ast 
import xgbr
import rfr
import xgbc
import lgbmr 
import svr 
import logreg 
import lgbm 
import logreg


def segmentation(config, list_title):
    directory = os.fsencode('data\post_processing\driver')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            name = filename.replace('_driver.csv', '')
            df = pd.read_csv(f'data\post_processing\driver\{filename}')
            col_list = ast.literal_eval(config['segmentation'][list_title])
            if name not in col_list:
                col_list.append(name)
            new_df = df[col_list]
            new_df = utils.apply_encoding(config , new_df, name)
            #Store new CSV 
            new_df.to_csv(fr'data\experimental_segmentation\top_five\{name}_topfive.csv')

#Runs specific best model with segments 
def best_model_runner(config):
    #Boolean check , if segmentation is not toggled in config, this function will not run 
    if config.getboolean('segmentation','segmentation_training'):
        pass
    else: 
        raise Exception('Segmentation training not toggled. Please toggle in the config.ini file!')

    #Retrieve the trust, segment list, and segmentation model dictionary 
    trust_list = ast.literal_eval(config['general']['trust_list'])
    segment_list = ast.literal_eval(config['segmentation']['segment_list'])
    segment_model_dict = ast.literal_eval(config['segmentation']['segmentation_model_dictionary']) 
    #For evey segment in the segments chosen to be trained on   
    for segment in segment_list:
        for model in segment_model_dict:
            print(f'BEGINNING SEGMENTED FOR MODEL: {model.key()}')
            segment_model_list = model.values()
            for trust in segment_model_list:
                print(f'BEGINNING SEGMENTED TRAINING FOR {trust} -------------------------------------------------------------------------------------------')
                df = pd.read_csv(fr'data\experimental_segmentation\top_five\{trust}_{segment}.csv')
                #REGRESSORS----------------------------------------------
                if config.getboolean('XGB', 'regressor'):
                    y = df[f'{trust}']
                    #Drop encoded and nonencoded version of the target variable 
                    X= df.drop(columns =[f'{trust}.encoded', trust], axis=1)
                    #XBG Regressor 
                    xgbr.xgb_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('MLR', 'regressor'):
                    y = df[f'{trust}']
                    #Drop encoded and nonencoded version of the target variable 
                    X= df.drop(columns =[f'{trust}.encoded', trust], axis=1)
                    #XBG Regressor 
                    mlr.mlr_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('SVR', 'regressor'):
                    y = df[f'{trust}']
                    #Drop encoded and nonencoded version of the target variable 
                    X= df.drop(columns =[f'{trust}.encoded', trust], axis=1)
                    #XBG Regressor 
                    svr.svr_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('RFR', 'regressor'):
                    y = df[f'{trust}']
                    #Drop encoded and nonencoded version of the target variable 
                    X= df.drop(columns =[f'{trust}.encoded', trust], axis=1)
                    #XBG Regressor 
                    rfr.rfr_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('LGBMR', 'regressor'):
                    y = df[f'{trust}']
                    #Drop encoded and nonencoded version of the target variable 
                    X= df.drop(columns =[f'{trust}.encoded', trust], axis=1)
                    #XBG Regressor 
                    lgbmr.lgbmr_complete(config, X, y, trust)
                else:
                    pass
                #CLASSIFIERS----------------------------------------------
                if config.getboolean('XGBClassifier', 'classifier'): 
                    #temporary fix--- 
                    y = df[f'{trust}.encoded']
                    if 'Composite.Trust.Human.encoded' == f'{trust}.encoded':
                        y =  y.replace(2, 1)
                    else:
                        pass
                    #Drop encoded and nonencoded version of the target variable
                    X = utils.get_X(config, df, trust)
                    if 'Composite.Trust.Human.encoded' in X.columns.tolist():
                        X['Composite.Trust.Human.encoded'] =  X['Composite.Trust.Human.encoded'].replace(2, 1)
                    else:
                        pass
                    #XGB Classifier
                    xgbc.xgbc_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('RandomForest', 'classifier'): 
                    y = df[f'{trust}.encoded']
                    #Drop encoded and nonencoded version of the target variable
                    X = utils.get_X(config, df, trust)
                    #XGB Classifier
                    rf.rfc_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('LogisticRegression', 'classifier'): 
                    y = df[f'{trust}.encoded']
                    #Drop encoded and nonencoded version of the target variable
                    X = utils.get_X(config, df, trust)
                    #XGB Classifier
                    logreg.logreg_complete(config, X, y, trust)
                else:
                    pass
                
                if config.getboolean('LightGBM', 'classifier'): 
                    y = df[f'{trust}.encoded']
                    #Drop encoded and nonencoded version of the target variable
                    X = utils.get_X(config, df, trust)
                    #XGB Classifier
                    lgbm.lgbm_complete(config, X, y, trust)
                else:
                    pass

                if config.getboolean('NativeBayes', 'classifier'): 
                    y = df[f'{trust}.encoded']
                    #Drop encoded and nonencoded version of the target variable
                    X = utils.get_X(config, df, trust)
                    #XGB Classifier
                    nb.nb_complete(config, X, y, trust)
                else:
                    pass