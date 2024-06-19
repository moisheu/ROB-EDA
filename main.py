import utils
import pandas as pd
from sklearn.svm import SVR
import ast
import warnings 
from utils import split_into_kfolds
import utils
#in use models
import models.IU_models.xgbr as xgbr
import models.IU_models.xgbc as xgbc
import models.IU_models.rfc as rfc
import models.IU_models.rfr as rfr 
import models.IU_models.logreg as logreg
import models.IU_models.svr_lk as svr_lk
import models.IU_models.svc_lk as svc_lk
import models.IU_models.svc_rk as svc_rk
import models.NIU_models.lgbmr as lgbmr
import models.IU_models.dtc as dtc
import models.IU_models.nb as nb
import models.NIU_models.mlr as mlr 
#not in use models 
import models.NIU_models.lgbm
import models.IU_models.linreg as linreg 

warnings.filterwarnings("ignore")
import os 

#get experiment queue
directory = 'experiments'
experiment_queue = os.listdir(directory)

#clear dirs for test/train dfs
utils.clear_directory('data/kfold/train')
utils.clear_directory('data/kfold/test')

#init eval dfs 
evaluation_df_reg = pd.DataFrame(columns = ['Name', 'Model','Target','Metric', 'Mean', 'Std'])
evaluation_df_clf = pd.DataFrame(columns = ['Name', 'Model','Target','Metric', 'Mean', 'Std'])

for experiment in experiment_queue:
    
    experiment_path= f'{directory}/{experiment}'
    print(experiment_path)

    #set up configs 
    config = utils.config_setup(experiment_path)
    std_df, raw_df = utils.load_dataframe(config)

    #drop columns 
    df = utils.drop_columns(raw_df, config)

    #apply binary encoding 
    df = utils.apply_encoding(config , df)

    #handle all nans by replacing with the mean 
    df = utils.bootstrap_nans(df)

    #establish target variable from config 
    target = config['general']['target']

    #retrieve k number of folds from config 
    k = config.getint('preprocessing','num_of_folds')
    split_into_kfolds(df, target, k)

    #print distribution of participant trust score clf across folds
    enc_target = f'{target}.encoded'
    for i in range(1, k+1):
        train_df = pd.read_csv(f'data/kfold/train/train_fold_{i}.csv')
        test_df = pd.read_csv(f'data/kfold/test/test_fold_{i}.csv')
        print(f'Distribution of trust classes in training set : {train_df[enc_target].value_counts()}')
        print(f'Distribution of trust classes in test set : {test_df[enc_target].value_counts()}')

    #print experiment name
    name = config['general']['exp_name']
    print(f'STARTING TRAINING FOR: {name}')

    #REGRESSORS----------------------------------------------
    if config.getboolean('XGBR', 'regressor'):
        xgbr_df = xgbr.kfolds_xgbr(k, config, target)   
        evaluation_df_reg = pd.concat([evaluation_df_reg, xgbr_df],ignore_index=True)
    else:
        pass

    if config.getboolean('MLR', 'regressor'):
        y = df[f'{target}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{target}_encoded', target], axis=1)
        #XBG Regressor 
        mlr.mlr_complete(config, X, y, target)
    else:
        pass

    if config.getboolean('SVRLK', 'regressor'):
        svr_lk_df = svr_lk.kfolds_svr_lk(k, config, target)   
        evaluation_df_reg = pd.concat([evaluation_df_reg, svr_lk_df],ignore_index=True)
    else:
        pass

    if config.getboolean('RFR', 'regressor'):
        rfr_df = rfr.kfolds_rfr(k, config, target)   
        evaluation_df_reg = pd.concat([evaluation_df_reg, rfr_df],ignore_index=True) 
    else:
        pass

    if config.getboolean('LGBMR', 'regressor'):
        y = df[f'{target}']
        #Drop encoded and nonencoded version of the target variable 
        X= df.drop(columns =[f'{target}_encoded', target], axis=1)
        #XBG Regressor 
        lgbmr.lgbmr_complete(config, X, y, target)
    else:
        pass

    if config.getboolean('LINREG', 'regressor'):
        linreg_df = linreg.kfolds_linear_regression(k, config, target)
        evaluation_df_reg = pd.concat([evaluation_df_reg, linreg_df],ignore_index=True)
    else:
        pass

    #CLASSIFIERS----------------------------------------------
    if config.getboolean('XGBClassifier', 'classifier'): 
        xgbc_df = xgbc.kfolds_xgbc(k, config, target, name)
        evaluation_df_clf = pd.concat([evaluation_df_clf, xgbc_df], ignore_index=True)
    else:
        pass

    if config.getboolean('RFC', 'classifier'): 
        rfc_df = rfc.kfolds_rfc(k, config, target, name)
        evaluation_df_clf = pd.concat([evaluation_df_clf, rfc_df], ignore_index=True)
    else:
        pass

    if config.getboolean('LogisticRegression', 'classifier'): 
        logreg_df = logreg.kfolds_logreg(k, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, logreg_df],ignore_index=True)
    else:
        pass

    if config.getboolean('DTC', 'classifier'): 
        dtc_df = dtc.kfolds_dtc(k, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, dtc_df],ignore_index=True)
    else:
        pass

    # if config.getboolean('LightGBM', 'classifier'): 
    #     y = df[f'{target}_encoded']
    #     #Drop encoded and nonencoded version of the target variable
    #     X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #     #XGB Classifier
    #     lgbm.lgbm_complete(config, X, y, target)
    # else:
    #     pass

    if config.getboolean('NaiveBayes', 'classifier'): 
        nb_df = nb.kfolds_nb(k, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, nb_df],ignore_index=True)
    else:
        pass

    if config.getboolean('SVCLK', 'classifier'): 
        svc_lk_df = svc_lk.kfolds_svc_lk(k, config, target)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, svc_lk_df],ignore_index=True)
    else:
        pass

    if config.getboolean('SVCRK', 'classifier'): 
        svc_rk_df = svc_rk.kfolds_svc_rk(k, config, target)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, svc_rk_df],ignore_index=True)
    else:
        pass

#final export 
evaluation_df_reg.to_csv('results/Metrics/evaluation_df_reg.csv', index=False)
evaluation_df_clf.to_csv('results/Metrics/evaluation_df_clf.csv', index=False)