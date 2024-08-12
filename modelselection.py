import utils
import pandas as pd
from sklearn.svm import SVR
import ast
import warnings 
from utils import split_into_kfolds
import utils
#in use models
import models.NIU_models.xgbr as xgbr
import models.IU_models.xgbc as xgbc
import models.IU_models.rfc as rfc
import models.NIU_models.rfr as rfr 
import models.IU_models.logreg as logreg
import models.NIU_models.svr_lk as svr_lk
import models.IU_models.svc_lk as svc_lk
import models.IU_models.svc_rk as svc_rk
import models.NIU_models.lgbmr as lgbmr
import models.IU_models.dtc as dtc
import models.IU_models.nb as nb
import models.NIU_models.mlr as mlr 
#not in use models 
import models.NIU_models.lgbm
import models.NIU_models.linreg as linreg 

warnings.filterwarnings("ignore")
import os 


#get experiment queue
directory = 'experiments'
experiment_queue = os.listdir(directory)

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

    #print distribution of participant trust score clf across folds
    enc_target = f'{target}.encoded'

    #print experiment name
    name = config['general']['exp_name']
    print(f'STARTING TRAINING FOR: {name}')

    #CLASSIFIERS----------------------------------------------
    if config.getboolean('XGBClassifier', 'classifier'): 
        xgbc_df = xgbc.full_df_xgbc(df, config, target, name)
        evaluation_df_clf = pd.concat([evaluation_df_clf, xgbc_df], ignore_index=True)
    else:
        pass

    if config.getboolean('RFC', 'classifier'): 
        rfc_df = rfc.full_df_rfc(df, config, target, name)
        evaluation_df_clf = pd.concat([evaluation_df_clf, rfc_df], ignore_index=True)
    else:
        pass

    if config.getboolean('LogisticRegression', 'classifier'): 
        logreg_df = logreg.full_df_logreg(df, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, logreg_df],ignore_index=True)
    else:
        pass

    if config.getboolean('DTC', 'classifier'): 
        dtc_df = dtc.full_df_dtc(df, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, dtc_df],ignore_index=True)
    else:
        pass

    if config.getboolean('NaiveBayes', 'classifier'): 
        nb_df = nb.full_df_nb(df, config, target, name)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, nb_df],ignore_index=True)
    else:
        pass

    if config.getboolean('SVCLK', 'classifier'): 
        svc_lk_df = svc_lk.full_df_svclk(df, config, target)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, svc_lk_df],ignore_index=True)
    else:
        pass

    if config.getboolean('SVCRK', 'classifier'): 
        svc_rk_df = svc_rk.full_df_svcrk(df, config, target)   
        evaluation_df_clf = pd.concat([evaluation_df_clf, svc_rk_df],ignore_index=True)
    else:
        pass

#final export 
evaluation_df_reg.to_csv('results/Metrics/evaluation_df_reg.csv', index=False)
evaluation_df_clf.to_csv('results/Metrics/evaluation_df_clf.csv', index=False)

