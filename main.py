import utils
import pandas as pd
from sklearn.svm import SVR
import ast
import warnings 
from utils import split_into_kfolds
#in use models
import models.IU_models.xgbr as xgbr
import models.IU_models.xgbc as xgbc
import models.IU_models.rf as rf
import models.IU_models.rfr as rfr 
import models.NIU_models.logreg as logreg
#not in use models 
import models.NIU_models.lgbm
import models.NIU_models.nb as nb
import models.NIU_models.mlr as mlr 
import models.NIU_models.svr as svr
import models.NIU_models.lgbmr as lgbmr


warnings.filterwarnings("ignore")

#Setting up configuration settings 
config = utils.config_setup()
std_df, raw_df = utils.load_dataframe(config)

#clear dirs 
utils.clear_directory('data/kfold/train')
utils.clear_directory('data/kfold/test')

#drop columns 
df = utils.drop_columns(raw_df, config)

#apply binary encoding 
df = utils.apply_encoding(config , df)

evaluation_df_reg = pd.DataFrame(columns = ['Name', 'Model','Target','Metric', 'Mean', 'Std'])
evaluation_df_clf = pd.DataFrame(columns = ['Name', 'Model','Target','Metric', 'Mean', 'Std'])

target = config['general']['target']

split_into_kfolds(df, config.getint('preprocessing','num_of_folds'))

k = config.getint('preprocessing','num_of_folds')

#REGRESSORS----------------------------------------------
if config.getboolean('XGBR', 'regressor'):
    xgbr_df = xgbr.kfolds_xgbr(k, config, target)
    xgbr_df
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

if config.getboolean('SVR', 'regressor'):
    y = df[f'{target}']
    #Drop encoded and nonencoded version of the target variable 
    X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #XBG Regressor 
    svr.svr_complete(config, X, y, target)
else:
    pass

if config.getboolean('RFR', 'regressor'):
    y = df[f'{target}']
    #Drop encoded and nonencoded version of the target variable 
    X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #XBG Regressor 
    rfr.rfr_complete(config, X, y, target)
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
#CLASSIFIERS----------------------------------------------
if config.getboolean('XGBClassifier', 'classifier'): 
    xgbc_df = xgbc.kfolds_xgbc(k, config, target)
    evaluation_df_clf = pd.concat([evaluation_df_clf, xgbc_df], ignore_index=True)
else:
    pass

if config.getboolean('RandomForest', 'classifier'): 
    y = df[f'{target}_encoded']
    #Drop encoded and nonencoded version of the target variable
    X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #XGB Classifier
    rf.rfc_complete(config, X, y, target)
else:
    pass

if config.getboolean('LogisticRegression', 'classifier'): 
    y = df[f'{target}_encoded']
    #Drop encoded and nonencoded version of the target variable
    X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #XGB Classifier
    logreg.logreg_complete(config, X, y, target)
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

if config.getboolean('NativeBayes', 'classifier'): 
    y = df[f'{target}_encoded']
    #Drop encoded and nonencoded version of the target variable
    X= df.drop(columns =[f'{target}_encoded', target], axis=1)
    #XGB Classifier
    nb.nb_complete(config, X, y, target)
else:
    pass

#final export 
evaluation_df_reg.to_csv('results/Metrics/evaluation_df_reg.csv', index=False)
evaluation_df_clf.to_csv('results/Metrics/evaluation_df_clf.csv', index=False)