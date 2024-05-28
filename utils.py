import logging as logger 
import pandas as pd 
import ast
import pandas as pd
import configparser 
import ast 
from sklearn.model_selection import KFold
from sklearn import preprocessing
import shutil

def config_setup():
      config = configparser.ConfigParser()
      config.read('config.ini')
      return config 

def load_dataframe(config):
    std_df = pd.read_csv(config['preprocessing']['std_path'])
    raw_df = pd.read_csv(config['preprocessing']['raw_path'])
    return std_df , raw_df

import os
import shutil

def clear_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return
    #iterate dirs 
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        #if file del
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            #if dir, clear and del 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def merge_dataframes_for_encoding(df_standardized, df_raw, config):
    #Take raw dataframe target cols and responseID from raw df
    category_dict = ast.literal_eval(config['general']['category_dictionary'])
    target_col_list = category_dict['trust']

    #Encode the target values, 0 if score is x<3 , 1 if score is x>3
    def encode(x):
        if int(x) > 3:
            return 1
        else:
            return 0
        
    for column in target_col_list:
        df_raw[column] = df_raw[column].apply(encode)        
    #Extract the target and responseID columns post merge
    df_raw_target_cols = df_raw[target_col_list]

    #Change col names to identify encoding 
    temp_names_list = df_raw_target_cols.columns.tolist()
    new_names_list = [f'{col}_encoded' if col != 'ResponseID' else col for col in temp_names_list]

    df_raw_target_cols.columns = new_names_list

    #Merge on unique ResponseId 
    merged_df = df_standardized.merge(
        df_raw_target_cols,
        # on = 'ResponseId',
        left_index=True, 
        right_index=True
       # how = 'left'
    )

    return merged_df

def drop_columns(df, config):
    keeplist = ast.literal_eval(config['general']['keeplist'])
    df = df.filter(items=keeplist)
    return df

def get_total_df(df):
    col_list = df.columns.tolist()
    composite_list = list()
    noncomposite_list = list()
    total_list = list()

    for col in col_list:
        if ('Composite' in col):
            composite_list.append(col)
        elif ('Total' in col):   
            total_list.append(col)
        else:
            noncomposite_list.append(col)

    if len(composite_list)+len(noncomposite_list)+len(total_list) == len(col_list):
        #comp_df = pd.DataFrame(data = df[composite_list], columns= composite_list)
        total_df = pd.DataFrame(data = df[total_list], columns= total_list)
        #noncomp_df = pd.DataFrame(data = df[noncomposite_list], columns= noncomposite_list)
    
    return total_df    

def get_dataframes(config):
        df = load_dataframe(config)
        drop_columns(df, config)

def apply_encoding(config,df):
    column = config['general']['target']
    #encode based on 2 classes
    def encode(x):
        quantiles = df[column].quantile([.5])
        threshold = quantiles.loc[0.5]
        if x >= threshold:
            return 1
        else:
            return 0
    df.loc[:, f'{column}.encoded'] = df[column].apply(encode)
    return df    

def split_into_kfolds(df, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_indices = list(kf.split(df))

    for i, (train_index, test_index) in enumerate(fold_indices):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        train_path = f'data/kfold/train/train_fold_{i+1}.csv'
        test_path = f'data/kfold/test/test_fold_{i+1}.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f'Saved fold {i+1} to {train_path} and {test_path}')
    
    return fold_indices

