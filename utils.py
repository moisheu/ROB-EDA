import logging as logger 
import pandas as pd 
import ast
import pandas as pd
import configparser 
import ast 
from sklearn import preprocessing

def config_setup():
      config = configparser.ConfigParser()
      config.read('config.ini')
      return config 

def load_dataframe(config):    
    df = pd.read_csv(config['preprocessing']['dataframe_path'])
    non_standard_df = pd.read_csv(config['preprocessing']['nonstandard_dataframe_path'])
    return df , non_standard_df

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
    category_dict = ast.literal_eval(config['general']['category_dictionary'])
    #global drops 
    df = df.drop(columns = ['Unnamed: 0', 'Duration.Seconds'])
    df = df.drop(columns = category_dict['totals_to_drop'])

    #conditional drops, to change drops change in config.ini file 
    if config.getboolean('preprocessing','drop_parent'):
        logger.info('Dropping parent categories')
        df = df.drop(columns = category_dict['parents_categories'])

    #create dataframes     
    for component in category_dict['trust']: 
         logger.info(f'Creating dataframes for {component}')
         temp_df = df 
         temp_list = category_dict['trust'][:]
         temp_list_encoded = [f'{trust}_encoded' for trust in temp_list]
         if component != 'Composite.Trust.Human':
             temp_list.remove(component)
             temp_list.remove('Composite.Trust.Human')
         else: 
             temp_list.remove(component)
         
         temp_list_encoded.remove(f'{component}_encoded')
         temp_df = temp_df.drop(columns = temp_list)
         temp_df = temp_df.drop(columns = temp_list_encoded)
         driver_df = temp_df.dropna(axis = 0)
         non_driver_df = temp_df[df.isna().any(axis=1)]
         
         #exporting new dfs
         d_export_path  = config['preprocessing']['driver_export_filepath']
         nd_export_path  = config['preprocessing']['nondriver_export_filepath']
         driver_df.to_csv(d_export_path+fr'\{component}_driver.csv', index=False)
         non_driver_df.to_csv(nd_export_path+fr'\{component}_nondriver.csv', index=False)
         logger.info(f'Created and exported {component} dataframes!')

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

def apply_encoding(config, df=None, name=None):
    def encode(x):      
        quantiles = df[column].quantile([0.33, 0.66]) 
        threshold_lowmed = quantiles.loc[0.33]
        threshold_medhigh = quantiles.loc[0.66]
        if x >= threshold_medhigh:
            return '2'
        elif threshold_lowmed <= x < threshold_medhigh:
            return '1'
        else:
            return '0'
        
    if config.getboolean('segmentation', 'segmentation_training'):
        column = name
        print(f"Encoding column: {column}")
        df[f'{column}_encoded'] = df[column].apply(encode)
        df = df.drop(columns=[column])
    else:
        category_dict = ast.literal_eval(config['general']['category_dictionary'])
        target_col_list = category_dict['trust']
        for column in target_col_list:
            print(f"Encoding column: {column}")
            df[f'{column}_encoded'] = df[column].apply(encode)
            if len(df[f'{column}_encoded'].unique()) == 2:
                df[f'{column}_encoded'] = df[f'{column}_encoded'].replace({'2': '1'})
    
    return df

def get_X(config, df, trust):
    if config.getboolean('segmentation','segmentation_training'):
        X= df.drop(columns =[f'{trust}_encoded'], axis=1)
    else:
        X= df.drop(columns =[f'{trust}_encoded', trust], axis=1)
    return X


