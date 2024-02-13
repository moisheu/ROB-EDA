import pandas as pd
import numpy as np 
import utils 
import os 
import ast 



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




