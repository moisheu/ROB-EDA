import numpy as np
import pandas as pd 

class preprocessing: 
    def __init__(self, df, drop_cols, target_col, variance_threshold, pca_bool, driver_bool):
        self.df = df #the dataframe that is to be preprocessed
        self.drop_cols = drop_cols #list of column names to drop from df 
        self.target_col = target_col #single target column name from df 
        self.variance_threshold = variance_threshold #float <1 to determine variance threshold
        self.pca_cool = pca_bool #boolean T/F, T to run PCA, F otherwise 
        self.driver_bool = driver_bool #boolean T/F, T to seperate drivers and nondrivers, F otherwise
        self.driver_df= None #dataframe of drivers
        self.nondriver_df = None #dataframe of non drivers 

    #drop coluns in df, return df
    def drop(self):
        df = self.df
        df.drop(columns = self.drop_cols)
        return df
    
    #seperate drivers and non-drivers , return dataframes respectively
    def d_or_nd(self):
        if self.driver_bool: 
            


