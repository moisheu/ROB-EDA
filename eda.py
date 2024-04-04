import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Correlation Analysis 
#compute the vif for all given features
def compute_vif(df, considered_features):
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

def compute_corr_variable_pairs(df, threshold):
    corr_matrix = df.corr()

    #Create a list to hold pairs of highly correlated features
    high_corr_var_list = []

    #Iterate over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)): # i+1 to not include self-correlation
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in high_corr_var_list):
                high_corr_var_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                high_corr_var_list.append(high_corr_var_pair)


    feature_list = list()
    #Print out the list of high correlation pairs
    for var_pair in high_corr_var_list:
        feature_list.append(var_pair[0])
        feature_list.append(var_pair[1])
        print(f"Variables {var_pair[0]} and {var_pair[1]} have a correlation of {var_pair[2]:.2f}")\
        
    
    feature_list = list(set(feature_list))
    return feature_list

#PCA 

def determine_PC_num(df, config):
    pca = PCA().fit(df)

    #Calculate the explained variance ratios
    explained_var_ratios = pca.explained_variance_ratio_

    #Calculate the difference in explained variance from one component to the next
    explained_var_diff = np.diff(explained_var_ratios)

    # Identify the elbow point where the explained variance stops increasing significantly
    elbow_point = np.argmin(explained_var_diff) + 1  # plus one since differences are between components

    def print_PCA_contributors():
        if config.getboolean('EDA','print_component_contributors'):
            # Find the most important feature (the one with the highest absolute loading on the first PC)
            loadings = pca.components_.T  # Transpose so that columns are features
            abs_loadings_first_pc = np.abs(loadings[:, 0])  # Only the loadings for the first PC
            most_important_feature_index = np.argmax(abs_loadings_first_pc)
            most_important_feature = df.columns[most_important_feature_index]

            print(f"The column that explains the most amount of data (contributes the most to PC1) is: {most_important_feature}")

            sorted_indices = np.argsort(abs_loadings_first_pc)[::-1]
            print("\nVariables contributing to PC1 sorted by absolute loading:")
            for i in sorted_indices:
                print(f"{df.columns[i]}: {loadings[i, 0]}")

            #Only print out positive contributions for PC1
            positive_loadings_indices = ([i for i in range(len(loadings[:, 0])) if loadings[i, 0] > 0])\

            #Extract the column names and loadings into a list of tuples
            positive_loadings = [(df.columns[i], loadings[i, 0]) for i in positive_loadings_indices if loadings[i, 0] > 0]

            #sort the list of tuples by the loading value (in ascending order)
            positive_loadings_sorted_by_value = sorted(positive_loadings, key=lambda x: x[1], reverse = True)

            #print the sorted loadings
            print("\nVariables with positive contributions to PC1 sorted by loading value:")
            for column, loading in positive_loadings_sorted_by_value:
                print(f"{column}: {loading:.4f}")
        else:
            pass

    print(f"Suggested number of principal components: {elbow_point}")
    if config.getboolean('EDA','automate_component_selection'):
        print_PCA_contributors()
        return elbow_point
    
    else:
        #Plot the explained variances
        features = range(pca.n_components_)
        plt.bar(features, explained_var_ratios, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)

        #Draw the cumulative explained variance plot
        plt.figure()
        plt.plot(np.cumsum(explained_var_ratios))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()
        print_PCA_contributors()
        manual_component_num = input('How many components for PCA?')
        return int(manual_component_num)

def plot_pca(df, target, n_components):
    X = df
    y = df[target]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print('Shape before PCA: ', X.shape)
    print('Shape after PCA: ', X_pca.shape)

    #creating PCA df 
    pca_df = pd.DataFrame(data = X_pca, columns = [f'PC_{x+1}' for x in range(0,n_components)])

    return pca_df

    # plt.scatter(X_pca[:, 0], X_pca[:,1], )
    

def automate_pca(df, component, config):
    trust_list = config['general']['trust_list']

    for component in trust_list: 
        n_components = determine_PC_num(df,config)
        plot_pca(df, component, n_components)


#utility functions for factor analysis 
def determine_factor_count(df):
    # Create factor analysis object and perform factor analysis
    print('DETERMINING FACTOR COUNT...-----------------------------------------')
    fa = FactorAnalyzer()
    try: 
        fa.fit(df, 25)
    except Exception as E:
        fa.fit(df, 10)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    print(ev)

    #visualize this data 
    #Create scree plot using matplotlib
    plt.scatter(range(1,df.shape[1]+1),ev)
    plt.plot(range(1,df.shape[1]+1),ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

def apply_factor_analysis(df, num_of_factors):
    fa = FactorAnalyzer(n_factors=num_of_factors, rotation=None)
    fa.fit(df)
    loadings = fa.loadings_
    col = df.columns.tolist()
    loading_df = pd.DataFrame(data = loadings,index=col,  columns = [f"FA{i}" for i in range(1, num_of_factors+1)])
    print(f'LOAD DF FOR {num_of_factors} FACTORS-----------------------------------------')
    print(loading_df)

     #Initialize a dictionary to hold significant variables for each factor
    significant_variables = {}

    #Iterate over the columns (factors) in the loading dataframe
    for factor in loading_df.columns:
        #Get variables with loadings above the threshold for the factor
        significant_loadings = loading_df[loading_df[factor].abs() >= 0.6]
        significant_variables[factor] = significant_loadings[factor]

    #Print the significant variables and their loadings
    print("Significant Variables")
    for factor, variables in significant_variables.items():
        print(f"\n{factor}:")
        for variable, loading in variables.items():
            print(f"{variable}: {loading:.2f}")

    fv = fa.get_factor_variance()
    print(f'FACTOR VARIANCE FOR {num_of_factors} FACTORS-----------------------------------------')
    print(fv)

    return loading_df, loadings

def factor_analysis(df):
    determine_factor_count(df)
    num  = input("Enter the number of factors... Recall significance is a Eigen val >1")
    num = int(num)
    loading_df, loadings = apply_factor_analysis(df, num)
    #loading_df.head()
    return loading_df, loadings