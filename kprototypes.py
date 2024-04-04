import numpy as np 
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes

def elbow(df):
    costs = []
    K = range(1,10) 
    for k in K:
        kproto = KPrototypes(n_clusters=k, init='Cao')
        categorical_col = df.columns.tolist()
        index = [idx for idx, s in enumerate(categorical_col) if '_category' in s]
        kproto.fit_predict(df, categorical=index)
        costs.append(kproto.cost_)

    plt.plot(K, costs, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def kprototypes(df):
    #elbow(df)
    #user_num = int(input('Enter number of clusters'))
    kproto =KPrototypes(n_clusters = 3, init = 'Cao')
    categorical_col = df.columns.tolist()
    index = [idx for idx, s in enumerate(categorical_col) if '_category' in s]
    kproto.fit_predict(df, categorical = index)
    centroids = kproto.cluster_centroids_
    labels =kproto.labels_

    print("Cluster Centroids:\n", centroids)
    print("Cluster Labels:\n", labels)

    (unique, counts) = np.unique(labels, return_counts=True)
    clusters_count = dict(zip(unique, counts))
    print("Cluster Sizes:\n", clusters_count)

    df['Cluster'] = labels

    print(df)

    temp = df.columns.tolist()
    numeric_df = df.drop(columns = temp[index[0]])

    # For numerical features
    numerical_stats = numeric_df.groupby('Cluster').agg(['mean', 'median', 'std'])

    # For categorical features
    categorical_counts = df.groupby('Cluster').agg(lambda x: x.value_counts().index[0])

    print("Numerical Features Statistics:\n", numerical_stats)
    print("\nCategorical Features Most Common:\n", categorical_counts)