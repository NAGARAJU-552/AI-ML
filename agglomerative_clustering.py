# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:44:50 2019

@author: sundar.p.jayaraman
"""


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('./Mall Customers2.csv')

X = dataset.iloc[:, [3, 4]].values


dataset.describe()

from sklearn.cluster import KMeans

final_K = 5
kmeans=KMeans(n_clusters=final_K)
kmeans.fit(X)

kmeans_cluster_val = kmeans.predict(X)
kmeans_cluster_val = kmeans_cluster_val.reshape(kmeans_cluster_val.shape[0],1)
kmeans_X1 = pd.DataFrame(np.concatenate((X, kmeans_cluster_val), axis=1), columns = ['x', 'y', 'cluster'])



color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(final_K):
    plt.scatter(kmeans_X1.loc[kmeans_X1['cluster'] == k]['x'],kmeans_X1.loc[kmeans_X1['cluster'] == k]['y'],s=100,c=color[k],label=labels[k])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=500,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

### Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=final_K)
agglo.fit(X)

agglo_cluster_val = agglo.fit_predict(X)
agglo_cluster_val = agglo_cluster_val.reshape(agglo_cluster_val.shape[0],1)
agglo_X1 = pd.DataFrame(np.concatenate((X, agglo_cluster_val), axis=1), columns = ['x', 'y', 'cluster'])


color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(final_K):
    plt.scatter(agglo_X1.loc[agglo_X1['cluster'] == k]['x'],agglo_X1.loc[agglo_X1['cluster'] == k]['y'],s=100,c=color[k],label=labels[k])
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=500,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
