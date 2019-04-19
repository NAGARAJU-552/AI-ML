# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:04:41 2019

@author: sundar.p.jayaraman
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('./Mall Customers2.csv')

X = dataset.iloc[:, [3, 4]].values


dataset.describe()

from sklearn.cluster import KMeans, MiniBatchKMeans

WCSS_array=np.array([])
for K in range(1,11):
    kmeans=KMeans(n_clusters = K)
    kmeans.fit(X)
    Centroids=kmeans.predict(X)
    WCSS_array=np.append(WCSS_array,kmeans.inertia_)
    
    
K_array=np.arange(1,11,1)
plt.plot(K_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()

final_K = 5

import datetime
a = datetime.datetime.now()
kmeans=KMeans(n_clusters=final_K)
kmeans.fit(X)
print(datetime.datetime.now() - a)

cluster_val = kmeans.predict(X)
cluster_val = cluster_val.reshape(cluster_val.shape[0],1)
X1 = pd.DataFrame(np.concatenate((X, cluster_val), axis=1), columns = ['x', 'y', 'cluster'])



color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(final_K):
    plt.scatter(X1.loc[X1['cluster'] == k]['x'],X1.loc[X1['cluster'] == k]['y'],s=100,c=color[k],label=labels[k])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=500,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print(kmeans.inertia_)

a = datetime.datetime.now()
mbk = MiniBatchKMeans(n_clusters=final_K, batch_size=1000)
mbk.fit(X)
print(datetime.datetime.now() - a)

cluster_val = mbk.predict(X)
cluster_val = cluster_val.reshape(cluster_val.shape[0],1)
X1 = pd.DataFrame(np.concatenate((X, cluster_val), axis=1), columns = ['x', 'y', 'cluster'])



color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(final_K):
    plt.scatter(X1.loc[X1['cluster'] == k]['x'],X1.loc[X1['cluster'] == k]['y'],s=100,c=color[k],label=labels[k])
plt.scatter(mbk.cluster_centers_[:,0],mbk.cluster_centers_[:,1],s=500,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print(mbk.inertia_)

