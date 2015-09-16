# -*- coding: utf-8 -*-

import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

NUMBER_OF_CLUSTERS = 15

def plot_cluster(X_train, km):
	fig = plt.figure(1, figsize=(4,3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	plt.cla()
	labels=km.labels_
	ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=labels.astype(np.float))
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Genre')
	ax.set_ylabel('Price')
	ax.set_zlabel('Large area name')
	plt.show()

# Load training coupons
filename_train = 'data/csv/coupon_list_train.csv'
df_train = pd.read_csv(filename_train, header=0)

# Extract features
# limited_df_train = df_train[['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE']]
limited_df_train = df_train[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
# headers = list(df_train.columns.values)

# Encode categories into integers
genre_encoder = LabelEncoder()
large_area_name_encoder = LabelEncoder()
limited_df_train.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_train['GENRE_NAME'])
limited_df_train.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_train['large_area_name'])

# Clusterize using kmeans
X_train = limited_df_train.as_matrix()
km = KMeans(n_clusters=NUMBER_OF_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
km.fit(X_train)

# Plot clusters for observation
# plot_cluster(X_train, km)

# TODO: Maybe map COUPON_ID_hash -> cluster_index
coupon_id_hashes = df_train['COUPON_ID_hash']
train_cluster_indexes = km.predict(X_train)
coupon_id_hash_cluster_index_map = pd.DataFrame(dict(
	COUPON_ID_hash=coupon_id_hashes, 
	cluster_index=train_cluster_indexes))

# Load buying history set
filename_detail = 'data/csv/coupon_detail_train.csv'
df_detail = pd.read_csv(filename_detail, header=0)

# Add clusters to df_detail
df_detail_clustered = pd.merge(df_detail, coupon_id_hash_cluster_index_map, on='COUPON_ID_hash')

# Group purchases by user
user_purchase_history = df_detail_clustered.groupby('USER_ID_hash')
grouped_user_purchase_history = user_purchase_history.groups

# Or by user and cluster
user_cluster = df_detail_clustered.groupby(['USER_ID_hash', 'cluster_index'])
grouped_user_cluster = user_cluster.groups

# Todo: Map USER_ID_hash -> cluster_index based on purchases

# # Load testing coupons
# filename_test = 'data/csv/coupon_list_test.csv'
# df_test = pd.read_csv(filename_test, header=0)

# # Extract training coupons features
# limited_df_test = df_test[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
# limited_df_test.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_test['GENRE_NAME'])
# limited_df_test.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_test['large_area_name'])
# X_test = limited_df_test.as_matrix()

# # Classify within trainin clusters
# predictions_test = km.predict(X_test)
