# -*- coding: utf-8 -*-

import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
# Extract features
filename_train = 'data/csv/coupon_list_train.csv'
df_train = pd.read_csv(filename_train, header=0)
# limited_df_train = df_train[['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE']]
limited_df_train = df_train[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
# np_array = df_train.as_matrix()
# headers = list(df_train.columns.values)
# Encode categories into integers
genre_encoder = LabelEncoder()
large_area_name_encoder = LabelEncoder()
coupon_id_encoder = LabelEncoder()
# Fit label encoder and return encoded labels
# labels = genre_encoder.fit_transform(limited_df_train['GENRE_NAME'])
# classes = label_encoder.classes_
# print label_encoder.transform(['グルメ','グルメ'])
# label_encoder.fit(limited_df_train)
# # Similarly, we can encode the original mutable data frame
limited_df_train.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_train['GENRE_NAME'])
limited_df_train.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_train['large_area_name'])
# limited_df_train.loc[:,('COUPON_ID_hash')] = coupon_id_encoder.fit_transform(limited_df_train['COUPON_ID_hash'])
# Use One Hot Encoder to convert categorical features to binary features
# matrice = OneHotEncoder().fit_transform(limited_df_train.as_matrix)
## Clusterize using kmeans
X_train = limited_df_train.as_matrix()
km = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=1)
km.fit(X_train)
# Finally, observe the result
# print km.labels_
# plot_cluster(X_train, km)

# Load testing coupons
# Classify within trainin clusters
filename_test = 'data/csv/coupon_list_test.csv'
df_test = pd.read_csv(filename_test, header=0)
limited_df_test = df_test[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
limited_df_test.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_test['GENRE_NAME'])
limited_df_test.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_test['large_area_name'])
X_test = limited_df_test.as_matrix()
predictions_test = km.predict(X_test)
