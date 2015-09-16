# -*- coding: utf-8 -*-

import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Load training coupons
# Extract features
filename = 'data/csv/coupon_list_train.csv'
df = pd.read_csv(filename, header=0)
# limited_df = df[['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE']]
limited_df = df[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
# np_array = df.as_matrix()
# headers = list(df.columns.values)
# Encode categories into integers
genre_encoder = LabelEncoder()
large_area_name_encoder = LabelEncoder()
# coupon_id_encoder = LabelEncoder()
# Fit label encoder and return encoded labels
# labels = genre_encoder.fit_transform(limited_df['GENRE_NAME'])
# classes = label_encoder.classes_
# print label_encoder.transform(['グルメ','グルメ'])
# label_encoder.fit(limited_df)
# # Similarly, we can encode the original mutable data frame
limited_df.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df['GENRE_NAME'])
limited_df.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df['large_area_name'])
# limited_df.loc[:,('COUPON_ID_hash')] = coupon_id_encoder.fit_transform(limited_df['COUPON_ID_hash'])
# Use One Hot Encoder to convert categorical features to binary features
# matrice = OneHotEncoder().fit_transform(limited_df.as_matrix)
## Clusterize using kmeans
X = limited_df.as_matrix()
km = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
# Finally, observe the result
# print km.labels_
fig = plt.figure(1, figsize=(4,3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
labels=km.labels_
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))
plt.show()
# Load testing coupons
# Classify within cluster
# -----------------------
# Play
# -----------------------
# # Manual instanstiation of a DataFrame from a dictionary
# features = {
# 	'pet': ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
# 	'children': [4., 6, 3, 3, 2, 3, 5, 4],
# 	'salary':   [90, 24, 44, 27, 32, 59, 36, 27]}
# data = pd.DataFrame(features)
