# -*- coding: utf-8 -*-

import csv

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Load training coupons
# Extract features
filename = 'data/csv/coupon_list_train.csv'
df = pd.read_csv(filename, header=0)
limited_df = df[['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE']][:10]
# np_array = df.as_matrix()
headers = list(df.columns.values)
# Encode categories into integers
genre_encoder = LabelEncoder()
coupon_id_encoder = LabelEncoder()
# Fit label encoder and return encoded labels
# labels = label_encoder.fit_transform(limited_df['GENRE_NAME'])
# classes = label_encoder.classes_
# print label_encoder.transform(['グルメ','グルメ'])
# label_encoder.fit(limited_df)
# # Similarly, we can encode the original mutable data frame
limited_df['GENRE_NAME'] = genre_encoder.fit_transform(limited_df['GENRE_NAME'])
limited_df['COUPON_ID_hash'] = coupon_id_encoder.fit_transform(limited_df['COUPON_ID_hash'])
# Use One Hot Encoder to convert categorical features to binary features
matrice = OneHotEncoder().fit_transform(limited_df.as_matrix())
# Clusterize using kmeans
km = KMeans(n_clusters=1, init='k-means++', max_iter=100, n_init=1)
km.fit(matrice.data)
# Finally, observe the result:
# as expected, it doesn't make sense. :)
print km.labels_

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
