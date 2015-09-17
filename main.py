# -*- coding: utf-8 -*-

import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

NUMBER_OF_CLUSTERS = 8

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
print "Loading training coupons..."
filename_train = 'data/csv/coupon_list_train.csv'
df_train = pd.read_csv(filename_train, header=0)

print "Extracting training coupons features..."
# Extract features
# limited_df_train = df_train[['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE']]
limited_df_train = df_train[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
# headers = list(df_train.columns.values)

print "Encoding training coupons categories..."
# Encode categories into integers
genre_encoder = LabelEncoder()
large_area_name_encoder = LabelEncoder()
limited_df_train.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_train['GENRE_NAME'])
limited_df_train.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_train['large_area_name'])

print "Clustering training coupons..."
# Clusterize using kmeans
X_train = limited_df_train.as_matrix()
km = KMeans(n_clusters=NUMBER_OF_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
km.fit(X_train)

# Plot clusters for observation
# print "Plotting training coupons clusters..."
# plot_cluster(X_train, km)

# TODO: Maybe map COUPON_ID_hash -> cluster_index
print "Mapping training coupons to training clusters..."
coupon_id_hashes = df_train['COUPON_ID_hash']
train_cluster_indexes = km.predict(X_train)
coupon_id_hash_cluster_index_map = pd.DataFrame(dict(
	COUPON_ID_hash=coupon_id_hashes, 
	cluster_index=train_cluster_indexes))

print "Loading purchase history..."
# Load buying history set
filename_detail = 'data/csv/coupon_detail_train.csv'
df_detail = pd.read_csv(filename_detail, header=0)

print "Adding clusters to purchase history..."
# Add clusters to df_detail
df_detail_clustered = pd.merge(df_detail, coupon_id_hash_cluster_index_map, on='COUPON_ID_hash')

# Group purchases by user
user_purchase_history = df_detail_clustered.groupby('USER_ID_hash')
grouped_user_purchase_history = user_purchase_history.groups

# Or by user and cluster
user_cluster = df_detail_clustered.groupby(['USER_ID_hash', 'cluster_index'])
# grouped_user_cluster = user_cluster.groups

print "Mapping users to training clusters..."
# Map USER_ID_hash -> cluster_index based on purchases
# TODO: Optimize this part
user_cluster_map = dict()
for user_id_cluster, coupon_id_hahes in user_cluster:
	user_id_hash = user_id_cluster[0] 
	user_cluster = user_id_cluster[1]
	if user_id_hash in user_cluster_map:
		if user_cluster > user_cluster_map[user_id_hash]:
			user_cluster_map[user_id_hash] = user_cluster
	else:
		user_cluster_map[user_id_hash] = user_cluster

print "Loading testing coupons..."
# Load testing coupons
filename_test = 'data/csv/coupon_list_test.csv'
df_test = pd.read_csv(filename_test, header=0)

# Extract testing coupons features
print "Extracting testing coupons features..."
limited_df_test = df_test[['GENRE_NAME', 'PRICE_RATE', 'large_area_name']]
limited_df_test.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_test['GENRE_NAME'])
limited_df_test.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_test['large_area_name'])
X_test = limited_df_test.as_matrix()

# Classify within training clusters
print "Predicting testing coupons clusters..."
predictions_test = km.predict(X_test)
test_coupon_id_hashes = df_test['COUPON_ID_hash']
test_coupon_id_hash_cluster_index_map = pd.DataFrame(dict(
	COUPON_ID_hash=test_coupon_id_hashes,
	cluster_index=predictions_test))
# df_test_clustered = pd.merge(df_test, test_coupon_id_hash_cluster_index_map, on='COUPON_ID_hash')

# Group test coupons by cluster
# clustered_test_coupons = df_test_clustered.groupby('cluster_index')

# Group test coupon->cluster map by cluster
grouped_test_coupons = test_coupon_id_hash_cluster_index_map.groupby('cluster_index')

# Create user->test_coupons map
print "Mapping %s users to test coupons..." % len(user_cluster_map)
user_id_hash_test_coupons_map = dict()
for user_id_hash, cluster_index in user_cluster_map.iteritems():
	try:
		user_id_hash_test_coupons_map[user_id_hash] = " ".join(list(grouped_test_coupons.get_group(cluster_index)['COUPON_ID_hash']))
	except:
		print "Cluster %s doesn't exist!" % cluster_index
		user_id_hash_test_coupons_map[user_id_hash] = ""

user_id_hash = user_id_hash_test_coupons_map.keys()

# Fill missing users
print "Filling missing users..."
user_list_filename = 'data/csv/user_list.csv'
df_users = pd.read_csv(user_list_filename, header=0)
df_users = df_users['USER_ID_hash']
users_list = list(df_users)
user_miss = 0
for user in users_list:
	if user not in user_id_hash:
		user_miss += 1
		user_id_hash.append(user)
print "Added %s missing users!" % user_miss

coupon_id_hash = user_id_hash_test_coupons_map.values()
for i in range(0, user_miss):
	coupon_id_hash.append("")

result = pd.DataFrame(dict(
    USER_ID_hash=user_id_hash,
    PURCHASED_COUPONS=coupon_id_hash))
result = result[['USER_ID_hash', 'PURCHASED_COUPONS']]

print "Writing results..."
# Write it to CSV
result.to_csv('submission.csv', index=False)

print "Done!"
