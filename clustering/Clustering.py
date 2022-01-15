!pip install prince
!pip install kmodes

import numpy as np 
import pandas as pd
import warnings

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import networkx as nx

from sklearn.cluster import KMeans
from kmodes import kmodes

from sklearn.decomposition import PCA
import prince

from gensim import corpora, similarities,models
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import psutil
import os


#Pre-processing

train = pd.read_csv('/content/drive/MyDrive/kaggle_market/datafiles/train.csv')
features = pd.read_csv('/content/drive/MyDrive/kaggle_market/datafiles/features.csv')
train = train.astype('float32')

ft = train.iloc[:, 7:-1]
ft_names = ft.columns
ft = StandardScaler().fit_transform(ft)
ft = pd.DataFrame(ft, columns = ft_names)
ft.mean().mean()


#Kmeans Clustering based on Pearson Correlation

ft_corr = ft.corr(method='pearson')
ft_corr_dist = np.sqrt(0.5*(1 - ft_corr))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(23,9))

sns.heatmap(ft_corr, ax = ax1, cmap='coolwarm');
sns.heatmap(ft_corr_dist, ax = ax2, cmap='Greys');
ax1.title.set_text('Correlation Matrix')
ax2.title.set_text('Distance Matrix')
plt.show()
memory_state()

ft_corr_np = ft_corr.to_numpy()

wcss = []
max_num_clusters = 15
for i in range(1, max_num_clusters):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(ft_corr_np)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, max_num_clusters), wcss)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ft_corr_labels = kmeans.fit_predict(ft_corr_np)

ft_corr_clust_km = pd.DataFrame(np.c_[ft_names, ft_corr_labels])
ft_corr_clust_km.columns = ["feature", "cluster"]
ft_corr_clust_km['feature_list'] = ft_corr_clust_km.groupby(["cluster"]).transform(lambda x: ', '.join(x))
ft_corr_clust_km = ft_corr_clust_km.groupby(["cluster", "feature_list"]).size().reset_index(name = 'feature_count')
ft_corr_clust_km.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_corr_clust_km.txt')


#Hierarchical Clustering based on Pearson Correlation


ft_corr_link = sch.linkage(ft_corr_dist, 'average')

fig = plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Feature')
plt.ylabel('Distance')
plt.hlines(1.5, 0, 1320)
ft_corr_dn = sch.dendrogram(ft_corr_link, leaf_rotation=90., leaf_font_size=11.)
plt.show()

max_d = 1.6
clusters = fcluster(ft_corr_link, t=max_d, criterion='distance')
ft_corr_clust_dn = pd.DataFrame({'Cluster':clusters, 'Features':ft_names})
ft_corr_clust_dn['feature_list'] = ft_corr_clust_dn.groupby(["Cluster"]).transform(lambda x: ', '.join(x))
ft_corr_clust_dn = ft_corr_clust_dn.groupby(["Cluster", "feature_list"]).size().reset_index(name = 'feature_count')
ft_corr_clust_dn['Cluster'] = ft_corr_clust_dn['Cluster'].apply(lambda x: x - 1)
ft_corr_clust_dn

ft_corr_clust_dn.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_corr_clust_dn.txt')


#DBSCAN Clustering based on Spearman Correlation

ft = ft.fillna(0)
ft_corr_spr = ss.spearmanr(ft).correlation

ft_corr_spr_dist = np.sqrt(0.5*(1 - ft_corr_spr))
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(23,9))

sns.heatmap(ft_corr_spr, ax = ax1, cmap='coolwarm');
sns.heatmap(ft_corr_spr_dist, ax = ax2, cmap='coolwarm');
ax1.title.set_text('Correlation Matrix')
ax2.title.set_text('Distance Matrix')
plt.show()


db_1 = DBSCAN(eps = 0.75)
ft_db = db.fit_predict(ft_corr_spr_dist)
ft_clust_db = pd.DataFrame(np.c_[ft_names, ft_db])
ft_clust_db.columns = ["feature", "cluster"]
ft_clust_db['feature_list'] = ft_clust_db.groupby(["cluster"]).transform(lambda x: ', '.join(x))
ft_clust_db = ft_clust_db.groupby(["cluster", "feature_list"]).size().reset_index(name = 'feature_count')
ft_clust_db['cluster'] += 1

ft_clust_db.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_clust_db.txt')


#Hierarchical Clustering based on Spearman Correlation

ft_corr_spr_link = sch.linkage(ft_corr_spr_dist, 'average')

fig = plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram for Spearman Correlation')
plt.xlabel('Feature')
plt.ylabel('Distance')
plt.hlines(2, 0, 1320)
ft_corr_dn = sch.dendrogram(ft_corr_spr_link, leaf_rotation=90., leaf_font_size=11.)
plt.show()

max_d = 2.0
clusters = fcluster(ft_corr_spr_link, t=max_d, criterion='distance')
ft_corr_spr_clust_dn = pd.DataFrame({'Cluster':clusters, 'Features':ft_names})
ft_corr_spr_clust_dn['feature_list'] = ft_corr_spr_clust_dn.groupby(["Cluster"]).transform(lambda x: ', '.join(x))
ft_corr_spr_clust_dn = ft_corr_spr_clust_dn.groupby(["Cluster", "feature_list"]).size().reset_index(name = 'feature_count')
ft_corr_spr_clust_dn['Cluster'] = ft_corr_spr_clust_dn['Cluster'].apply(lambda x: x - 1)

ft_corr_spr_clust_dn.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_corr_spr_clust_dn.txt')


#Clustering based on NaN numbers

ft_nan = pd.DataFrame(columns = ['feature' , 'nan_num'])

for i in range(130):
  nan_num = len(train[train[f'feature_{i}'].isna()])
  ft_nan.loc[i] = [f'feature_{i}', nan_num]

ft_nan.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_nan.txt')

ft_nan['feature_list'] = ft_nan.groupby(['nan_num']).transform(lambda x: ', '.join(x))
ft_nan = ft_nan.groupby(["nan_num", "feature_list"]).size().reset_index(name = 'feature_count')
ft_nan_clust = ft_nan.iloc[:, 1:]

ft_nan_clust['Cluster'] = ft_nan_clust.index
ft_nan_clust = ft_nan_clust[['Cluster', 'feature_list', 'feature_count']]
ft_nan_clust.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_nan_clust.txt')


#Kmode-Clustering based on Tags

features.replace({False: "False", True: "True"}, inplace = True)

cost = []
max_clust = 30
for num_clusters in list(range(1, max_clust)):
    kmode = kmodes.KModes(n_clusters = num_clusters, init = "Cao", n_init = 5, verbose=1)
    kmode.fit_predict(features.iloc[:, 7:])
    cost.append(kmode.cost_)

plt.plot(range(1, max_clust), cost)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.show()

n_clusters_kmodes = 7
kmode_cao = kmodes.KModes(n_clusters = n_clusters_kmodes, init = "Cao", n_init = 10, verbose=1)
features_labels = kmode_cao.fit_predict(features.iloc[:, 7:])

# Preparing a dataframe to collect some cluster stats
feature_tags_clust = pd.DataFrame(np.c_[ft_names, features_labels])
feature_tags_clust.columns = ["feature", "cluster"]
feature_tags_clust['feature_list'] = feature_tags_clust.groupby(["cluster"]).transform(lambda x: ', '.join(x))
feature_tags_clust = feature_tags_clust.groupby(["cluster", "feature_list"]).size().reset_index(name = 'feature_count')
feature_tags_clust.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_tags_clust.txt')


#Clustering based on LSI model for Tags 

features_data_types_dict = {
    'feature_id': 'str',
    'tags': 'str'
}

features_df = pd.read_csv('/content/drive/MyDrive/kaggle_market/charmesul/truelist.csv').transpose()
features_df.reset_index(inplace = True)

features_df.rename(columns = {'index' : 'feature_id', 0 : 'tags'}, inplace = True)
features_df['tags'].fillna("-1",inplace=True)
features_df = features_df.astype(features_data_types_dict)


class tag_dic_c(object):
    def __iter__(self):
        for index, doc in enumerate(features_df['tags']):
            yield doc.split(' ')
tag_dic = tag_dic_c()
dictionary = corpora.Dictionary(tag_dic)

class vectorizer(object):
    def __iter__(self):
        for index,doc in enumerate(features_df['tags']):
            yield dictionary.doc2bow(doc.split(' '))
corpus = vectorizer()
# corpus = [dictionary.doc2bow(text) for text in self.text]
tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
corpus_tfidf = tfidf_model[corpus]

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics = 12)
#lsi_model.save("tag_lsi_model.lsi")
#lsi_model = models.LsiModel.load('tag_lsi_model.lsi')


def get_arg_max(single_list):
    max_index=0
    max_num=single_list[0][1]
    for index in range(len(single_list)-1):
        if max_num<single_list[index+1][1]:
            max_num=single_list[index+1][1]
            max_index=index+1
    return max_index

all_data_lsi_1=[]

for tags in features_df['tags']:
    vec_bow=dictionary.doc2bow(tags.split(' '))
    vec_lsi_1=list(lsi_model[tfidf_model[vec_bow]])
    if len(vec_lsi_1)==0:
        all_data_lsi_1.append(0)
    else:
        all_data_lsi_1.append(get_arg_max(vec_lsi_1))


features_df['tags_lsi_1']=all_data_lsi_1

print(features_df.head(50))
features_df.drop(columns=['tags'], inplace=True)

features_df.to_csv("tag_lsi.csv", sep=',')

features_df['feature_id'] = features_df.groupby(["tags_lsi_1"]).transform(lambda x: ', '.join(x))
features_df = features_df.groupby(["tags_lsi_1", 'feature_id']).size().reset_index(name = 'feature_count')

features_df.rename(columns = {"tags_lsi_1" : "Cluster", 'feature_id' : "feature_list"}, inplace = True)
features_df.reset_index(inplace = True)
features_df = features_df.drop(['index'], axis = 1)
features_df.to_csv('/content/drive/MyDrive/kaggle_market/charmesul/feature_tag_lsi.txt')
