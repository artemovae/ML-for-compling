# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
from os import path
import glob
import pickle
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import *
from sklearn.cluster import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt


def execute_pipeline(data_model, alg):
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=150)),
        ('norm', Normalizer()),
        ('clust', alg(n_clusters=28))
    ])
    pipeline.fit(data_model)
    return pipeline

def svd_varience(pipeline):
    explained_variance = pipeline.named_steps['svd'].explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

def print_eval_metrics(pipeline, raw_news_filepath ='raw_news.csv'):
    clust_labels = pipeline.named_steps['clust'].labels_

    with codecs.open(raw_news_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        labels = [row[1] for row in reader]
        labels = [int(l) for l in labels[1:]]

    print("Homogeneity:", homogeneity_score(labels, clust_labels))
    print("Completeness:", completeness_score(labels, clust_labels))
    print("V-measure",  v_measure_score(labels, clust_labels))
    print("Adjusted Rand-Index:",  adjusted_rand_score(labels, clust_labels))
    #print(confusion_matrix(labels, clust_labels))
    #plotClusters(data_model)


def plotClusters(a):
    z = hac.linkage(a, method='ward')
    hac.dendrogram(z)
    plt.tight_layout()
    plt.show()

list_of_alg = [KMeans, AgglomerativeClustering, SpectralClustering]
for filename in glob.iglob('vectors_d4v__from_*.pickle', recursive=True):
    with open(filename, 'rb') as f:
        data_model = pickle.load(f)
        for alg in list_of_alg:
            print(alg, filename.replace('vectors_d4v__from', ''))
            pipeline = execute_pipeline(data_model, alg)
            svd_varience(pipeline)
            print_eval_metrics(pipeline)
            print('==============')
