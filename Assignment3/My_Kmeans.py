import numpy as np
from scipy.spatial.distance import cdist
import random


class My_Kmeans:
    '''自己实现的Kmeans聚类模型'''
    
    def __init__(self,n_cluster,max_iter=300,distance = None) -> None:
        '''初始化'''
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        if distance == None:
            self.distance = self.euclidean
        else:
            self.distance = distance
    
    '''欧氏距离'''
    @staticmethod
    def euclidean(A,B):
        vec_dist = A-B
        scale_distance = vec_dist.dot(vec_dist.T)
        return scale_distance
    
    def initial_centroids(self):
        '''初始化中心点'''
        id = []
        first_id = random.sample(range(self.data_shape[0]), 1)
        id.append(first_id[0])
        center = self.samples[first_id]
        for j in range(self.n_cluster-1):
            all_distance = np.zeros(self.data_shape[0])
            for i in range(self.data_shape[0]):
                if not i in id:
                    all_distance[i] = self.distance(self.samples[i],center)
            dist_p = all_distance/all_distance.sum()
            next_id = np.random.choice(range(self.data_shape[0]), 1, p=dist_p)
            id.append(next_id[0])
            center = np.average(self.samples[id,:],axis=0)
        centroids = self.samples[id,:]
        return centroids
    
    def Exception(self):
        '''E步给样本贴标签'''
        for k in range(self.data_shape[0]):
            sample = self.samples[k]
            nearest_centroid = -1
            nearest_distance = 0
            for i in range(self.n_cluster):
                center = self.centroids[i]
                dist = self.distance(sample,center)
                if dist < nearest_distance or nearest_centroid==-1:
                    nearest_centroid = i
                    nearest_distance = dist
            self.labels[k] = nearest_centroid            
    
    def Maximum(self):
        '''最大化似然'''
        for i in range(self.n_cluster):
            self.centroids[i] = np.average(self.samples[self.labels == i],axis=0)
    
    def fit(self,X):
        '''拟合数据'''
        self.data_shape = X.shape
        self.samples = X.copy()
        self.centroids = self.initial_centroids()
        self.labels = np.zeros((len(self.samples),))
        
        for i in range(self.max_iter):
            self.Exception()
            old_score = self.score()
            self.Maximum()
            if old_score == self.score():
                break
    
    def score(self):
        '''评价标准 mean distortion'''
        return sum(np.min(cdist(self.samples, self.centroids, 'euclidean'), axis=1))/self.data_shape[0]
