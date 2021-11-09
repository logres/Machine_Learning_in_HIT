import numpy as np
import random
from scipy.spatial.distance import euclidean

class Gaussion:
    '''高斯模型'''
    
    def __init__(self,N,cov) -> None:
        self.N = N
        self.means = np.zeros((N,))
        self.cov = np.eye(N)*cov
    
    def update(self,means,cov):
        '''更新均值、协方差'''
        self.means = means
        self.cov = cov
        return self
    
    def pdf(self,A):
        '''概率密度'''
        Z = A-self.means
        half = (np.power((2*np.pi),-self.N/2)*
                np.power(np.linalg.det(self.cov),-1/2))
        half2 = np.exp(-1/2*Z.T.dot(np.linalg.inv(self.cov)).dot(Z))
        return half*half2
    
    def means(self):
        '''返回均值'''
        return self.means

class My_GMM:
    
    def __init__(self,n_cluster,max_iter=300,cov=0.5) -> None:
        '''初始化'''
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.cov = cov
    
    @staticmethod
    def euclidean(A,B):
        '''欧氏距离'''
        vec_dist = A-B
        scale_distance = vec_dist.dot(vec_dist.T)
        return scale_distance
    
    def initial_gaussions(self):
        '''初始化高斯'''
        id = []
        first_id = random.sample(range(self.data_shape[0]), 1)
        id.append(first_id[0])
        center = self.samples[first_id]
        for j in range(self.n_cluster-1):
            all_distance = np.zeros(self.data_shape[0])
            for i in range(self.data_shape[0]):
                if not i in id:
                    all_distance[i] = euclidean(self.samples[i],center)
            dist_p = all_distance/all_distance.sum()
            next_id = np.random.choice(range(self.data_shape[0]), 1, p=dist_p)
            id.append(next_id[0])
            center = np.average(self.samples[id,:],axis=0)
        means = self.samples[id,:]
        gaussions = [Gaussion(self.data_shape[1],self.cov).update(means[i],np.eye(self.data_shape[1])* np.random.rand(1)) for i in range(self.n_cluster)]
        return gaussions        
    
    def update_phi(self):
        '''更新phi'''
        self.phi = self.weight.sum(axis=0)/self.data_shape[0]
    
    def update_Gaussions(self):
        '''更新高斯模型'''
        for j in range(self.n_cluster):
            gaussion = self.gaussions[j]
            means = self.samples.T.dot(self.weight[:,j])/(self.phi[j]*self.data_shape[0])
            Z = self.samples-means
            cov = np.dot(Z.T*self.weight[:,j], Z) / (self.phi[j]*self.data_shape[0])+0.001*np.eye(self.data_shape[1])
            gaussion.update(means,cov)
    
    def update_weight(self):
        '''更新权重'''
        for i in range(self.data_shape[0]):
            sample = self.samples[i]
            for j in range(self.n_cluster):
                gaussion = self.gaussions[j]
                self.weight[i,j]=gaussion.pdf(sample)*self.phi[j]
            self.weight[i,:] /= sum(self.weight[i,:])
    
    def Exception(self):
        '''期望'''
        self.update_weight()  
    
    def Maximum(self):
        '''最大化'''
        self.update_phi()
        self.update_Gaussions()
    
    def fit(self,X):
        '''拟合数据'''
        self.data_shape = X.shape
        self.samples = X.copy()
        self.gaussions = self.initial_gaussions()
        self.weight = np.ones((self.data_shape[0],self.n_cluster))/self.n_cluster
        self.phi = np.zeros((self.n_cluster,))
        self.update_phi()
        
        for i in range(self.max_iter):
            self.Exception()
            old_score = self.score()
            self.Maximum()
            if old_score == self.score():
                break
            
        self.labels = np.argmax(self.weight,axis=1)
    
    def score(self):
        '''评分'''
        score = 0
        for i in range(self.data_shape[0]):
            temp = 0
            for j in range(self.n_cluster):
                temp = temp + self.gaussions[j].pdf(self.samples[i])*self.phi[j]
            score = score + np.log(temp)
        return score/self.data_shape[0]
    
    def means(self):
        '''均值'''
        return  np.array([self.gaussions[i].means for i in range(self.n_cluster)])