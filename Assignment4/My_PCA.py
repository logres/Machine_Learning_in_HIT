import numpy as np

class PCA:
    '''主成分分析'''
    
    def __init__(self,n_components):
        self.n_components = n_components
    
    def fit(self,X):
        '''拟合数据'''
        self.m = len(X)
        self.n = len(X.T)
        self.mean = np.mean(X,axis=0)
        st_X = X-self.mean
        eigen_value,eigen_vector = self.eigen(st_X)#求特征值和特征向量
        '''print(eigen_value)
        print(eigen_vector)
        print("\n")'''
        sortIndex = np.flipud( np.argsort(eigen_value) ) #特征值排序
        self.explained_variance_ = eigen_value[sortIndex[0:self.n_components]]
        self.explained_variance_ratio_ = eigen_value[sortIndex[0:self.n_components]]/np.sum(eigen_value)
        self.components= eigen_vector[sortIndex[0:self.n_components]]
    
    def eigen(self,X):
        '''计算特征向量、值'''
        Covariance_matrix = np.cov(X,rowvar=False)
        eigen_value,eigen_vector = np.linalg.eig(Covariance_matrix)
        return np.real(eigen_value),np.real(eigen_vector.T)
    
    def transform(self,X):
        '''降维'''
        X = X-self.mean
        Y = X.dot(self.components.T)
        return Y
    
    def inverse_transform(self,Y):
        '''重构'''
        if len(Y.shape)>1:
            X = np.zeros((Y.shape[0],self.n))
            for i in range(Y.shape[0]):
                for j in range(self.n_components):
                    X[i] += Y[i,j]*self.components[j]
        else:
            X = np.zeros((self.n,))
            for i in range(self.n_components):
                X += Y[i]*self.components[i]
        X = X+self.mean
        return X    