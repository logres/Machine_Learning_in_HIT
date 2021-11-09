import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from My_Kmeans import My_Kmeans
from My_GMM import My_GMM
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster._kmeans import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import os

def make_blobs(n,means,covs,sample_numbers):
    '''生成高斯数据'''
    X = []
    y_true = []
    for i in range(n):
        x = multivariate_normal(means[i],covs[i],sample_numbers)
        X.append(x)
        y_true.append(np.ones(sample_numbers)*i)
    X = np.concatenate(X,axis=0)
    y_true = np.concatenate(y_true)
    return X,y_true

def show_correlation(components,name):
    '''展示原数据与主成分的相关性'''
    df_cm = pd.DataFrame(np.abs(components), columns=name)
    plt.figure(figsize = (8,4))
    ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
    # 设置y轴的字体的大小
    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_tick_params(labelsize=15)
    plt.title('PCA', fontsize='xx-large')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.show()

def k_means(k):
    '''Kmeans模型测试'''
    X, y_true = make_blobs(4,[[0,1],[2,3],[3,5],[-1,2]],\
        [[[0.1,0],[0,0.1]],[[0.8,0.2],[0.2,0.8]],[[0.3,0.4],[0.4,0.3]],[[0.5,0.3],[0.3,0.1]]],30)
    distortion = np.Infinity
    for i in range(10):
        k_means = My_Kmeans(k,300)
        k_means.fit(X)
        score = k_means.score()
        print(score)
        if distortion > score:
            distortion = score
            labels = k_means.labels
            centers = k_means.centroids
            
    color = ['b','g','r','orange','gray']
    plt.figure(1)
    for index in range(k):
        A = X[(labels==index),0]
        B = X[(labels==index),1]
        plt.scatter(A,B,c=color[index])
    plt.scatter(centers[:,0],centers[:,1])
    plt.title("My Kmeans")
    
    k_means = KMeans(k,max_iter=500)
    k_means.fit(X)
    labels = k_means.predict(X)
        
    color = ['b','g','r','orange','gray']
    plt.figure(2)
    for index in range(k):
        A = X[(labels==index),0]
        B = X[(labels==index),1]
        plt.scatter(A,B,c=color[index])
    plt.title("Standar Kmeans")

    plt.figure(3)
    for index in range(k):
        A = X[(y_true==index),0]
        B = X[(y_true==index),1]
        plt.scatter(A,B,c=color[index])   
    plt.title("origin data")
    plt.show()

def Gmm(k):
    '''高斯混合模型测试'''
    X, y_true = make_blobs(4,[[0,1],[2,3],[3,5],[-1,2]],\
        [[[0.1,0],[0,0.1]],[[0.8,0.2],[0.2,0.8]],[[0.3,0.4],[0.4,0.3]],[[0.5,0.3],[0.3,0.1]]],30)
    
    likelihood = -np.Infinity
    for i in range(3):
        gmm = My_GMM(k,500,cov=1)
        gmm.fit(X)
        score = gmm.score()
        print(score)
        if likelihood < score:
            likelihood = score
            labels = gmm.labels
            means = gmm.means()
    
    color = ['b','g','r','orange','gray']
    plt.figure(1)
    for index in range(k):
        A = X[(labels==index),0]
        B = X[(labels==index),1]
        plt.scatter(A,B,c=color[index])
    plt.scatter(means[:,0],means[:,1])
    plt.title("My GMM")
    
    gmm = GMM(n_components=4).fit(X)
    labels = gmm.predict(X)
    color = ['b','g','r','orange','gray']
    plt.figure(2)
    for index in range(k):
        A = X[(labels==index),0]
        B = X[(labels==index),1]
        plt.scatter(A,B,c=color[index])
    plt.title("Standar GMM")
    
    plt.figure(3)
    for index in range(k):
        A = X[(y_true==index),0]
        B = X[(y_true==index),1]
        plt.scatter(A,B,c=color[index])
    plt.title("origin data")
    plt.show()
    
def UCI():
    '''UCI数据测试'''
    data = pd.read_csv(os.path.join(os.getcwd(),"tripadvisor_review.csv"),header=0)
    feature_name = data.columns[1:]
    X = np.array(data.iloc[:,1:])

    pca = PCA(3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    component = pca.components_
    new_X = pca.transform(X)


    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')

    color = ['b','g','r','orange','gray']
    gmm_ = My_GMM(4,400)
    gmm_.fit(new_X)
    labels_ = gmm_.labels
    means = gmm_.means()

    for i in range(4):
        ax.scatter(new_X[(labels_==i),0],new_X[(labels_==i),1],new_X[(labels_==i),2],c=color[i])

    plt.show()
    show_correlation(components=component,name=feature_name)

    attrs = pca.inverse_transform(means)
    plt.xticks(ticks=range(10),labels=["art galleries","dance clubs","juice bars","restaurants","museums",\
        "resorts","parks/picnic spots","beaches","theaters","religious institutions"])
    for i in range(len(means)):
        attr = attrs[i]
        plt.scatter(range(10),attr,c = color[i])
    plt.show()

k_means(4)
Gmm(4)
UCI()
