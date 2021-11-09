from My_PCA import PCA
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk

from PIL import Image

def PSNR(source,result,max_value):
    '''峰值信噪比'''
    mse = np.mean((source-result)**2)
    psnr = 10*np.log10(max_value**2/mse)
    return psnr

def test1():
    '''二维降一维'''
    X = multivariate_normal([-1,4],[[1,0],[0,0.2]],size = 30)
    pca = PCA(1)
    pca.fit(X)
    Y = pca.transform(X)
    new_X = pca.inverse_transform(Y)
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1])#原始数据
    plt.scatter(new_X[:,0],new_X[:,1])#构成数据
    vector = pca.components[0]
    vector_to_draw = np.vstack([vector*-5,vector*5])+pca.mean
    plt.plot(vector_to_draw[:,0],vector_to_draw[:,1])
    plt.show()
    
def test2():
    '''三维降二维'''
    X = multivariate_normal([-1,4,5],[[1,0,0],[0,1,0],[0,0,0.01]],size = 100)
    pca = PCA(2)
    pca.fit(X)
    Y = pca.transform(X)
    new_X = pca.inverse_transform(Y)
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2])#原始数据
    ax.scatter(new_X[:,0],new_X[:,1],new_X[:,2])#构成数据
    vector = pca.components[0]
    vector_to_draw = np.vstack([vector*-5,vector*5])+pca.mean
    ax.plot(vector_to_draw[:,0],vector_to_draw[:,1],vector_to_draw[:,2])
    vector = pca.components[1]
    vector_to_draw = np.vstack([vector*-5,vector*5])+pca.mean
    ax.plot(vector_to_draw[:,0],vector_to_draw[:,1],vector_to_draw[:,2])
    plt.show()
    
def Face_PCA_training(n_componets):
    '''动漫头像模型训练'''
    sample_directory = os.path.join(os.getcwd(),"trainning_sample")#训练样本文件夹
    model_dirctory = os.path.join(os.getcwd(),"model")#模型文件夹
    file_list = os.listdir(sample_directory)
    X=[]
    for file in file_list:
        im = Image.open(os.path.join(sample_directory,file))
        X.append(np.array(im).reshape((-1,)))
    X = np.array(X).reshape((len(file_list),-1))
    pca = PCA(n_components=n_componets)
    pca.fit(X)
    with open(os.path.join(model_dirctory,"PCA{}".format(n_componets)),"wb") as f:
        pk.dump(pca,f)
    
def Face():
    '''动漫头像测试'''
    test_directory = os.path.join(os.getcwd(),"test_sample")#测试样本文件夹
    model_directory = os.path.join(os.getcwd(),"model")#模型文件夹
    sample_list = os.listdir(test_directory)
    model_list = os.listdir(model_directory)
    sample_dict = {}
    model_dict = {}
    for sample in sample_list:
        im = Image.open(os.path.join(test_directory,sample))
        sample_dict[sample] = np.array(im).reshape((-1,))
    for model in model_list:
        with open(os.path.join(model_directory,model),"rb") as f:
            model_dict[model] = pk.load(f)
    model_list = ["PCA1","PCA3","PCA5","PCA10","PCA20","PCA40","PCA80","PCA100","PCA200",\
        "PCA300","PCA1000"]
    psnrs = {}
    result_directory = os.path.join(os.getcwd(),"result")
    for sample in sample_list:
        X = sample_dict[sample]
        psnr = []
        for model in model_list:
            pca = model_dict[model]
            Y = pca.transform(X.reshape((1,-1)))
            '''print(sample,model,Y)'''
            new_X = pca.inverse_transform(Y)
            psnr.append(PSNR(X,new_X,255))
            new_X = new_X.reshape((50,50))  
            im = Image.fromarray(new_X)
            im = im.convert('L')
            im.save(os.path.join(result_directory,sample.replace(".jpg","_")+model+".jpg"))
        psnrs[sample]=psnr
    
    plt.figure(0)
    for k in psnrs.keys():
        plt.plot(range(len(psnrs[k])),psnrs[k],label = k)
    plt.xticks(range(len(psnrs[k])),model_list)
    plt.legend()
    plt.xlabel("Model")
    plt.ylabel("PSNR (dB)")
    plt.show()

if __name__ == "__main__":
    test1()
    test2()
    '''for n in [1,3,5,10,20,40,80,100,200,300,1000]:
        Face_PCA_training(n)'''#训练用代码
    Face()
    