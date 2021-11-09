import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import os
from sklearn.metrics import confusion_matrix


def generate_data(mean1,cov1,mean2,cov2,size,name):
    '''生成数据集'''
    data1 = np.random.multivariate_normal(mean=mean1,cov=cov1,size = size)
    data2 = np.random.multivariate_normal(mean=mean2,cov=cov2,size = size)
    data1 = np.c_[data1,np.ones(len(data1))]
    data2 = np.c_[data2,np.zeros(len(data2))]
    plt.scatter(data1[:,0],data1[:,1])
    plt.scatter(data2[:,0],data2[:,1])
    plt.title(name)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()
    plt.scatter(data1[:,0],data1[:,1])
    plt.scatter(data2[:,0],data2[:,1])
    data = np.append(data1,data2,axis=0)
    feature = np.append(np.ones((len(data),1)),data[:,0:-1],axis=1)
    value = data[:,-1].reshape(-1,)
    return feature,value

def generate_theta(dimesion):
    '''生成theta初始值'''
    theta = np.array([value + np.random.normal(loc=0,scale=0.01) \
        for value in np.zeros(shape = (dimesion,))])
    return theta    

def sigmoid(Z):   
    '''做了防溢出处理的sigmoid'''
    mask = (Z > 0)
    positive_out = np.zeros_like(Z, dtype='float64')
    negative_out = np.zeros_like(Z, dtype='float64')
    positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
    positive_out[~mask] = 0
    expZ = np.exp(Z,negative_out,where=~mask)
    negative_out = expZ / (1+expZ)
    negative_out[mask] = 0
    return positive_out + negative_out
    
def logistic_gradient(X,y,theta,_lambda):
    '''逻辑回归的梯度函数'''
    m = X.shape[0]
    return 1/m*(X.T.dot(sigmoid(X.dot(theta))-y)\
        +_lambda*theta)

def logistic_cost(X,y,theta,_lambda):
    '''逻辑回归的代价函数'''
    m = X.shape[0]
    return 1/m*(-y.T.dot(np.log(sigmoid(X.dot(theta))))-\
                (1-y).T.dot(np.log(1-sigmoid(X.dot(theta)))))+_lambda/(2*m)*theta.T.dot(theta)

def batch_gradient_descent(X,Y,theta,_lambda,alpha,iter_num,accuracy,gradient,cost,is_obvious):
    '''实现批量梯度下降'''
    c = cost(X,Y,theta,_lambda)
    cost_history = []
    new_theta = theta.copy()
    for i in range(0,iter_num):
        new_theta = new_theta - alpha * gradient(X,Y,new_theta,_lambda)
        if cost(X,Y,new_theta,_lambda)<c:
            old_c = c
            c = cost(X,Y,new_theta,_lambda)
            cost_history.append(c)
            if old_c - c<accuracy:
                break
        else:
            alpha=alpha/1.1
        if is_obvious:
            if i%10==0:
                print(alpha,c)
    return new_theta,cost_history

def Show_Confusion_Matrix(classes,confusion):
    '''绘制混淆矩阵'''
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('True_Label')
    plt.title("混淆矩阵热度图")
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index],fontsize=7)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

def predict(feature,theta):
    '''预测函数'''
    res = sigmoid(feature.dot(theta))
    prediction = np.zeros(len(feature))
    for i in range(0,len(res)):
        if(res[i]>=0.5):
            prediction[i]=1
    return prediction

def bounder_y(x,theta):
    '''打印决策边界'''
    return (-theta[0]-theta[1]*x)/theta[2] 

def simple_test():
    '''高斯数据实验（贝叶斯假设/非贝叶斯假设）'''
    Gaussian_data_test([0,2],[[1.5,0],[0,0.8]],[2,1],[[0.8,0],[0,2]],100,0,"贝叶斯假设")
    Gaussian_data_test([0,2],[[1.5,0],[0,0.8]],[2,1],[[0.8,0],[0,2]],100,0.01,"贝叶斯假设-正则lambda0.01")
    Gaussian_data_test([0,2],[[2,1],[1,2]],[2,1],[[1,0.5],[0.5,1]],100,0,"非贝叶斯假设")

def Gaussian_data_test(mean1,cov1,mean2,cov2,size,_lambda,title):
    '''高斯数据集实验'''
    feature,value=generate_data(mean1,cov1,mean2,cov2,size,"训练集")
    theta = generate_theta(3)
    theta1,cost_history = batch_gradient_descent(feature,value,theta,_lambda,0.01,100000000,0.000001,logistic_gradient,logistic_cost,True)
    interval = np.linspace(-3,3,100)
    plt.plot(interval,[bounder_y(x,theta1) for x in interval])
    plt.title(title+"对训练集拟合成果")
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()
    plt.cla()
    generate_data(mean1,cov1,mean2,cov2,size,"测试集")
    plt.plot(interval,[bounder_y(x,theta1) for x in interval])
    plt.rcParams['axes.unicode_minus']=False
    plt.title(title+"在测试集效果")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()
    plt.cla()
    
def UCI_test():
    '''UCI数据集实验'''
    feature,label = sk.load_svmlight_file(os.path.join(os.getcwd(),"farm-ads-vect"))#使用特定库解析文本形式的数据集
    feature = feature.toarray()
    for i in range(0,len(label)): #修正-1标签为0
        if label[i]==-1:
            label[i]=0
    feature = np.append(np.ones((len(feature),1)),feature,axis=1)
    theta = generate_theta(len(feature.T))
    training_set = (feature[0:3000],label[0:3000])
    test_set = (feature[3000:],label[3000:])
    #train
    theta,cost= batch_gradient_descent(training_set[0],training_set[1],theta,0,1,1000000000,0.00001,logistic_gradient,logistic_cost,True)
    #test
    prediction=predict(test_set[0],theta)
    #show
    confuse = confusion_matrix(test_set[1],prediction,labels=[0,1])
    Show_Confusion_Matrix(["accept","reject"],confuse)


if __name__ == "__main__":
    simple_test()
    UCI_test()
    