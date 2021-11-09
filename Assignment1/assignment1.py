import matplotlib.pyplot as plt
import numpy as np

#初始化Theta
def get_theta(n):
    return np.array([value + np.random.normal(loc=0,scale=1) \
        for value in np.zeros(shape = (n,))])

#计算代价
def cost(X,y,theta,_lambda):
    return 1/(2*len(y))*((X.dot(theta)-y).T.\
        dot((X.dot(theta)-y).T)+_lambda*theta.T.dot(theta))

#获取数据
def get_data(sample_number,degree,is_scatter=True,function=np.sin):
    base = np.linspace(start = 0, stop= 2, \
        num = sample_number, endpoint = True)
    Y = np.array([_y + np.random.normal(loc=0,scale=0.2) \
        for _y in function(base*np.pi)])
    X = np.array([[np.power(x,l) for l in range(0,degree+1)] for x in base])
    if(is_scatter):
        plt.scatter(base,Y)
    return X,Y

#假设方程
def hypnosis_X(theta):
    def hypnosis(x):
        return np.dot(np.power(x,range(0,len(theta))).T,theta)
    return hypnosis

def normal_equation(X,Y,_lambda):
    return np.linalg.inv(X.T.dot(X)+_lambda*np.eye(X.shape[1])).dot(X.T).dot(Y)

#求梯度
def gradient(theta, X, Y,_lambda):
    m = X.shape[0]
    gradient = X.T.dot((X.dot(theta) - Y)) + _lambda*theta
    return gradient/m

#批量梯度下降
def batch_gradient_descent(X,Y,theta,_lambda,alpha,iter_num,accuracy,gradient,cost,is_obvious):
    c = cost(X,Y,theta,_lambda)
    cost_history = []
    new_theta = theta.copy()
    for i in range(0,iter_num):
        new_theta = new_theta - alpha * gradient(new_theta,X,Y,_lambda)
        if cost(X,Y,new_theta,_lambda)<c:
            old_c = c
            c = cost(X,Y,new_theta,_lambda)
            cost_history.append(c)
            if old_c - c<accuracy:
                break
        else:
            alpha=alpha/1.2
        if is_obvious:
            if i%10000==0:
                print(alpha,c)
    return new_theta,cost_history

#共轭梯度下降
def conjugate(X,Y,_lambda,accuracy):
    A = X.T.dot(X)+_lambda*np.eye(X.shape[1])
    b = X.T.dot(Y)
    x = np.zeros(shape=(X.shape[1],))
    r = b
    p = r
    while True:
        alpha = r.T.dot(r) / (p.T.dot(A).dot(p))
        x = x + alpha*p
        next_r = r - alpha * A.dot(p)
        if next_r.T.dot(next_r) < accuracy:
            break
        beta = next_r.T.dot(next_r)/(r.T.dot(r))
        p = next_r + beta * p
        r = next_r
    return x

#绘图
def plot(start,end,step,type,functions,title,single_deal = True):
    x = np.arange(start,end,step)
    if type =="plot":
        for function,name in functions.items():
            plt.plot(x,[function(_x) for _x in x],label= name)
            plt.title(title)
        if single_deal:
            plt.legend(loc = "best")
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False 
            plt.show()

#正规方程
def Normal_1(X,Y):
    theta1 = normal_equation(X,Y,0)
    theta2 = normal_equation(X,Y,0.1)
    theta3 = normal_equation(X,Y,0.5)
    plot(0,2,0.001,"plot",{hypnosis_X(theta1):"normal_equation: lambda = 0",\
                            hypnosis_X(theta2):"normal_equation: lambda = 0.1",\
                            hypnosis_X(theta3):"normal_equation: lambda = 0.5",\
                            lambda x:np.sin(x*np.pi):"sin"},"正规方程拟合曲线")
    
#第二次正规方程
def Normal_2(X,Y):
    theta1 = normal_equation(X,Y,0)
    theta2 = normal_equation(X,Y,0.0007)
    plot(0,2,0.001,"plot",{hypnosis_X(theta1):"normal_equation: lambda = 0",\
                            hypnosis_X(theta2):"normal_equation: lambda = 0.0007",\
                            lambda x:np.sin(x*np.pi):"sin"},"正规方程拟合曲线")


#梯度下降实验
def Grad_(X,Y,accuracy):#13 9 7
    theta1,_ = batch_gradient_descent(X,Y,get_theta(10),\
        0,0.0001,10000000000,accuracy,gradient,cost,True)
    theta2,_ = batch_gradient_descent(X,Y,get_theta(10),\
        0.0007,0.0001,10000000000,accuracy,gradient,cost,True)
    theta3 = conjugate(X,Y,0,accuracy)
    theta4 = conjugate(X,Y,0.0007,accuracy)
    plot(0,2,0.001,"plot",{hypnosis_X(theta1):"batch_gradient: lambda = 0",\
                            hypnosis_X(theta2):"batch_gradient: lambda = 0.0007",\
                            hypnosis_X(theta3):"conjugate: lambda = 0",\
                            hypnosis_X(theta4):"conjugate: lambda = 0.0007",\
                            lambda x:np.sin(x*np.pi):"sin"},"优化方法")

#探究最佳lambda
def lambda_chooser(X,Y):
    base = np.linspace(0,0.01,30,endpoint=True)
    test_x = np.arange(0,2,0.0001)
    test_y = np.sin(test_x*np.pi)
    costs = []
    for _lambda in base:
        theta = normal_equation(X,Y,_lambda)
        costs.append(cost(np.array([[np.power(_x,l)\
            for l in range(0,10)] for _x in test_x]),test_y,theta,0))
    plt.plot(base,costs,label = "cost of lambda")
    plt.xlabel("lambda")
    plt.ylabel("cost")
    plt.title("lambda 与 cost 关系图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False 
    plt.show()
    
#对比实验
def compare_(degrees,sample_numbers):
    
    x_len = len(degrees)
    y_len = len(sample_numbers)
    accuracy = 0.000000001
    
    plt.subplots(x_len,y_len,sharex = True, sharey = True)
    for degree in degrees:
        for sample_number in sample_numbers:
            plt.subplot(x_len,y_len,degrees.index(degree)*y_len+sample_numbers.\
                index(sample_number)+1)
            X,Y = get_data(sample_number,degree,True)
            theta1,_ = batch_gradient_descent(X,Y,get_theta(degree+1),\
                0,0.0001,10000000000,accuracy,gradient,cost,True)
            theta2,_ = batch_gradient_descent(X,Y,get_theta(degree+1),\
                0.0007,0.0001,10000000000,accuracy,gradient,cost,True)
            theta3 = conjugate(X,Y,0,accuracy)
            theta4 = conjugate(X,Y,0.0007,accuracy)
            theta5 = normal_equation(X,Y,0)
            theta6 = normal_equation(X,Y,0.0007)
            plot(0,2,0.001,"plot",{hypnosis_X(theta1):"BD",\
                                    hypnosis_X(theta2):"BDR",\
                                    hypnosis_X(theta3):"CGD",\
                                    hypnosis_X(theta4):"CGDR",\
                                    hypnosis_X(theta5):"NE",\
                                    hypnosis_X(theta6):"NER",\
                                    lambda x:np.sin(x*np.pi):"sin"},\
                                        "对比试验：degree:{},sample_number:{}"\
                                        .format(degree,sample_number),False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.subplot(x_len,y_len,1)
    plt.legend(bbox_to_anchor=(0,0.8,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
    plt.show()

if __name__ == "__main__":
    
    #第一次解析解
    X,Y = get_data(20,9)
    Normal_1(X,Y)
    plt.cla()
    
    #求解最佳lambda
    '''lambda_chooser(X,Y)'''
    
    #使用最佳lambda求解析解
    X,Y = get_data(20,9)
    Normal_2(X,Y)
    plt.cla()
    
    #梯度下降，精度0.00001
    X,Y = get_data(20,9)
    Grad_(X,Y,0.00001)
    plt.cla()
    
    #梯度下降，精度0.000000001
    X,Y = get_data(20,9)
    Grad_(X,Y,0.0000000001)
    plt.cla()

    #对比试验
    degrees = [3,5,9]
    sample_numbers = [20,60,120]
    compare_(degrees,sample_numbers)