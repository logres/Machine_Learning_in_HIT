import matplotlib.pyplot as plt
import numpy as np

def Predict(x,Theta,level):
    return np.dot(np.power(x,level).T,Theta)

def normal_equation(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T.dot((X.dot(theta) - y)) 
    return inner / m

def gradient_descent(X,y,iter_num,Theta,alpha):
    for i in range(0,iter_num):
        print(cost(X,y,Theta,False))
        gradient = X.T.dot(X.dot(Theta)-y)
        Theta = Theta - (alpha/len(y)) * gradient
    return Theta

def batch_gradient_decent(X, y, epoch,theta, alpha=0.01):
    _theta = theta.copy() 
    c = cost(X,y,_theta)
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        if cost(X,y,_theta)<0.001:
            break
        if cost(X,y,_theta)<c:
            c = cost(X,y,_theta)
        else:
            alpha=alpha/2
        '''if _%1000==0:
            print(c)'''
    return _theta

def cost(X,y,Theta):
    return 1/(2*len(y))*(X.dot(Theta)-y).T.dot((X.dot(Theta)-y).T)

base = np.linspace(start = 0, stop= 2, num = 100, endpoint = True)
y = np.sin(base*np.pi)
y = [_y + np.random.normal(loc=0,scale=0.1) for _y in y]
y = np.array(y)

level = [0,1,2,3,4,5,6,7,8,9]

X = np.array([[np.power(x,l) for l in level] for x in base])
Theta1 = normal_equation(X,y)

Theta2 = np.zeros(shape = (10,))
Theta2 = [value + np.random.normal(loc=0,scale=1) for value in Theta2]
Theta2 = np.array(Theta2)

print(Theta2)

Theta2 = batch_gradient_decent(X,y,100000000,Theta2,0.0001)

print(Theta1)
print(Theta2)

a = np.arange(0,2,0.001)
b1 = np.array([Predict(value,Theta1,level=level) for value in a])
b2 = np.array([Predict(value,Theta2,level=level) for value in a])
c = np.array([np.sin(value*np.pi) for value in a])

print(cost(X,y,Theta1))

plt.scatter(base,y)
plt.plot(a,b1)
plt.plot(a,b2)
#plt.plot(a,c)
plt.show()










