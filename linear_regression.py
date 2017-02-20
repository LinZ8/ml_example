from sklearn import datasets
import numpy as np
import pandas as pd

#read in dataset from sklearn 
#boston housing price dataset
#CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n

def readindata():
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns = \
                             ['CRIM','ZN','INDUS','CHAS','NOX','RM','\
                             AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
    price = pd.Series(boston.target)
    n_df = len(boston_df)
    n_test = np.rint(n_df/3.0)
    n_train = np.int(n_df - n_test)
    boston_train = boston_df.sample(n_train)
    boston_test = boston_df.drop(boston_train.index)
    price_train = price[boston_train.index]
    price_test = price.drop(boston_train.index)
    return boston_train, boston_test, price_train, price_test

#update theta
def update_theta(theta,alfa,der_j):
    theta = theta - alfa*der_j
    return theta

def minsquare_j(h,y):
    cost_j = 0.
    for j in range(len(y)):
        cost_j = cost_j + 1./2./len(y)*(h[j] - y[j])**2.0
    return cost_j
    

def deri_j(h,theta,y,x):
    der_j = np.zeros(len(theta)) 
    for i in range(len(theta)):
        if i != 0:
            for j in range(len(x)):
                der_j[i] = der_j[i] + (h[j] - y[j])*x[j,i]
        else:
            for j in range(len(x)):
                der_j[i] = der_j[i] + (h[j] - y[j])          
    #print(der_j)
    der_j = 1./(len(x))*der_j
    #print(der_j)
    return der_j
            

def linear_regre(theta,x_train):
    return np.dot(theta,x_train.T)

    
#df_train, df_test, dfy_train, dfy_test = readindata()
house = [1400.,1600.,1700.,1875.,1100.,1550.,2350.,2450.,1425.,1700.]
price = [245.,312.,279.,308.,199.,219.,405.,324.,319.,255.]
df_train = pd.DataFrame(house)
dfy_train = pd.Series(price)
    
x_train = np.ones((len(df_train),len(df_train.columns)+1))
#x_test = np.ones((len(df_test),len(df_test.columns)+1))                 
x_train[:,1:] =np.array(df_train.values)
#x_test[:,1:] = np.array(df_test.values)
y_train = np.array(dfy_train.values)
#y_test = np.array(dfy_test.values)
#set the initial guess of parameters and the learning rate
#theta = np.ones(len(df_train.columns)+1)
theta = np.array([90,0.5])
der_j = np.zeros(len(df_train.columns)+1)
cost_j = 1.0
cost_j_hist = []
alfa = 0.001
h = linear_regre(theta,x_train)
minsquare_j(h,y_train)
cost_j_hist.append(cost_j)

for i in range(50):
    der_j = deri_j(h,theta,y_train,x_train)
    #break
    theta = update_theta(theta,alfa,der_j)
    h = linear_regre(theta,x_train)
    #break
    cost_j = minsquare_j(h,y_train)
    cost_j_hist.append(cost_j)

                           