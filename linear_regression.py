from sklearn import datasets
import numpy as np
import pandas as pd

#read in dataset from sklearn 
#boston housing price dataset
#CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n


#readin from sklearn boston house price dataset,only the DIS coloumn is input.
#data are seperated into two sets (train data 66.6% and test data 33.3%)
#outputs are in the form pandas dataframe and series
def readindata():
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data[:,7]) 
                             #columns = \
                             #['CRIM','ZN','INDUS','CHAS','NOX','RM','\
                             #AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
    price = pd.Series(boston.target)
    n_df = len(boston_df)
    n_test = np.rint(n_df/3.0)
    n_train = np.int(n_df - n_test)
    boston_train = boston_df.sample(n_train)
    boston_test = boston_df.drop(boston_train.index)
    price_train = price[boston_train.index]
    price_test = price.drop(boston_train.index)
    return boston_train, boston_test, price_train, price_test

#update theta using derivative of j and learning rate alfa
#return is the theta value in form of array or numpy array (n+1)
def update_theta(theta,alfa,der_j):
    theta = theta - alfa*der_j
    return theta

#evaluate loss function J, using minimum square method
#return is the loss function J in form of a double number
def minsquare_j(h,y):
    cost_j = 0.
    for j in range(len(y)):
        cost_j = cost_j + 1./2./len(y)*(h[j] - y[j])**2.0
    return cost_j
    
#evaluate derivative of loss function J.
#return is is the derivative of loss function in form a numpy array (n+1)
def deri_j(h,theta,y,x):
    der_j = np.zeros(len(theta)) 
    for i in range(len(theta)):
        for j in range(len(x)):
            der_j[i] = der_j[i] + (h[j] - y[j])*x[j,i]      
    der_j = 1./(len(x))*der_j
    return der_j
            
#evaluate the hypothesis using linear regression method
#return is the hypothesis in form of np array (m)
def linear_regre(theta,x_train):
    return np.dot(theta,x_train.T)

#normalization method
#input could be any vector
#output is the normalized vector
def normalize(x):
    x = (x - x.mean())/(x.max()-x.min())
    return x

#denormalization method
#input the normalized theta value and original dataset
#output the denormalized theta value in form of numpy array (n+1)
def theta_denormal(theta,x,y):
    theta_deno =np.zeros(len(theta))
    for i in range(len(theta)-1):
        theta_deno[i+1] = theta[i+1]*(y.max()-y.min())/(x[:,i+1].max()-x[:,i+1].min())
        theta_deno[0] = theta_deno[0] - theta[i+1]*x[:,i+1].mean()/(x[:,i+1].max()-
                                                                    x[:,i+1].min())
    theta_deno[0] = (theta[0]+theta_deno[0])*(y.max()-y.min())+y.mean()
    return theta_deno
    
#one simple test value
#house = [1400.,1600.,1700.,1875.,1100.,1550.,2350.,2450.,1425.,1700.]
#price = [245.,312.,279.,308.,199.,219.,405.,324.,319.,255.]
#df_train = pd.DataFrame(house)
#dfy_train = pd.Series(price)

#read in data
df_train, df_test, dfy_train, dfy_test = readindata()

#tranfer data into numpy array 
x_train = np.ones((len(df_train),len(df_train.columns)+1))
x_train[:,1:] =np.array(df_train.values)
y_train = np.array(dfy_train.values)
x_test = np.ones((len(df_test),len(df_test.columns)+1))                 
x_test[:,1:] = np.array(df_test.values)
y_test = np.array(dfy_test.values)

#set the initial guess of parameters and the learning rate
theta = np.ones(len(df_train.columns)+1)
der_j = np.zeros(len(df_train.columns)+1)
cost_j_hist = []
alfa = 0.3
x_train_norm = np.ones((len(y_train),len(theta)))
y_train_norm = np.ones(len(y_train))

#normalize each column of dataset
for i in range(len(theta)-1):
    x_train_norm[:,i+1] = normalize(x_train[:,i+1])
y_train_norm[:] = normalize(y_train)

#iteration to find minimum J
h = linear_regre(theta,x_train_norm)
cost_j = minsquare_j(h,y_train_norm)
cost_j_hist.append(cost_j)
for i in range(1000):
    der_j = deri_j(h,theta,y_train_norm,x_train_norm)
    #break
    theta = update_theta(theta,alfa,der_j)
    h = linear_regre(theta,x_train_norm)
    #break
    cost_j = minsquare_j(h,y_train_norm)
    cost_j_hist.append(cost_j)

#denormalize the theta and evaluate the hypothesis    
theta_denor = theta_denormal(theta,x_train,y_train)
y_denor = linear_regre(theta_denor,x_test)
loss = minsquare_j(y_denor,y_test)
                           