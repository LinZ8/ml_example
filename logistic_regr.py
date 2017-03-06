from sklearn import datasets
import numpy as np
import pandas as pd

#'Iris Plants Database\n\nNotes\n-----\nData Set Characteristics:\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThis is a copy of UCI ML iris datasets.\nhttp://archive.ics.uci.edu/ml/datasets/Iris\n\nThe famous Iris database, first used by Sir R.A Fisher\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\nReferences\n----------\n   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...\n'

#hypothesis for logistic regression
#return is the hypothesis in form of np array (m)
def log_regr(theta,x):
    g = np.dot(theta,x.T)
    return 1.0/(1.0+np.exp(-g))

#cost function (or minus max liklihood function)
#return is loss function J in form of a double number
def cost(h,y):
    cost_j = 0.
    m = len(y)
    for j in range(m):
        cost_j = -1.0/m*(y[j]*np.log(h[j])+(1-y[j])*np.log(1-h[j]))
    return cost_j

#derivative of loss function J.
#return is in form of a numpy array (n+1)
def deri_j(h,theta,y,x):
    der_j = np.zeros(len(theta)) 
    for i in range(len(theta)):
        for j in range(len(x)):
            der_j[i] = der_j[i] + (y[j] - h[j])*x[j,i]      
    der_j = -1./(len(x))*der_j
    return der_j

#update theta
#return is in form of numpy array (n+1)
def update_theta(theta,alfa,der_j):
    theta = theta - alfa*der_j
    return theta

#readin data from sklearn iris dataset
#use 1/3 of dataset as test set, 2/3 as train set
#use pandas dataframe as the store approach in the first step
iris = datasets.load_iris()
total_x = pd.DataFrame(iris.data[:,:2])
total_y = pd.Series(iris.target)
n_x = len(total_x)
n_test = np.rint(n_x/3.0)
n_train = np.int(n_x-n_test)
x_train_df = total_x.sample(n_train)
x_test_df = total_x.drop(x_train_df.index)
y_train_df = total_y[x_train_df.index]
y_test_df = total_y.drop(x_train_df.index)

#target value is a three classifications' dataset (0,1,2)
#for now, test seperating 0 and others
#consider all target of 2 as 1
y_train_no2 = y_train_df.replace(2,1)
y_test_no2 = y_test_df.replace(2,1)

#setup the initial value
theta = np.zeros(len(x_train_df.columns)+1)
der_j = np.zeros(len(x_train_df.columns)+1)
cost_j_hist = []
alfa = 0.3
x_train = np.ones((len(y_train_df),len(theta)))
x_test = np.ones((len(y_test_df),len(theta)))
x_train[:,1:] = np.array(x_train_df)
x_test[:,1:] = np.array(x_test_df)
y_train = np.array(y_train_no2)
y_test = np.array(y_test_no2)

#main calculation
h = log_regr(theta,x_train)
cost_j = cost(h,y_train)
cost_j_hist.append(cost_j)
for i in range(3000):
    der_j = deri_j(h,theta,y_train,x_train)
    #break
    theta = update_theta(theta,alfa,der_j)
    h = log_regr(theta,x_train)
    #break
    cost_j = cost(h,y_train)
    cost_j_hist.append(cost_j)

#for ploting the decision boundary line given a 2D case.    
x1 = np.zeros(10)
x2 = np.zeros(10)
for i in range(10):
    x1[i] = i
    x2[i] = (theta[0]+theta[1]*x1[i])/(-theta[2])
    
























