import numpy as np
import random as rnd

class svm_smo():
    
    def __init__(self, c=1.0,maxiter = 1000, tol = 0.00001):
        self.c = c
        self.maxiter = maxiter
        self.tol = tol
    
    def linear_kernel(self, x1,x2):
        return np.dot(x1,x2.T)
    
    def clip_alpha(self, alpha, h,l):
        if alpha>= h:
            alpha = h
        elif alpha <= l:
            alpha = l
        return alpha
    
    def h_value(self, alphas,x_train,y_train,x,b):
        h = 0.0
        for i in range(x_train.shape[0]):
            h += alphas[i]*y_train[i]*self.linear_kernel(x_train[i,:], x)
        h += b
        return h
    
    def comput_l_h(self, alphai, alphaj, C, yi,yj):
        if yi != yj:
            l = max(0,alphaj-alphai)
            h = min(C,C+alphaj-alphai)
        else:
            l = max(0,alphaj+alphai-C)
            h = min(C,alphaj+alphai)
        return l, h
    
    def e_value(self, alphas,x_train,y_train,x,b,i):
        #print b
        #print self.h_value(alphas,x_train,y_train,x,b)
        return self.h_value(alphas,x_train,y_train,x,b)-y_train[i]
    
    def random_j(self,i,a,b):
        j =i
        while j ==i:
            j = rnd.randint(a,b)
        return j
   
    def calc_w(self,alphas,y_train,x_train):
        return np.dot(alphas*y_train,x_train)
    
    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        n = x_train.shape[1]
        alphas = np.zeros(m)
        b=0.0
        w = np.zeros(n)
        niter = 0
        while niter<self.maxiter:
            niter += 1
            alphas_prev = np.copy(alphas)
            for i in range(m):
                j = self.random_j(i,0,m-1)
                eta = self.linear_kernel(x_train[i,:],x_train[i,:])+ \
                      self.linear_kernel(x_train[j,:],x_train[j,:])- \
                      2.0*self.linear_kernel(x_train[i,:],x_train[j,:])
                if eta <= 0.0:
                    continue
                
                (l,h) = self.comput_l_h(alphas[i],alphas[j],self.c,y_train[i],y_train[j])
                if l ==h:
                    continue
                ei = self.e_value(alphas,x_train,y_train,x_train[i,:],b,i)
                ej = self.e_value(alphas,x_train,y_train,x_train[j,:],b,j)
                #print ei,ej
                alphaoldj = alphas[j].copy()
                alphaoldi = alphas[i].copy()
                
                alphas[j] = alphas[j] + y_train[j]*(ei-ej)/eta
                alphas[j] = self.clip_alpha(alphas[j],h,l)
                
                alphas[i] = alphas[i] + y_train[i]*y_train[j]*(alphaoldj-alphas[j])
                
                b1 = b - ei - y_train[i]*(alphas[i]-alphaoldi)*self.linear_kernel(x_train[i,:],x_train[i,:])- \
                     y_train[j]*(alphas[j]-alphaoldj)*self.linear_kernel(x_train[i,:],x_train[j,:]) 
                                                                              
                b2 = b - ej - y_train[i]*(alphas[i]-alphaoldi)*self.linear_kernel(x_train[i,:],x_train[j,:])- \
                     y_train[j]*(alphas[j]-alphaoldj)*self.linear_kernel(x_train[j,:],x_train[j,:])
                
                #print j
                if (alphas[i]>0.0) and (alphas[i]<self.c):
                    b = b1
                elif (alphas[j]>0.0) and (alphas[j]<self.c):  
                    b = b2
                else:
                    b = (b1+b2)/2.0    
                #print b
            diff = np.linalg.norm(alphas-alphas_prev)
            if diff<self.tol:
                print "niter = %i" % niter 
                break
        w = self.calc_w(alphas,y_train,x_train)
                                                                         
        return w, b
        

                                                                         
        