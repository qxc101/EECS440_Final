#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import scipy.optimize as optimize


# In[1]:


class GLAD:
    def __init__(self, label_Matrix):
        self.label_Matrix = label_Matrix.copy()
        self.I = label_Matrix.shape[0]
        self.J = label_Matrix.shape[1]
        self.alpha = np.random.randn(self.I)
        self.beta = np.ones((self.J, ))
        
    def eStep(self):
        self.posterior = np.empty((2,self.J))
        for j in range(self.J):
            index = (self.label_Matrix[:, j] != -1)
            labelsJ = self.label_Matrix[index,j]
            self.posterior[0,j] = np.prod(bern(logistic(self.alpha[index] * self.beta[j]), 1-labelsJ))
            self.posterior[1,j] = np.prod(bern(logistic(self.alpha[index] * self.beta[j]), labelsJ))
        self.posterior /= np.sum(self.posterior, axis = 0)
        
    def mStep(self):
        self.updatetb()
        self.obj = -self.objm(np.hstack((self.alpha,np.log(self.beta))))
        
    def updatetb(self):
        def grad(theta):
            alpha = theta[:self.I]
            beta = np.exp(theta[-self.J:])
            grad_alpha = np.zeros((self.I,))
            grad_beta = np.zeros((self.J,))
            for j in range(self.J):
                index = (self.label_Matrix[:, j] != -1)
                labelsJ = self.label_Matrix[index,j]
                temp = logistic(alpha[index]*beta[j])
                grad_alpha[index] = grad_alpha[index] + beta[j] * (labelsJ * self.posterior[1,j] + (1-labelsJ) * self.posterior[0,j] - temp)
                grad_beta[j] = beta[j] * np.sum(alpha[index] * (labelsJ * self.posterior[1,j] + (1-labelsJ) * self.posterior[0,j] - temp))
            return -np.hstack((grad_alpha, grad_beta))
        thetai = np.hstack((self.alpha, np.log(self.beta)))
        thetah, _, d = optimize.fmin_l_bfgs_b(self.objm, thetai, fprime=grad, disp=0)
        self.alpha = thetah[:self.I]
        self.beta = np.exp(thetah[-self.J:])
        
    def objm(self, theta):
        alpha = theta[:self.I]
        beta = np.exp(theta[-self.J:])
        obj = 0
        for j in range(self.J):
            index = (self.label_Matrix[:, j] != -1)
            labelsJ = self.label_Matrix[index,j]
            temp = logistic(alpha[index]*beta[j])
            obj = obj + np.sum(((1-labelsJ) * np.log(temp+0.000000001) + labelsJ * np.log(1-temp+0.000000001)) * self.posterior[0,j])
            obj = obj + np.sum(((1-labelsJ) * np.log(temp+0.000000001) + labelsJ * np.log(1-temp+0.000000001)) * self.posterior[1,j])
        return -obj


# In[2]:


def logistic(x):
    result = 1./(1+np.exp(-x))
    return result


# In[3]:


def bern(x,l):
    return x ** l * (1-x) ** (1-l)

