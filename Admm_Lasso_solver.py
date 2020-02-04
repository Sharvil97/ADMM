# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:52:43 2019

@author: Chanakya-vc
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

class ADMM:
    def __init__(self,A,b,tau):
        self.A=A
        self.b=b
        self.tau=tau
        self.X=np.random.randn(np.size(A,1),1)
        self.Z=np.zeros((np.size(A,1),1))
        self.rho=3
        self.alpha = 0.01
        self.dual = np.zeros((np.size(A,1), 1))
    
    def solve(self,algo):
# =============================================================================
# Using the closed form solution for X take the ADMM step for solving X
# =============================================================================
        print(self.X)
        self.X= inv(self.A.T.dot(self.A) + self.rho).dot(self.A.T.dot(self.b) + self.rho * self.Z - self.dual)
# =============================================================================
#         Update Z according to the lasso or the ridge solver
# =============================================================================
        if(algo=="lasso"):
            self.Z = self.X + self.dual / self.rho - (self.alpha / self.rho) *  np.sign(self.Z)
        elif(algo=="ridge"):
            self.Z=(self.rho*self.X+self.dual)/(2*self.alpha+self.rho)
# =============================================================================
#         Update the dual variable
# =============================================================================
        self.dual = self.dual + self.rho * (self.X - self.Z)
        
# =============================================================================
#     Define the objective function as 1/2|AX-B|**2+tau*|X|**2
# =============================================================================
    def LassoObjective(self,algo):
        if(algo=="lasso"):
            return 0.5 * norm(self.A.dot(self.X) - self.b)**2 + self.tau *  norm(self.X, 1)  
        elif(algo=="ridge"):
            return 0.5 * norm(self.A.dot(self.X) - self.b)**2 + self.tau *  norm(self.X)**2 
            
    def get_X(self,iterations,algo):
        for i in range(iterations):
            self.solve(algo)
    def predict(self,test):
        
        b=np.ones((test.shape[0], 1))
        test = np.concatenate([test, b], axis = 1)
        return test.dot(self.X)