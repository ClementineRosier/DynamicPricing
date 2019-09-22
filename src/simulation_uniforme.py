import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class UniformSimulation():
    """
    Simple demand simulation class following a continuous uniform with support [a,b]

    Args : 
        a(float) : lowest value of the distribution 
        b(float) : largest value fo the distribution 
    """
    def __init__(self,a,b):
        assert a >= 0
        assert b >= a
        self.a = a
        self.b = b
        self.mu, self.sigma = self.mean_variance()
        self.optimal_price = self.get_optimal_price()
        self.max_revenue = self.compute_revenue(self.optimal_price)
               
    def mean_variance(self):
        """
        Compute mean and std for truncated normal
        """
        mu = (self.a +self.b)/2
        sigma = math.sqrt((self.a -self.b)**2/12)
        return mu, sigma

    def get_optimal_price(self):
        """
        Computes the optimal price given the underlying distribution
        (optimal price is the price that maximizes revenue)
        """
        res = scipy.optimize.minimize(self.compute_revenue, self.mu, method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p):
        if p<=self.a:
            return -1*p
        elif p >=self.b:
            return 0
        else:
            return -p*(self.b-p)/(self.b -self.a)
        

    def _simulate(self):
        """
        Hidden method for random sampling
        """
        s = -1
        while s < 0:
            s=np.random.uniform( self.a,self.b, 1)
        return s

    def evaluate(self,p):
        """
        Return bool : True if buy False either
        """
        sample = self._simulate()
        return bool(sample >= p)
    
    

