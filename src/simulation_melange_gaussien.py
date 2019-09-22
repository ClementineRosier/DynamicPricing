import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class GMixSimulation():
    """
    Simple demand simulation class based on a mixture of two normal distribution

    Args : 
        mu_1(float) : mean of the first normal
        sigma_1(float) : std of the first normal
        mu_2(float) : mean of the 2nd normal
        sigma_2(float) : std of the 2nd normal
    """
    def __init__(self,mu_1,sigma_1,mu_2,sigma_2,weight):
        self.mu_1=mu_1
        self.sigma_1=sigma_1
        self.mu_2=mu_2
        self.sigma_2=sigma_2
        self.weight =weight
        self.gm_mu, self.gm_sigma = self.gm()
        self.optimal_price = self.get_optimal_price()
        self.max_revenue = self.compute_revenue(self.optimal_price)
               
    def gm(self):
        """A VERIFIER MAIS PAS NECESSAIRE IMMEDIATEMENT
        """
        gm_mu = self.weight*self.mu_1 +(1-self.weight)*self.mu_2
        gm_sigma = self.weight*(self.mu_1**2 +self.sigma_1) +(1-self.weight)*(self.mu_2**2+self.sigma_2)
        return gm_mu, gm_sigma

    def get_optimal_price(self):
        """
        Computes the optimal price given the underlying distribution
        (optimal price is the price that maximizes revenue)
        """
        res = scipy.optimize.minimize(self.compute_revenue, self.gm_mu, method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p):
        return -p*(1-(self.weight*scipy.stats.norm.cdf(p,loc=self.mu_1,scale=self.sigma_1)+(1-self.weight)*scipy.stats.norm.cdf(p,loc=self.mu_2,scale=self.sigma_2)))

    def _simulate(self):
        """
        Hidden method for random sampling
        """
        s = -1
        while s < 0:
            w=np.random.binomial(1,self.weight)
            s=w*np.random.normal( self.mu_1,self.sigma_1, 1)+(1-w)*np.random.normal( self.mu_2,self.sigma_2, 1)
        return s

    def evaluate(self,p):
        """
        Return bool : True if buy False either
        """
        sample = self._simulate()
        return bool(sample >= p)
    



    
    

