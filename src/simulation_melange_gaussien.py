import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class GSumSimulation():
    """
    Simple demand simulation class based on a sum of independant normal variables
    nb: possibilité de mettre des poids sur les normales ou d'avoir plus de normale
    
    A VERIFIER AU NIVEAU DES PROPRIETES DE LA LOI

    Args : 
        mu_1(float) : mean of the first normal
        sigma_1(float) : std of the first normal
        mu_2(float) : mean of the 2nd normal
        sigma_2(float) : std of the 2nd normal
    """
    def __init__(self,mu_1,sigma_1,mu_2,sigma_2):
        self.mu_1=mu_1
        self.sigma_1=sigma_1
        self.mu_2=mu_2
        self.sigma_2=sigma_2
        self.gs_mu, self.gs_sigma = self.gs()
        self.optimal_price = self.get_optimal_price()
        self.max_revenue = self.compute_revenue(self.optimal_price)
               
    def gs(self):
        """
        Compute mean and std for truncated normal
        """
        gs_mu = self.mu_1 +self.mu_2
        gs_sigma = self.sigma_1 +self.sigma_2
        return gs_mu, gs_sigma

    def get_optimal_price(self):
        """
        Computes the optimal price given the underlying distribution
        (optimal price is the price that maximizes revenue)
        """
        res = scipy.optimize.minimize(self.compute_revenue, self.gs_mu, method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p):
        return -p*(1-scipy.stats.norm.cdf(p,loc = self.gs_mu,scale=slef.gm_sigma))

    def _simulate(self):
        """
        Hidden method for random sampling
        """
        s = -1
        while s < 0:
            s=np.random.normal( self.mu_1,self.sigma_1, 1)+np.random.normal( self.mu_2,self.sigma_2, 1)
        return s

    def evaluate(self,p):
        """
        Return bool : True if buy False either
        """
        sample = self._simulate()
        return bool(sample >= p)
    
    

