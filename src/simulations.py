import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import math

class SimpleSimulation():
    """
    Simple demand simulation class following a truncated gaussian

    Args : 
        mu(float) : mean of non truncated normal
        sigma(float) : std of non truncated normal
    """
    def __init__(self,mu,sigma):
        self.mu=mu
        self.sigma=sigma
        self.truncated_mu, self.truncated_sigma = self.truncated_normal()
        self.optimal_price = self.get_optimal_price()
        self.max_revenue = self.compute_revenue(self.optimal_price)
               
    def truncated_normal(self):
        """
        Compute mean and std for truncated normal
        """
        alpha = -self.mu /self.sigma
        lambda_alpha =norm.pdf(alpha)/(1-norm.cdf(alpha))
        truncated_mu = self.mu +self.sigma*lambda_alpha
        
        truncated_sigma =math.sqrt(self.sigma**2*(1 - lambda_alpha*(lambda_alpha-alpha)))
        return truncated_mu, truncated_sigma

    def get_optimal_price(self):
        """
        Computes the optimal price given the underlying distribution
        (optimal price is the price that maximizes revenue)
        """
        
        res = scipy.optimize.minimize(self.compute_revenue, self.mu, method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p):
        alpha = -self.mu /self.sigma
        return -p*(1 -1/norm.cdf(-alpha)*(norm.cdf((p-self.mu)/self.sigma)-norm.cdf(alpha)) )

    def _simulate(self):
        """
        Hidden method for random sampling
        """
        s = -1
        while s < 0:
            s=np.random.normal( self.mu,self.sigma, 1)
        return s

    def evaluate(self,p):
        """
        Return bool : True if buy False either
        """
        sample = self._simulate()
        return bool(sample >= p)
    
    

