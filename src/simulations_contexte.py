import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class ContextSimulation():
	"""
	Demand simulation for contextual bandit v=BX +E
	with X=(X_c X_d) and X_c ~G() ;X_d ~B() and ? E~simple simul ?
	"""

	def __init__(self,beta,mu_c,sigma_c,n,theta):
		self.beta=beta
		self.mu_c=mu_c
		self.sigma_c=sigma_c
		self.n=n
		self.theta
		self.mu, self.sigma = self.mean_var()
        self.optimal_price = self.get_optimal_price()
        self.max_revenue = self.compute_revenue(self.optimal_price)
               
    def mean_var(self):
        """
        Compute mean and var for the demand
        """
        means=np.append(self.mu_c,self.theta*(self.n-1))
        vars=np.append(self.sigma_c,self.theta*(1-self.theta)*(self.n-1))
        mu = np.sum(beta*means)
        sigma =np.sum(beta**2*vars)
        return mu, sigma

    def get_optimal_price(self):
        """
        Computes the optimal price given the underlying distribution
        (optimal price is the price that maximizes revenue)
        """
        res = scipy.optimize.minimize(self.compute_revenue, self.mu, method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p):
    	#to be updated
        return -(p/(self.mu**2*scipy.stats.norm.cdf(self.mu/self.sigma,loc=0, scale=1))*scipy.stats.norm.cdf((-p+self.mu)/self.sigma, loc=0, scale=1))

    def _simulate(self):
    	#to be updated
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