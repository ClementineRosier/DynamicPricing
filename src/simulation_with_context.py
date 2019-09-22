import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math


class ContextualDemandSimulation():
    """
    Demand simulation for contextual bandit v=BX +E
    with X=(X_c X_d) and X_c ~G() ;X_d ~U() and ? E~simple simul ?
    """
    def __init__(self,beta_c,beta_d,mu_e,sigma_e):
        self.beta_c= np.array(beta_c)
        self.beta_d = beta_d #array with impact of each category
        self.mu_e= np.array(mu_e)
        self.sigma_e= np.array(sigma_e)
    
    def _get_context_impact(self, continuous_context, discrete_context):
        # Continuous impact
        cont = self.beta_c * np.array(continuous_context)
        disc = [sum(np.array(beta) * np.array(disc)) for beta,disc in zip(self.beta_d, discrete_context)]        
        return sum(cont) + sum(disc)
    
    def _simulate(self, continuous_context, discrete_context):
        cont = self._get_context_impact(continuous_context, discrete_context)
        s=-1
        while s < 0:
            s_i = np.random.normal(self.mu_e,self.sigma_e,1)
            s=s_i + cont
        return s[0]
    
    def evaluate(self,p, continuous_context, discrete_context):
        """
        Return bool : True if buy False either
        """
        sample = self._simulate(continuous_context, discrete_context)
        return int(sample >= p)

    def get_optimal_price(self,continuous_context, discrete_context):
        """
        Computes the optimal price given the underlying distribution for a given context
        (optimal price is the price that maximizes the expected revenue for a given context)
        """
        res = scipy.optimize.minimize(self.compute_revenue, self.mu_e, args=(continuous_context, discrete_context),method='nelder-mead')
        return res.x[0]

    def compute_revenue(self, p,continuous_context, discrete_context):
        """
        compute expected revenue for a given price and a given context
         E(R|p,context)=E(p*Buy|p,context)=p*P(v>=p|context)=p*P(espilon >=p-context)
                                                 = p*(1-P(epsilon<=p-context))
        """    
        context_impact=self._get_context_impact(continuous_context, discrete_context)

        return -p*(1-(scipy.stats.norm.cdf(p-context_impact,loc=self.mu_e,scale=self.sigma_e)))