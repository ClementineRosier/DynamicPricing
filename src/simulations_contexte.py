import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class ContextSimulation():
	"""
	Demand simulation for contextual bandit v=BX +E
	with X=(X_c X_d) and X_c ~G() ;X_d ~U() and ? E~simple simul ?
	"""
	def __init__(self,beta_c,mu_c,sigma_c,beta_d,n,mu_e,sigma_e):
		self.beta_c=beta_c
		self.mu_c=mu_c
		self.sigma_c=sigma_c
		self.beta_d = beta_d #array with impact of each category
		self.n=n
		self.mu_e=mu_e
		self.sigma_e=sigma_e
		self.context=[np.repeat(0,len(beta_c) + len(beta_d)),np.repeat(0,len(beta_c) + len(beta_d))]
		self.mu, self.sigma = self.mean_var()
		self.optimal_price = self.get_optimal_price()
		self.max_revenue = self.compute_revenue(self.optimal_price)
			   
	def mean_var(self):
		"""
		Compute mean and var for the demand
		"""
		theta=1/len(self.n)
		mean_d=np.sum((self.beta_d.T*theta).T,axis=1)
		mean_context=np.append(self.beta_c*self.mu_c,mean_d)
		var_d = np.sum((self.beta_d.T**2*theta).T,axis=1)
		var_context=np.append(self.beta_c**2*self.sigma_c,var_d)
		mu = mean_context+self.mu_e
		sigma =var_context+self.sigma_e
		return mu, sigma

	def get_optimal_price(self):
		"""
		Computes the optimal price given the underlying distribution for a given context X
		(optimal price is the price that maximizes revenue)
		"""
		res = scipy.optimize.minimize(self.compute_revenue, self.mu_e, method='nelder-mead')
		return res.x[0]

	def compute_revenue(self, X,p):
		#compute expected revenue for a given context ie E(R)=p*E(A=1|X)=p*P(p-BX<=V) with V ~N(mu_e,sigma_e)
		return -p*np.random.cdf(p-self.self.context[0],loc=self.mu_e,scale=sigma_e)

	def _simulate_context(self):

		s_c=np.random.normal(self.mu_c,self.sigma_c,len(self.mu_c))
		
		a=np.repeat(1,len(self.n))
		s_d =[np.random.random_integers(a[i],self.n[i]) for i in range(len(self.n))]
		context_cat=np.append(s_c,s_d)
		context_value=np.append(self.beta_c*s_c,[self.beta_d[i,s_d[i]-1] for i in range(len(s_d))])
		self.context=[context_value,context_cat]
		return self.context


	def _simulate(self):
		"""
		Hidden method for random sampling of the context and the individual demand
		"""
		s = -1
		context=self._simulate_context()
		while s < 0:
			s_i = np.random.normal(self.mu_e,self.sigma_e,1)
			s=s_i +beta*self.context[0]
		return context[1],s

	def evaluate(self,p):
		"""
		Return bool : True if buy False either
		"""
		sample = self._simulate()[1]
		return bool(sample >= p)