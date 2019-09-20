import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math

class ContextGenerator():
	"""
	Class creating random contexts
	Context is composed of real values and categorical values
	Sampling is done via : 
		- draw from a normal(mu, sigma) for each real variable
		- uniform choice for each categorical
	Args : 
		mu(np.array) : vector of means for continuous variables
		sigma(np.array) : vector of std for continuous variables
		n_discrete(np.array) : vector containing the number of categories for each discrete variable
	"""

	def __init__(self, mu, sigma, n_discrete):
		self.mu = mu
		self.sigma = sigma
		self.n_discrete = n_discrete

	def simulate(self):
		continuous = np.random.normal(self.mu, self.sigma).tolist()

		dummies = [np.zeros(i) for i in self.n_discrete]
		for variable in dummies : 
		    random_index = np.random.randint(len(variable))
		    variable[random_index] = 1
		dummies = [x.tolist() for x in dummies] # flat list

		return continuous, dummies