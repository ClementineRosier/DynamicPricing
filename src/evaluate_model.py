import pandas as pd
import numpy as np
import scipy
from scipy.stats import beta
import math

class EvaluateBandit():
	"""
	Evaluation of models through regret using the following formula:
	pseudo-regret = TxE_p*(R) - T_p1xE_p1(R) - T_p2xE_p2(R) ...- T_pKE_pK(R)
	
	Args :
		k_p: list of size k (number of arms) defining the price for each arm
		n_obs: number of trials for each arm

		compute_revenue: compute expected outcome (1*p or 0*p) for a given price
		optimal_price : price maximizing the expected revenue

	"""
	def __init__(self, bandit ,simulation):
		self.k_p=bandit.k_p
		self.compute_revenue=simulation.compute_revenue
		self.exp_revenue= self._get_expected_revenue_arm()
		self.best_action = np.argmax(self.exp_revenue) #arm with the highest expected revenue
		self.best_price=self.k_p[self.best_action] #price to be played to maximize expected revenue
		self.regret=[]
		self.p_max = max(self.k_p)
	def _get_expected_revenue_arm(self):
		"""
		Compute the expected revenue for each arm
		"""
		return [-1*self.compute_revenue(p) for p in self.k_p ]

	def get_regret(self,n_obs):
		"""
		Compute pseudo regret of the model
		"""

		regret_t= 1/self.p_max *(np.sum(n_obs)*self.exp_revenue[self.best_action] - np.sum([n_obs[i]*self.exp_revenue[i]  for i in range(len(self.k_p))]))
		self.regret.append(regret_t)
		return regret_t
		