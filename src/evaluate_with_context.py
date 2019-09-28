import pandas as pd
import numpy as np
import scipy
from scipy.stats import beta
import math

class EvaluateBanditContext():
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
		self.regret=[]
		self.regret_t=0
		self.p_max = max(self.k_p)
	def _get_expected_revenue_arm(self,continuous_context, discrete_context):
		"""
		Compute the expected revenue for each arm
		"""
		return [-1*self.compute_revenue(p,continuous_context, discrete_context) for p in self.k_p ]

	def get_regret(self,n_obs,chosen_action,continuous_context, discrete_context):
		"""
		Compute pseudo regret of the model
		"""
		exp_revenue= self._get_expected_revenue_arm(continuous_context, discrete_context)
		best_action = np.argmax(exp_revenue)
		self.regret_t+= 1/self.p_max *exp_revenue[best_action] - exp_revenue[chosen_action]
		self.regret.append(self.regret_t)
		return self.regret_t