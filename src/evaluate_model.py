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
	def __init__(self, k_p,n_obs ,compute_revenue,optimal_price ):
		self.T=np.sum(n_obs)
		self.k_p=k_p
		self.n_obs=n_obs
		self.compute_revenue=compute_revenue
		self.optimal_price=optimal_price

		self.regret=[]

	def regret(self):
		"""
		Compute pseudo regret of the model
		"""

		regret_t= T*compute_revenue(optimal_price) - np.sum([n_obs[i]*compute_revenue(k_p[i]) for i in range(len(k_p))])
		self.regret.append(regret_t)
		return
		

