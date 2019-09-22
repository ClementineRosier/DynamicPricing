import pandas as pd
import numpy as np
import scipy
from scipy.stats import beta
import math


class BinomialBandit():
    """
    Implementation of a classic multi armed bandit problem for a price selection problem.
    Each arm represent a different price. If arm n (corresponding to price p) is chosen
    then price p is presented to customer. Bandit receives reward r = p * 1_buy=1 
    
    Buying probability is modeled as a bernouilli distribution : buy ~ B(theta)
    We assume beta prior for theta : theta ~ beta(alpha, beta)
    Following that modelisation : P(theta | buy) = P(buy | theta) * P(theta)
    Posterior computation : 
        - alpha_n = alpha_0 + sum(buy) = alpha_0 + sum(positive_occurrences)
        - beta_n = beta_0 + n - sum(buy)  = beta_0 + sum(negative_occurrences)

    Action selection policy : Thompson sampling

    Args : 
        k_p(list) : list of size k (number of arms) defining the price for each arm
        alpha_0 (list) : list of size k (number of arms)
        beta_0 (list) : list of size k (number of arms)
    """

    def __init__(self, k_p, alpha_0, beta_0):

        assert len(k_p) == len(alpha_0) == len(beta_0), "k_mu, k_sigma and k_p must all be same length" 
        self.k_p = k_p
        self.k = len(self.k_p)
        self.alpha_0 = np.array(alpha_0)
        self.beta_0 = np.array(beta_0)
        self.n_pos = np.repeat(0, self.k)
        self.n_obs = np.repeat(0, self.k) # number of trials for each arm

        self.alpha_n = np.array(self.alpha_0)
        self.beta_n = np.array(self.beta_0)
        print(f"BinomialBandit model instanciated with {self.k} arms.")

    def update(self, k, reward):
        """
        Update priors for arm k given observation of reward 

        Args : 
            k (int) : index of the arm played
            buy (int) : 1 if reward > 0, else 0
        """
        self.n_obs[k] += 1
        self.n_pos[k] += int(reward > 0)
        self.alpha_n[k] = self.alpha_0[k] + self.n_pos[k]
        self.beta_n[k] = self.beta_0[k] + self.n_obs[k] - self.n_pos[k]

    def thompson_sampling(self):
        """
        Random sampling over each arm's probability distribution
        Return : 
            int : argmax over sampling
        """
        # Sample theta
        theta = beta.rvs(self.alpha_n, self.beta_n)
        # Compute expected reward for each arm
        exp_r = theta * self.k_p
        return np.argmax(exp_r)


    def chose_action(self, method = "thompson", force_action = None):
        """
        Choose an action
        Args : 
            method :
                - thompson : chose via thompson sampling
                - random : pure random choice
            force_action(int): force model to play arm n

        Return :
            int : index of the arm played
        """
        if force_action is not None :
            assert 0 <= force_action <= self.k - 1, f"Action must be in range [0, {self.k - 1}]"
            self.action = force_action
            return
        
        assert method is not None, "Provide a selection method" 
        if method == "thompson":
            self.action = self.thompson_sampling()
        elif method == "random":
            self.action = np.random.randint(0,self.k)
        return 

