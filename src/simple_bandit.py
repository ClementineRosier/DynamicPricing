import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm, invgamma
import math


class SimpleBandit():
    """
    Implementation of a classic multi armed bandit problem for a price selection problem.
    Each arm is set to be a given price : the model estimates the reward probability for each arm
    We model reward probability as a gaussian distribution with parameters (mu, sigma)
    P(mu|sigma) ~ Norm(alpha, sigma)
    P(sigma) ~ InvGamma(a_0, b_0)

    Following that modelisation : P(mu, sigma | R) ~= P(R | mu, sigma) * P(mu | sigma) * P(sigma)

    Action selection via Thompson sampling

    Args : 
        k_p(list) : list of size k (number of arms) defining the price for each arm
        alpha_0 (list) : list of size k (number of arms)
        beta_0 (list) : list of size k (number of arms)
        a_0 (list): list of size k (number of arms)
        b_0 (list): list of size k (number of arms)
    """

    def __init__(self, k_p, alpha_0, beta_0, a_0, b_0):

        assert len(k_p) == len(alpha_0) == len(beta_0) == len(a_0) == len(b_0), "k_mu, k_sigman and k_p must all be same length" 
        self.k_p = k_p
        self.k = len(self.k_p)
        self.alpha_0 = np.array(alpha_0)
        self.beta_0 = np.array(beta_0)
        self.a_0 = np.array(a_0)
        self.b_0 = np.array(b_0)
        self.n_obs = np.repeat(0, self.k) # number of trials for each arm

        self.alpha_n = self.alpha_0
        self.a_n = np.array(self.a_0)
        self.b_n = np.array(self.b_0)

        print(f"SimpleBandit model instanciated with {self.k} arms.")


    def update(self, k, reward):
        """
        Update priors for arm k given observation of reward 

        Args : 
            k (int) : index of the arm played
            reward (float) : value of the observed reward
        """
        n = self.n_obs[k]
        beta_0 = self.beta_0[k] 
        alpha_0 = self.alpha_0[k]
        a_0 = self.a_0[k]
        b_0 = self.b_0[k]

        self.alpha_n[k] = 1/ ( n+1 / beta_0) * ( reward + 1/beta_0 * alpha_0)
        self.a_n[k] = n + a_0
        self.b_n[k] = 1 /self.a_n[k] * ( (reward - alpha_0 )**2 / (1 + beta_0) +  a_0/b_0)


    def thompson_sampling(self):
        """
        Random sampling over each arm's probability distribution
        Return : 
            int : argmax over sampling
        """
        # Sample sigma
        sigma = self.b_n * invgamma.rvs(self.a_n)
        # Sample mu
        mu = np.random.normal(self.alpha_n, sigma * self.beta_0)
        return np.argmax(mu)


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