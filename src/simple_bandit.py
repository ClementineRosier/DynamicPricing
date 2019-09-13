import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm
import math


class SimpleBandit():
    """
    Implementation of a classic multi armed bandit problem for a price selection problem.
    Each arm is set to be a given price : the model estimates the reward probability for each arm
    We model reward probability as a gaussian distribution with parameters (mu, sigma)
    Action selection via Thompson sampling

    Args : 
        k_mu(list) : list of size k (number of arms) defining the mean for the gaussian prior for each arm
        k_sigma(list) : list of size k (number of arms) defining the std for the gaussian prior for each arm
        k_p(list) : list of size k (number of arms) defining the price for each arm
    """

    def __init__(self, k_mu, k_sigma, k_p):
        assert len(k_mu) == len(k_sigma) == len(k_p), "k_mu, k_sigman and k_p must all be same length" 
        self.k_mu = k_mu
        self.k_sigma = k_sigma
        self.k_p = k_p
        self.k = len(self.k_mu)
        print(f"SimpleBandit model instanciated with {self.k} arms.")

    def thompson_sampling(self):
        """
        Random sampling over each arm's probability distribution
        Return : 
            int : argmax over sampling
        """
        return np.argmax([np.random.normal(loc = self.k_mu[i], scale = self.k_sigma[i]) for i in range(self.k)])

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

