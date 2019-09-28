import pandas as pd
import numpy as np
import scipy
from scipy.stats import beta
import math


class UCBBandit():
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

    Action selection policy : UCB (cf Auer 2002)

    Args : 
        k_p(list) : list of size k (number of arms) defining the price for each arm
        d (integer) : strictly positive and must be smaller than the difference between expected profit with optimal price and the expected profit of any other price
        c (inter) : strictly positive
    """

    def __init__(self, k_p):

        self.k_p = k_p
        self.k = len(self.k_p)
        self.n_pos = np.repeat(0, self.k)
        self.n_obs = np.repeat(0, self.k) # number of trials for each arm
        self.p_max = max(self.k_p)


        print(f"BinomialBandit model for UCB instanciated with {self.k} arms.")

    def update(self,k, reward):
        """
        Update parameters, and notaby epsilon_n and average profit obtained per arm

        Args : 
            k (int) : index of the arm played
            buy (int) : 1 if reward > 0, else 0
        """
        self.n_obs[k] += 1
        self.n_pos[k] += int(reward > 0)


    def ucb(self):
        """
        Select with proba Epsilon_n a random arm and otherwise the arm with the highest average reward
        #if not tested we put by default the highest value possible ie pmax * 1
        """
        # compute upper bound
        average_reward = self.k_p*np.nan_to_num(self.n_pos/(self.n_obs))/self.p_max
        self.B= [1*(self.n_obs[i]!=0)*(average_reward[i]/+math.sqrt(2*math.log(np.sum(self.n_obs)+1)/self.n_obs[i])) + 1*(self.n_obs[i]==0)*1 for i in range(self.k)]
        #select the highest
        return np.argmax(self.B)


    def chose_action(self, method = "UCB", force_action = None):
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
        if method == "UCB":
            self.action = self.ucb()
        elif method == "random":
            self.action = np.random.randint(0,self.k)
        return 

