import pandas as pd
import numpy as np
import scipy
from scipy.stats import beta
import math


class EpsilonGreedyBandit():
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

    Action selection policy : epsilon-greedy (cf Auer 2002)

    Args : 
        k_p(list) : list of size k (number of arms) defining the price for each arm
        epsilon(float) : probability of not playing greedy ant to explore (between 0 and 1) at each step
    """

    def __init__(self, k_p, epsilon):

        self.k_p = k_p
        self.k = len(self.k_p)
        self.epsilon = epsilon
        self.n_pos = np.repeat(0, self.k)
        self.n_obs = np.repeat(0, self.k) # number of trials for each arm


        print(f"BinomialBandit model for espilon-greedy instanciated with {self.k} arms.")

    def update(self,k, reward):
        """
        Update parameters and average profit obtained per arm

        Args : 
            k (int) : index of the arm played
            buy (int) : 1 if reward > 0, else 0
        """
        self.n_obs[k] += 1
        self.n_pos[k] += int(reward > 0)

    def epsilon_greedy(self):
        """
        Select with proba Epsilon_n a random arm and otherwise the arm with the highest average reward
        """
        # Wich random selection ? Bernouilli ( Epsilon)
        random_selection = np.random.binomial(1,self.epsilon)
        #print(self.epsilon)
        #print(random_selection)
        if random_selection == 1:
            #selct randomly an arm
            return np.random.randint(0,self.k)
        else:
            average_reward = self.k_p*np.nan_to_num(self.n_pos/(self.n_obs))
            return np.argmax(average_reward)


    def chose_action(self, method = "greedy", force_action = None):
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
        if method == "greedy":
            self.action = self.epsilon_greedy()
        elif method == "random":
            self.action = np.random.randint(0,self.k)
        return 