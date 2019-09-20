import pandas as pd
import numpy as np
import scipy
from scipy.stats import truncnorm, invgamma
import math


class ContextBandit():
    """
    Implementation of a classic multi armed bandit problem for a price selection problem.
    Each arm is set to be a given price : the model estimates the reward probability for each arm
    We consider context of the consumer as impacting the valuation and hence the probability of buying

    We modelize the problem based on logistic regression, and consider that Buy=1/(1+e-Y)
    where y is a latent variable and Y= beta*X, with X the context

    
   


    Action selection via Thompson sampling

    Args : 
        k_p(list) : list of size k (number of arms) defining the price for each arm
        
        size_context (list) : number of features of the context + 1 (intercept)
        m_0 (list) : array of size size_context with initial mean of the weight of each features, by default = 0
        q_0 (list): array of size size_context with initial variance of the weight of each features
    """

    def __init__(self,k_p,size_context,m_0=np.zeros(shape=(len(k_p),size_context)),q_0):
        #assess size_context==len(m)==len(q)
        self.k_p =k_p   
        self.k = len(self.k_p)    
        self.size_context=size_context
        self.m_0=m_0
        self.q_0=q_0

        self.context=[np.repeat(0,len(size_context)),np.repeat(0,len(size_context))]
        
        self.n_obs = np.repeat(0, self.k) # number of trials for each arm

        self.m_n = np.array(self.m_0)
        self.q_n = np.array(self.q_0)

        print(f"ContextBandit model instanciated with {self.k} arms.")

    def get_weight(self, weight):
        y=int(reward > 0)
        q=self.q[k]
        m=self.m[k]
        context = self.context
        return 0.5*q*(w-m)+math.log(1+math.exp(-y*np.dot(w.T,context)))
    
    def update(self, k, reward):
        """
        Update priors for arm k given observation of reward 

        Args : 
            k (int) : index of the arm played
            reward (float) : value of the observed reward
        """

        min = scipy.optimize.minimize(self.get_weight, self.w[k], method='nelder-mead')
        w=min.x
        self.m[k]=min.x
        self.q[k]=self.q[k] +self.context**2*np.linalg.inv(1+math.exp(-np.dot(w.T,self.context)))*(1-np.linalg.inv(1+math.exp(-np.dot(w.T,self.context))))
        


    def thompson_sampling(self):
        """
        Random sampling over each arm's probability distribution
        Return : 
            int : argmax over sampling
        """
        # Sample weight
        self.w=np.random.normal(self.m,self.q)
        # compute theta*p for each arm
        exp_r = p*(1+math.exp(-math.dot(w,context.T)))
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