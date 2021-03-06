'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random
import numpy as np

from .td_learner_base import TDLearnerBase

class SarsaLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)

    def initialize(self, q):
        self._q = q

    def _learn_online_from_episode(self):
        
        # Storing variable for episode returns
        R_episode = 0

        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
                   
        # Pick the first action
        A = self._q.policy().sample_action(S[0], S[1])
         
        # Main loop
        done = False
        num_steps = 1
           
        while done is False:

            # Terminate if terminal state
            if(A==9 or A==8):
                break

            S_prime, R, done, info = self._environment.step(A)


            # Q3a: Replace with code to implement SARSA
            A_prime = self._q.policy().sample_action(S_prime[0], S_prime[1])
            
            q = self._q.value(S[0],S[1],A)
            q_prime = self._q.value(S_prime[0],S_prime[1],A_prime)
            new_q = q + self.alpha()*(R + self.gamma()*q_prime - q)
            
            self._q.set_value(S[0], S[1], A, new_q)
   
            # Store the state                
            S = S_prime
            A = A_prime

            # Store the return
            R_episode = R_episode + new_q
            num_steps += 1
        
        return int(R_episode/num_steps)
               
            