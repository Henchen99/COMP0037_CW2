'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random

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
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
                   
        # Pick the first action
        A = self._q.policy().sample_action(S[0], S[1])
         
        # Main loop
        done = False
           
        while done is False:
    
            S_prime, R, done, info = self._environment.step(A)
    
            # Q3a: Replace with code to implement SARSA
            A_prime = A
            new_q = 0
            
            self._q.set_value(S[0], S[1], A, new_q)
   
            # Store the state                
            S = S_prime
            A = A_prime
               
            