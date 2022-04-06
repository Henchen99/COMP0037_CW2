'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random

from airport.driving_actions import DrivingActionType

from .td_learner_base import TDLearnerBase

class QLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)
        
        self._q = None

    def initialize(self, q):
        self._q = q
        
        if self._q.policy() is not None:
            self._q.policy().set_epsilon(self._epsilon)

    def _learn_online_from_episode(self):
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
        
        # Main loop
        done = False
        
        while done is False:
                        
            # Sample the action
            A = self._q.policy().sample_action(S[0], S[1])
           
            # Step the environment
            S_prime, R, done, info = self._environment.step(A)

            # Q3b : Replace with code to implement Q-learning
            new_q = 0
            
            self._q.set_value(S[0], S[1], A, new_q)
           
            # Store the state                
            S = S_prime


        