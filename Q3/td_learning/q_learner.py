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
        
        # Storing variable for episode returns
        R_episode = 0
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
        
        # Main loop
        done = False
        num_steps = 1
        
        while done is False:
                        
            # Sample the action
            A = self._q.policy().sample_action(S[0], S[1])

            # Terminate if terminal state
            if(A==9 or A==8):
                break
           
            # Step the environment
            S_prime, R, done, info = self._environment.step(A)

            # Q3b : Replace with code to implement Q-learning
            q = self._q.value(S[0],S[1],A)
            maxq = max(self._q.values_of_actions(S_prime[0],S_prime[1]))
            new_q = q + self.alpha()*(R + self.gamma()*maxq - q)
            
            self._q.set_value(S[0], S[1], A, new_q)
           
            # Store the state                
            S = S_prime

            # Store the return
            R_episode = R_episode + new_q
            num_steps += 1
        
        return int(R_episode/num_steps)


        