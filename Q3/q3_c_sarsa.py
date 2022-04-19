#!/usr/bin/env python3

'''
Created on 21 Mar 2022

@author: ucacsjj
'''

import math
import matplotlib.pyplot as plt

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from airport.airport_map_drawer import AirportMapDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer
from airport.driving_actions import DrivingActionType
from airport.driving_q_grid import DrivingQGrid

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from td_learning.sarsa_learner import SarsaLearner

if __name__ == '__main__':
    
    # Create test environment
    airport_map, drawer_height = corridor_scenario()
    airport = AirportDrivingEnvironment(airport_map)
    airport.set_nominal_direction_probability(0.8)
    
    q_grid = DrivingQGrid('Sarsa Learner', airport_map)
    q_grid.show()
    
    learner = SarsaLearner(airport)    
    learner.initialize(q_grid)
    learner.set_gamma(1)
    learner.set_alpha(1e-3)
    learner.set_epsilon(1)

    returns = []
            
    # Bind the drawer with the solver
    # policy_drawer = DrivingPolicyDrawer(q_grid.policy(), drawer_height)
    # learner.set_policy_drawer(policy_drawer)
    
    # value_function_drawer = ValueFunctionDrawer(q_grid.value_function(), drawer_height)
    # learner.set_value_function_drawer(value_function_drawer)
    
    # Run the learning algorithm.
    for i in range(10000):
        learner.set_epsilon(1/math.sqrt(i+1))
        reward = learner.learn_online_policy()
        # q_grid.show()
        returns.append(reward)
        if(i%100==0):
            print("episode: " + str(i))
    print(returns)
    # policy_drawer.wait_for_key_press()
    plt.figure()
    plt.plot(returns, color = 'red', label = 'Average Return')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Average Return per Episode')
    
    plt.show()
    plt.pause(0.001)
    input()
