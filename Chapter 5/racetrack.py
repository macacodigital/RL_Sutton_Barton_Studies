"""
This is the RaceTrack Environment

"""
from typing import Tuple
import numpy as np
import scipy.stats

class racetrack_turn_1():
    
    finish_line = [np.array([16,26]), np.array([16,27]), np.array([16,28]), np.array([16,29]), np.array([16,30]) np.array([16,31])]
    track_boundaries_left = [np.array([2,0]), np.array([2,1]), np.array([2,2]),np.array([1,3]), np.array([1,4]), np.array([1,5]), 
                       np.array([1,6]), np.array([1,7]), np.array([1,9]), np.array([0,10]), np.array([0,11]), np.array([0,12]), 
                       np.array([0,13]), np.array([0,14]), np.array([0,15]), np.array([0,16]), np.array([0,17]), np.array([-1,18]),
                       np.array([-1,19]), np.array([-1,20]), np.array([-1,21]), np.array([-1,22]), np.array([-1,23]), np.array([-1,24]), 
                       np.array([-1,25]), np.array([-1,26]), np.array([-1,27]), np.array([-1,28]), np.array([0,29]), np.array([1,30]), np.array([1,31])]
    track_boundaries_top = [np.array([2,32]), np.array([3,32]), np.array([4,32]), np.array([5,32]), np.array([6,32]), np.array([7,32]), np.array([8,32]), np.array([9,32]), 
                       np.array([10,32]), np.array([11,32]), np.array([12,32]), np.array([13,32]), np.array([14,32]), np.array([15,32]), np.array([16,32])]
    track_boundaries_right = [np.array([9,0]), np.array([9,1]), np.array([9,2]), np.array([9,3]), np.array([9,4]), np.array([9,5]), np.array([9,6]), np.array([9,7]), np.array([9,8]),
                              np.array([9,9]), np.array([9,10]), np.array([9,11]), np.array([9,12]), np.array([9,13]), np.array([9,14]), np.array([9,15]), np.array([9,16]), 
                              np.array([9,17]), np.array([9,18]), np.array([9,19]), np.array([9,20]), np.array([9,21]), np.array([9,22]), np.array([9,23]), np.array([9,24]), 
                              np.array([10,25]), np.array([17,26]), np.array([17,27]), np.array([17,28]), np.array([17,29]), np.array([17,30]) np.array([17,31])]
    min_velocity = 0
    max_velocity = 4
    
    action_space = [-1, 0 ,1]
    
    gamma = 1
    
    def __init__(self):
        
        self.velocity= np.array([0,0])
        
        self.reset()
        
    def reset(self):
        
        self.state = np.array([0,np.random.randint(3, 9)])            
        return self.state
    
    def step(self, action):
        
        reward = 0
        done = False
        
        self.velocity = self.velocity + action
        
        self.velocity = min(self.velocity, self.max_velocity)
        self.velocity = max(self.velocity, self.min_velocity)
        
        new_state = self.state + self.velocity
        
        finish_line = np.all(self.finish_line < new_state[1], axis=1)
        boundary_left = np.all(self.track_boundaries_left > new_state[0], axis=0)
        boundary_top = np.all(self.track_boudaries_topline < new_state[1], axis=1)
        boundary_right = np.all(self.track_boundaries_right < new_state[0], axis=0)
        
        if finish_line.any():
            reward = 0
            done = True
        else:
            if new_state[0] > 31:
                reward = 0
                self.state = self.reset(self)
            else:
                
        
        return self.state, reward, done

    def expected_return(self, state_values, action, state, gamma=1):
        
        capital = state
        bet = action
        # bet = min(action, capital)
        # bet = min(bet, 100 - capital)
        
        ps = np.zeros(2, dtype=float)
        ps = (1-self.ph, self.ph)
        
        rewards = np.zeros(2, dtype=float)    
        
        terminals = np.zeros(2, dtype=bool) 
        
        capitals = np.zeros((2, 1), dtype=int)
        capitals[0, 0] = max(capital - bet, self.terminals[0])
        terminals[0] = (capitals[0, 0] == self.terminals[0])
        capitals[1, 0] = min(capital + bet, self.terminals[1])
        terminals[1] = (capitals[1, 0] == self.terminals[1])
        
        #print(f"{capitals}")
        
        # Reward Calculation
        rewards = (0.0, float(capitals[1, 0] // 100))
        
        returns = rewards + gamma * state_values[tuple(capitals.T)]
        
        if terminals[0]:
            returns[0] = rewards[0]
        
        if terminals[1]:
            returns[1] = rewards[1]
              
        #print(f"Returns: {returns}")
        
        expected_return = np.sum(ps * returns)
        

        '''
        if (state==51):
            print(f"Bet: {bet}, Rewards: {rewards}, State Value: {state_values[tuple(capitals.T)]}, Return: {expected_return}")
        '''
              
        #print(f"Expected Return: {expected_return}")
        
        return expected_return