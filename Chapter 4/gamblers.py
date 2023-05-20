"""
This is Gamblers Environment

"""
from typing import Tuple
import numpy as np
import scipy.stats

class gamblers():
    
    terminals = [np.array([0]), np.array([100])]
    
    gamma = 1
    
    ph = 0.5
    
    def __init__(self, ph=0.5):
        
        #caching probabilities of requests and returns

        self.ph = ph
        self.reset()
        
    def reset(self):
        
        self.state = np.random.randint(1, 100)            
        return self.state
    
    def step(self, action):
        
        reward = 0
        done = False
        
        win = np.random.choice(2,1,p=[1-self.ph, self.ph]) * 2 * action
        
        self.state = min(self.state - action + win, 100)
        self.state = max(self.state, 0)
        
        if self.state == 100:
            reward = 1   
            done = True
            
        if self.state == 0:
            done = True
        
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