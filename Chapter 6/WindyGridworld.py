"""
This is the Windy GRidworld Environment

"""
from typing import Tuple
import numpy as np
import scipy.stats
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class windygridworld():
    
    observation_space = np.array([7,10])    
    
    action_labels = ["up", "down", "right", "left", "up-right", "up-left", "down_right", "down_left", "stop"]
    
    base_actions = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
    king_actions = [np.array([1,1]), np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])]
    stop_action = [np.array([0,0])]
    
    rewards = [-1, 0]
    
    start_state = np.array([3,0])
    terminal_state = np.array([3,7])
    
    wind_factor = [0,0,0,1,1,1,2,2,1,0]
    
    map = np.zeros(tuple(observation_space))
    map[tuple(terminal_state)] = 2
    map[tuple(start_state)] = 1
    
    def __init__(self, king=False, stop=False, stochastic=False):

        self.stochastic = stochastic
        self.king = king
        self.stop = stop
        
        self.ax = None
        self.arrow = None
        
        self.actions = self.base_actions[:]

        if self.king:
            self.actions += self.king_actions
        if self.stop:
            self.actions += self.stop_action
            
        self.action_space = len(self.actions)
        self.state = self.reset()

    def reset(self):
    
        self.state = self.start_state.copy()
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.state

    def step(self, action):

        done = False

        new_state = self.state + self.actions[action]
        
        #Clipping position before adding wind
        new_state = np.maximum(new_state, np.array([0,0]))
        new_state = np.minimum(new_state, self.observation_space - 1)
        
        #Adding Wind
        wind = self.wind_factor[new_state[1]]
        if self.stochastic:
            wind += np.random.choice([-1, 0, +1])
        new_state = new_state + np.array([wind, 0])
        
        #Clipping position after adding wind
        new_state = np.maximum(new_state, np.array([0,0]))
        new_state = np.minimum(new_state, self.observation_space - 1)
        
        self.arrow = new_state - self.state
        
        self.state = new_state.copy()

        if np.all(new_state == self.terminal_state):
            reward = 0
            done = True
        else:
            reward = -1

        return self.state, reward, done

    def render(self, mode='human', reset=None):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()
            
            # Background colored by wind strength
            wind = np.vstack([self.wind_factor] * self.observation_space[0])
            self.ax.imshow(wind, aspect='equal', origin='lower', cmap='Blues')
            
            # Annotations at start and goal positions
            self.ax.annotate("G", self.terminal_state, size=25, color='gray', ha='center', va='center')
            self.ax.annotate("S", self.start_state, size=25, color='gray', ha='center', va='center')

            #background with custom colormap
            cmap = mcolors.ListedColormap(['gray', 'red', 'green'])
            self.ax.imshow((self.map), aspect='equal', origin='lower', cmap=cmap)
            
            # Major tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.wind_factor)))
            self.ax.set_xticklabels(self.wind_factor)
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.map.shape[1]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.map.shape[0]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='minor', length=0)
            self.ax.set_frame_on(True)

        position = np.flip(self.state)
        arrow = np.flip(self.arrow)
    
        # Arrow pointing from the previous to the current position
        if (arrow == 0).all():
            patch = mpatches.Circle(position, radius=0.05, color='black', zorder=1)
        else:
            patch = mpatches.FancyArrow(*(position - arrow), *arrow, color='black',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
        self.ax.add_patch(patch)