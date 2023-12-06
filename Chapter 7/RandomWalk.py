"""
This is the Random Walk Environment

"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class randomwalk():
    
    observation_space = np.array([1, 7])    
    
    action_labels = ["right", "left"]
    
    base_actions = [np.array([0, 1]), np.array([0, -1])]
    
    rewards = [0, 1]
    
    start_state = np.array([0, 3])
    terminal_states = [np.array([0, 0]),np.array([0, 6])]
    
    map = np.zeros(tuple(observation_space))
    map[tuple(terminal_states[0])] = 2
    map[tuple(terminal_states[1])] = 3
    map[tuple(start_state)] = 1
    
    def __init__(self):
        
        self.ax = None
        self.arrow = None
        
        self.actions = self.base_actions[:]
        #print(f"Actions: {self.actions}")
            
        self.action_space = len(self.actions)
        self.state = self.reset()
        
        #self.rng = np.random.default_rng(7)
        self.seed()

    def reset(self):
    
        self.state = self.start_state.copy()
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.state

    def step(self, action):

        done = False
        
        reward = 0
        
        if action == 2:
            action = np.random.choice([-1,1])
        
            if action == -1:
                action = 1
            else:
                action = 0

        new_state = self.state + self.actions[action]
        
        self.arrow = new_state - self.state
        
        self.state = new_state.copy()

        if np.all(new_state == self.terminal_states[0]):
            reward = 0
            done = True
        elif np.all(new_state == self.terminal_states[1]):
            reward = 1
            done = True

        return self.state, reward, done
    
    def policy(self):
        #return self.rng.choice([0,1], p=[0.5,0.5])
        return np.random.choice([0, +1])

    def render(self, mode='human', reset=None):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()
            
            #background with custom colormap
            cmap = mcolors.ListedColormap(['gray', 'red', 'yellow', 'green'])
            self.ax.imshow((self.map), aspect='equal', origin='lower', cmap=cmap)
            
            # Annotations at start and goal positions
            self.ax.annotate("0", np.flip(self.terminal_states[0]), size=12, color='gray', ha='center', va='center')
            self.ax.annotate("1", np.flip(self.terminal_states[1]), size=12, color='gray', ha='center', va='center')

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.map.shape[1]), minor=True)
            self.ax.set_xticklabels(["0", "T", "A", "B", "C", "D", "E", "T"])
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
        
    def seed(self, seed=None):
        seed = np.random.seed(seed)
        return [seed]