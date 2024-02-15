"""
This is the Maze Environment
After 3000 steps it should open a new path

"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class maze_c():

    debug = False
    
    observation_space = [6, 9]
    start_state = (0, 3)
    terminal_states = [(5, 8)]
    
    shortcut_timestep = 3_000

    walls = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]

    action_labels = ["up", "right", "down", "left"]
    base_actions = [(+1, 0), (0, +1), (-1, 0), (0, -1)]
    
    rewards = [-1]

    def __init__(self):
        self.ax = None
        self.arrow = None
        
        self.actions = self.base_actions[:]
        if self.debug:
            print(f"Actions: {self.actions}")
            
        self.action_space = len(self.actions)
        self.state = self.reset()
        self.walls = self.walls[:]

        self.timestep = 0
        #self.position = None
        
        #self.rng = np.random.default_rng(7)
        self.seed()

    def reset(self):
    
        self.state = np.array(self.start_state)
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.state

    def step(self, action):
    
            #assert self.actions.contains(action)
    
            # Check whether the shortcut should be opened
            if self.timestep == self.shortcut_timestep:
                del self.walls[-1]
            self.timestep += 1
    
            # Calculate move vector and new position
            delta = self.actions[action]
            position = self.state + np.array(delta)

            if self.debug:
                print(f"position: {position}")

            if self.debug:
                print(f"Walls: {self.walls}")
    
            # Check for collisions with walls
            if tuple(position) in self.walls:
                position = self.state
            else:
                position[0] = max(position[0], 0)
                position[0] = min(position[0], 5)
                position[1] = max(position[1], 0)
                position[1] = min(position[1], 8)
    
            # Store position for the next step and calculate arrow for rendering
            self.arrow = position - self.state
            self.state = position
    
            # Check for terminal state
            done = (position == self.terminal_states).all()
            reward = int(done)

            return position, reward, done, {}

    def render(self, mode='human', reset=None):

        size = (tuple(self.observation_space))

        map = np.zeros(size)
        map[(tuple(zip(*self.walls)))] = 1.0
        map[(self.terminal_states[0])] = 2.0
        map[(self.start_state)] = 2.0
        map[2,8] = 0.5 #Part of the wall that opens
        
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()
            
            #background with custom colormap
            #cmap = mcolors.ListedColormap(['gray', 'red', 'yellow', 'green'])
            self.ax.imshow((map), aspect='equal', origin='lower', cmap='Greys')
            
            # Annotations at start and goal positions
            for terminal_state in self.terminal_states:
                self.ax.annotate("G", np.flip(terminal_state), size=12, color='gray', ha='center', va='center')
            self.ax.annotate("S", np.flip(self.start_state), size=12, color='gray', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks([])
            self.ax.set_xticklabels([])
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, size[1]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, size[0]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
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