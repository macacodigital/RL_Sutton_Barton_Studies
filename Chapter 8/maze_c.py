"""
This is the Maze Environment

"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class maze_c():
    observation_space = np.array([6, 9])
    start_state = np.array([0, 3])
    terminal_states = [np.array([5, 8])]
    
    shortcut_timestep = 3_000

    walls = [np.array([2, 1]), np.array([2, 2]), np.array([2, 3]), np.array([2, 4]), np.array([2, 5]), np.array([2, 6]), np.array([2, 7]), np.array([2, 8])]

    action_labels = ["down", "right", "up", "left"]
    base_actions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
    
    rewards = [-1]
    
    map = np.zeros(tuple(observation_space))
    map[tuple(terminal_states[0])] = 2
    map[tuple(start_state)] = 1

    def __init__(self):
        self.ax = None
        self.arrow = None
        
        self.actions = self.base_actions[:]
        #print(f"Actions: {self.actions}")
            
        self.action_space = len(self.actions)
        self.state = self.reset()
        self.walls = self.walls[:]

        self.timestep = 0
        self.position = None
        
        #self.rng = np.random.default_rng(7)
        self.seed()

    def reset(self):
    
        self.state = self.start_state.copy()
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.state

    def step(self, action):
    
            assert self.actions.contains(action)
    
            # Check whether the shortcut should be opened
            if self.timestep == self.shortcut_timestep:
                del self.walls[-1]
            self.timestep += 1
    
            # Calculate move vector and new position
            delta = self.actions[action]
            position = self.state + delta
    
            # Check for collisions with walls
            if tuple(position) in self.walls:
                position = self.state
            else:
                position = np.clip(position, 0, self.observation_space.nvec - 1)
    
            # Store position for the next step and calculate arrow for rendering
            self.arrow = position - self.state
            self.state = position
    
            # Check for terminal state
            done = (position == self.terminal_states).all()
            reward = int(done)
    
            assert self.observation_space.contains(position)
            return position, reward, done, {}

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