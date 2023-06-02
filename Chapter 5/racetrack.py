"""
This is the RaceTrack Environment

"""
from typing import Tuple
import numpy as np
import scipy.stats
import skimage.draw

# import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class racetrack():
    
    start_line = [[np.array([0,3]), np.array([0,4]), np.array([0,5]), np.array([0,6]), np.array([0,7]), np.array([0,8]), np.array([0,9])],
                  [np.array([0,0]), np.array([0,1]), np.array([0,2]), np.array([0,3]), np.array([0,4]), np.array([0,5]), np.array([0,6]), np.array([0,7]), np.array([0,8]), 
                  np.array([0,9]), np.array([0,10]), np.array([0,11]), np.array([0,12]), np.array([0,13]), np.array([0,14]), np.array([0,15]), np.array([0,16]), np.array([0,17]),
                  np.array([0,18]), np.array([0,19]), np.array([0,20]), np.array([0,21]), np.array([0,22])]]
    track_boundaries_left = [[np.array([0,3]), np.array([1,3]), np.array([2,3]), 
                              np.array([3,2]), np.array([4,2]), np.array([5,2]), np.array([6,2]), np.array([7,2]), np.array([8,2]), np.array([9,2]), 
                               np.array([10,1]), np.array([11,1]), np.array([12,1]), np.array([13,1]), np.array([14,1]), np.array([15,1]), np.array([16,1]), np.array([17,1]), 
                               np.array([18,0]), np.array([19,0]), np.array([20,0]), np.array([21,0]), np.array([22,0]), np.array([23,0]), np.array([24,0]), np.array([25,0]), 
                               np.array([26,0]), np.array([27,0]), 
                               np.array([28,1]), 
                               np.array([29,2]), np.array([30,2]), 
                               np.array([31,3])],
                             [np.array([2,0]), np.array([3,1]), np.array([4,2]), np.array([5,3]), np.array([6,4]), np.array([7,5]), np.array([8,6]), np.array([9,7]), 
                              np.array([10,8]), np.array([11,9]), np.array([12,10]), np.array([13,11]), np.array([14,12]), np.array([15,13]), 
                              np.array([16,14]), np.array([17,14]), np.array([18,14]), np.array([19,14]), np.array([20,14]), 
                              np.array([21,13]), 
                              np.array([22,12]), 
                              np.array([23,11]), np.array([24,11]), np.array([25,11]), np.array([26,11]), 
                              np.array([27,12]), 
                              np.array([28,13]), 
                              np.array([29,16])]]
    track_boundaries_top = [[np.array([32,0]), np.array([32,1]), np.array([32,2]), np.array([32,3]), np.array([32,4]), np.array([32,5]), np.array([32,6]), np.array([32,7]), 
                              np.array([32,8]), np.array([32,9]), np.array([32,10]), np.array([32,11]), np.array([32,12]), np.array([32,13]), np.array([32,14]), np.array([32,15]), 
                              np.array([32,16]), np.array([32,17])], 
                              [np.array([30,16]), np.array([30,17]), np.array([30,18]), np.array([30,19]), np.array([30,20]), np.array([30,21]), np.array([30,22]), np.array([30,23]), 
                               np.array([30,24]), 
                              np.array([30,25]), np.array([30,26]), np.array([30,27]), np.array([30,28]), np.array([30,29]), np.array([30,30])]]
    finish_line = [[np.array([26,16]), np.array([27,16]), np.array([28,16]), np.array([29,16]), np.array([30,16]), np.array([31,16])], 
                   [np.array([29,31]), np.array([28,31]), np.array([27,31]), np.array([26,31]), np.array([25,31]), np.array([24,31]), np.array([23,31]), np.array([22,31]), 
                    np.array([21,31])]]
    track_boundaries_right = [[np.array([0,9]), np.array([1,9]), np.array([2,9]), np.array([3,9]), np.array([4,9]), np.array([5,9]), np.array([6,9]), np.array([7,9]), 
                                np.array([8,9]), np.array([9,9]), np.array([10,9]), np.array([11,9]), np.array([12,9]), np.array([13,9]), np.array([14,9]), np.array([15,9]), 
                                np.array([16,9]), np.array([17,9]), np.array([18,9]), np.array([19,9]), np.array([20,9]), np.array([21,9]), np.array([22,9]), np.array([23,9]), 
                                np.array([24,9]), 
                                np.array([25,10]), 
                                np.array([26,17]), np.array([27,17]), np.array([28,17]), np.array([29,17]), np.array([30,17]), np.array([31,17])], 
                              [np.array([20,31]), np.array([20,30]), 
                               np.array([19,29]), np.array([19,28]), np.array([19,27]), 
                               np.array([18,26]),  
                               np.array([17,24]), np.array([17,25]), 
                               np.array([16,23]), np.array([15,23]), np.array([14,23]), np.array([13,23]), np.array([12,23]), np.array([11,23]), np.array([10,23]), 
                               np.array([9,23]), np.array([8,23]), np.array([7,23]), np.array([6,23]), np.array([5,23]), np.array([4,23]), np.array([3,23]), np.array([2,23]), 
                               np.array([1,23]) , np.array([0,23])]]

    
    min_velocity = np.array([0,0])
    max_velocity = np.array([5,5])
    
    action_space = [np.array([-1,-1]), np.array([-1,0]), np.array([-1,1]), np.array([0,-1]), np.array([0,0]), np.array([0,1]), np.array([1,-1]), np.array([1,0]), np.array([1,1])]
    
    gamma = 1
    
    state = [np.array([0,0]), np.array([0,0])]
    
    def __init__(self, curve_number, noisy=True):
        
        self.curve_number = curve_number
        
        if curve_number == 1:
            self.curve = np.full((32, 17), 1)
        else:
            self.curve = np.full((30, 32), 1)
        
        # Finish Line == 3
        # Start Line == 2
        # Inside Track == 1
        # Off-Track == 0
        
        for box in self.finish_line[curve_number - 1]:
            self.curve[tuple(box)] = 3
        for box in self.start_line[curve_number - 1]:
            self.curve[tuple(box)] = 2
        for box in self.track_boundaries_left[curve_number - 1]:
            for i in range(0, box[1]):
                self.curve[box[0], i] = 0
        for box in self.track_boundaries_right[curve_number - 1]:
            for i in range(box[1], self.curve.shape[1]):
                self.curve[box[0], i] = 0
        for box in self.track_boundaries_top[curve_number - 1]:
            for i in range(box[0], self.curve.shape[0]):
                self.curve[i, box[1]] = 0
                    
        self.noisy = noisy
        self.ax = None
        
        self.state = self.reset()
        
    def reset(self):
        
        if self.curve_number == 1:
            state = [np.array([0, np.random.randint(3,9)]), np.array([0,0])]  
        else:
            state = [np.array([0, np.random.randint(0,22)]), np.array([0,0])] 
            
        self.state = state.copy()
        
        return state
    
    def step(self, action):
        
        #print(f"-----Step------")
        #print(f"State: {self.state}")
        #print(f"Action: {action}")
        
        reward = 0
        done = False
        
        #Convert action number to acceleration
        if self.noisy and np.random.rand() < 0.1:
            acceleration = np.array([0, 0])
        else:
            acceleration = self.action_space[action]
            
        #print(f"Acceleration: {acceleration}")
        
        velocity = self.state[1] + acceleration
        
        # Limiting velocity between min and max
        velocity = np.minimum(velocity, self.max_velocity)
        velocity = np.maximum(velocity, self.min_velocity)
        
        # Prevent Zero Speed
        if (velocity == 0).all():
            velocity = np.array([0,1])
            
        self.state[1] = velocity
        
        #print(f"Velocity: {self.state[1]}")
            
        new_position = self.state[0] + self.state[1]
        
        # Calculate intersection of the speed vector and the racetrack
        ys, xs = skimage.draw.line(*self.state[0], *new_position)
        ys_within_track = np.clip(ys, 0, self.curve.shape[0] - 1)
        xs_within_track = np.clip(xs, 0, self.curve.shape[1] - 1)
        collisions = self.curve[ys_within_track, xs_within_track]
        
        if (collisions == 3).any():
            reward = 0
            done = True
            
            #print(f"Finish Line")
            
            # Clip position to racetrack limits (applies when crossing the finish line)
            self.state[0] = np.minimum(new_position, np.array(self.curve.shape))
            
        else:
            reward = -1
            within_track_limits = (xs == xs_within_track).all() and (ys == ys_within_track).all()
            if (collisions == 0).any() or (not within_track_limits):
                self.state = self.reset()
                #print(f"Out of Track")
            else:
                self.state[0] = new_position
        
        #print(f"State: {self.state}")
        
        return self.state, reward, done
 
    def render(self, mode='human', reset=None):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Racetrack background with custom colormap
            cmap = mcolors.ListedColormap(['white', 'gray', 'red', 'green'])
            self.ax.imshow(self.curve, aspect='equal', origin='lower', cmap=cmap)

            # Major tick marks max_speed step apart
            self.ax.set_xticks(np.arange(0, self.curve.shape[1], self.max_velocity[1]), minor=False)
            self.ax.set_yticks(np.arange(0, self.curve.shape[0], self.max_velocity[0]), minor=False)

            margin = 1
            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5 - margin, self.curve.shape[1] + margin, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5 - margin, self.curve.shape[0] + margin, 1), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.05)
            self.ax.tick_params(which='minor', length=0)
            self.ax.set_frame_on(False)

        position = np.flip(self.state[0])
        speed = np.flip(self.state[1])

        if reset is not None:
            # Reset arrow pointing from the reset position to the current car position
            new_state = reset(self.curve_number)
            reset_position = np.flip(new_state[0])
            reset_speed = np.flip(new_state[1])
            patch = mpatches.FancyArrow(*reset_position, *(position - reset_position), color='blue',
                                        fill=False, width=0.10, head_width=0.25, length_includes_head=True)
        else:
            # Speed arrow pointing to the the current car position
            if (speed == 0).all():
                patch = mpatches.Circle(position, radius=0.1, color='black', zorder=1)
            else:
                patch = mpatches.FancyArrow(*(position - speed), *speed, color='black',
                                            zorder=2, fill=True, width=0.05, head_width=0.25, length_includes_head=True)
        self.ax.add_patch(patch)
        return self.ax