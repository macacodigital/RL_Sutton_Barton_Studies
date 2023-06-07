"""
This is a on-policy-sarsa_agent implementation

"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def initialize_action_values(shape):
    
    #shape is a 3 element tuple
    action_values = np.zeros(shape)
    
    return action_values

def policy(state, action_values, action_space, epsilon=0.1):
        
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    else:
        av = action_values[state[0], state[1]]
        return np.argmax(av)

def run_episode(env, action_values, epsilon=0.2, render=True):
    steps = 0
    transitions = []

    state = env.reset()

    done = False

    while not done and (steps < 3000):
        action = policy(env.state, action_values, env.action_space, epsilon)
        next_state, reward, done = env.step(action)
        steps += 1
        transitions.append([state, action, reward])
        #print(f"State: {state}, Action: {env.action_labels[action]}, Reward: {reward}, Next State: {next_state}")
        state = next_state.copy()
        if render:
            env.render()

    #if render:
        #print(f"Steps: {steps}")
        
    return transitions

def on_policy_td_control(env, action_values, episodes, alpha=0.5, gamma=1, epsilon=0.1, render=False):
    
    steps = 0
    total_steps = 0
    previous_episode = 0
        
    if render and (episodes > 10):
        render = False
        print(f"Too many episodes to render!")
    
    for episode in range(1, episodes + 1):
        
        done = False
    
        state = env.reset()
        
        action = policy(state, action_values, env.action_space, epsilon)
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = policy(next_state, action_values, env.action_space, epsilon)
            
            qsa = action_values[state[0]][state[1]][action]
            
            if not done:
                next_qsa = action_values[next_state[0]][next_state[1]][next_action]
            else:
                next_qsa = 0
                
            action_values[state[0]][state[1]][action] = qsa + alpha * (reward + gamma * next_qsa - qsa)
            
            state = next_state.copy()
            action = next_action
            steps += 1
                                                               
        if (episode % (episodes/10) == 0):
            print(f"Episode: {episode} Finished | Average Steps: {steps/(episode - previous_episode)}")
            total_steps += steps
            steps = 0
            previous_episode = episode
            
    print(f"Total number of episodes: {episodes}")
    print(f"Total number of steps: {total_steps}")
    print(f"Total Average of Steps Per Episode: {total_steps/episodes}")
            

def plot_results(observation_space, env_actions, action_values):
    
    #observation_space is a np.array([X, Y])
    
    matplotlib.rcParams['figure.figsize'] = [10, 10]
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Optimal Value Function and Policy")
    
    action_values = np.copy(action_values)
    unvisited = np.where(action_values == 0)
    v = np.max(action_values, axis=2).reshape(observation_space)
    ax.imshow(v, origin='lower')

    xr = []
    yr = []
    
    actions = np.argmax(action_values,axis=2)
    arrows = np.empty((np.shape(actions)[0], np.shape(actions)[1],2))
    for row in range(len(actions)):
        for col in range(len(actions[row])):
            arrows[row][col][0] = env_actions[actions[row][col]][0]
            arrows[row][col][1] = env_actions[actions[row][col]][1]
            yr.append(row)
            xr.append(col)
    ax.quiver(xr, yr, arrows[:, :, 1], arrows[:, :, 0], pivot='mid')
    