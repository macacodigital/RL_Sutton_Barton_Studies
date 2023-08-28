"""
This is a on-policy-sarsa_agent implementation

"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def initialize_state_values(shape):
    
    #shape is tuple
    state_values = np.ones(shape, dtype=float) * 0.5
    
    return state_values

def initialize_action_values(shape):
    
    #shape is a tuple
    action_values = np.zeros(shape)
    
    return action_values

def policy(state, action_values, action_space, epsilon=0.1):
        
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    else:
        av = action_values[state[0], state[1]]
        return np.argmax(av)

def run_episode(env, epsilon=0.2, render=True):
    steps = 0
    transitions = []

    state = env.reset()

    done = False

    while not done and (steps < 3000):
        action = env.policy()
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

def n_step_td_estimating(env, state_values, episodes, n=2, alpha=0.5, gamma=1, epsilon=0.1, render=False, debug=False):
    
    steps = 0
    total_steps = 0
    previous_episode = 0
    history = []
        
    if render and (episodes > 10):
        render = False
        print(f"Too many episodes to render!")
        
    if debug:
        episodes = 10
    
    for episode in range(1, episodes + 1):
        
        transitions = []
        rewards = []
        
        done = False

        state = env.reset()
        transitions.append(state.copy())
        rewards.append(0)
        
        T = float('inf')
        step = 0
        G = 0
        time = 0
        
        while (time != (T - 1)):
            
            
            print(f"---------------------------") if debug else 0
            print(f"Step: {step}") if debug else 0
            print(f"T: {T}") if debug else 0
            
            G = 0
            
            if step < T:
                action = env.policy()
                print(f"Action: {action}") if debug else 0
                next_state, reward, done = env.step(action)
                transitions.append(next_state)
                print(f"Transitions: {transitions}") if debug else 0
                rewards.append(reward)
                print(f"Rewards: {rewards}") if debug else 0
                if done:
                    T = step + 1
            time = step - n + 1
            print(f"time: {time}") if debug else 0
                
            if (time >= 0):
                for i in range(time + 1, min(time + n, T) + 1):
                    G += gamma ** (i - time  - 1) * rewards[i]
                    print(f"G: {G} : gamma: {gamma} | exp: {i - time  - 1} | gamma**exp: {gamma**(i - time  - 1)} | Reward: {rewards[i]}") if debug else 0
                if ((time + n) < T):
                    G += (gamma ** n) * state_values[tuple(transitions[time + n])]
                    print(f"G Final: {G}") if debug else 0
                state_values[tuple(transitions[time])] += alpha * (G - state_values[tuple(transitions[time])])
                print(f"G: {G} | State Value {transitions[time]}: {state_values[tuple(transitions[time])]}") if debug else 0

            state = next_state.copy()
            step += 1
                                                               
        if (episode % (episodes/10) == 0):
            print(f"Episode: {episode} Finished | Average Steps: {step/(episode - previous_episode)}")
            total_steps += step
            step = 0
            previous_episode = episode
            
        print(f"State_Values: {state_values}") if debug else 0
        print(f"State_Values: {state_values[0][1:-1]}") if debug else 0
        history.append(state_values[0][1:-1].copy())
        print(f"History: {history}") if debug else 0
            
    print(f"Total number of episodes: {episodes}")
    print(f"Total number of steps: {total_steps}")
    print(f"Total Average of Steps Per Episode: {total_steps/episodes}")
    
    return history

def sum_td_errors_estimating(env, state_values, episodes, n=2, alpha=0.5, gamma=1, epsilon=0.1, render=False, debug=False):
    
    steps = 0
    total_steps = 0
    previous_episode = 0
    history = []
        
    if render and (episodes > 10):
        render = False
        print(f"Too many episodes to render!")
        
    if debug:
        episodes = 10
    
    for episode in range(1, episodes + 1):
        
        transitions = []
        rewards = []
        
        done = False
    
        state = env.reset()
        transitions.append(state.copy())
        rewards.append(0)
        
        step = 0
        
        while not done:
            
            print(f"---------------------------") if debug else 0
            print(f"Step: {step}") if debug else 0
            
            action = env.policy()
            print(f"Action: {action}") if debug else 0
            next_state, reward, done = env.step(action)
            transitions.append(next_state)
            print(f"Transitions: {transitions}") if debug else 0
            rewards.append(reward)
            print(f"Rewards: {rewards}") if debug else 0
            state = next_state.copy()
            step += 1
            
        for i in range(0, step + 1):
            
            err_sum = 0
            
            for j in range(i, step + 1):
                
                n_step_reward = 0
                start_step = j + 1
                stop_step = min(j + n, step)
                
                print(f"Transitions {j}-{stop_step}: {transitions[j:stop_step+1]}") if debug else 0
                print(f"Rewards {j}-{stop_step}: {rewards[j:stop_step+1]}") if debug else 0
                
                for s in range (start_step, stop_step+1):
                    n_step_reward += gamma ** (s - start_step) * rewards[s]
                    print(f"n_step_reward: {n_step_reward}") if debug else 0
                if stop_step < step:    
                    err_sum += gamma ** (j - i) * (n_step_reward + gamma ** (stop_step - start_step) * state_values[tuple(transitions[stop_step])] - state_values[tuple(transitions[j])])
                else:
                    err_sum += gamma ** (j - i) * (n_step_reward - state_values[tuple(transitions[j])])
                print(f"err_sum: {err_sum}") if debug else 0
            state_values[tuple(transitions[i])] = state_values[tuple(transitions[i])] + alpha * (err_sum)
            print(f"err_sum: {err_sum} | State Value {transitions[i]}: {state_values[tuple(transitions[i])]}") if debug else 0
                                                               
        if (episode % (episodes/10) == 0):
            print(f"Episode: {episode} Finished | Average Steps: {step/(episode - previous_episode)}")
            total_steps += step
            step = 0
            previous_episode = episode
            
        print(f"State_Values: {state_values}") if debug else 0
        print(f"State_Values: {state_values[0][1:-1]}") if debug else 0
        history.append(state_values[0][1:-1].copy())
        print(f"History: {history}") if debug else 0
            
    print(f"Total number of episodes: {episodes}")
    print(f"Total number of steps: {total_steps}")
    print(f"Total Average of Steps Per Episode: {total_steps/episodes}")
    
    return history

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
    