import numpy as np


def dyna_q(env, n, num_episodes, eps=0.1, alpha=0.5, gamma=0.95):
    """ Tabular Dyna-Q algorithm per Chapter 8.2 """
    history = [0]
    debug = False

    # Number of available actions and maximal state ravel index
    n_state, n_action = env.observation_space, len(env.base_actions)
    if debug:
        print(f"n_state: {n_state}")
        print(f"n_action: {n_action}")

    # Initialization of action value function
    q = np.zeros([n_state[0], n_state[1], n_action], dtype=float)
    if debug:
        print(f"q: {q}")

    # Initialize policy to equal-probable random
    policy = np.ones([n_state[0], n_state[1], n_action], dtype=float) / n_action
    if debug:
        print(f"policy: {policy}")
    
    assert np.allclose(np.sum(policy[0], axis=1), 1)

    # Model of a deterministic environment
    model = {}
    if debug:
        print(f"model: {model}")

    for episode in range(num_episodes):
        state = env.reset()
        p_state= [0,0]

        done = False
        while not done:
            # Sample action according to the current policy and step the environment forward
            action = np.random.choice(n_action, p=policy[state[0]][state[1]])
            next, reward, done, info = env.step(action)
            history += [reward]

            # Update q value with a q-learning update and reset visit counter
            q[state[0], state[1], action] += alpha * (reward + gamma * np.max(q[next[0]][next[1]]) - q[state[0], state[1], action])
            model[state[0], state[1], action] = next[0], next[1], reward
            if debug:
                print(f"model: {model}")

            # Planning with previously visited state-action pairs
            transitions = list(model.keys())
            if debug:
                print(f"Transitions: {transitions}")
            for i in range(n):
                p_next = [0,0]
                p_state[0], p_state[1], p_action = transitions[np.random.choice(len(transitions))]
                p_next[0], p_next[1], p_reward = model[p_state[0], p_state[1], p_action]
                q[p_state[0], p_state[1], p_action] += alpha * (p_reward + gamma * np.max(q[p_next[0], p_next[1]]) - q[p_state[0], p_state[1], p_action])

            if debug:
                print(f"q: {q}")

            # Extract eps-greedy policy from the updated q values
            policy[state[0], state[1], :] = eps / n_action
            policy[state[0], state[1], np.argmax(q[state[0], state[1]])] = 1 - eps + eps / n_action
            #assert np.allclose(np.sum(policy, axis=1), 1)
            if debug:
                print(f"policy: {policy}")

            # Prepare the next q update and increase visit counter for all states
            state = next

    if debug:
        print(f"Q: {q}")
    if debug:
        print(f"policy: {policy}")
    return q, policy, history


def dyna_q_plus(env, n, num_episodes, eps=0.1, alpha=0.5, gamma=0.95, kappa=1e-4, action_only=False):
    """ Tabular Dyna-Q+ algorithm per Chapter 8.3 (action_only=False) or Exercise 8.4 (action_only=True). """
    history = [0]

    # Number of available actions and maximal state ravel index
    n_state, n_action = env.observation_space, len(env.base_actions)

    # Initialization of action value function and visit counter
    q = np.zeros([n_state[0], n_state[1], n_action], dtype=float)
    tau = np.zeros([n_state[0], n_state[1], n_action], dtype=int)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state[0], n_state[1], n_action], dtype=float) / n_action
    #print(f"policy: {policy}")
    assert np.allclose(np.sum(policy[0], axis=1), 1)

    # Model of a deterministic environment
    model = {}

    for episode in range(num_episodes):
        state = env.reset()
        

        done = False
        while not done:
            # Sample action according to the current policy and step the environment forward
            action = np.random.choice(n_action, p=policy[state[0],state[1]])
            next, reward, done, info = env.step(action)
            history += [reward]

            # Update q value with a q-learning update and reset visit counter
            q[state[0], state[1], action] += alpha * (reward + gamma * np.max(q[next[0], next[1]]) - q[state[0], state[1], action])
            model[state[0], state[1], action] = next[0], next[1], reward
            tau[state[0], state[1], action] = 0

            # Planning that allows taking unvisited actions from visited states
            #print(f"Model: {model}")
            states = list(model.keys())
            #print(f"States: {states}")
            for i in range(n):
                p_next =[0,0]
                p_state = states[np.random.choice(len(states))]
                #print(f"p_state: {p_state}")
                #print(f"tuple(p_state): {tuple(p_state)}")
                p_action = p_state[2]
                #print(f"p_action: {p_action}")
                #print(f"model[p_state[0], p_state[1], p_action]: {model[p_state[0], p_state[1], p_action]}")
                #print(f"model.get(tuple(p_state), (p_state[0], p_state[1], 0)): {model.get(tuple(p_state), (p_state[0], p_state[1], 0))}")
                p_next[0], p_next[1], p_reward = model.get(tuple(p_state), (p_state[0], p_state[1], 0))
                bonus = 0 if action_only else kappa * np.sqrt(tau[p_state[0], p_state[1], p_action])
                q[p_state[0], p_state[1], p_action] += alpha * (p_reward + bonus + gamma * np.max(q[p_next[0], p_next[1]]) - q[p_state[0], p_state[1], p_action])

            # Extract eps-greedy policy from the updated q values and exploration bonus
            bonus = kappa * np.sqrt(tau[state[0], state[1]]) if action_only else 0
            policy[state[0], state[1], :] = eps / n_action
            #print(f"policy[state[0], state[1], :]: {policy[state[0], state[1], :]}")
            policy[state[0], state[1], np.argmax(q[state[0], state[1]] + bonus)] = 1 - eps + eps / n_action
            #print(f"policy[state[0], state[1], np.argmax(q[state[0], state[1]] + bonus)]: {policy[state[0], state[1], np.argmax(q[state[0], state[1]] + bonus)]}")
            assert np.allclose(np.sum(policy[0], axis=1), 1)

            # Prepare the next q update and increase visit counter for all states
            state = next
            tau += 1

    return q, policy, history
