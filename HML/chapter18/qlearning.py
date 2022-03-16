import numpy as np
from mdpExample import transition_probabilities, rewards, possible_actions


def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    # Transition randomly, using a (state, action) pair.
    reward = rewards[state][action][next_state]
    return next_state, reward


def exploration_policy(state):
    return np.random.choice(possible_actions[state])


Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0


alpha0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.90  # discount factor
state = 0  # initial state

for iterations in range(10000):
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state])
    alpha = alpha0 / (1+iterations*decay)

    # The formulation I am used to:
    # TD = (reward + gamma*next_value - Q_values[state, action])
    # Q_values[state, action] = Q_values[state, action] + alpha*TD

    # This is from the book, rather weird to not split off the TD error.
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state
