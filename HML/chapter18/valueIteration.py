import numpy as np
from mdpExample import transition_probabilities, rewards, possible_actions

Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0

gamma = 0.9  # discount factor

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp]
                * rewards[s][a][sp] + gamma*np.max(Q_prev[sp])
                for sp in range(3)])
