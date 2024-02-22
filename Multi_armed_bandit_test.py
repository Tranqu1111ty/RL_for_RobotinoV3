import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MultiArmedBandit:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.q_values = np.zeros((n_rows, n_cols, 8))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(range(len(self.actions)))
        else:
            return np.argmax([self.q_values[state[0], state[1], a] for a in range(len(self.actions))])

    def update_q_value(self, state, action, reward, next_state, alpha, gamma):
        max_next_q_value = np.max([self.q_values[next_state[0], next_state[1], a] for a in range(len(self.actions))])
        self.q_values[state[0], state[1], action] += alpha * (reward + gamma * max_next_q_value - self.q_values[state[0], state[1], action])

    def train(self, episodes, alpha, gamma, epsilon):
        q_values_history = []
        for _ in range(episodes):
            state = (np.random.randint(self.n_rows), np.random.randint(self.n_cols))
            while True:
                action = self.select_action(state, epsilon)
                next_state = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
                if 0 <= next_state[0] < self.n_rows and 0 <= next_state[1] < self.n_cols:

                    reward = 1 if next_state == (self.n_rows - 1, self.n_cols - 1) else 0
                    self.update_q_value(state, action, reward, next_state, alpha, gamma)
                    state = next_state
                    if next_state == (self.n_rows - 1, self.n_cols - 1):
                        break
            q_values_history.append(self.q_values.copy())
        return q_values_history


n_rows = 25
n_cols = 25
bandit = MultiArmedBandit(n_rows, n_cols)

q_values_history = bandit.train(episodes=100000, alpha=0.1, gamma=0.9, epsilon=0.2)

def update_plot(frame):
    ax.clear()
    ax.imshow(q_values_history[frame].max(axis=2), cmap='viridis', origin='lower')
    ax.set_title(f'Episode {frame+1}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    return


fig, ax = plt.subplots()
animation = FuncAnimation(fig, update_plot, frames=len(q_values_history), repeat=False)
plt.show()
