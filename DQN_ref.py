import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from NM_w_angle_ref import NeurophysicalModel


class RobotEnv(gym.Env):
    def __init__(self):
        self.grid_size = 0.05
        self.min_position = np.array([0, 0])
        self.max_position = np.array([2, 2])

        self.observation_space = spaces.Discrete(41 * 41)
        self.action_space = spaces.Discrete(125)

        self.current_position = np.array([1, 1, 0], dtype=np.float32)
        self.target_position = np.array([0.5, 0.3], dtype=np.float32)
        self.type_surface = 2
        self.time = 0.5

        self.robot_x = []
        self.robot_y = []
        self.possible_speeds = [-40, -20, 0, 20, 40]

    def reset(self):
        self.current_position = np.array([1, 1, 0], dtype=np.float32)
        return self.get_state_index(self.current_position[:2])

    def step(self, action_index):
        print(action_index)
        speed_combination_index = divmod(action_index, 25)
        speed_indices = [divmod(speed_combination_index[0], 5)[0], divmod(speed_combination_index[0], 5)[1],
                         divmod(speed_combination_index[1], 5)[1]]
        velocities = [self.possible_speeds[i] for i in speed_indices]

        print(velocities[0], velocities[1], velocities[2])

        delta_x, delta_y, angle = NeurophysicalModel(velocities[0], velocities[1], velocities[2], self.type_surface,
                                                     self.time, self.current_position[2])

        self.current_position[0] += delta_x
        self.current_position[1] += delta_y
        self.current_position[2] += angle

        distance_to_target = np.linalg.norm(self.current_position[:2] - self.target_position[:2])
        reward = -distance_to_target

        done = False

        if distance_to_target < 0.2:
            done = True
            reward = 50

        if not (0 <= self.current_position[0] <= 2) or not (0 <= self.current_position[1] <= 2):
            done = True
            reward = -100 - distance_to_target

        print(reward)

        print(self.current_position[0], self.current_position[1])

        self.robot_x.append(self.current_position[0])
        self.robot_y.append(self.current_position[1])

        return self.get_state_index(self.current_position[:2]), reward, done, {}

    def get_state_index(self, position):
        x_index = int((position[0] - self.min_position[0]) / self.grid_size)
        y_index = int((position[1] - self.min_position[1]) / self.grid_size)
        print(x_index, y_index)
        return x_index + y_index * int((self.max_position[0] - self.min_position[0]) / self.grid_size)

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)

        if len(self.robot_x) > 0:
            for i in range(len(self.robot_x) - 1):
                distance = np.linalg.norm(
                    np.array([self.robot_x[i], self.robot_y[i]]) - np.array([self.robot_x[i + 1], self.robot_y[i + 1]]))
                if distance <= 0.5:
                    self.ax.plot([self.robot_x[i], self.robot_x[i + 1]], [self.robot_y[i], self.robot_y[i + 1]], 'b-')

        if len(self.robot_x) > 0:
            self.ax.plot(self.robot_x[-1], self.robot_y[-1], 'gx')

        self.ax.plot(self.target_position[0], self.target_position[1], 'ro')
        self.ax.plot(1, 1, 'go')

        plt.draw()
        plt.pause(0.01)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.tensor([state_to_one_hot(s, env.observation_space.n) for s in state_batch], dtype=torch.float32)
    action_batch = torch.tensor(action_batch, dtype=torch.int64)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    next_state_batch = torch.tensor([state_to_one_hot(s, env.observation_space.n) for s in next_state_batch], dtype=torch.float32)
    done_batch = torch.tensor(done_batch, dtype=torch.float32)

    q_values = policy_net(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
    next_q_values = target_net(next_state_batch).max(dim=1)[0]
    expected_q_values = reward_batch + (gamma * next_q_values) * (1 - done_batch)

    loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


batch_size = 64
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
eps_decay = 10000
target_update = 10

env = RobotEnv()

policy_net = DQN(env.observation_space.n, env.action_space.n)
target_net = DQN(env.observation_space.n, env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.RMSprop(policy_net.parameters())

memory = deque(maxlen=50000)
steps_done = 0

def state_to_one_hot(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return one_hot


for episode in range(1, 50000):
    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        steps_done += 1
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay)

        if random.random() < eps_threshold:
            action = random.randint(0, env.action_space.n - 1)
        else:
            with torch.no_grad():
                one_hot_state = state_to_one_hot(state, env.observation_space.n)
                state_tensor = torch.tensor([one_hot_state], dtype=torch.float32)
                action = policy_net(state_tensor).max(dim=1)[1].item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        memory.append((state, action, reward, next_state, done))

        optimize_model(policy_net, target_net, memory, optimizer)
        state = next_state

        if done:
            break

    print(f"Episode {episode}: Total Reward = {total_reward}")

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())


torch.save(policy_net.state_dict(), "DQN.pth")

env.close()
