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
        self.grid_size = 0.01
        self.min_position = np.array([0, 0])
        self.max_position = np.array([0.3, 0.3])

        self.possible_speeds = [-40, -34.641, -28.2843, -25.7115, -20, -16.9047, -10.3528, -6.94593, 0,
                                40, 34.641, 28.2843, 25.7115, 20, 16.9047, 10.3528, 6.94593]

        self.all_speed_combinations = [(v1, v2, v3) for v1 in self.possible_speeds for v2 in self.possible_speeds for v3 in
                                  self.possible_speeds]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.all_speed_combinations))

        self.current_position = np.array([0.15, 0.15, 0], dtype=np.float32)
        self.circle_radius = 0.1
        self.circle_points = self.generate_points_on_circle(self.circle_radius, num_points=24)
        self.target_position_index = 0
        self.target_position = np.array([0.15, 0.15], dtype=np.float32) + self.circle_points[self.target_position_index]
        self.type_surface = 2
        self.time = 0.5

        self.successful_episodes = 0
        self.max_steps = 5

        self.robot_x = []
        self.robot_y = []

        self.target_reached_count = 0
        self.max_target_reached_count = 1

    def generate_points_on_circle(self, radius, num_points):
        points = []
        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append((x, y))
        return points

    def set_target_point_index(self, point_index):
        self.target_position_index = point_index
        self.target_position = np.array([0.15, 0.15], dtype=np.float32) + self.circle_points[self.target_position_index]

    def reset(self):
        self.current_position = np.array([0.15, 0.15, 0], dtype=np.float32)
        self.step_count = 0
        return self.get_state_index(self.current_position[:2])

    def step(self, action_index):
        speed_combination = self.all_speed_combinations[action_index]
        velocities = list(speed_combination)
        print("скорости робота:", velocities)
        print("количество успешных эпизодов: ", self.successful_episodes)
        delta_x, delta_y, angle, speed_motor_1, speed_motor_2, speed_motor_3, current_first_motor_on_grey, \
        current_second_motor_on_grey, current_third_motor_on_grey, slippage_first_grey, slippage_second_grey, \
        slippage_third_grey = NeurophysicalModel(velocities[0], velocities[1], velocities[2], self.type_surface,
                                                     self.time, self.current_position[2])

        self.current_position[0] += delta_x
        self.current_position[1] += delta_y
        self.current_position[2] += angle

        distance_to_target = np.linalg.norm(self.current_position[:2] - self.target_position[:2])
        reward = -distance_to_target

        done = False

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True
            reward = -10

        if distance_to_target < 0.02:
            self.successful_episodes += 1
            self.target_reached_count += 1
            if self.target_reached_count >= self.max_target_reached_count:
                self.target_position_index = (self.target_position_index + 1) % len(self.circle_points)
                self.target_reached_count = 0

            self.target_position = np.array([0.15, 0.15], dtype=np.float32) + self.circle_points[self.target_position_index]
            done = True
            reward = 10

        if not (0 <= self.current_position[0] <= 0.3) or not (0 <= self.current_position[1] <= 0.3):
            done = True
            reward = -10 - distance_to_target

        self.robot_x.append(self.current_position[0])
        self.robot_y.append(self.current_position[1])

        print(*self.circle_points[self.target_position_index],
            speed_motor_1,
            speed_motor_2,
            speed_motor_3,
            current_first_motor_on_grey,
            current_second_motor_on_grey,
            current_third_motor_on_grey,
            slippage_first_grey,
            slippage_second_grey,
            slippage_third_grey)

        state = np.array([
            *self.circle_points[self.target_position_index],
            speed_motor_1,
            speed_motor_2,
            speed_motor_3,
            current_first_motor_on_grey,
            current_second_motor_on_grey,
            current_third_motor_on_grey,
            slippage_first_grey,
            slippage_second_grey,
            slippage_third_grey
        ], dtype=np.float32)

        return state, reward, done, {}

    def get_state_index(self, position):
        x_index = int((position[0] - self.min_position[0]) / self.grid_size)
        y_index = int((position[1] - self.min_position[1]) / self.grid_size)
        return x_index + y_index * int((self.max_position[0] - self.min_position[0]) / self.grid_size)

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(0, 0.3)
        self.ax.set_ylim(0, 0.3)

        self.ax.grid(True)

        self.ax.set_xlabel('X coordinate (metres)')
        self.ax.set_ylabel('Y coordinate (metres)')

        if len(self.robot_x) > 0:
            start_index = max(0, len(self.robot_x) - 200)
            for i in range(start_index, len(self.robot_x) - 1):
                distance = np.linalg.norm(
                    np.array([self.robot_x[i], self.robot_y[i]]) - np.array([self.robot_x[i + 1], self.robot_y[i + 1]]))
                if distance <= 0.1:
                    self.ax.plot([self.robot_x[i], self.robot_x[i + 1]], [self.robot_y[i], self.robot_y[i + 1]], 'b-')

        if len(self.robot_x) > 0:
            self.ax.plot(self.robot_x[-1], self.robot_y[-1], 'gx')
            self.ax.text(self.robot_x[-1], self.robot_y[-1], 'Current location', fontsize=8, color='r')

        self.ax.plot(self.target_position[0], self.target_position[1], 'ro')
        self.ax.plot(0.15, 0.15, 'go')
        self.ax.plot(1, 1, 'go')

        self.ax.text(self.target_position[0], self.target_position[1], 'Target', fontsize=12, color='r')

        self.ax.text(0.15, 0.15, 'Start', fontsize=12, color='g')

        plt.draw()

        if len(self.robot_x) % 1000 == 0:
            plt.savefig(f"step_{self.step_count}.png")

        plt.pause(1)

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


def state_to_one_hot(state):
    shape = env.observation_space.shape

    state_scaled = (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)

    state_indices = np.digitize(state_scaled, bins=np.linspace(0, 1, num=shape[0])) - 1

    num_states = np.prod(shape)
    one_hot = np.zeros(num_states)
    one_hot[state_indices] = 1

    return one_hot

def load_model(model_path, input_dim, output_dim):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Перевод модели в режим оценки
    return model

# Инициализация среды
env = RobotEnv()

# Загрузка обученной модели
model_path = 'DQN_20_000.pth'  # Укажите путь к вашей обученной модели
policy_net = load_model(model_path, env.observation_space.shape[0], env.action_space.n)

# Тестирование модели
for episode in range(1, 100):  # Вы можете изменить количество эпизодов для тестирования
    state = env.reset()
    total_reward = 0
    while True:
        env.render()

        # Выбор действия исключительно на основе модели без исследовательской стратегии
        with torch.no_grad():
            one_hot_state = state_to_one_hot(state)
            state_tensor = torch.tensor([one_hot_state], dtype=torch.float32)
            action = policy_net(state_tensor).max(dim=1)[1].item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

        state = next_state

    print(f"Episode {episode}: Total Reward = {total_reward}")

env.close()
