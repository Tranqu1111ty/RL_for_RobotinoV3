import gym
import numpy as np
import matplotlib.pyplot as plt
from NM_w_angle_ref import NeurophysicalModel
from gym import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from torch.optim import Adam

class RobotEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([0, 0, -2]), high=np.array([2, 2, 2]), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-40, -40, -40], dtype=np.float64), high=np.array([40, 40, 40], dtype=np.float64))

        self.current_position = np.array([0, 0, 0], dtype=np.float32)
        self.target_position = np.array([1, 1, 0], dtype=np.float32)
        self.type_surface = 2
        self.time = 0.5

        self.robot_x = []
        self.robot_y = []

    def reset(self):
        self.current_position = np.array([0, 0, 0], dtype=np.float32)
        return self.current_position

    def step(self, action):
        velocity_1, velocity_2, velocity_3 = action

        delta_x, delta_y, angle = NeurophysicalModel(velocity_1, velocity_2, velocity_3, self.type_surface,
                                                     self.time, self.current_position[2])

        invalid_action = False
        if -5 < velocity_1 < 5:
            invalid_action = True
        if -5 < velocity_2 < 5:
            invalid_action = True
        if -5 < velocity_3 < 5:
            invalid_action = True

        self.current_position[0] += delta_x
        self.current_position[1] += delta_y
        self.current_position[2] += angle
        print(velocity_1, velocity_2, velocity_3)
        print(self.current_position[2])
        print(angle)

        distance_to_target = np.linalg.norm(self.current_position[:2] - self.target_position[:2])
        reward = -distance_to_target

        if invalid_action:
            reward -= 10

        done = False

        if distance_to_target < 0.1:
            done = True
            reward = 20

        if not (0 <= self.current_position[0] <= 2) or not (0 <= self.current_position[1] <= 2):
            done = True
            reward = -10 - distance_to_target

        self.robot_x.append(self.current_position[0])
        self.robot_y.append(self.current_position[1])

        return self.current_position, reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(-0.5, 2.5)

        if len(self.robot_x) > 1:
            self.ax.plot(self.robot_x[:-1], self.robot_y[:-1], 'b-')

        self.ax.plot(self.robot_x[-1], self.robot_y[-1], 'bx')

        self.ax.plot(self.target_position[0], self.target_position[1], 'ro')

        plt.draw()
        plt.pause(0.1)


env = DummyVecEnv([lambda: RobotEnv()])

policy_kwargs = {
    'net_arch': [256, 256, 256],
}
model = A2C("MlpPolicy", env, learning_rate=0.3, ent_coef=0.1, policy_kwargs=policy_kwargs)

obs = env.reset()
for i in range(500000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    plt.pause(0.1)
    plt.draw()

model.save("A2C_MlpPolicy_500000_steps.zip")

# Для метрик
rewards = []
episode_lengths = []
action_counts = []

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)

    rewards.append(reward)
    episode_lengths.append(info.get('episode')['l'])
    action_counts.append(len(action))

    env.render()
    if done:
        obs = env.reset()

    plt.pause(0.001)
    plt.draw()


average_reward = np.mean(rewards)
average_episode_length = np.mean(episode_lengths)
average_action_count = np.mean(action_counts)

print("Average Reward:", average_reward)
print("Average Episode Length:", average_episode_length)
print("Average Action Count:", average_action_count)
