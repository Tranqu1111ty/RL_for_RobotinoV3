import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from NM_w_angle_ref import NeurophysicalModel
import matplotlib.pyplot as plt
from gym import spaces
from collections import deque

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MADDPGAgent:
    def __init__(self, actor_input_dim, critic_input_dim, action_dim, lr_actor=0.01, lr_critic=0.02, gamma=0.95,
                 tau=0.01):
        self.actor = Actor(actor_input_dim, action_dim)
        self.critic = Critic(critic_input_dim)
        self.target_actor = Actor(actor_input_dim, action_dim)
        self.target_critic = Critic(critic_input_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau

    def update(self, observations, actions, rewards, next_observations, next_actions, done, agents):
        observations_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_observations_tensor = torch.FloatTensor(next_observations)
        next_actions_tensor = torch.FloatTensor(next_actions)

        Q_targets_next = self.target_critic(torch.cat([next_observations_tensor, next_actions_tensor], dim=1))
        Q_targets = rewards_tensor + (self.gamma * Q_targets_next * (1 - done))
        Q_expected = self.critic(torch.cat([observations_tensor, actions_tensor], dim=1))
        critic_loss = nn.functional.mse_loss(Q_expected, Q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = [
            agent.actor(observations_tensor) if i == agents.index(self) else agent.actor(observations_tensor).detach()
            for i, agent in enumerate(agents)]
        predicted_actions = torch.cat(predicted_actions, dim=1)
        actor_loss = -self.critic(torch.cat([observations_tensor, predicted_actions], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action


class MultiAgentEnv:
    def __init__(self, num_agents):

        self.num_agents = num_agents
        self.agents = [RobotEnv() for _ in range(num_agents)]

        self.observation_space = spaces.Tuple([agent.observation_space for agent in self.agents])
        self.action_space = spaces.Tuple([agent.action_space for agent in self.agents])

    def reset(self):

        return np.array([agent.reset() for agent in self.agents])

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []
        for agent, action in zip(self.agents, actions):
            ob, reward, done, info = agent.step(action)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.array(obs), np.array(rewards), np.array(dones), infos

    def render(self, mode='human'):
        for agent in self.agents:
            agent.render(mode=mode)


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
        reward = -distance_to_target - 1
        if invalid_action:
            reward -= 20

        done = False

        if distance_to_target < 0.1:
            done = True
            reward = 1

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
        plt.pause(0.01)



num_agents = 2  # Пример: 2 робота
actor_input_dim = ...  # Размерность входных данных для актера
critic_input_dim = ...  # Размерность входных данных для критика
action_dim = ...  # Размерность действий
num_episodes = ...  # Количество эпизодов обучения

agents = [MADDPGAgent(actor_input_dim, critic_input_dim, action_dim) for _ in range(num_agents)]
env = MultiAgentEnv()  # Инициализация мультиагентного окружения
batch_size = 64
buffer = deque(maxlen=10000)

# Цикл обучения
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    while True:
        actions = [agent.act(obs[i]) for i, agent in enumerate(agents)]
        next_obs, rewards, dones, _ = env.step(actions)
        buffer.append((obs, actions, rewards, next_obs, dones))

        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = zip(*batch)
            next_actions_batch = [agent.target_actor(next_obs).detach() for agent in agents]
            next_actions_batch = torch.cat(next_actions_batch, dim=1)
            for i, agent in enumerate(agents):
                other_agents = agents[:i] + agents[i + 1:]
                agent.update(obs_batch, actions_batch, rewards_batch, next_obs_batch, next_actions_batch, dones_batch,
                             other_agents)

        obs = next_obs
        episode_reward += sum(rewards)

        if np.any(dones):
            print(f'Episode {episode} Reward: {episode_reward}')
            break