import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from env_hiv import HIVPatient


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch)))
        )

    def __len__(self):
        return len(self.data)


env = HIVPatient(domain_randomization=False)


class ProjectAgent:
    def act(self, observation, use_random=False):
        return 0

    def save(self, path):
        pass

    def load(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass


class RandomAgent(ProjectAgent):
    def act(self, observation, use_random=True):
        return env.action_space.sample()


class AlwaysPrescribe(ProjectAgent):
    def act(self, observation, use_random=True):
        return 3


dqn_architecture = torch.nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, env.action_space.n),
)
dqn_config = {
    "buffer_size": 1e7,
    "batch_size": 100,
    "gamma": 0.9,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay_period": 20000,
    "epsilon_delay_decay": 1000,
    "learning_rate": 0.001,
    "gradient_steps": 5,
    "update_target_freq": 100,
}


class DQN(ProjectAgent):
    def __init__(self, config=dqn_config, model=dqn_architecture):
        # Init config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = config["gamma"] if "gamma" in config.keys() else 0.95
        self.batch_size = config["batch_size"] if "batch_size" in config.keys() else 100
        buffer_size = (
            config["buffer_size"] if "buffer_size" in config.keys() else int(1e5)
        )
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = (
            config["epsilon_max"] if "epsilon_max" in config.keys() else 1.0
        )
        self.epsilon_min = (
            config["epsilon_min"] if "epsilon_min" in config.keys() else 0.01
        )
        self.epsilon_stop = (
            config["epsilon_decay_period"]
            if "epsilon_decay_period" in config.keys()
            else 10000
        )
        self.epsilon_delay = (
            config["epsilon_delay_decay"]
            if "epsilon_delay_decay" in config.keys()
            else 1000
        )
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = (
            config["criterion"] if "criterion" in config.keys() else torch.nn.MSELoss()
        )
        lr = config["learning_rate"] if "learning_rate" in config.keys() else 0.001
        self.optimizer = (
            config["optimizer"]
            if "optimizer" in config.keys()
            else torch.optim.Adam(self.model.parameters(), lr=lr)
        )
        self.nb_gradient_steps = (
            config["gradient_steps"] if "gradient_steps" in config.keys() else 1
        )
        self.update_target_strategy = (
            config["update_target_strategy"]
            if "update_target_strategy" in config.keys()
            else "replace"
        )
        self.update_target_freq = (
            config["update_target_freq"]
            if "update_target_freq" in config.keys()
            else 20
        )
        self.update_target_tau = (
            config["update_target_tau"]
            if "update_target_tau" in config.keys()
            else 0.005
        )
        # Init iterative variables
        self.step = 0
        self.target_step = 0
        self.epsilon = self.epsilon_max

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.detach().item()

    def remember(self, state, action, reward, next_state, done):
        return self.memory.append(state, action, reward, next_state, done)

    def learn(self):
        # train
        tot_loss = 0
        for _ in range(self.nb_gradient_steps):
            loss = self.gradient_step()
            if loss is not None:
                tot_loss += loss / self.nb_gradient_steps
        # update target network if needed
        if self.update_target_strategy == "replace":
            if self.target_step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        if self.update_target_strategy == "ema":
            target_state_dict = self.target_model.state_dict()
            model_state_dict = self.model.state_dict()
            tau = self.update_target_tau
            for key in model_state_dict:
                target_state_dict[key] = (
                    tau * model_state_dict + (1 - tau) * target_state_dict
                )
            self.target_model.load_state_dict(target_state_dict)
        self.target_step += 1
        return tot_loss

    def act(self, observation, use_random=False):
        if self.step > self.epsilon_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if use_random and np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        else:
            Q = self.model(
                torch.tensor(
                    observation, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            )
            action = torch.argmax(Q).item()
            # if self.step > 2000:
            #    print("Q: ", Q)

        self.step += 1
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load("./model.pt"))
        self.model.eval()
