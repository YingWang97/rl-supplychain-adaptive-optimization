import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from ..utils.replay import ReplayBuffer

class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)  # discrete approximation: we discretize actions later
        )

    def forward(self, x):
        return self.net(x)

class DiscreteWrapper:
    """
    Wraps a 2D continuous Box action into a discrete grid of KxK actions.
    """
    def __init__(self, k_per_dim=11, max_regular=100, max_expedite=30):
        self.k = k_per_dim
        self.reg_vals = np.linspace(0, max_regular, k_per_dim, dtype=np.int32)
        self.exp_vals = np.linspace(0, max_expedite, k_per_dim, dtype=np.int32)
        self.actions = np.array([(r, e) for r in self.reg_vals for e in self.exp_vals], dtype=np.int32)

    @property
    def n(self):
        return len(self.actions)

    def decode(self, idx: int):
        return self.actions[idx]

class DQNAgent:
    def __init__(self, obs_dim, act_space, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.act = act_space
        self.q = QNet(obs_dim, self.act.n).to(device)
        self.qt = QNet(obs_dim, self.act.n).to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer(50000, obs_dim)
        self.step_count = 0

    def select(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(self.act.n)
        with torch.no_grad():
            q = self.q(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    def update(self, batch_size=64):
        if len(self.buf) < batch_size:
            return 0.0
        obs, act, rew, nxt, done = self.buf.sample(batch_size, self.device)
        q = self.q(obs).gather(1, act.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.qt(nxt).max(1)[0]
            target = rew + (1 - done) * self.gamma * q_next
        loss = nn.functional.mse_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.step_count += 1
        if self.step_count % 200 == 0:
            self.qt.load_state_dict(self.q.state_dict())
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return float(loss.item())
