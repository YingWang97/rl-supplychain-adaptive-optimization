import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.log_std.exp()

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.v(x)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, device="cpu"):
        self.device = device
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip

    def select(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std = self.actor(obs)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(dim=1)
        return a.squeeze(0).detach().cpu().numpy(), logp.detach().cpu().numpy()

    def update(self, batch):
        obs, act, adv, ret, logp_old = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch]
        mu, std = self.actor(obs)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act).sum(dim=1, keepdim=True)
        ratio = torch.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
        loss_a = -torch.min(surr1, surr2).mean()
        val = self.critic(obs).squeeze(1)
        loss_c = ((val - ret) ** 2).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        return float(loss_a.item()), float(loss_c.item())
