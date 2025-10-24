import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.nxt = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, o, a, r, n, d):
        i = self.ptr % self.capacity
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.nxt[i] = n
        self.done[i] = d
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int, device="cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.tensor(self.obs[idx], dtype=torch.float32, device=device)
        act = torch.tensor(self.act[idx], dtype=torch.int64, device=device)
        rew = torch.tensor(self.rew[idx], dtype=torch.float32, device=device)
        nxt = torch.tensor(self.nxt[idx], dtype=torch.float32, device=device)
        done = torch.tensor(self.done[idx], dtype=torch.float32, device=device)
        return obs, act, rew, nxt, done
