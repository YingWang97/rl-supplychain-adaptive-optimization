import argparse, os, time, csv
import numpy as np
import torch

from rlsupplychain.envs.supply_chain_env import SupplyChainEnv, CostParams
from rlsupplychain.policies.dqn import DQNAgent, DiscreteWrapper
from rlsupplychain.policies.ppo import PPOAgent
from rlsupplychain.policies.classic import base_stock_policy, sS_policy
from rlsupplychain.utils.seed import set_seed

def run_dqn(args):
    env = SupplyChainEnv(horizon=args.horizon, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act = DiscreteWrapper(k_per_dim=args.k, max_regular=env.max_order, max_expedite=env.max_expedite)
    agent = DQNAgent(obs_dim, act, lr=args.lr, device=args.device)
    run_dir = os.path.join("runs", args.experiment)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train.csv"), "w", newline="")
    writer = csv.DictWriter(log, fieldnames=["episode","return","loss","epsilon"])
    writer.writeheader()
    for ep in range(args.episodes):
        o,_ = env.reset(seed=args.seed + ep)
        ep_ret, loss = 0.0, 0.0
        for t in range(env.horizon):
            a_idx = agent.select(o)
            a = act.decode(a_idx)
            n, r, term, trunc, info = env.step(a)
            agent.buf.push(o, a_idx, r, n, float(term or trunc))
            l = agent.update(args.batch)
            loss = l or loss
            o = n; ep_ret += r
            if term or trunc: break
        writer.writerow({"episode": ep, "return": ep_ret, "loss": loss, "epsilon": agent.eps})
    torch.save(agent.q.state_dict(), os.path.join(run_dir, "dqn.pt"))
    log.close()

def run_ppo(args):
    env = SupplyChainEnv(horizon=args.horizon, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, lr=args.lr, device=args.device)
    run_dir = os.path.join("runs", args.experiment)
    os.makedirs(run_dir, exist_ok=True)
    T = args.episodes * env.horizon
    # Simple on-policy rollouts (toy)
    obs_buf, act_buf, rew_buf, logp_buf, val_buf = [], [], [], [], []
    o,_ = env.reset(seed=args.seed)
    ep_len = 0; ep_ret = 0.0
    for t in range(T):
        a, logp = agent.select(o)
        n, r, term, trunc, info = env.step(a)
        obs_buf.append(o); act_buf.append(a); rew_buf.append(r); logp_buf.append([logp])
        o = n; ep_ret += r; ep_len += 1
        if term or trunc:
            # GAE-lambda returns (simple baseline v=0 for brevity)
            adv = np.array(rew_buf, dtype=np.float32)
            ret = adv.copy()
            # update (toy single-batch)
            batch = (
                np.array(obs_buf, dtype=np.float32),
                np.array(act_buf, dtype=np.float32),
                adv[:, None],
                ret,
                np.array(logp_buf, dtype=np.float32),
            )
            agent.update(batch)
            # reset episode
            o,_ = env.reset(seed=args.seed + t)
            obs_buf, act_buf, rew_buf, logp_buf, val_buf = [], [], [], [], []
            ep_len, ep_ret = 0, 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("+experiment", dest="experiment", default="toy_dqn")
    p.add_argument("--algo", choices=["dqn","ppo"], default="dqn")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=52)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--k", type=int, default=11, help="DQN action grid per dim")
    args = p.parse_args()
    set_seed(args.seed)
    if args.algo == "dqn":
        run_dqn(args)
    else:
        run_ppo(args)

if __name__ == "__main__":
    main()
