import argparse, os, json, csv
import numpy as np
import torch
from rlsupplychain.envs.supply_chain_env import SupplyChainEnv
from rlsupplychain.policies.dqn import DQNAgent, DiscreteWrapper
from rlsupplychain.policies.classic import base_stock_policy, sS_policy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--policy", choices=["dqn","base","sS"], default="dqn")
    args = p.parse_args()

    env = SupplyChainEnv(horizon=52, seed=123)
    obs_dim = env.observation_space.shape[0]
    act = DiscreteWrapper()
    if args.policy == "dqn":
        agent = DQNAgent(obs_dim, act)
        agent.q.load_state_dict(torch.load(os.path.join(args.run_dir, "dqn.pt"), map_location="cpu"))
    with open(os.path.join(args.run_dir, "eval.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","return"])
        writer.writeheader()
        for ep in range(args.episodes):
            o,_ = env.reset(seed=123+ep)
            ep_ret = 0.0
            for t in range(env.horizon):
                if args.policy == "dqn":
                    a = act.decode(agent.select(o))
                elif args.policy == "base":
                    a = base_stock_policy(o)
                else:
                    a = sS_policy(o)
                o, r, term, trunc, info = env.step(a)
                ep_ret += r
                if term or trunc: break
            writer.writerow({"episode": ep, "return": ep_ret})
    print("Saved:", os.path.join(args.run_dir, "eval.csv"))

if __name__ == "__main__":
    main()
