import numpy as np

def base_stock_policy(obs, S=80):
    # order up to S ignoring expedite
    on_hand = obs[0]
    order = max(0, S - int(on_hand))
    return np.array([order, 0], dtype=np.float32)

def sS_policy(obs, s=30, S=90):
    on_hand = obs[0]
    if on_hand < s:
        return np.array([max(0, S - int(on_hand)), 0], dtype=np.float32)
    return np.array([0, 0], dtype=np.float32)
