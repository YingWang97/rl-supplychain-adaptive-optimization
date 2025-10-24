from rlsupplychain.envs.supply_chain_env import SupplyChainEnv

def test_env_reset_and_step():
    env = SupplyChainEnv(horizon=5, seed=7)
    o,_ = env.reset()
    assert o.shape[0] >= 2
    n, r, term, trunc, info = env.step([0,0])
    assert isinstance(r, float)
    assert "cost_breakdown" in info
