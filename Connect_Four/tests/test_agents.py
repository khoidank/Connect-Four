from env.game_env import ConnectFourEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent


def test_all_agents_return_legal_moves():
    env = ConnectFourEnv()
    agents = [
        RandomAgent(seed=42),
        HeuristicAgent(seed=42),
        MinimaxAgent(depth=3, use_alpha_beta=True),
        MCTSAgent(num_rollouts=100, seed=42),
    ]
    for agent in agents:
        env.reset()
        action = agent.select_action(env)
        assert action in env.legal_actions(), f"{agent.name} trả về nước đi sai!"


def test_alpha_beta_prunes_nodes():
    env = ConnectFourEnv()
    env.step(3)
    pure = MinimaxAgent(depth=3, use_alpha_beta=False)
    ab = MinimaxAgent(depth=3, use_alpha_beta=True)

    pure.select_action(env)
    ab.select_action(env)

    assert ab.get_metrics()["nodes_expanded"] < pure.get_metrics()["nodes_expanded"]