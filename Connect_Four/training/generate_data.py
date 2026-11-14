import os
import pickle
import random
import numpy as np
from env.game_env import ConnectFourEnv
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent


def generate_expert_dataset(num_games: int = 2000, save_path: str = "logs/expert_data.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = ConnectFourEnv()
    minimax = MinimaxAgent(depth=3, use_alpha_beta=True)
    mcts = MCTSAgent(num_rollouts=200)

    X_states, y_actions = [], []

    for g in range(1, num_games + 1):
        env.reset()
        game_history = []

        while not env.is_terminal():
            legal = env.legal_actions()
            if len(legal) == 1:
                action = legal[0]
            elif random.random() < 0.5:
                action = minimax.select_action(env)
            else:
                action = mcts.select_action(env)

            canonical_state = env.get_canonical_state()
            game_history.append((canonical_state, action, env.current_player))
            env.step(action)

        if env.winner in (1, -1):
            for state, action, player in game_history:
                if player == env.winner:
                    X_states.append(state)
                    y_actions.append(action)
                    X_states.append(np.fliplr(state).copy())
                    y_actions.append(6 - action)

        if g % 200 == 0:
            print(f"Đã mô phỏng {g}/{num_games} ván | Tổng mẫu thu được: {len(X_states)}")

    with open(save_path, "wb") as f:
        pickle.dump((np.array(X_states, dtype=np.float32), np.array(y_actions, dtype=np.int64)), f)
    print(f"Hoàn tất! Đã lưu dữ liệu tại {save_path}")


if __name__ == "__main__":
    generate_expert_dataset(num_games=2000)