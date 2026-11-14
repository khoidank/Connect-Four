import math
import time
import random
import numpy as np
from agents.base_agent import BaseAgent
from agents.nn_agent import NNAgent
from env.game_env import ConnectFourEnv


class AlphaGoNode:
    def __init__(self, env: ConnectFourEnv, parent=None, action: int | None = None, prior: float = 0.0):
        self.env = env
        self.parent = parent
        self.action = action
        self.prior = prior
        self.player_who_moved = -env.current_player
        self.children: list[AlphaGoNode] = []
        self.visits = 0
        self.wins = 0.0

    def best_child_puct(self, c_puct: float) -> "AlphaGoNode":
        sqrt_parent = math.sqrt(max(1, self.visits))
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits if child.visits > 0 else 0.0)
            + c_puct * child.prior * (sqrt_parent / (1 + child.visits)),
        )


class AlphaGoAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "checkpoints/policy_rl.pt",
        num_rollouts: int = 400,
        time_limit: float | None = None,
        c_puct: float = 1.5,
        seed: int | None = None,
    ):
        budget = f"{time_limit}s" if time_limit else f"{num_rollouts} rollouts"
        super().__init__(name=f"Level 7 - Mixed MCTS / AlphaGo ({budget})")
        self.policy_net = NNAgent(model_path=model_path)
        self.num_rollouts = num_rollouts
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.rng = random.Random(seed)

    def _expand(self, node: AlphaGoNode):
        probs = self.policy_net.get_action_probs(node.env)
        for action in node.env.legal_actions():
            next_env = node.env.clone()
            next_env.step(action)
            node.children.append(
                AlphaGoNode(next_env, parent=node, action=action, prior=float(probs[action]))
            )

    def _rollout(self, env: ConnectFourEnv) -> int | None:
        sim_env = env.clone()
        while not sim_env.is_terminal():
            sim_env.step(self.rng.choice(sim_env.legal_actions()))
        return sim_env.winner

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if len(legal) == 1:
            return legal[0]

        root = AlphaGoNode(env.clone())
        self._expand(root)

        start_time = time.perf_counter()
        rollouts_done = 0

        while True:
            if self.time_limit is not None:
                if time.perf_counter() - start_time >= self.time_limit:
                    break
            elif rollouts_done >= self.num_rollouts:
                break

            node = root
            while node.children and not node.env.is_terminal():
                node = node.best_child_puct(self.c_puct)

            if not node.env.is_terminal() and node.visits > 0:
                self._expand(node)
                node = node.best_child_puct(self.c_puct)

            winner = self._rollout(node.env)

            while node is not None:
                node.visits += 1
                if winner == 0:
                    node.wins += 0.5
                elif winner == node.player_who_moved:
                    node.wins += 1.0
                node = node.parent

            rollouts_done += 1

        best_child = max(root.children, key=lambda c: c.visits)
        self.metrics = {
            "rollouts_executed": rollouts_done,
            "best_win_rate": round(best_child.wins / max(1, best_child.visits), 4),
            "prior_prob": round(best_child.prior, 4),
        }
        return best_child.action