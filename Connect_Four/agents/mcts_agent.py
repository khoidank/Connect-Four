import math
import time
import random
from agents.base_agent import BaseAgent
from env.game_env import ConnectFourEnv


class MCTSNode:
    def __init__(self, env: ConnectFourEnv, parent: "MCTSNode | None" = None, action: int | None = None):
        self.env = env
        self.parent = parent
        self.action = action

        self.player_who_moved = -env.current_player
        self.children: list[MCTSNode] = []
        self.untried_actions: list[int] = env.legal_actions()
        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child_uct(self, c_param: float) -> "MCTSNode":
        log_parent_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits)
            + c_param * math.sqrt(log_parent_visits / child.visits),
        )


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        num_rollouts: int = 1000,
        time_limit: float | None = None,
        c_param: float = 1.414,
        seed: int | None = None,
    ):
        budget_str = f"{time_limit}s" if time_limit else f"{num_rollouts} rollouts"
        super().__init__(name=f"Level 4 - MCTS ({budget_str})")
        self.num_rollouts = num_rollouts
        self.time_limit = time_limit
        self.c_param = c_param
        self.rng = random.Random(seed)

    def _rollout(self, env: ConnectFourEnv) -> int | None:
        sim_env = env.clone()
        while not sim_env.is_terminal():
            actions = sim_env.legal_actions()
            sim_env.step(self.rng.choice(actions))
        return sim_env.winner

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env.legal_actions()
        if len(legal) == 1:
            self.metrics = {"rollouts_executed": 0}
            return legal[0]

        root = MCTSNode(env.clone())
        start_time = time.perf_counter()
        rollouts_done = 0

        while True:
            if self.time_limit is not None:
                if time.perf_counter() - start_time >= self.time_limit:
                    break
            elif rollouts_done >= self.num_rollouts:
                break

            node = root

            while not node.env.is_terminal() and node.is_fully_expanded():
                node = node.best_child_uct(self.c_param)

            if not node.env.is_terminal() and not node.is_fully_expanded():
                action = self.rng.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_env = node.env.clone()
                next_env.step(action)
                child_node = MCTSNode(next_env, parent=node, action=action)
                node.children.append(child_node)
                node = child_node

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
            "best_win_rate": round(best_child.wins / best_child.visits, 4),
        }
        return best_child.action