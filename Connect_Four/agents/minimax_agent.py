import math
import numpy as np
from config import ROWS, COLS, CONNECT_N, EMPTY
from agents.base_agent import BaseAgent
from env.game_env import ConnectFourEnv


class MinimaxAgent(BaseAgent):
    def __init__(self, depth: int = 4, use_alpha_beta: bool = True):
        mode = "AlphaBeta" if use_alpha_beta else "PureMinimax"
        super().__init__(name=f"Level 3 - {mode} (d={depth})")
        self.depth = depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_expanded = 0
        self.move_order = [3, 2, 4, 1, 5, 0, 6]

    def _evaluate_window(self, window: np.ndarray, player: int) -> float:
        opp = -player
        p_count = np.count_nonzero(window == player)
        o_count = np.count_nonzero(window == opp)
        e_count = np.count_nonzero(window == EMPTY)

        if p_count == 4:
            return 10000.0
        elif p_count == 3 and e_count == 1:
            return 10.0
        elif p_count == 2 and e_count == 2:
            return 3.0

        if o_count == 3 and e_count == 1:
            return -12.0
        elif o_count == 2 and e_count == 2:
            return -3.0
        return 0.0

    def _evaluate_board(self, grid: np.ndarray, player: int) -> float:
        score = 0.0
        center_col = grid[:, COLS // 2]
        score += np.count_nonzero(center_col == player) * 4.0

        for r in range(ROWS):
            for c in range(COLS - CONNECT_N + 1):
                score += self._evaluate_window(grid[r, c:c + CONNECT_N], player)

        for c in range(COLS):
            for r in range(ROWS - CONNECT_N + 1):
                score += self._evaluate_window(grid[r:r + CONNECT_N, c], player)

        for r in range(ROWS - CONNECT_N + 1):
            for c in range(COLS - CONNECT_N + 1):
                diag_main = np.array([grid[r + i, c + i] for i in range(CONNECT_N)])
                diag_anti = np.array([grid[r + 3 - i, c + i] for i in range(CONNECT_N)])
                score += self._evaluate_window(diag_main, player)
                score += self._evaluate_window(diag_anti, player)

        return score

    def _ordered_actions(self, env: ConnectFourEnv) -> list[int]:
        legal = set(env.legal_actions())
        return [c for c in self.move_order if c in legal]

    def _minimax(
        self, env: ConnectFourEnv, depth: int, alpha: float, beta: float,
        maximizing: bool, root_player: int
    ) -> tuple[int | None, float]:
        self.nodes_expanded += 1

        if env.is_terminal():
            if env.winner == root_player:
                return None, 100000.0 + depth * 100
            elif env.winner == -root_player:
                return None, -100000.0 - depth * 100
            else:
                return None, 0.0

        if depth == 0:
            return None, self._evaluate_board(env.board.grid, root_player)

        actions = self._ordered_actions(env)
        best_col = actions[0]

        if maximizing:
            max_eval = -math.inf
            for col in actions:
                child_env = env.clone()
                child_env.step(col)
                _, eval_score = self._minimax(
                    child_env, depth - 1, alpha, beta, False, root_player
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return best_col, max_eval
        else:
            min_eval = math.inf
            for col in actions:
                child_env = env.clone()
                child_env.step(col)
                _, eval_score = self._minimax(
                    child_env, depth - 1, alpha, beta, True, root_player
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return best_col, min_eval

    def select_action(self, env: ConnectFourEnv) -> int:
        self.nodes_expanded = 0
        legal = env.legal_actions()
        if len(legal) == 1:
            self.metrics = {"nodes_expanded": 1, "best_score": 0.0}
            return legal[0]

        best_col, best_score = self._minimax(
            env, self.depth, -math.inf, math.inf, True, env.current_player
        )
        self.metrics = {
            "nodes_expanded": self.nodes_expanded,
            "best_score": best_score,
        }
        return best_col